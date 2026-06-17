import os
import sys
import time
import json
import queue
import threading
import logging
import traceback
from typing import Iterator, Optional, Dict, Any, List, Callable
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import grpc

from generated import drone_service_pb2 as pb2
from generated import drone_service_pb2_grpc as pb2_grpc
from modules.gps import GPSData
from modules.detector import Detection
from modules.persistence_queue import SQLitePersistenceQueue
from modules.reliable_transport import (
    ReliableGRPCTransport,
    ReliableTransportConfig,
)


logger = logging.getLogger(__name__)


class DroneGRPCClient:
    """
    gRPC 客户端（增强版）：
      - 调用 send_detection → 同步写入 SQLite 持久化队列（保证不丢）
      - ReliableTransport 从队列拉取 → 发 gRPC 双向流
      - 收到 ServerAck → ReliableTransport 命中 inflight → 标记 ACKED
      - 超时未 ACK → 指数退避重传（500ms → 60s 封顶，最多 50 次）
      - 断网→恢复→自动从 SQLite 补传全部 PENDING
      - 任务调度：接收云端下发的补喷任务，更新航点路径
    """

    def __init__(
        self,
        config: Dict[str, Any],
        drone_id: str,
        on_command: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_mission_update: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self._server_addr = config.get("server_address", "localhost:50051")
        self._use_tls = config.get("use_tls", False)
        self._tls_cert_path = config.get("tls_cert_path", "")
        self._reconnect_interval = config.get("reconnect_interval", 5)
        self._max_reconnect = config.get("max_reconnect_attempts", 20)
        self._send_queue_size = config.get("send_queue_size", 1000)

        reliability_cfg = config.get("reliability", {})
        self._persistence_enabled = reliability_cfg.get(
            "persistence_enabled", True
        )
        self._db_path = reliability_cfg.get(
            "db_path",
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data",
                "msg_queue.sqlite3",
            ),
        )
        self._max_db_size_mb = reliability_cfg.get("max_db_size_mb", 512)
        self._max_retention_days = reliability_cfg.get("max_retention_days", 7)
        self._graceful_shutdown_wait_s = reliability_cfg.get(
            "graceful_shutdown_wait_s", 30
        )

        self._config = config
        self._drone_id = drone_id
        self._on_command = on_command
        self._on_mission_update = on_mission_update

        self._current_mission: Optional[Dict[str, Any]] = None
        self._mission_lock = threading.Lock()
        self._current_waypoint_index: int = 0
        self._mission_execution_status: str = "IDLE"

        self._battery_level_pct: float = 100.0
        self._chemical_level_pct: float = 100.0
        self._home_latitude: float = 0.0
        self._home_longitude: float = 0.0
        self._cruise_speed_m_s: float = 8.0
        self._spray_rate_l_per_s: float = 0.5

        self._current_latitude: float = 0.0
        self._current_longitude: float = 0.0
        self._current_altitude: float = 50.0

        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[pb2_grpc.DroneDetectionServiceStub] = None
        self._stream_call = None
        self._stream_lock = threading.RLock()

        self._wire_send_queue: "queue.Queue[bytes]" = queue.Queue(
            maxsize=self._send_queue_size
        )
        self._command_queue: "queue.Queue[pb2.ServerCommand]" = queue.Queue()

        self._persistence_queue: Optional[SQLitePersistenceQueue] = None
        self._reliable_tx: Optional[ReliableGRPCTransport] = None

        self._running = False
        self._connected = False
        self._stop_event = threading.Event()

        self._send_thread: Optional[threading.Thread] = None
        self._recv_thread: Optional[threading.Thread] = None
        self._cmd_thread: Optional[threading.Thread] = None
        self._status_thread: Optional[threading.Thread] = None
        self._monitor_thread: Optional[threading.Thread] = None

        self._stats_lock = threading.Lock()
        self._stats = {
            "frames_submitted": 0,
            "frames_sent_wire": 0,
            "bytes_sent_wire": 0,
            "acks_received_wire": 0,
            "reconnects": 0,
            "stream_errors": 0,
        }

    def start(self) -> bool:
        if self._running:
            return True

        self._running = True
        self._stop_event.clear()

        try:
            if self._persistence_enabled:
                self._persistence_queue = SQLitePersistenceQueue(
                    db_path=self._db_path,
                    drone_id=self._drone_id,
                    max_size_mb=self._max_db_size_mb,
                    max_age_days=self._max_retention_days,
                )
                reliability_config = ReliableTransportConfig(
                    self._config.get("reliability", {})
                )
                self._reliable_tx = ReliableGRPCTransport(
                    persistence_queue=self._persistence_queue,
                    config=reliability_config,
                    drone_id=self._drone_id,
                    proto_frame_class=pb2.FrameDetection,
                    proto_ack_class=pb2.ServerAck,
                    send_fn=self._send_proto_bytes_to_stream,
                    stats_callback=self._on_reliable_stats,
                )
                self._reliable_tx.start()
                logger.info(
                    f"Persistence queue enabled at {self._db_path}"
                )
            else:
                logger.warning(
                    "Persistence queue DISABLED - data loss possible on network failures"
                )
        except Exception as e:
            logger.error(f"Failed to init reliable transport: {e}")
            traceback.print_exc()

        if not self._connect():
            logger.warning("Initial gRPC connection failed, will retry in background")

        self._send_thread = threading.Thread(
            target=self._send_loop, daemon=True, name="GRPCWireSendThread"
        )
        self._recv_thread = threading.Thread(
            target=self._recv_loop, daemon=True, name="GRPCWireRecvThread"
        )
        self._cmd_thread = threading.Thread(
            target=self._command_dispatch_loop, daemon=True, name="GRPCCmdThread"
        )
        self._status_thread = threading.Thread(
            target=self._status_stream_loop, daemon=True, name="GRPCStatusThread"
        )
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="GRPCMonitorThread"
        )

        for t in [
            self._send_thread,
            self._recv_thread,
            self._cmd_thread,
            self._status_thread,
            self._monitor_thread,
        ]:
            t.start()

        logger.info(
            f"gRPC client (reliable) started for drone {self._drone_id} "
            f"→ {self._server_addr}"
        )
        return True

    def stop(self) -> None:
        if not self._running:
            return

        logger.info(
            f"Stopping gRPC client (graceful_wait={self._graceful_shutdown_wait_s}s)..."
        )
        self._running = False
        self._stop_event.set()

        if self._reliable_tx:
            try:
                drained = self._reliable_tx.wait_until_empty(
                    timeout_s=self._graceful_shutdown_wait_s
                )
                logger.info(
                    f"Graceful drain {'SUCCESS' if drained else 'TIMEOUT'} "
                    f"after {self._graceful_shutdown_wait_s}s"
                )
            except Exception:
                pass

        if self._stream_call:
            try:
                self._stream_call.cancel()
            except Exception:
                pass

        for t in [
            self._send_thread,
            self._recv_thread,
            self._cmd_thread,
            self._status_thread,
            self._monitor_thread,
        ]:
            if t and t.is_alive():
                t.join(timeout=3.0)

        if self._reliable_tx:
            try:
                self._reliable_tx.stop()
            except Exception:
                pass

        if self._persistence_queue:
            try:
                self._persistence_queue.close()
            except Exception:
                pass

        if self._channel:
            try:
                self._channel.close()
            except Exception:
                pass
            self._channel = None
            self._stub = None

        logger.info("gRPC client (reliable) stopped")

    # =================================================================
    # gRPC 连接管理
    # =================================================================

    def _connect(self) -> bool:
        try:
            with self._stream_lock:
                if self._channel:
                    try:
                        self._channel.close()
                    except Exception:
                        pass

                options = [
                    ("grpc.max_send_message_length", 1024 * 1024 * 16),
                    ("grpc.max_receive_message_length", 1024 * 1024 * 4),
                    ("grpc.keepalive_time_ms", 10000),
                    ("grpc.keepalive_timeout_ms", 5000),
                    ("grpc.keepalive_permit_without_calls", True),
                    ("grpc.http2.max_pings_without_data", 0),
                ]

                if self._use_tls and os.path.exists(self._tls_cert_path):
                    with open(self._tls_cert_path, "rb") as f:
                        cert_data = f.read()
                    creds = grpc.ssl_channel_credentials(
                        root_certificates=cert_data
                    )
                    self._channel = grpc.secure_channel(
                        self._server_addr, creds, options=options
                    )
                else:
                    self._channel = grpc.insecure_channel(
                        self._server_addr, options=options
                    )

                self._stub = pb2_grpc.DroneDetectionServiceStub(self._channel)
                self._stream_call = self._stub.StreamDetections(
                    self._request_generator()
                )

            self._connected = True
            logger.info(
                f"Connected to gRPC server {self._server_addr}, stream established"
            )
            return True

        except Exception as e:
            logger.error(f"gRPC connect failed: {e}")
            self._connected = False
            self._channel = None
            self._stub = None
            self._stream_call = None
            return False

    def _reconnect_loop(self) -> None:
        attempts = 0
        while self._running and attempts < self._max_reconnect:
            attempts += 1
            with self._stats_lock:
                self._stats["reconnects"] += 1

            backoff = min(self._reconnect_interval * (2 ** min(attempts - 1, 5)), 60)
            logger.info(
                f"gRPC reconnect attempt {attempts}/{self._max_reconnect} "
                f"(backoff {backoff:.0f}s)"
            )

            if self._connect():
                return

            self._stop_event.wait(backoff)

        logger.error("Max gRPC reconnect attempts reached - will keep trying")
        while self._running:
            self._stop_event.wait(self._reconnect_interval * 4)
            if self._connect():
                return

    # =================================================================
    # ReliableTransport ↔ gRPC 双向流 桥接
    # =================================================================

    def _send_proto_bytes_to_stream(self, proto_bytes: bytes) -> bool:
        """ReliableTransport 调用此函数将字节推到 wire 队列"""
        if proto_bytes is None:
            return True
        if len(proto_bytes) == 0:
            return True
        try:
            self._wire_send_queue.put_nowait(proto_bytes)
            return True
        except queue.Full:
            try:
                self._wire_send_queue.get_nowait()
            except Exception:
                pass
            try:
                self._wire_send_queue.put_nowait(proto_bytes)
                return True
            except queue.Full:
                return False

    def _request_generator(self) -> Iterator[pb2.FrameDetection]:
        while self._running and not self._stop_event.is_set():
            try:
                raw_bytes = self._wire_send_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                msg = pb2.FrameDetection()
                msg.ParseFromString(raw_bytes)
                yield msg
                with self._stats_lock:
                    self._stats["frames_sent_wire"] += 1
                    self._stats["bytes_sent_wire"] += len(raw_bytes)
            except Exception as e:
                logger.warning(f"Failed to deserialize proto bytes: {e}")
                continue

        logger.debug("request_generator exited")

    # =================================================================
    # Wire 层收发循环
    # =================================================================

    def _send_loop(self) -> None:
        """
        注意：实际 wire 发送是在 request_generator 中（作为 gRPC stream 输入迭代器）完成。
        这个线程的职责是：监控连接状态，断线时触发重连。
        """
        logger.info("gRPC wire monitor loop started")
        while self._running:
            try:
                if not self._connected:
                    self._reconnect_loop()
                    if not self._running:
                        break
                    continue

                self._stop_event.wait(0.5)

            except Exception as e:
                logger.error(f"Wire monitor loop error: {e}")
                traceback.print_exc()
                self._connected = False
                self._stop_event.wait(1.0)

        logger.info("gRPC wire monitor loop exited")

    def _recv_loop(self) -> None:
        logger.info("gRPC wire recv loop started")
        while self._running:
            try:
                if not self._connected or self._stream_call is None:
                    time.sleep(0.5)
                    continue

                try:
                    ack = next(self._stream_call)
                    with self._stats_lock:
                        self._stats["acks_received_wire"] += 1

                    if self._reliable_tx:
                        self._reliable_tx.on_ack_received(ack)

                except StopIteration:
                    logger.warning("gRPC stream closed by server")
                    with self._stats_lock:
                        self._stats["stream_errors"] += 1
                    self._connected = False
                    self._stop_event.wait(self._reconnect_interval)
                    continue
                except grpc.RpcError as e:
                    code = e.code() if hasattr(e, "code") else "UNKNOWN"
                    if code != grpc.StatusCode.CANCELLED:
                        logger.warning(
                            f"gRPC stream recv error [{code}]: {e}"
                        )
                    with self._stats_lock:
                        self._stats["stream_errors"] += 1
                    self._connected = False
                    self._stop_event.wait(self._reconnect_interval)
                    continue

            except Exception as e:
                logger.error(f"Wire recv loop error: {e}")
                traceback.print_exc()
                self._connected = False
                time.sleep(1.0)

        logger.info("gRPC wire recv loop exited")

    # =================================================================
    # 命令 & 状态流（独立的 unary-stream RPC）
    # =================================================================

    def _command_dispatch_loop(self) -> None:
        logger.info("Command dispatch loop started")
        while self._running:
            try:
                cmd = self._command_queue.get(timeout=1.0)
                if self._on_command:
                    try:
                        params = {}
                        if cmd.parameters:
                            try:
                                params = json.loads(cmd.parameters)
                            except json.JSONDecodeError:
                                params = {"raw": cmd.parameters}

                        mission_plan = None
                        if cmd.HasField("mission_plan") and cmd.mission_plan and cmd.mission_plan.mission_id:
                            mission_plan = self._parse_mission_plan(cmd.mission_plan)
                            self._handle_mission_update(mission_plan)

                        heatmap = None
                        if cmd.HasField("heatmap") and cmd.heatmap and cmd.heatmap.field_id:
                            heatmap = self._parse_heatmap_data(cmd.heatmap)

                        self._on_command({
                            "command_id": cmd.command_id,
                            "command_type": cmd.command_type,
                            "parameters": params,
                            "timestamp": cmd.timestamp,
                            "mission_plan": mission_plan,
                            "heatmap": heatmap,
                        })
                    except Exception as e:
                        logger.error(f"Command handler error: {e}")
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Command dispatch error: {e}")

        logger.info("Command dispatch loop exited")

    def _parse_mission_plan(self, mp_proto) -> Dict[str, Any]:
        """解析 protobuf MissionPlan 为字典"""
        waypoints = []
        for wp in mp_proto.waypoints:
            waypoints.append({
                "waypoint_id": wp.waypoint_id,
                "latitude": wp.latitude,
                "longitude": wp.longitude,
                "altitude": wp.altitude,
                "speed": wp.speed,
                "action": wp.action,
                "spray_density": wp.spray_density,
                "estimated_arrival": wp.estimated_arrival,
            })

        return {
            "mission_id": mp_proto.mission_id,
            "mission_type": mp_proto.mission_type,
            "description": mp_proto.description,
            "created_at": mp_proto.created_at,
            "waypoints": waypoints,
            "estimated_distance_m": mp_proto.estimated_distance_m,
            "estimated_duration_s": mp_proto.estimated_duration_s,
            "estimated_battery_used_pct": mp_proto.estimated_battery_used_pct,
            "estimated_chemical_used_pct": mp_proto.estimated_chemical_used_pct,
            "priority": mp_proto.priority,
        }

    def _parse_heatmap_data(self, hm_proto) -> Dict[str, Any]:
        """解析 protobuf HeatmapData 为字典"""
        cells = []
        for cell in hm_proto.cells:
            cells.append({
                "latitude": cell.latitude,
                "longitude": cell.longitude,
                "density": cell.density,
                "severity_score": cell.severity_score,
            })

        return {
            "field_id": hm_proto.field_id,
            "generated_at": hm_proto.generated_at,
            "grid_size": hm_proto.grid_size,
            "cells": cells,
            "min_density": hm_proto.min_density,
            "max_density": hm_proto.max_density,
            "avg_severity": hm_proto.avg_severity,
        }

    def _handle_mission_update(self, mission_plan: Dict[str, Any]) -> None:
        """处理任务更新，保存当前任务状态"""
        with self._mission_lock:
            self._current_mission = mission_plan
            self._current_waypoint_index = 0
            self._mission_execution_status = "PENDING"

        mission_id = mission_plan.get("mission_id", "")
        num_waypoints = len(mission_plan.get("waypoints", []))
        logger.info(
            f"Mission updated: {mission_id} "
            f"({num_waypoints} waypoints, "
            f"est_battery={mission_plan.get('estimated_battery_used_pct', 0):.1f}%, "
            f"est_chemical={mission_plan.get('estimated_chemical_used_pct', 0):.1f}%)"
        )

        if self._on_mission_update:
            try:
                self._on_mission_update(mission_plan)
            except Exception as e:
                logger.error(f"Mission update callback error: {e}")

    def get_current_mission(self) -> Optional[Dict[str, Any]]:
        """获取当前任务"""
        with self._mission_lock:
            return dict(self._current_mission) if self._current_mission else None

    def get_mission_waypoints(self) -> List[Dict[str, Any]]:
        """获取当前任务的航点列表"""
        with self._mission_lock:
            if self._current_mission:
                return list(self._current_mission.get("waypoints", []))
            return []

    def set_drone_state(
        self,
        battery_level_pct: Optional[float] = None,
        chemical_level_pct: Optional[float] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        altitude: Optional[float] = None,
        home_latitude: Optional[float] = None,
        home_longitude: Optional[float] = None,
    ) -> None:
        """更新无人机状态（供外部模块调用）"""
        if battery_level_pct is not None:
            self._battery_level_pct = float(battery_level_pct)
        if chemical_level_pct is not None:
            self._chemical_level_pct = float(chemical_level_pct)
        if latitude is not None:
            self._current_latitude = float(latitude)
        if longitude is not None:
            self._current_longitude = float(longitude)
        if altitude is not None:
            self._current_altitude = float(altitude)
        if home_latitude is not None:
            self._home_latitude = float(home_latitude)
        if home_longitude is not None:
            self._home_longitude = float(home_longitude)

    def _status_stream_loop(self) -> None:
        logger.info("Status stream loop started")
        while self._running:
            try:
                if not self._connected or self._stub is None:
                    time.sleep(1.0)
                    continue

                status = self._build_status()
                try:
                    cmd_iterator = self._stub.StreamStatus(status)
                    for cmd in cmd_iterator:
                        self._command_queue.put_nowait(cmd)
                except grpc.RpcError as e:
                    code = e.code() if hasattr(e, "code") else "UNKNOWN"
                    if code != grpc.StatusCode.CANCELLED:
                        logger.debug(f"Status stream error [{code}]: {e}")
                except Exception as e:
                    logger.debug(f"Status stream exception: {e}")

                self._stop_event.wait(5.0)

            except Exception as e:
                logger.error(f"Status stream loop error: {e}")
                time.sleep(5.0)

        logger.info("Status stream loop exited")

    def _monitor_loop(self) -> None:
        logger.info("Reliability monitor loop started")
        last_report = time.time()
        while self._running and not self._stop_event.is_set():
            self._stop_event.wait(15.0)
            if not self._running:
                break

            try:
                rtx_stats = (
                    self._reliable_tx.get_stats() if self._reliable_tx else {}
                )
                queue_counts = rtx_stats.get("queue_counts", {})

                pending = queue_counts.get("pending", 0)
                inflight_queue = queue_counts.get("inflight", 0)
                inflight_rtx = rtx_stats.get("inflight_count", 0)
                oldest_s = rtx_stats.get("oldest_pending_age_s", 0)
                acked = rtx_stats.get("acked_total", 0)
                retries = rtx_stats.get("retries_total", 0)
                retry_exhausted = rtx_stats.get("retry_exhausted", 0)

                if (
                    pending > 0
                    or inflight_queue > 0
                    or inflight_rtx > 0
                    or retries > 0
                    or (time.time() - last_report > 60)
                ):
                    logger.info(
                        "[RELIABILITY] "
                        f"pending={pending} inflight(q)={inflight_queue} "
                        f"inflight(tx)={inflight_rtx} acked={acked} "
                        f"retries={retries} exhausted={retry_exhausted} "
                        f"oldest_age={oldest_s}s "
                        f"grpc_connected={self._connected}"
                    )
                    last_report = time.time()

                if oldest_s > 0 and oldest_s > 60 and self._connected:
                    logger.warning(
                        f"OLDEST pending message is {oldest_s}s old! "
                        f"Network may be congested or server slow"
                    )

            except Exception as e:
                logger.error(f"Reliability monitor error: {e}")

        logger.info("Reliability monitor loop exited")

    # =================================================================
    # 对外主接口
    # =================================================================

    def send_detection(
        self,
        frame_id: int,
        timestamp_ns: int,
        gps_data: GPSData,
        detections: List[Detection],
        frame_width: int,
        frame_height: int,
        multispectral_band: str,
        inference_latency_ms: float,
    ) -> bool:
        """
        【关键】此调用同步持久化到 SQLite，成功返回后消息已保证不丢失（除非磁盘故障）。
        后续的 gRPC 发送、ACK、重试全部由 ReliableTransport 异步处理。
        """
        if not self._running:
            return False

        try:
            frame_msg = self._build_frame_detection(
                frame_id=frame_id,
                timestamp_ns=timestamp_ns,
                gps_data=gps_data,
                detections=detections,
                frame_width=frame_width,
                frame_height=frame_height,
                multispectral_band=multispectral_band,
                inference_latency_ms=inference_latency_ms,
            )

            if self._reliable_tx:
                ok, _msg_id = self._reliable_tx.submit_frame(frame_msg)
                if ok:
                    with self._stats_lock:
                        self._stats["frames_submitted"] += 1
                return ok
            else:
                try:
                    raw = frame_msg.SerializeToString()
                    self._send_proto_bytes_to_stream(raw)
                    with self._stats_lock:
                        self._stats["frames_submitted"] += 1
                    return True
                except Exception:
                    return False

        except Exception as e:
            logger.error(f"Failed to submit detection: {e}")
            traceback.print_exc()
            return False

    # =================================================================
    # 辅助函数
    # =================================================================

    def detection_to_json_dict(
        self,
        frame_id: int,
        timestamp_ns: int,
        gps_data: GPSData,
        detections: List[Detection],
        frame_width: int,
        frame_height: int,
        multispectral_band: str,
        inference_latency_ms: float,
    ) -> Dict[str, Any]:
        return {
            "drone_id": self._drone_id,
            "frame_id": frame_id,
            "timestamp": timestamp_ns,
            "gps": {
                "latitude": gps_data.latitude,
                "longitude": gps_data.longitude,
                "altitude": gps_data.altitude,
                "speed": gps_data.speed,
                "heading": gps_data.heading,
                "satellites": gps_data.satellites,
                "hdop": gps_data.hdop,
            },
            "frame_size": {
                "width": frame_width,
                "height": frame_height,
            },
            "multispectral_band": multispectral_band,
            "inference_latency_ms": inference_latency_ms,
            "detections": [
                {
                    "bbox": {
                        "x1": d.x1,
                        "y1": d.y1,
                        "x2": d.x2,
                        "y2": d.y2,
                    },
                    "confidence": d.confidence,
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "severity_score": d.severity_score,
                    "severity_level": d.severity_level,
                }
                for d in detections
            ],
            "detection_count": len(detections),
        }

    def _build_frame_detection(
        self,
        frame_id: int,
        timestamp_ns: int,
        gps_data: GPSData,
        detections: List[Detection],
        frame_width: int,
        frame_height: int,
        multispectral_band: str,
        inference_latency_ms: float,
    ) -> pb2.FrameDetection:
        gps_msg = pb2.GPSCoordinate(
            latitude=gps_data.latitude,
            longitude=gps_data.longitude,
            altitude=gps_data.altitude,
            speed=gps_data.speed,
            heading=gps_data.heading,
            timestamp=gps_data.timestamp,
        )

        det_msgs = [
            pb2.DetectionBox(
                x1=int(round(d.x1)),
                y1=int(round(d.y1)),
                x2=int(round(d.x2)),
                y2=int(round(d.y2)),
                confidence=d.confidence,
                class_name=d.class_name,
                class_id=int(d.class_id),
                severity_score=d.severity_score,
                severity_level=d.severity_level,
            )
            for d in detections
        ]

        return pb2.FrameDetection(
            drone_id=self._drone_id,
            frame_id=frame_id,
            timestamp=timestamp_ns,
            gps=gps_msg,
            detections=det_msgs,
            frame_width=frame_width,
            frame_height=frame_height,
            multispectral_band=multispectral_band,
            inference_latency_ms=inference_latency_ms,
        )

    def _build_status(self) -> pb2.DroneStatus:
        with self._stats_lock:
            stats = dict(self._stats)

        rtx_stats = (
            self._reliable_tx.get_stats() if self._reliable_tx else {}
        )
        queue_counts = rtx_stats.get("queue_counts", {})

        pending = queue_counts.get("pending", 0) + queue_counts.get("inflight", 0)
        total_det = rtx_stats.get("acked_total", 0) * 3

        with self._mission_lock:
            mission_id = self._current_mission.get("mission_id", "") if self._current_mission else ""

        return pb2.DroneStatus(
            drone_id=self._drone_id,
            status="running" if self._connected else "disconnected",
            timestamp=int(time.time() * 1e9),
            battery_level=self._battery_level_pct,
            chemical_level=self._chemical_level_pct,
            current_position=pb2.GPSCoordinate(
                latitude=self._current_latitude,
                longitude=self._current_longitude,
                altitude=self._current_altitude,
                speed=8.5,
                heading=90.0,
                timestamp=int(time.time() * 1e9),
            ),
            home_position=pb2.GPSCoordinate(
                latitude=self._home_latitude,
                longitude=self._home_longitude,
                altitude=0.0,
                speed=0.0,
                heading=0.0,
                timestamp=int(time.time() * 1e9),
            ),
            total_frames_processed=stats.get("frames_submitted", 0),
            total_detections=total_det,
            cruise_speed=self._cruise_speed_m_s,
            spray_rate=self._spray_rate_l_per_s,
            current_mission_id=mission_id,
        )

    def _on_reliable_stats(self, stats: Dict[str, Any]) -> None:
        pass

    def get_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            wire = dict(self._stats)

        result = {
            "wire": wire,
            "grpc_connected": self._connected,
        }
        if self._reliable_tx:
            result["reliable"] = self._reliable_tx.get_stats()
        if self._persistence_queue:
            result["persistence"] = self._persistence_queue.get_stats()

        return result

    def is_connected(self) -> bool:
        return self._connected

    def is_running(self) -> bool:
        return self._running
