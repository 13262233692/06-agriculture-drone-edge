import os
import sys
import time
import json
import uuid
import queue
import threading
import logging
import traceback
from collections import OrderedDict
from typing import Iterator, Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime, timezone


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import grpc
from concurrent import futures

from generated import drone_service_pb2 as pb2
from generated import drone_service_pb2_grpc as pb2_grpc
from modules.influx_writer import InfluxDBWriter
from modules.mission_scheduler import DynamicMissionScheduler


logger = logging.getLogger(__name__)


class IdempotencyWindow:
    """
    基于 drone_id → 有序字典(frame_id → ack_nonce) 的幂等性去重。
    滑动窗口：每个无人机最多保留最近 WINDOW_SIZE 个已处理 frame_id。
    对窗口内命中的重复 frame，直接返回缓存的 ACK，不重复写入。
    """

    def __init__(self, window_size_per_drone: int = 5000, max_drones: int = 100):
        self._window_size = window_size_per_drone
        self._max_drones = max_drones
        self._windows: "OrderedDict[str, OrderedDict[int, Tuple[bool, str]]]" = OrderedDict()
        self._lock = threading.RLock()

        self._stats_lock = threading.Lock()
        self._duplicates_filtered = 0
        self._new_entries = 0
        self._evictions = 0

    def check_and_mark(
        self, drone_id: str, frame_id: int
    ) -> Tuple[bool, Optional[str]]:
        """
        返回：(是否重复命中, 上次成功时的 ack_nonce/None)
        True = 重复，调用方应跳过写入，直接返回 ACK
        False = 新帧，调用方应继续写入 InfluxDB
        """
        with self._lock:
            win = self._windows.get(drone_id)
            if win is None:
                if len(self._windows) >= self._max_drones:
                    self._windows.popitem(last=False)
                    with self._stats_lock:
                        self._evictions += 1
                win = OrderedDict()
                self._windows[drone_id] = win

            if frame_id in win:
                win.move_to_end(frame_id)
                with self._stats_lock:
                    self._duplicates_filtered += 1
                processed, nonce = win[frame_id]
                return (processed, nonce)

            if len(win) >= self._window_size:
                win.popitem(last=False)
                with self._stats_lock:
                    self._evictions += 1

            win[frame_id] = (False, "")
            self._windows.move_to_end(drone_id)
            with self._stats_lock:
                self._new_entries += 1
            return (False, None)

    def mark_processed(
        self, drone_id: str, frame_id: int, ack_nonce: str
    ) -> None:
        """InfluxDB 写入成功后，把窗口条目标记为已处理，缓存 ack_nonce"""
        with self._lock:
            win = self._windows.get(drone_id)
            if win and frame_id in win:
                win[frame_id] = (True, ack_nonce)
                win.move_to_end(frame_id)

    def get_stats(self) -> Dict[str, int]:
        total_windows = 0
        total_entries = 0
        with self._lock:
            total_windows = len(self._windows)
            total_entries = sum(len(w) for w in self._windows.values())

        with self._stats_lock:
            return {
                "active_drones_tracked": total_windows,
                "total_entries": total_entries,
                "duplicates_filtered": self._duplicates_filtered,
                "new_entries": self._new_entries,
                "window_evictions": self._evictions,
            }

    def clear(self, drone_id: Optional[str] = None) -> None:
        with self._lock:
            if drone_id:
                self._windows.pop(drone_id, None)
            else:
                self._windows.clear()


class DroneDetectionServicer(pb2_grpc.DroneDetectionServiceServicer):
    def __init__(
        self,
        influx_writer: InfluxDBWriter,
        detection_config: Dict[str, Any],
        mission_scheduler: Optional[DynamicMissionScheduler] = None,
        on_frame_received: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self._influx = influx_writer
        self._detection_cfg = detection_config
        self._on_frame_received = on_frame_received
        self._min_confidence = detection_config.get("min_confidence", 0.5)
        self._class_mapping = detection_config.get("class_mapping", {})
        self._mission_scheduler = mission_scheduler

        dedup_cfg = detection_config.get("idempotency", {})
        self._dedup = IdempotencyWindow(
            window_size_per_drone=dedup_cfg.get("window_size_per_drone", 5000),
            max_drones=dedup_cfg.get("max_drones", 100),
        )

        self._drone_sessions: Dict[str, Dict[str, Any]] = {}
        self._sessions_lock = threading.Lock()

        self._command_queues: Dict[str, "queue.Queue[pb2.ServerCommand]"] = {}
        self._command_lock = threading.Lock()

        self._stats_lock = threading.Lock()
        self._global_stats = {
            "total_frames": 0,
            "total_detections": 0,
            "total_bytes": 0,
            "active_drones": set(),
            "frames_per_drone": {},
            "detections_per_drone": {},
            "by_severity": {"mild": 0, "moderate": 0, "severe": 0, "unknown": 0},
            "duplicates_skipped": 0,
            "unique_writes": 0,
        }

        self._frame_log_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "frames",
        )
        os.makedirs(self._frame_log_dir, exist_ok=True)

    def StreamDetections(
        self,
        request_iterator: Iterator[pb2.FrameDetection],
        context: grpc.ServicerContext,
    ) -> Iterator[pb2.ServerAck]:
        peer = context.peer()
        drone_id = "UNKNOWN"

        try:
            for frame in request_iterator:
                try:
                    drone_id = frame.drone_id or "UNKNOWN"
                    frame_id = int(frame.frame_id)

                    self._register_drone(drone_id, peer)

                    if frame_id == -1:
                        ack = pb2.ServerAck(
                            received_frame_id=-1,
                            server_timestamp=int(time.time() * 1e9),
                            success=True,
                            message="HEARTBEAT_OK",
                        )
                        yield ack
                        continue

                    is_dup, cached_nonce = self._dedup.check_and_mark(
                        drone_id, frame_id
                    )

                    if is_dup:
                        with self._stats_lock:
                            self._global_stats["duplicates_skipped"] += 1
                        logger.debug(
                            f"[DEDUP] {drone_id} frame={frame_id} "
                            f"already processed → returning cached ACK"
                        )
                        yield pb2.ServerAck(
                            received_frame_id=frame_id,
                            server_timestamp=int(time.time() * 1e9),
                            success=True,
                            message=f"DEDUP_OK cached={cached_nonce}",
                        )
                        continue

                    result = self._process_frame(frame)

                    ack_nonce = uuid.uuid4().hex[:12]
                    self._dedup.mark_processed(drone_id, frame_id, ack_nonce)

                    if self._on_frame_received:
                        try:
                            self._on_frame_received(result)
                        except Exception as e:
                            logger.error(f"Frame callback error: {e}")

                    if self._mission_scheduler:
                        try:
                            self._mission_scheduler.notify_frame_received(
                                drone_id=drone_id,
                                frame_id=frame_id,
                                detection_count=result.get("detection_count", 0),
                            )
                        except Exception as e:
                            logger.debug(f"Mission scheduler notify failed: {e}")

                    self._log_frame_json(result)

                    with self._stats_lock:
                        self._global_stats["unique_writes"] += 1

                    yield pb2.ServerAck(
                        received_frame_id=frame_id,
                        server_timestamp=int(time.time() * 1e9),
                        success=True,
                        message=(
                            f"OK: {result.get('detection_count', 0)} detections "
                            f"stored nonce={ack_nonce}"
                        ),
                    )

                except Exception as e:
                    logger.error(f"Frame processing error: {e}")
                    traceback.print_exc()
                    yield pb2.ServerAck(
                        received_frame_id=getattr(frame, "frame_id", -1),
                        server_timestamp=int(time.time() * 1e9),
                        success=False,
                        message=str(e),
                    )

        except grpc.RpcError as e:
            code = e.code() if hasattr(e, "code") else "UNKNOWN"
            if code != grpc.StatusCode.CANCELLED:
                logger.warning(f"Stream ended [{code}]: {e} (drone={drone_id})")
        except Exception as e:
            logger.error(f"Stream error for drone {drone_id}: {e}")
            traceback.print_exc()
        finally:
            self._unregister_drone(drone_id)
            logger.info(f"Stream closed for drone {drone_id} ({peer})")

    def StreamStatus(
        self,
        request: pb2.DroneStatus,
        context: grpc.ServicerContext,
    ) -> Iterator[pb2.ServerCommand]:
        drone_id = request.drone_id or "UNKNOWN"
        peer = context.peer()

        logger.info(
            f"Drone status stream opened: {drone_id} "
            f"status={request.status} battery={request.battery_level}% "
            f"chemical={request.chemical_level}% "
            f"peer={peer}"
        )

        self._register_drone(drone_id, peer)

        if self._mission_scheduler:
            try:
                self._mission_scheduler.register_drone(drone_id)
                self._mission_scheduler.update_drone_status({
                    "drone_id": drone_id,
                    "battery_level": request.battery_level,
                    "chemical_level": request.chemical_level,
                    "current_position": {
                        "latitude": request.current_position.latitude,
                        "longitude": request.current_position.longitude,
                        "altitude": request.current_position.altitude,
                    },
                    "home_position": {
                        "latitude": request.home_position.latitude if request.HasField("home_position") else request.current_position.latitude,
                        "longitude": request.home_position.longitude if request.HasField("home_position") else request.current_position.longitude,
                    },
                    "cruise_speed": request.cruise_speed,
                    "spray_rate": request.spray_rate,
                    "current_mission_id": request.current_mission_id,
                })
            except Exception as e:
                logger.error(f"Failed to register drone with scheduler: {e}")

        cmd_queue: "queue.Queue[pb2.ServerCommand]" = queue.Queue()
        with self._command_lock:
            self._command_queues[drone_id] = cmd_queue

        try:
            keepalive_cmd = pb2.ServerCommand(
                command_id=f"keepalive-{uuid.uuid4().hex[:8]}",
                command_type="KEEPALIVE",
                parameters=json.dumps({"interval": 30}),
                timestamp=int(time.time() * 1e9),
            )
            yield keepalive_cmd

            while True:
                try:
                    cmd = cmd_queue.get(timeout=30.0)
                    yield cmd
                except queue.Empty:
                    yield pb2.ServerCommand(
                        command_id=f"ping-{uuid.uuid4().hex[:8]}",
                        command_type="PING",
                        parameters="{}",
                        timestamp=int(time.time() * 1e9),
                    )

        except grpc.RpcError as e:
            code = e.code() if hasattr(e, "code") else "UNKNOWN"
            if code != grpc.StatusCode.CANCELLED:
                logger.debug(f"Status stream ended for {drone_id}: {code}")
        finally:
            with self._command_lock:
                self._command_queues.pop(drone_id, None)
            self._unregister_drone(drone_id)
            if self._mission_scheduler:
                try:
                    self._mission_scheduler.unregister_drone(drone_id)
                except Exception as e:
                    logger.debug(f"Failed to unregister drone from scheduler: {e}")

    def send_command(
        self,
        drone_id: str,
        command_type: str,
        parameters: Dict[str, Any] = None,
        mission_plan: Dict[str, Any] = None,
        heatmap: Dict[str, Any] = None,
    ) -> bool:
        cmd = pb2.ServerCommand(
            command_id=f"cmd-{uuid.uuid4().hex[:12]}",
            command_type=command_type,
            parameters=json.dumps(parameters or {}),
            timestamp=int(time.time() * 1e9),
        )

        if mission_plan:
            cmd.mission_plan.CopyFrom(self._build_mission_plan_proto(mission_plan))

        if heatmap:
            cmd.heatmap.CopyFrom(self._build_heatmap_proto(heatmap))

        with self._command_lock:
            q = self._command_queues.get(drone_id)
            if q is None:
                logger.warning(f"Cannot send command: drone {drone_id} not connected")
                return False
            try:
                q.put_nowait(cmd)
                logger.info(
                    f"Sent command {command_type} to {drone_id} (id={cmd.command_id})"
                )
                return True
            except queue.Full:
                logger.warning(f"Command queue full for drone {drone_id}")
                return False

    def _build_mission_plan_proto(self, mission_plan: Dict[str, Any]) -> pb2.MissionPlan:
        """将 mission plan 字典转换为 protobuf 消息"""
        waypoint_protos = []
        for wp in mission_plan.get("waypoints", []):
            wp_proto = pb2.Waypoint(
                waypoint_id=int(wp.get("waypoint_id", 0)),
                latitude=float(wp.get("latitude", 0.0)),
                longitude=float(wp.get("longitude", 0.0)),
                altitude=float(wp.get("altitude", 50.0)),
                speed=float(wp.get("speed", 8.0)),
                action=str(wp.get("action", "SPRAY")),
                spray_density=float(wp.get("spray_density", 1.0)),
                estimated_arrival=int(wp.get("estimated_arrival", 0)),
            )
            waypoint_protos.append(wp_proto)

        return pb2.MissionPlan(
            mission_id=str(mission_plan.get("mission_id", "")),
            mission_type=str(mission_plan.get("mission_type", "")),
            description=str(mission_plan.get("description", "")),
            created_at=int(mission_plan.get("created_at", 0)),
            waypoints=waypoint_protos,
            estimated_distance_m=float(mission_plan.get("estimated_distance_m", 0.0)),
            estimated_duration_s=float(mission_plan.get("estimated_duration_s", 0.0)),
            estimated_battery_used_pct=float(mission_plan.get("estimated_battery_used_pct", 0.0)),
            estimated_chemical_used_pct=float(mission_plan.get("estimated_chemical_used_pct", 0.0)),
            priority=str(mission_plan.get("priority", "normal")),
        )

    def _build_heatmap_proto(self, heatmap: Dict[str, Any]) -> pb2.HeatmapData:
        """将 heatmap 字典转换为 protobuf 消息"""
        cell_protos = []
        for cell in heatmap.get("cells", [])[:200]:
            cell_proto = pb2.HeatmapCell(
                latitude=float(cell.get("latitude", 0.0)),
                longitude=float(cell.get("longitude", 0.0)),
                density=float(cell.get("density", 0.0)),
                severity_score=float(cell.get("severity_score", 0.0)),
            )
            cell_protos.append(cell_proto)

        return pb2.HeatmapData(
            field_id=str(heatmap.get("field_id", "")),
            generated_at=int(heatmap.get("generated_at", 0)),
            grid_size=int(heatmap.get("grid_size", 0)),
            cells=cell_protos,
            min_density=float(heatmap.get("min_density", 0.0)),
            max_density=float(heatmap.get("max_density", 0.0)),
            avg_severity=float(heatmap.get("avg_severity", 0.0)),
        )

    def _process_frame(self, frame: pb2.FrameDetection) -> Dict[str, Any]:
        gps = frame.gps
        gps_dict = {
            "latitude": gps.latitude,
            "longitude": gps.longitude,
            "altitude": gps.altitude,
            "speed": gps.speed,
            "heading": gps.heading,
            "timestamp": gps.timestamp,
            "satellites": 0,
            "hdop": 0.0,
        }

        detections: List[Dict[str, Any]] = []
        for d in frame.detections:
            if d.confidence < self._min_confidence:
                continue

            cls_info = self._class_mapping.get(str(d.class_id), {})
            effective_class = cls_info.get("name", d.class_name)
            effective_severity = cls_info.get("severity_level", d.severity_level)

            detection = {
                "bbox": {
                    "x1": d.x1,
                    "y1": d.y1,
                    "x2": d.x2,
                    "y2": d.y2,
                },
                "confidence": d.confidence,
                "class_id": d.class_id,
                "class_name": effective_class,
                "severity_score": d.severity_score,
                "severity_level": effective_severity,
            }
            detections.append(detection)

            sev = effective_severity or "unknown"
            with self._stats_lock:
                self._global_stats["by_severity"][sev] = (
                    self._global_stats["by_severity"].get(sev, 0) + 1
                )

        drone_id = frame.drone_id or "UNKNOWN"
        frame_data = {
            "drone_id": drone_id,
            "frame_id": frame.frame_id,
            "timestamp": frame.timestamp,
            "datetime": datetime.fromtimestamp(
                frame.timestamp / 1e9, tz=timezone.utc
            ).isoformat(),
            "gps": gps_dict,
            "frame_width": frame.frame_width,
            "frame_height": frame.frame_height,
            "multispectral_band": frame.multispectral_band,
            "inference_latency_ms": frame.inference_latency_ms,
            "detections": detections,
            "detection_count": len(detections),
            "received_at": int(time.time() * 1e9),
        }

        self._influx.write_detection_frame(
            drone_id=drone_id,
            frame_id=frame.frame_id,
            timestamp_ns=frame.timestamp,
            gps_data=gps_dict,
            detections=detections,
            frame_width=frame.frame_width,
            frame_height=frame.frame_height,
            multispectral_band=frame.multispectral_band,
            inference_latency_ms=frame.inference_latency_ms,
        )

        with self._stats_lock:
            self._global_stats["total_frames"] += 1
            self._global_stats["total_detections"] += len(detections)
            self._global_stats["total_bytes"] += frame.ByteSize()
            self._global_stats["frames_per_drone"][drone_id] = (
                self._global_stats["frames_per_drone"].get(drone_id, 0) + 1
            )
            self._global_stats["detections_per_drone"][drone_id] = (
                self._global_stats["detections_per_drone"].get(drone_id, 0)
                + len(detections)
            )

        return frame_data

    def _log_frame_json(self, frame_data: Dict[str, Any]) -> None:
        try:
            drone_id = frame_data["drone_id"]
            ts = frame_data["timestamp"]
            date_dir = datetime.fromtimestamp(ts / 1e9).strftime("%Y%m%d")
            dir_path = os.path.join(self._frame_log_dir, date_dir)
            os.makedirs(dir_path, exist_ok=True)

            fname = (
                f"{drone_id}_{frame_data['frame_id']}_"
                f"{int(ts)}.json"
            )
            fpath = os.path.join(dir_path, fname)

            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(frame_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.debug(f"Frame JSON log failed: {e}")

    def _register_drone(self, drone_id: str, peer: str) -> None:
        with self._sessions_lock:
            self._drone_sessions[drone_id] = {
                "peer": peer,
                "connected_at": time.time(),
                "last_seen": time.time(),
            }
        with self._stats_lock:
            self._global_stats["active_drones"].add(drone_id)

        logger.info(f"Drone registered: {drone_id} from {peer}")

    def _unregister_drone(self, drone_id: str) -> None:
        with self._sessions_lock:
            self._drone_sessions.pop(drone_id, None)
        with self._stats_lock:
            self._global_stats["active_drones"].discard(drone_id)

    def get_active_drones(self) -> List[Dict[str, Any]]:
        with self._sessions_lock:
            return [
                {
                    "drone_id": did,
                    **info,
                    "connected_seconds": time.time() - info.get("connected_at", 0),
                }
                for did, info in self._drone_sessions.items()
            ]

    def get_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            base = {
                "total_frames": self._global_stats["total_frames"],
                "total_detections": self._global_stats["total_detections"],
                "total_bytes_mb": round(
                    self._global_stats["total_bytes"] / (1024 * 1024), 2
                ),
                "active_drones": sorted(self._global_stats["active_drones"]),
                "active_drone_count": len(self._global_stats["active_drones"]),
                "frames_per_drone": dict(
                    sorted(self._global_stats["frames_per_drone"].items())
                ),
                "detections_per_drone": dict(
                    sorted(self._global_stats["detections_per_drone"].items())
                ),
                "by_severity": dict(self._global_stats["by_severity"]),
                "duplicates_skipped": self._global_stats["duplicates_skipped"],
                "unique_writes": self._global_stats["unique_writes"],
                "dedup_effectiveness": (
                    round(
                        100
                        * self._global_stats["duplicates_skipped"]
                        / max(
                            1,
                            self._global_stats["duplicates_skipped"]
                            + self._global_stats["unique_writes"],
                        ),
                        2,
                    )
                ),
            }
        base["dedup"] = self._dedup.get_stats()
        return base


class CloudServer:
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self._config = config
        self._logger = logger

        server_cfg = config.get("server", {})
        self._host = server_cfg.get("host", "0.0.0.0")
        self._port = server_cfg.get("port", 50051)
        self._max_workers = server_cfg.get("max_workers", 32)
        self._use_tls = server_cfg.get("use_tls", False)
        self._tls_cert = server_cfg.get("tls_cert_path", "")
        self._tls_key = server_cfg.get("tls_key_path", "")
        self._max_streams = server_cfg.get("max_concurrent_streams", 100)

        self._influx = InfluxDBWriter(config.get("influxdb", {}))
        self._mission_scheduler: Optional[DynamicMissionScheduler] = None
        self._servicer: Optional[DroneDetectionServicer] = None
        self._grpc_server: Optional[grpc.Server] = None

        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        self._logger.info("=" * 60)
        self._logger.info("  Cloud Server Starting (Reliable Mode + Dedup)")
        self._logger.info(f"  gRPC: {self._host}:{self._port}")
        self._logger.info(f"  Timestamp: {datetime.now().isoformat()}")
        self._logger.info("=" * 60)

        try:
            self._influx.start()

            self._mission_scheduler = DynamicMissionScheduler(
                config=self._config,
                send_command_fn=self.send_command,
            )
            self._mission_scheduler.start()
            logger.info("Mission scheduler initialized")

            self._servicer = DroneDetectionServicer(
                influx_writer=self._influx,
                detection_config=self._config.get("detection", {}),
                mission_scheduler=self._mission_scheduler,
                on_frame_received=self._on_frame_callback,
            )

            server_opts = [
                ("grpc.max_send_message_length", 1024 * 1024 * 4),
                ("grpc.max_receive_message_length", 1024 * 1024 * 16),
                (
                    "grpc.max_concurrent_streams",
                    self._max_streams,
                ),
                ("grpc.keepalive_time_ms", 15000),
                ("grpc.keepalive_timeout_ms", 10000),
                ("grpc.keepalive_permit_without_calls", True),
                ("grpc.http2.max_pings_without_data", 0),
                ("grpc.http2.min_ping_interval_without_data_ms", 5000),
            ]

            self._grpc_server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=self._max_workers),
                options=server_opts,
            )

            pb2_grpc.add_DroneDetectionServiceServicer_to_server(
                self._servicer, self._grpc_server
            )

            bind_addr = f"{self._host}:{self._port}"
            if self._use_tls and os.path.exists(self._tls_cert):
                with open(self._tls_cert, "rb") as f:
                    cert = f.read()
                with open(self._tls_key, "rb") as f:
                    key = f.read()
                creds = grpc.ssl_server_credentials([(key, cert)])
                self._grpc_server.add_secure_port(bind_addr, creds)
                self._logger.info(f"Secure gRPC server on {bind_addr}")
            else:
                self._grpc_server.add_insecure_port(bind_addr)
                self._logger.info(f"Insecure gRPC server on {bind_addr}")

            self._grpc_server.start()
            self._running = True

            self._monitor_thread = threading.Thread(
                target=self._monitor_loop, daemon=True, name="MonitorThread"
            )
            self._monitor_thread.start()

            self._logger.info("Cloud server (reliable) started successfully")
            return True

        except Exception as e:
            self._logger.error(f"Cloud server start failed: {e}")
            traceback.print_exc()
            self.stop()
            return False

    def stop(self) -> None:
        if not self._running:
            return

        self._logger.info("Shutting down cloud server...")
        self._running = False

        if self._grpc_server:
            try:
                self._grpc_server.stop(grace=5.0).wait(timeout=10.0)
            except Exception:
                pass
            self._grpc_server = None

        self._influx.stop()

        if self._mission_scheduler:
            try:
                self._mission_scheduler.stop()
            except Exception:
                pass

        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)

        self._logger.info("Cloud server stopped")

    def wait(self) -> None:
        try:
            while self._running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.stop()

    def _on_frame_callback(self, frame_data: Dict[str, Any]) -> None:
        if frame_data["detection_count"] > 0:
            sev_summary = {}
            for d in frame_data["detections"]:
                lvl = d["severity_level"]
                sev_summary[lvl] = sev_summary.get(lvl, 0) + 1

            self._logger.info(
                f"[{frame_data['drone_id']}] "
                f"Frame {frame_data['frame_id']}: "
                f"{frame_data['detection_count']} detections "
                f"[{', '.join(f'{k}={v}' for k, v in sev_summary.items())}] "
                f"@ ({frame_data['gps']['latitude']:.5f}, "
                f"{frame_data['gps']['longitude']:.5f}) "
                f"inf={frame_data['inference_latency_ms']:.1f}ms"
            )

    def _monitor_loop(self) -> None:
        self._logger.info("Server monitor loop started")
        last_report = time.time()

        while self._running:
            try:
                time.sleep(30.0)
                if not self._running:
                    break

                now = time.time()
                interval = now - last_report
                last_report = now

                serv_stats = self._servicer.get_stats() if self._servicer else {}
                influx_stats = self._influx.get_stats()

                active = serv_stats.get("active_drones", [])
                total_frames = serv_stats.get("total_frames", 0)
                total_det = serv_stats.get("total_detections", 0)
                by_sev = serv_stats.get("by_severity", {})
                dup = serv_stats.get("duplicates_skipped", 0)
                unique = serv_stats.get("unique_writes", 0)
                dedup_stats = serv_stats.get("dedup", {})

                fps = total_frames / interval if interval > 0 else 0

                self._logger.info(
                    "\n" + "=" * 60 + "\n"
                    f"  SERVER STATUS (Reliable Mode)\n"
                    + "-" * 60 + "\n"
                    f"  Active Drones:      {len(active)} {active}\n"
                    f"  Total Frames:       {total_frames}\n"
                    f"  Total Detections:   {total_det}\n"
                    f"  Process Rate:       {fps:.1f} frames/s\n"
                    f"  Data Received:      {serv_stats.get('total_bytes_mb', 0)} MB\n"
                    f"  Severity Breakdown: {by_sev}\n"
                    + "-" * 60 + "\n"
                    f"  [RELIABILITY]\n"
                    f"  Unique Writes:      {unique}\n"
                    f"  Duplicates Skipped: {dup}\n"
                    f"  Dedup Effectiveness:{serv_stats.get('dedup_effectiveness', 0)}%\n"
                    f"  Dedup Window Size:  {dedup_stats.get('total_entries', 0)} "
                    f"({dedup_stats.get('active_drones_tracked', 0)} drones tracked)\n"
                    f"  InfluxDB:           connected={influx_stats.get('connected', False)} "
                    f"written={influx_stats.get('points_written', 0)} "
                    f"errors={influx_stats.get('write_errors', 0)}\n"
                    + "=" * 60
                )

            except Exception as e:
                self._logger.error(f"Monitor loop error: {e}")

        self._logger.info("Server monitor loop exited")

    def send_command(self, drone_id: str, command_type: str, params=None) -> bool:
        if self._servicer:
            return self._servicer.send_command(drone_id, command_type, params)
        return False

    def get_stats(self) -> Dict[str, Any]:
        return {
            "server": {
                "running": self._running,
                "host": self._host,
                "port": self._port,
            },
            "servicer": self._servicer.get_stats() if self._servicer else {},
            "influxdb": self._influx.get_stats(),
            "mission_scheduler": (
                self._mission_scheduler.get_stats() if self._mission_scheduler else {}
            ),
        }

    @property
    def running(self) -> bool:
        return self._running
