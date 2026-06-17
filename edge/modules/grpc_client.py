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


logger = logging.getLogger(__name__)


class DroneGRPCClient:
    def __init__(
        self,
        config: Dict[str, Any],
        drone_id: str,
        on_command: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self._server_addr = config.get("server_address", "localhost:50051")
        self._use_tls = config.get("use_tls", False)
        self._tls_cert_path = config.get("tls_cert_path", "")
        self._reconnect_interval = config.get("reconnect_interval", 5)
        self._max_reconnect = config.get("max_reconnect_attempts", 20)
        self._ack_timeout = config.get("ack_timeout", 2.0)
        self._send_queue_size = config.get("send_queue_size", 1000)

        self._drone_id = drone_id
        self._on_command = on_command

        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[pb2_grpc.DroneDetectionServiceStub] = None
        self._stream_call = None

        self._send_queue: "queue.Queue[pb2.FrameDetection]" = queue.Queue(
            maxsize=self._send_queue_size
        )
        self._ack_queue: "queue.Queue[pb2.ServerAck]" = queue.Queue(
            maxsize=self._send_queue_size
        )
        self._command_queue: "queue.Queue[pb2.ServerCommand]" = queue.Queue()

        self._running = False
        self._connected = False
        self._stop_event = threading.Event()

        self._send_thread: Optional[threading.Thread] = None
        self._recv_thread: Optional[threading.Thread] = None
        self._cmd_thread: Optional[threading.Thread] = None
        self._status_thread: Optional[threading.Thread] = None

        self._stats_lock = threading.Lock()
        self._stats = {
            "frames_sent": 0,
            "bytes_sent": 0,
            "acks_received": 0,
            "reconnects": 0,
            "errors": 0,
        }

    def start(self) -> bool:
        if self._running:
            return True

        self._running = True
        self._stop_event.clear()

        if not self._connect():
            logger.warning("Initial connection failed, will retry in background")

        self._send_thread = threading.Thread(
            target=self._send_loop, daemon=True, name="GRPCSendThread"
        )
        self._recv_thread = threading.Thread(
            target=self._recv_loop, daemon=True, name="GRPCRecvThread"
        )
        self._cmd_thread = threading.Thread(
            target=self._command_dispatch_loop, daemon=True, name="GRPCCmdThread"
        )
        self._status_thread = threading.Thread(
            target=self._status_stream_loop, daemon=True, name="GRPCStatusThread"
        )

        self._send_thread.start()
        self._recv_thread.start()
        self._cmd_thread.start()
        self._status_thread.start()

        logger.info(f"gRPC client started for drone {self._drone_id}")
        return True

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()

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
        ]:
            if t and t.is_alive():
                t.join(timeout=3.0)

        if self._channel:
            try:
                self._channel.close()
            except Exception:
                pass
            self._channel = None
            self._stub = None

        logger.info("gRPC client stopped")

    def _connect(self) -> bool:
        try:
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
                creds = grpc.ssl_channel_credentials(root_certificates=cert_data)
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
            logger.info(f"Connected to gRPC server at {self._server_addr}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
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

            logger.info(
                f"Reconnect attempt {attempts}/{self._max_reconnect}"
            )

            if self._connect():
                return

            self._stop_event.wait(self._reconnect_interval)

        logger.error("Max reconnect attempts reached")
        self._running = False

    def _request_generator(self) -> Iterator[pb2.FrameDetection]:
        while self._running and not self._stop_event.is_set():
            try:
                item = self._send_queue.get(timeout=0.5)
                yield item
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Request generator error: {e}")
                break

    def _send_loop(self) -> None:
        logger.info("gRPC send loop started")
        while self._running:
            try:
                if not self._connected:
                    self._reconnect_loop()
                    if not self._connected:
                        break
                    continue

                item = self._send_queue.get(timeout=1.0)
                size = item.ByteSize()

                with self._stats_lock:
                    self._stats["frames_sent"] += 1
                    self._stats["bytes_sent"] += size

            except queue.Empty:
                continue
            except grpc.RpcError as e:
                code = e.code() if hasattr(e, "code") else "UNKNOWN"
                logger.warning(f"gRPC send error [{code}]: {e}")
                with self._stats_lock:
                    self._stats["errors"] += 1
                self._connected = False
                self._stop_event.wait(self._reconnect_interval)
            except Exception as e:
                logger.error(f"Send loop error: {e}")
                traceback.print_exc()
                with self._stats_lock:
                    self._stats["errors"] += 1
                self._connected = False
                self._stop_event.wait(self._reconnect_interval)

        logger.info("gRPC send loop exited")

    def _recv_loop(self) -> None:
        logger.info("gRPC recv loop started")
        while self._running:
            try:
                if not self._connected or self._stream_call is None:
                    time.sleep(0.5)
                    continue

                try:
                    ack = next(self._stream_call)
                    self._ack_queue.put_nowait(ack)
                    with self._stats_lock:
                        self._stats["acks_received"] += 1
                except StopIteration:
                    logger.warning("Stream ended by server")
                    self._connected = False
                    time.sleep(self._reconnect_interval)
                    continue
                except grpc.RpcError as e:
                    code = e.code() if hasattr(e, "code") else "UNKNOWN"
                    if code != grpc.StatusCode.CANCELLED:
                        logger.warning(f"gRPC recv error [{code}]: {e}")
                    self._connected = False
                    time.sleep(self._reconnect_interval)
                    continue

            except Exception as e:
                logger.error(f"Recv loop error: {e}")
                traceback.print_exc()
                self._connected = False
                time.sleep(self._reconnect_interval)

        logger.info("gRPC recv loop exited")

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

                        self._on_command({
                            "command_id": cmd.command_id,
                            "command_type": cmd.command_type,
                            "parameters": params,
                            "timestamp": cmd.timestamp,
                        })
                    except Exception as e:
                        logger.error(f"Command handler error: {e}")
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Command dispatch error: {e}")

        logger.info("Command dispatch loop exited")

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

                time.sleep(5.0)

            except Exception as e:
                logger.error(f"Status stream loop error: {e}")
                time.sleep(5.0)

        logger.info("Status stream loop exited")

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

            try:
                self._send_queue.put_nowait(frame_msg)
                return True
            except queue.Full:
                try:
                    self._send_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._send_queue.put_nowait(frame_msg)
                    return True
                except queue.Full:
                    logger.warning("Send queue is full, dropping frame")
                    return False

        except Exception as e:
            logger.error(f"Failed to send detection: {e}")
            return False

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
                x1=d.x1,
                y1=d.y1,
                x2=d.x2,
                y2=d.y2,
                confidence=d.confidence,
                class_name=d.class_name,
                class_id=d.class_id,
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

        return pb2.DroneStatus(
            drone_id=self._drone_id,
            status="running" if self._connected else "disconnected",
            timestamp=int(time.time() * 1e9),
            battery_level=85.0,
            current_position=pb2.GPSCoordinate(
                latitude=39.9042,
                longitude=116.4074,
                altitude=50.0,
                speed=8.5,
                heading=90.0,
                timestamp=int(time.time() * 1e9),
            ),
            total_frames_processed=stats["frames_sent"],
            total_detections=stats["frames_sent"] * 2,
        )

    def get_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            return dict(self._stats)

    def is_connected(self) -> bool:
        return self._connected

    def is_running(self) -> bool:
        return self._running
