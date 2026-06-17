import os
import sys
import time
import json
import signal
import logging
import threading
import argparse
import traceback
from typing import Dict, Any, Optional
from datetime import datetime

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(system_cfg: Dict[str, Any], drone_id: str) -> logging.Logger:
    log_level = getattr(
        logging, system_cfg.get("log_level", "INFO").upper(), logging.INFO
    )
    log_file = system_cfg.get("log_file", f"./logs/{drone_id}.log")

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger("drone_edge")
    logger.setLevel(log_level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(log_level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


class DroneEdgeApp:
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self._config = config
        self._logger = logger

        drone_cfg = config.get("drone", {})
        self._drone_id = drone_cfg.get("id", "UNKNOWN-DRONE")

        system_cfg = config.get("system", {})
        self._frame_skip = system_cfg.get("frame_skip_interval", 0)
        self._save_images = system_cfg.get("save_detection_images", False)
        self._output_dir = system_cfg.get("detection_output_dir", "./output")
        self._enable_profiling = system_cfg.get("enable_profiling", False)

        self._camera = None
        self._detector = None
        self._gps = None
        self._grpc_client = None

        self._running = False
        self._main_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

        self._stats_lock = threading.Lock()
        self._stats = {
            "start_time": time.time(),
            "frames_processed": 0,
            "frames_skipped": 0,
            "total_detections": 0,
            "avg_inference_ms": 0.0,
            "avg_fps": 0.0,
            "last_report_time": time.time(),
            "last_report_frames": 0,
        }

        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        def handler(signum, frame):
            self._logger.info(f"Received signal {signum}, shutting down...")
            self.stop()

        try:
            signal.signal(signal.SIGINT, handler)
            signal.signal(signal.SIGTERM, handler)
        except (ValueError, OSError):
            pass

    def start(self) -> bool:
        if self._running:
            return True

        self._logger.info("=" * 60)
        self._logger.info(f"  Drone Edge Application Starting")
        self._logger.info(f"  Drone ID: {self._drone_id}")
        self._logger.info(f"  Timestamp: {datetime.now().isoformat()}")
        self._logger.info("=" * 60)

        try:
            from modules.camera import MultispectralCamera
            from modules.detector import DiseaseDetector
            from modules.gps import GPSProvider
            from modules.grpc_client import DroneGRPCClient

            self._camera = MultispectralCamera(self._config.get("camera", {}))
            if not self._camera.start():
                self._logger.error("Failed to start camera")
                return False

            self._detector = DiseaseDetector(self._config.get("inference", {}))
            self._logger.info("Disease detector initialized")

            self._gps = GPSProvider(self._config.get("gps", {}))
            if not self._gps.start():
                self._logger.warning("GPS start failed, will use mock data")

            self._grpc_client = DroneGRPCClient(
                self._config.get("grpc", {}),
                self._drone_id,
                on_command=self._handle_server_command,
            )
            self._grpc_client.start()

            os.makedirs(self._output_dir, exist_ok=True)

            self._running = True
            self._shutdown_event.clear()

            self._main_thread = threading.Thread(
                target=self._main_loop, daemon=True, name="MainLoopThread"
            )
            self._main_thread.start()

            report_thread = threading.Thread(
                target=self._stats_report_loop, daemon=True, name="StatsReportThread"
            )
            report_thread.start()

            self._logger.info("Drone Edge application started successfully")
            return True

        except Exception as e:
            self._logger.error(f"Startup failed: {e}")
            traceback.print_exc()
            self.stop()
            return False

    def stop(self) -> None:
        if not self._running:
            return

        self._logger.info("Shutting down Drone Edge application...")
        self._running = False
        self._shutdown_event.set()

        if self._grpc_client:
            self._grpc_client.stop()
        if self._gps:
            self._gps.stop()
        if self._camera:
            self._camera.stop()

        if self._main_thread and self._main_thread.is_alive():
            self._main_thread.join(timeout=5.0)

        self._print_final_stats()
        self._logger.info("Drone Edge application stopped")

    def _main_loop(self) -> None:
        self._logger.info("Main processing loop started")

        frame_counter = 0

        while self._running and not self._shutdown_event.is_set():
            try:
                frame_item = self._camera.get_frame(timeout=1.0)
                if frame_item is None:
                    self._logger.debug("Waiting for camera frame...")
                    continue

                frame_id, frame, capture_ts = frame_item

                if self._frame_skip > 0:
                    frame_counter += 1
                    if frame_counter % (self._frame_skip + 1) != 0:
                        with self._stats_lock:
                            self._stats["frames_skipped"] += 1
                        continue

                detections, inf_latency = self._detector.detect(frame)

                gps_data = self._gps.get_current()
                frame_h, frame_w = frame.shape[:2]
                band = self._camera.get_active_band()
                ts_ns = int(time.time() * 1e9)

                if detections:
                    detection_json = self._grpc_client.detection_to_json_dict(
                        frame_id=frame_id,
                        timestamp_ns=ts_ns,
                        gps_data=gps_data,
                        detections=detections,
                        frame_width=frame_w,
                        frame_height=frame_h,
                        multispectral_band=band,
                        inference_latency_ms=inf_latency,
                    )
                    self._save_json_snapshot(detection_json, frame_id)

                    if self._save_images:
                        self._save_detected_image(frame, detections, frame_id)

                sent = self._grpc_client.send_detection(
                    frame_id=frame_id,
                    timestamp_ns=ts_ns,
                    gps_data=gps_data,
                    detections=detections,
                    frame_width=frame_w,
                    frame_height=frame_h,
                    multispectral_band=band,
                    inference_latency_ms=inf_latency,
                )

                with self._stats_lock:
                    self._stats["frames_processed"] += 1
                    self._stats["total_detections"] += len(detections)
                    self._update_running_avg("avg_inference_ms", inf_latency)

                    total = self._stats["frames_processed"]
                    elapsed = time.time() - self._stats["start_time"]
                    if elapsed > 0:
                        self._stats["avg_fps"] = total / elapsed

                if len(detections) > 0:
                    severity_summary = {}
                    for d in detections:
                        lvl = d.severity_level
                        severity_summary[lvl] = severity_summary.get(lvl, 0) + 1

                    self._logger.info(
                        f"Frame {frame_id}: {len(detections)} detections "
                        f"[{', '.join(f'{k}={v}' for k,v in severity_summary.items())}] "
                        f"| lat={gps_data.latitude:.6f} lon={gps_data.longitude:.6f} "
                        f"| inf={inf_latency:.1f}ms"
                        + (" | SENT" if sent else " | DROP")
                    )

            except Exception as e:
                self._logger.error(f"Main loop error: {e}")
                traceback.print_exc()
                time.sleep(0.1)

        self._logger.info("Main processing loop exited")

    def _update_running_avg(self, key: str, new_val: float, alpha: float = 0.05) -> None:
        cur = self._stats.get(key, 0.0)
        if cur == 0:
            self._stats[key] = new_val
        else:
            self._stats[key] = cur * (1 - alpha) + new_val * alpha

    def _handle_server_command(self, cmd: Dict[str, Any]) -> None:
        cmd_type = cmd.get("command_type", "")
        params = cmd.get("parameters", {})
        self._logger.info(f"Received server command: {cmd_type} params={params}")

        try:
            if cmd_type == "SWITCH_BAND":
                band = params.get("band", "RGB")
                if self._camera:
                    self._camera.set_active_band(band)
            elif cmd_type == "SET_CONF_THRESHOLD":
                threshold = params.get("threshold", 0.5)
                self._logger.info(f"Conf threshold change requested: {threshold}")
            elif cmd_type == "PAUSE_DETECTION":
                self._logger.info("Detection pause requested")
            elif cmd_type == "RESUME_DETECTION":
                self._logger.info("Detection resume requested")
        except Exception as e:
            self._logger.error(f"Command execution failed: {e}")

    def _save_json_snapshot(self, data: Dict[str, Any], frame_id: int) -> None:
        try:
            if not data.get("detections"):
                return

            date_str = datetime.now().strftime("%Y%m%d")
            dir_path = os.path.join(self._output_dir, "json", date_str)
            os.makedirs(dir_path, exist_ok=True)

            file_path = os.path.join(
                dir_path, f"{self._drone_id}_{frame_id}_{int(time.time())}.json"
            )
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self._logger.debug(f"Failed to save JSON snapshot: {e}")

    def _save_detected_image(
        self, frame, detections, frame_id: int
    ) -> None:
        try:
            import cv2
            import numpy as np

            annotated = frame.copy()
            if len(annotated.shape) == 2:
                annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
            elif annotated.shape[2] == 1:
                annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

            colors = {
                "mild": (0, 255, 0),
                "moderate": (0, 200, 255),
                "severe": (0, 0, 255),
            }

            for d in detections:
                color = colors.get(d.severity_level, (255, 255, 255))
                cv2.rectangle(
                    annotated,
                    (d.x1, d.y1),
                    (d.x2, d.y2),
                    color,
                    2,
                )
                label = f"{d.class_name} {d.confidence:.2f}"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    annotated,
                    (d.x1, max(0, d.y1 - th - 6)),
                    (d.x1 + tw + 4, d.y1),
                    color,
                    -1,
                )
                cv2.putText(
                    annotated,
                    label,
                    (d.x1 + 2, max(th + 2, d.y1 - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                )

            date_str = datetime.now().strftime("%Y%m%d")
            dir_path = os.path.join(self._output_dir, "images", date_str)
            os.makedirs(dir_path, exist_ok=True)

            file_path = os.path.join(
                dir_path, f"{self._drone_id}_{frame_id}_{int(time.time())}.jpg"
            )
            cv2.imwrite(file_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        except Exception as e:
            self._logger.debug(f"Failed to save image: {e}")

    def _stats_report_loop(self) -> None:
        while self._running and not self._shutdown_event.is_set():
            try:
                self._shutdown_event.wait(10.0)
                if not self._running:
                    break

                with self._stats_lock:
                    now = time.time()
                    elapsed = now - self._stats["last_report_time"]
                    frames_delta = (
                        self._stats["frames_processed"]
                        - self._stats["last_report_frames"]
                    )
                    inst_fps = frames_delta / elapsed if elapsed > 0 else 0.0

                    total_up = now - self._stats["start_time"]
                    hours, rem = divmod(int(total_up), 3600)
                    mins, secs = divmod(rem, 60)

                    grpc_stats = (
                        self._grpc_client.get_stats()
                        if self._grpc_client
                        else {}
                    )

                    self._logger.info(
                        "=" * 60 + "\n"
                        f"  UPTIME: {hours:02d}:{mins:02d}:{secs:02d}\n"
                        f"  FRAMES: processed={self._stats['frames_processed']} "
                        f"skipped={self._stats['frames_skipped']}\n"
                        f"  FPS:    avg={self._stats['avg_fps']:.1f} "
                        f"instant={inst_fps:.1f}\n"
                        f"  DETECT: total={self._stats['total_detections']} "
                        f"avg_per_frame="
                        f"{self._stats['total_detections'] / max(1, self._stats['frames_processed']):.2f}\n"
                        f"  INFER:  avg_ms={self._stats['avg_inference_ms']:.1f}\n"
                        f"  GRPC:   sent={grpc_stats.get('frames_sent', 0)} "
                        f"acks={grpc_stats.get('acks_received', 0)} "
                        f"reconnects={grpc_stats.get('reconnects', 0)} "
                        f"connected={self._grpc_client.is_connected() if self._grpc_client else False}\n"
                        + "=" * 60
                    )

                    self._stats["last_report_time"] = now
                    self._stats["last_report_frames"] = self._stats[
                        "frames_processed"
                    ]

            except Exception as e:
                self._logger.error(f"Stats report error: {e}")

    def _print_final_stats(self) -> None:
        try:
            with self._stats_lock:
                total_up = time.time() - self._stats["start_time"]
                hours, rem = divmod(int(total_up), 3600)
                mins, secs = divmod(rem, 60)

                self._logger.info(
                    "\n" + "=" * 60 + "\n"
                    f"  FINAL STATISTICS\n"
                    + "-" * 60 + "\n"
                    f"  Uptime:            {hours:02d}:{mins:02d}:{secs:02d}\n"
                    f"  Frames Processed:  {self._stats['frames_processed']}\n"
                    f"  Frames Skipped:    {self._stats['frames_skipped']}\n"
                    f"  Total Detections:  {self._stats['total_detections']}\n"
                    f"  Avg FPS:           {self._stats['avg_fps']:.2f}\n"
                    f"  Avg Inference:     {self._stats['avg_inference_ms']:.2f} ms\n"
                    + "=" * 60
                )
        except Exception:
            pass

    def wait(self) -> None:
        try:
            while self._running and not self._shutdown_event.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.stop()

    @property
    def running(self) -> bool:
        return self._running


def main():
    parser = argparse.ArgumentParser(description="Drone Edge Detection System")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "config", "config.yaml"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--drone-id",
        type=str,
        default=None,
        help="Override drone ID",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force mock mode for camera and GPS",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)

    if args.drone_id:
        config["drone"]["id"] = args.drone_id

    if args.mock:
        if "gps" in config:
            config["gps"]["mock_enabled"] = True
        if "camera" in config:
            pass

    logger = setup_logging(
        config.get("system", {}), config["drone"].get("id", "default")
    )

    app = DroneEdgeApp(config, logger)

    if not app.start():
        logger.error("Failed to start application")
        sys.exit(1)

    try:
        app.wait()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop()


if __name__ == "__main__":
    main()
