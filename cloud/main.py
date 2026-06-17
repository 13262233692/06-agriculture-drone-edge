import os
import sys
import time
import json
import signal
import logging
import argparse
import threading
import traceback
from typing import Dict, Any
from datetime import datetime

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    monitor_cfg = config.get("monitoring", {})
    log_level = getattr(
        logging, monitor_cfg.get("log_level", "INFO").upper(), logging.INFO
    )
    log_file = monitor_cfg.get("log_file", "./logs/cloud_server.log")

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger("cloud_server")
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


class CloudServerApp:
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self._config = config
        self._logger = logger

        from modules.grpc_server import CloudServer

        self._server = CloudServer(config, logger)

        self._running = False
        self._shutdown_event = threading.Event()

        self._setup_signal_handlers()
        self._cli_thread: threading.Thread = None

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
        if not self._server.start():
            return False

        self._running = True
        self._shutdown_event.clear()

        self._cli_thread = threading.Thread(
            target=self._cli_loop, daemon=True, name="AdminCLIThread"
        )
        self._cli_thread.start()

        return True

    def stop(self) -> None:
        if not self._running:
            return

        self._logger.info("Shutting down Cloud Server application...")
        self._running = False
        self._shutdown_event.set()

        self._server.stop()

        if self._cli_thread and self._cli_thread.is_alive():
            self._cli_thread.join(timeout=2.0)

        self._print_final_summary()
        self._logger.info("Cloud Server application stopped")

    def wait(self) -> None:
        try:
            while self._running and not self._shutdown_event.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.stop()

    def _cli_loop(self) -> None:
        self._logger.info(
            "Admin CLI started. Commands: [drones|stats|send <drone> <cmd> [args]|quit]"
        )
        while self._running and not self._shutdown_event.is_set():
            try:
                time.sleep(0.5)
            except Exception:
                break

    def _print_final_summary(self) -> None:
        try:
            stats = self._server.get_stats()
            servicer_stats = stats.get("servicer", {})
            influx_stats = stats.get("influxdb", {})

            self._logger.info(
                "\n" + "=" * 60 + "\n"
                f"  FINAL SERVER SUMMARY\n"
                + "-" * 60 + "\n"
                f"  Total Frames Received:    {servicer_stats.get('total_frames', 0)}\n"
                f"  Total Detections Stored:  {servicer_stats.get('total_detections', 0)}\n"
                f"  Total Data Received:      {servicer_stats.get('total_bytes_mb', 0)} MB\n"
                f"  Max Concurrent Drones:    {len(servicer_stats.get('frames_per_drone', {}))}\n"
                f"  InfluxDB Points Written:  {influx_stats.get('points_written', 0)}\n"
                f"  InfluxDB Write Errors:    {influx_stats.get('write_errors', 0)}\n"
                f"  InfluxDB Points Dropped:  {influx_stats.get('points_dropped', 0)}\n"
                + "=" * 60
            )
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Agriculture Drone Cloud Server")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "config", "config.yaml"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Override server host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override server port",
    )
    parser.add_argument(
        "--no-influx",
        action="store_true",
        help="Disable InfluxDB (use mock write mode)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)

    if args.host:
        config["server"]["host"] = args.host
    if args.port:
        config["server"]["port"] = args.port
    if args.no_influx:
        config["influxdb"]["token"] = ""

    logger = setup_logging(config)

    app = CloudServerApp(config, logger)

    if not app.start():
        logger.error("Failed to start cloud server application")
        sys.exit(1)

    try:
        app.wait()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop()


if __name__ == "__main__":
    main()
