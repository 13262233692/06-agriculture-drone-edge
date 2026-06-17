import os
import sys
import time
import queue
import json
import threading
import logging
import traceback
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone


logger = logging.getLogger(__name__)


@dataclass
class DetectionPoint:
    measurement: str
    tags: Dict[str, str]
    fields: Dict[str, Any]
    timestamp_ns: int


class InfluxDBWriter:
    def __init__(self, config: Dict[str, Any]):
        self._url = config.get("url", "http://localhost:8086")
        self._token = config.get("token", "")
        self._org = config.get("org", "agriculture")
        self._bucket = config.get("bucket", "drone_detections")
        self._batch_size = config.get("batch_size", 100)
        self._flush_interval_ms = config.get("flush_interval_ms", 1000)
        self._enable_gzip = config.get("enable_gzip", True)
        self._timeout_seconds = config.get("timeout_seconds", 30)

        self._client = None
        self._write_api = None
        self._query_api = None

        self._point_queue: "queue.Queue[DetectionPoint]" = queue.Queue(
            maxsize=10000
        )
        self._batch_buffer: List[DetectionPoint] = []
        self._buffer_lock = threading.Lock()

        self._running = False
        self._flush_thread: Optional[threading.Thread] = None
        self._last_flush = time.time()

        self._stats_lock = threading.Lock()
        self._stats = {
            "points_written": 0,
            "batches_flushed": 0,
            "write_errors": 0,
            "points_dropped": 0,
            "bytes_written": 0,
        }

        self._on_error_callback: Optional[Callable[[Exception], None]] = None
        self._connected = False

        self._try_connect()

    def _try_connect(self) -> bool:
        try:
            from influxdb_client import InfluxDBClient
            from influxdb_client.client.write_api import SYNCHRONOUS

            if not self._token:
                logger.warning(
                    "InfluxDB token not configured, using mock write mode"
                )
                self._connected = False
                return False

            self._client = InfluxDBClient(
                url=self._url,
                token=self._token,
                org=self._org,
                enable_gzip=self._enable_gzip,
                timeout=self._timeout_seconds * 1000,
            )

            self._write_api = self._client.write_api(
                write_options=SYNCHRONOUS
            )
            self._query_api = self._client.query_api()

            self._client.ping()
            self._connected = True
            logger.info(
                f"InfluxDB connected: url={self._url} "
                f"org={self._org} bucket={self._bucket}"
            )
            return True

        except ImportError as e:
            logger.warning(
                f"influxdb-client not installed ({e}), using mock write mode"
            )
            self._connected = False
            return False
        except Exception as e:
            logger.warning(
                f"InfluxDB connection failed: {e}, using mock write mode"
            )
            self._connected = False
            self._client = None
            return False

    def start(self) -> bool:
        if self._running:
            return True

        self._running = True
        self._flush_thread = threading.Thread(
            target=self._flush_loop, daemon=True, name="InfluxDBFlushThread"
        )
        self._flush_thread.start()

        logger.info("InfluxDB writer started")
        return True

    def stop(self) -> None:
        self._running = False

        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=10.0)

        self._force_flush()

        if self._write_api:
            try:
                self._write_api.close()
            except Exception:
                pass
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass

        logger.info("InfluxDB writer stopped")

    def set_error_callback(self, cb: Callable[[Exception], None]) -> None:
        self._on_error_callback = cb

    def write_detection_frame(
        self,
        drone_id: str,
        frame_id: int,
        timestamp_ns: int,
        gps_data: Dict[str, float],
        detections: List[Dict[str, Any]],
        frame_width: int,
        frame_height: int,
        multispectral_band: str,
        inference_latency_ms: float,
    ) -> None:
        frame_point = DetectionPoint(
            measurement="detection_frame",
            tags={
                "drone_id": drone_id,
                "multispectral_band": multispectral_band,
            },
            fields={
                "frame_id": frame_id,
                "detection_count": len(detections),
                "frame_width": frame_width,
                "frame_height": frame_height,
                "inference_latency_ms": inference_latency_ms,
                "gps_lat": float(gps_data.get("latitude", 0.0)),
                "gps_lon": float(gps_data.get("longitude", 0.0)),
                "gps_alt": float(gps_data.get("altitude", 0.0)),
                "gps_speed": float(gps_data.get("speed", 0.0)),
                "gps_heading": float(gps_data.get("heading", 0.0)),
                "gps_satellites": int(gps_data.get("satellites", 0)),
                "gps_hdop": float(gps_data.get("hdop", 0.0)),
            },
            timestamp_ns=timestamp_ns,
        )
        self._enqueue_point(frame_point)

        for idx, det in enumerate(detections):
            bbox = det.get("bbox", {})
            det_point = DetectionPoint(
                measurement="disease_spot",
                tags={
                    "drone_id": drone_id,
                    "class_name": det.get("class_name", "unknown"),
                    "severity_level": det.get("severity_level", "unknown"),
                    "class_id": str(det.get("class_id", -1)),
                    "multispectral_band": multispectral_band,
                },
                fields={
                    "frame_id": frame_id,
                    "detection_index": idx,
                    "confidence": float(det.get("confidence", 0.0)),
                    "severity_score": float(det.get("severity_score", 0.0)),
                    "bbox_x1": int(bbox.get("x1", 0)),
                    "bbox_y1": int(bbox.get("y1", 0)),
                    "bbox_x2": int(bbox.get("x2", 0)),
                    "bbox_y2": int(bbox.get("y2", 0)),
                    "bbox_width": int(bbox.get("x2", 0)) - int(bbox.get("x1", 0)),
                    "bbox_height": int(bbox.get("y2", 0)) - int(bbox.get("y1", 0)),
                    "gps_lat": float(gps_data.get("latitude", 0.0)),
                    "gps_lon": float(gps_data.get("longitude", 0.0)),
                    "gps_alt": float(gps_data.get("altitude", 0.0)),
                },
                timestamp_ns=timestamp_ns,
            )
            self._enqueue_point(det_point)

        if detections:
            severity_counts: Dict[str, int] = {}
            for d in detections:
                lvl = d.get("severity_level", "unknown")
                severity_counts[lvl] = severity_counts.get(lvl, 0) + 1

            for lvl, count in severity_counts.items():
                agg_point = DetectionPoint(
                    measurement="severity_aggregate",
                    tags={
                        "drone_id": drone_id,
                        "severity_level": lvl,
                        "multispectral_band": multispectral_band,
                    },
                    fields={
                        "frame_id": frame_id,
                        "count": count,
                        "gps_lat": float(gps_data.get("latitude", 0.0)),
                        "gps_lon": float(gps_data.get("longitude", 0.0)),
                    },
                    timestamp_ns=timestamp_ns,
                )
                self._enqueue_point(agg_point)

    def _enqueue_point(self, point: DetectionPoint) -> None:
        try:
            self._point_queue.put_nowait(point)
        except queue.Full:
            with self._stats_lock:
                self._stats["points_dropped"] += 1
            logger.debug("Point queue full, dropping point")

    def _flush_loop(self) -> None:
        logger.info("InfluxDB flush loop started")
        flush_interval_s = self._flush_interval_ms / 1000.0

        while self._running:
            try:
                time_since_flush = time.time() - self._last_flush

                with self._buffer_lock:
                    buffer_size = len(self._batch_buffer)

                batch_ready = buffer_size >= self._batch_size
                timeout_ready = time_since_flush >= flush_interval_s

                if batch_ready or timeout_ready:
                    if buffer_size > 0:
                        self._flush_batch()

                try:
                    point = self._point_queue.get(timeout=0.2)
                    with self._buffer_lock:
                        self._batch_buffer.append(point)
                except queue.Empty:
                    continue

            except Exception as e:
                logger.error(f"Flush loop error: {e}")
                traceback.print_exc()
                time.sleep(0.5)

        self._force_flush()
        logger.info("InfluxDB flush loop exited")

    def _flush_batch(self) -> None:
        with self._buffer_lock:
            if len(self._batch_buffer) == 0:
                return
            batch = list(self._batch_buffer)
            self._batch_buffer.clear()

        self._last_flush = time.time()

        try:
            points = self._to_line_protocol_batch(batch)

            if self._connected and self._write_api is not None:
                self._write_api.write(
                    bucket=self._bucket,
                    org=self._org,
                    record=points,
                    write_precision="ns",
                )
            else:
                self._mock_write(batch)

            with self._stats_lock:
                self._stats["points_written"] += len(batch)
                self._stats["batches_flushed"] += 1
                self._stats["bytes_written"] += sum(
                    len(json.dumps(p.fields)) for p in batch
                )

            if len(batch) > 0:
                logger.debug(
                    f"Flushed {len(batch)} points to InfluxDB "
                    f"(total={self._stats['points_written']})"
                )

        except Exception as e:
            logger.error(f"Failed to flush batch: {e}")
            if self._on_error_callback:
                try:
                    self._on_error_callback(e)
                except Exception:
                    pass

            with self._stats_lock:
                self._stats["write_errors"] += 1
                self._stats["points_dropped"] += len(batch)

    def _force_flush(self) -> None:
        with self._buffer_lock:
            if len(self._batch_buffer) == 0:
                remaining = []
            else:
                remaining = list(self._batch_buffer)
                self._batch_buffer.clear()

        while True:
            try:
                remaining.append(self._point_queue.get_nowait())
            except queue.Empty:
                break

        if remaining:
            try:
                points = self._to_line_protocol_batch(remaining)
                if self._connected and self._write_api is not None:
                    self._write_api.write(
                        bucket=self._bucket,
                        org=self._org,
                        record=points,
                        write_precision="ns",
                    )
                else:
                    self._mock_write(remaining)

                with self._stats_lock:
                    self._stats["points_written"] += len(remaining)
            except Exception as e:
                logger.error(f"Force flush error: {e}")
                with self._stats_lock:
                    self._stats["points_dropped"] += len(remaining)

    def _to_line_protocol_batch(
        self, batch: List[DetectionPoint]
    ) -> List[str]:
        lines = []
        for p in batch:
            tag_str = ",".join(
                f"{k}={_escape_value(v)}" for k, v in p.tags.items()
            )
            field_str = ",".join(
                f"{k}={_format_field_value(v)}" for k, v in p.fields.items()
            )

            if tag_str:
                line = f"{p.measurement},{tag_str} {field_str} {p.timestamp_ns}"
            else:
                line = f"{p.measurement} {field_str} {p.timestamp_ns}"
            lines.append(line)
        return lines

    def _mock_write(self, batch: List[DetectionPoint]) -> None:
        log_lines = []
        for p in batch[:3]:
            log_lines.append(
                f"  [{p.measurement}] tags={p.tags} "
                f"fields_count={len(p.fields)}"
            )
        if len(batch) > 3:
            log_lines.append(f"  ... and {len(batch) - 3} more points")

        if log_lines:
            logger.debug(
                f"[MOCK] Writing {len(batch)} points to InfluxDB:\n"
                + "\n".join(log_lines)
            )

        self._save_mock_batch(batch)

    def _save_mock_batch(self, batch: List[DetectionPoint]) -> None:
        try:
            log_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data",
                "influx_mock",
            )
            os.makedirs(log_dir, exist_ok=True)

            date_str = datetime.now().strftime("%Y%m%d")
            file_path = os.path.join(log_dir, f"points_{date_str}.jsonl")

            with open(file_path, "a", encoding="utf-8") as f:
                for p in batch:
                    record = {
                        "measurement": p.measurement,
                        "tags": p.tags,
                        "fields": p.fields,
                        "timestamp": p.timestamp_ns,
                        "datetime": datetime.fromtimestamp(
                            p.timestamp_ns / 1e9, tz=timezone.utc
                        ).isoformat(),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug(f"Mock save failed: {e}")

    def query(
        self, flux_query: str
    ) -> List[Any]:
        if not self._connected or self._query_api is None:
            logger.warning("InfluxDB not connected, cannot query")
            return []

        try:
            result = self._query_api.query(flux_query)
            return result
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            with self._buffer_lock:
                buffer_size = len(self._batch_buffer)
            return {
                **self._stats,
                "queue_size": self._point_queue.qsize(),
                "buffer_size": buffer_size,
                "connected": self._connected,
            }

    def is_connected(self) -> bool:
        return self._connected

    def is_running(self) -> bool:
        return self._running


def _escape_value(v: str) -> str:
    return (
        str(v)
        .replace(" ", "\\ ")
        .replace(",", "\\,")
        .replace("=", "\\=")
    )


def _format_field_value(v: Any) -> str:
    if isinstance(v, bool):
        return "t" if v else "f"
    elif isinstance(v, int):
        return f"{v}i"
    elif isinstance(v, float):
        return f"{v}"
    elif isinstance(v, str):
        escaped = v.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    else:
        escaped = str(v).replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
