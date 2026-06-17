import time
import math
import random
import threading
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class GPSData:
    latitude: float
    longitude: float
    altitude: float
    speed: float
    heading: float
    timestamp: int
    satellites: int = 0
    fix_quality: int = 0
    hdop: float = 0.0


class GPSProvider:
    def __init__(self, config: Dict[str, Any]):
        self._enabled = config.get("enabled", True)
        self._serial_port = config.get("serial_port", "/dev/ttyTHS1")
        self._baud_rate = config.get("baud_rate", 9600)
        self._protocol = config.get("protocol", "NMEA")
        self._mock_enabled = config.get("mock_enabled", False)
        mock_data = config.get("mock_data", {})
        self._mock_lat = mock_data.get("latitude", 39.9042)
        self._mock_lon = mock_data.get("longitude", 116.4074)
        self._mock_alt = mock_data.get("altitude", 50.0)
        self._mock_speed = mock_data.get("speed", 8.5)
        self._mock_heading = mock_data.get("heading", 90.0)

        self._current_data: Optional[GPSData] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._serial_conn = None
        self._lock = threading.Lock()
        self._update_count = 0

    def start(self) -> bool:
        if not self._enabled:
            logger.info("GPS disabled, using mock data")
            self._mock_enabled = True
            self._running = True
            self._thread = threading.Thread(
                target=self._mock_loop, daemon=True, name="GPSMockThread"
            )
            self._thread.start()
            return True

        if self._mock_enabled:
            logger.info("GPS mock mode enabled")
            self._running = True
            self._thread = threading.Thread(
                target=self._mock_loop, daemon=True, name="GPSMockThread"
            )
            self._thread.start()
            return True

        try:
            self._setup_serial()
            self._running = True
            self._thread = threading.Thread(
                target=self._read_loop, daemon=True, name="GPSThread"
            )
            self._thread.start()
            logger.info(f"GPS started on {self._serial_port} @ {self._baud_rate}")
            return True
        except Exception as e:
            logger.warning(
                f"Failed to init GPS serial: {e}, falling back to mock"
            )
            self._mock_enabled = True
            self._running = True
            self._thread = threading.Thread(
                target=self._mock_loop, daemon=True, name="GPSMockThread"
            )
            self._thread.start()
            return True

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._serial_conn:
            try:
                self._serial_conn.close()
            except Exception:
                pass
            self._serial_conn = None
        logger.info("GPS stopped")

    def get_current(self) -> GPSData:
        with self._lock:
            if self._current_data is None:
                return self._create_mock_data(int(time.time() * 1e9))
            return GPSData(**{**self._current_data.__dict__})

    def get_current_tuple(self) -> Tuple[float, float, float, float, float, int]:
        data = self.get_current()
        return (
            data.latitude,
            data.longitude,
            data.altitude,
            data.speed,
            data.heading,
            data.timestamp,
        )

    def _setup_serial(self) -> None:
        try:
            import serial

            self._serial_conn = serial.Serial(
                port=self._serial_port,
                baudrate=self._baud_rate,
                timeout=1.0,
            )
            logger.info(f"Serial port opened: {self._serial_port}")
        except ImportError:
            raise RuntimeError("pyserial not installed")

    def _read_loop(self) -> None:
        logger.info("GPS read loop started")
        buffer = ""

        while self._running:
            try:
                if self._serial_conn and self._serial_conn.in_waiting > 0:
                    raw = self._serial_conn.read(
                        self._serial_conn.in_waiting
                    ).decode("ascii", errors="ignore")
                    buffer += raw

                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if line:
                            self._parse_nmea_line(line)

                time.sleep(0.05)

            except Exception as e:
                logger.error(f"GPS read error: {e}")
                time.sleep(0.5)

        logger.info("GPS read loop exited")

    def _parse_nmea_line(self, line: str) -> None:
        try:
            import pynmea2

            if not line.startswith("$"):
                return

            msg = pynmea2.parse(line)

            lat = None
            lon = None
            alt = None
            speed = None
            heading = None
            satellites = 0
            fix_quality = 0
            hdop = 0.0

            if hasattr(msg, "latitude") and msg.latitude is not None:
                lat = float(msg.latitude)
            if hasattr(msg, "longitude") and msg.longitude is not None:
                lon = float(msg.longitude)
            if hasattr(msg, "altitude") and msg.altitude is not None:
                alt = float(msg.altitude)
            if hasattr(msg, "spd_over_grnd") and msg.spd_over_grnd is not None:
                speed = float(msg.spd_over_grnd) * 1.852
            if hasattr(msg, "true_course") and msg.true_course is not None:
                heading = float(msg.true_course)
            if hasattr(msg, "num_sats"):
                try:
                    satellites = int(msg.num_sats)
                except (ValueError, TypeError):
                    pass
            if hasattr(msg, "gps_qual"):
                try:
                    fix_quality = int(msg.gps_qual)
                except (ValueError, TypeError):
                    pass
            if hasattr(msg, "horizontal_dil"):
                try:
                    hdop = float(msg.horizontal_dil)
                except (ValueError, TypeError):
                    pass

            with self._lock:
                if lat is not None and lon is not None:
                    old = self._current_data
                    self._current_data = GPSData(
                        latitude=lat,
                        longitude=lon,
                        altitude=alt if alt is not None else (old.altitude if old else 0.0),
                        speed=speed if speed is not None else (old.speed if old else 0.0),
                        heading=heading if heading is not None else (old.heading if old else 0.0),
                        timestamp=int(time.time() * 1e9),
                        satellites=satellites,
                        fix_quality=fix_quality,
                        hdop=hdop,
                    )
                    self._update_count += 1

        except Exception as e:
            logger.debug(f"NMEA parse error: {e}")

    def _mock_loop(self) -> None:
        logger.info("GPS mock loop started")
        base_time = time.time()

        while self._running:
            try:
                elapsed = time.time() - base_time
                data = self._create_mock_data(int(time.time() * 1e9), elapsed)

                with self._lock:
                    self._current_data = data
                    self._update_count += 1

                time.sleep(0.1)

            except Exception as e:
                logger.error(f"GPS mock loop error: {e}")
                time.sleep(0.1)

        logger.info("GPS mock loop exited")

    def _create_mock_data(self, timestamp_ns: int, elapsed: float = 0.0) -> GPSData:
        speed_factor = elapsed * 0.001
        lat = self._mock_lat + (speed_factor * 0.0005) + random.uniform(-1e-6, 1e-6)
        lon = self._mock_lon + (speed_factor * 0.0005) + random.uniform(-1e-6, 1e-6)
        alt = self._mock_alt + random.uniform(-0.5, 0.5)
        speed = self._mock_speed + random.uniform(-0.5, 0.5)
        heading = (self._mock_heading + elapsed * 0.1) % 360.0

        return GPSData(
            latitude=round(lat, 8),
            longitude=round(lon, 8),
            altitude=round(alt, 2),
            speed=round(speed, 2),
            heading=round(heading, 2),
            timestamp=timestamp_ns,
            satellites=random.randint(8, 14),
            fix_quality=2,
            hdop=round(random.uniform(0.5, 1.5), 2),
        )

    def haversine_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        R = 6371000.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = (
            math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c
