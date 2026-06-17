import cv2
import time
import threading
import logging
import numpy as np
from queue import Queue, Full
from typing import Tuple, Optional


logger = logging.getLogger(__name__)


class MultispectralCamera:
    BAND_CHANNELS = {
        "RGB": 3,
        "NIR": 1,
        "RedEdge": 1,
    }

    def __init__(self, config: dict):
        self._device_index = config.get("device_index", 0)
        self._width = config.get("width", 1280)
        self._height = config.get("height", 720)
        self._fps = config.get("fps", 30)
        self._multispectral_cfg = config.get("multispectral", {})
        self._enabled = self._multispectral_cfg.get("enabled", False)
        self._bands = self._multispectral_cfg.get("bands", ["RGB"])
        self._active_band = self._multispectral_cfg.get("active_band", "RGB")

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_queue: Queue = Queue(maxsize=100)
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._frame_id = 0
        self._lock = threading.Lock()

    def start(self) -> bool:
        try:
            backend = cv2.CAP_GSTREAMER if self._detect_gstreamer() else cv2.CAP_ANY
            self._cap = cv2.VideoCapture(self._device_index, backend)

            if not self._cap.isOpened():
                logger.error(f"Failed to open camera device {self._device_index}")
                return False

            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            self._cap.set(cv2.CAP_PROP_FPS, self._fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            actual_w = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
            logger.info(
                f"Camera opened: {actual_w}x{actual_h} @ {actual_fps}fps "
                f"(band: {self._active_band})"
            )

            self._running = True
            self._capture_thread = threading.Thread(
                target=self._capture_loop, daemon=True, name="CameraThread"
            )
            self._capture_thread.start()
            return True

        except Exception as e:
            logger.error(f"Camera start failed: {e}")
            self.stop()
            return False

    def stop(self) -> None:
        self._running = False
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info("Camera stopped")

    def get_frame(self, timeout: float = 1.0) -> Optional[Tuple[int, np.ndarray, float]]:
        try:
            item = self._frame_queue.get(timeout=timeout)
            return item
        except Exception:
            return None

    def get_active_band(self) -> str:
        return self._active_band

    def set_active_band(self, band: str) -> bool:
        if band in self._bands:
            with self._lock:
                self._active_band = band
            logger.info(f"Switched multispectral band to: {band}")
            return True
        return False

    def get_resolution(self) -> Tuple[int, int]:
        return (self._width, self._height)

    def _capture_loop(self) -> None:
        logger.info("Capture loop started")
        frame_interval = 1.0 / self._fps if self._fps > 0 else 0.033

        while self._running:
            try:
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    logger.warning("Frame capture failed, retrying...")
                    time.sleep(0.01)
                    continue

                with self._lock:
                    active_band = self._active_band

                processed = self._process_multispectral(frame, active_band)

                with self._lock:
                    self._frame_id += 1
                    frame_id = self._frame_id

                timestamp = time.time()

                try:
                    self._frame_queue.put_nowait(
                        (frame_id, processed, timestamp)
                    )
                except Full:
                    try:
                        self._frame_queue.get_nowait()
                    except Exception:
                        pass
                    try:
                        self._frame_queue.put_nowait(
                            (frame_id, processed, timestamp)
                        )
                    except Exception:
                        pass

                if frame_interval > 0:
                    elapsed = time.time() - timestamp
                    sleep_time = max(0, frame_interval - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Capture loop error: {e}")
                time.sleep(0.01)

        logger.info("Capture loop exited")

    def _process_multispectral(
        self, frame: np.ndarray, band: str
    ) -> np.ndarray:
        if not self._enabled or frame is None:
            return frame

        try:
            if band == "RGB":
                return frame
            elif band == "NIR":
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    return frame[:, :, 2:3]
                return frame
            elif band == "RedEdge":
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    red = frame[:, :, 2].astype(np.float32)
                    green = frame[:, :, 1].astype(np.float32)
                    rededge = (red * 0.7 + green * 0.3).astype(np.uint8)
                    return rededge[:, :, np.newaxis]
                return frame
            else:
                return frame
        except Exception as e:
            logger.error(f"Multispectral processing error: {e}")
            return frame

    def _detect_gstreamer(self) -> bool:
        try:
            build_info = cv2.getBuildInformation()
            return "GStreamer: YES" in build_info
        except Exception:
            return False
