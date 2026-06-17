import os
import time
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str
    severity_score: float
    severity_level: str


class TensorRTEngine:
    def __init__(self, config: Dict[str, Any]):
        self._engine_path = config.get("engine_path", "")
        self._input_shape = tuple(config.get("input_shape", [1, 3, 640, 640]))
        self._output_shape = tuple(config.get("output_shape", [1, 25200, 85]))
        self._precision = config.get("precision", "fp16")

        self._engine = None
        self._context = None
        self._input_binding = None
        self._output_binding = None
        self._d_input = None
        self._d_output = None
        self._stream = None

        self._runtime = None
        self._loaded = False

        self._try_load_tensorrt()

    def _try_load_tensorrt(self) -> None:
        try:
            global tensorrt, cuda
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            logger.info("TensorRT and PyCUDA imported successfully")
        except ImportError as e:
            logger.warning(
                f"TensorRT/PyCUDA not available: {e}. "
                f"Falling back to mock inference mode."
            )
            self._loaded = False
            return

        if not os.path.exists(self._engine_path):
            logger.warning(
                f"Engine file not found: {self._engine_path}. "
                f"Using mock inference mode."
            )
            self._loaded = False
            return

        try:
            logger.info(f"Loading TensorRT engine: {self._engine_path}")
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            self._runtime = trt.Runtime(TRT_LOGGER)

            with open(self._engine_path, "rb") as f:
                engine_data = f.read()

            self._engine = self._runtime.deserialize_cuda_engine(engine_data)
            self._context = self._engine.create_execution_context()

            self._setup_bindings()
            self._loaded = True
            logger.info(
                f"TensorRT engine loaded successfully. "
                f"Input: {self._input_shape}, Output: {self._output_shape}"
            )

        except Exception as e:
            logger.error(f"Failed to load TensorRT engine: {e}")
            self._loaded = False

    def _setup_bindings(self) -> None:
        if self._engine is None:
            return

        for idx in range(self._engine.num_bindings):
            name = self._engine.get_binding_name(idx)
            if self._engine.binding_is_input(idx):
                self._input_binding = idx
                self._d_input = cuda.mem_alloc(
                    int(np.prod(self._input_shape) * np.float32().nbytes)
                )
            else:
                self._output_binding = idx
                self._d_output = cuda.mem_alloc(
                    int(np.prod(self._output_shape) * np.float32().nbytes)
                )

        self._stream = cuda.Stream()
        self._h_input = cuda.pagelocked_empty(
            self._input_shape, dtype=np.float32
        )
        self._h_output = cuda.pagelocked_empty(
            self._output_shape, dtype=np.float32
        )

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        if not self._loaded:
            return self._mock_infer(input_data)

        try:
            np.copyto(self._h_input, input_data)
            cuda.memcpy_htod_async(self._d_input, self._h_input, self._stream)

            bindings = [int(self._d_input), int(self._d_output)]
            self._context.execute_async_v2(bindings, self._stream.handle, None)

            cuda.memcpy_dtoh_async(self._h_output, self._d_output, self._stream)
            self._stream.synchronize()

            return self._h_output.copy()

        except Exception as e:
            logger.error(f"TensorRT inference error: {e}")
            return self._mock_infer(input_data)

    def _mock_infer(self, input_data: np.ndarray) -> np.ndarray:
        batch_size = self._output_shape[0]
        num_detections = self._output_shape[1]
        num_attrs = self._output_shape[2]

        output = np.zeros(
            (batch_size, num_detections, num_attrs), dtype=np.float32
        )

        rng = np.random.default_rng(int(time.time() * 1000) % 2**32)
        num_real = rng.integers(0, 8)

        for i in range(min(num_real, num_detections)):
            x_center = rng.uniform(0.1, 0.9)
            y_center = rng.uniform(0.1, 0.9)
            w = rng.uniform(0.05, 0.2)
            h = rng.uniform(0.05, 0.2)

            output[0, i, 0] = x_center
            output[0, i, 1] = y_center
            output[0, i, 2] = w
            output[0, i, 3] = h
            output[0, i, 4] = rng.uniform(0.5, 0.98)

            class_id = rng.integers(0, 3)
            output[0, i, 5 + class_id] = rng.uniform(0.6, 0.95)

        return output


class DiseaseDetector:
    def __init__(self, config: Dict[str, Any]):
        trt_config = config.get("tensorrt", {})
        self._engine = TensorRTEngine(trt_config)

        det_config = config.get("detection", {})
        self._conf_threshold = det_config.get("conf_threshold", 0.5)
        self._nms_threshold = det_config.get("nms_threshold", 0.45)
        self._max_detections = det_config.get("max_detections_per_frame", 50)
        self._target_classes = {
            cls["id"]: cls for cls in det_config.get("target_classes", [])
        }

        self._input_h = trt_config.get("input_shape", [1, 3, 640, 640])[2]
        self._input_w = trt_config.get("input_shape", [1, 3, 640, 640])[3]

    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        if frame is None or frame.size == 0:
            raise ValueError("Invalid input frame")

        if len(frame.shape) == 2:
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.shape[2] == 1:
            frame = np.concatenate([frame] * 3, axis=-1)

        original_h, original_w = frame.shape[:2]

        scale = min(
            self._input_w / original_w,
            self._input_h / original_h,
        )
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)

        resized = cv2_resize(frame, (new_w, new_h))

        padded = np.zeros(
            (self._input_h, self._input_w, 3), dtype=np.uint8
        )
        pad_x = (self._input_w - new_w) // 2
        pad_y = (self._input_h - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        blob = padded.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        blob = np.ascontiguousarray(blob)

        return blob, (scale, (pad_x, pad_y))

    def postprocess(
        self,
        output: np.ndarray,
        original_shape: Tuple[int, int],
        scale_info: Tuple[float, Tuple[int, int]],
    ) -> List[Detection]:
        original_h, original_w = original_shape
        scale, (pad_x, pad_y) = scale_info

        if output.ndim == 3:
            preds = output[0]
        else:
            preds = output

        boxes = []
        confidences = []
        class_ids = []

        for pred in preds:
            obj_conf = pred[4]
            if obj_conf < self._conf_threshold:
                continue

            class_scores = pred[5:]
            if len(class_scores) == 0:
                continue

            class_id = int(np.argmax(class_scores))
            class_conf = class_scores[class_id]

            if class_conf < self._conf_threshold:
                continue

            if class_id not in self._target_classes:
                continue

            cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]

            x1 = int((cx - w / 2) * self._input_w - pad_x) / scale
            y1 = int((cy - h / 2) * self._input_h - pad_y) / scale
            x2 = int((cx + w / 2) * self._input_w - pad_x) / scale
            y2 = int((cy + h / 2) * self._input_h - pad_y) / scale

            x1 = int(max(0, min(original_w - 1, x1)))
            y1 = int(max(0, min(original_h - 1, y1)))
            x2 = int(max(0, min(original_w - 1, x2)))
            y2 = int(max(0, min(original_h - 1, y2)))

            if x2 - x1 < 2 or y2 - y1 < 2:
                continue

            boxes.append([x1, y1, x2, y2])
            confidences.append(float(obj_conf * class_conf))
            class_ids.append(class_id)

        if len(boxes) == 0:
            return []

        try:
            boxes_np = np.array(boxes, dtype=np.float32)
            keep = nms_boxes(
                boxes_np,
                np.array(confidences, dtype=np.float32),
                self._nms_threshold,
            )
        except Exception as e:
            logger.warning(f"NMS error: {e}")
            keep = list(range(len(boxes)))

        detections = []
        for idx in keep[: self._max_detections]:
            cls_id = class_ids[idx]
            cls_info = self._target_classes.get(
                cls_id,
                {
                    "name": f"class_{cls_id}",
                    "severity_range": [0.0, 1.0],
                    "severity_level": "unknown",
                },
            )

            sev_min, sev_max = cls_info["severity_range"]
            conf = confidences[idx]
            severity_score = sev_min + (sev_max - sev_min) * conf

            detections.append(
                Detection(
                    x1=int(boxes[idx][0]),
                    y1=int(boxes[idx][1]),
                    x2=int(boxes[idx][2]),
                    y2=int(boxes[idx][3]),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_info["name"],
                    severity_score=float(severity_score),
                    severity_level=cls_info["severity_level"],
                )
            )

        return detections

    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], float]:
        start_time = time.time()

        try:
            original_shape = frame.shape[:2]
            blob, scale_info = self.preprocess(frame)
            output = self._engine.infer(blob)
            detections = self.postprocess(output, original_shape, scale_info)

            latency_ms = (time.time() - start_time) * 1000.0
            return detections, latency_ms

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            latency_ms = (time.time() - start_time) * 1000.0
            return [], latency_ms


def cv2_resize(img: np.ndarray, dsize: Tuple[int, int]) -> np.ndarray:
    try:
        import cv2

        return cv2.resize(img, dsize, interpolation=cv2.INTER_LINEAR)
    except ImportError:
        return _simple_resize(img, dsize)


def _simple_resize(
    img: np.ndarray, dsize: Tuple[int, int]
) -> np.ndarray:
    h, w = img.shape[:2]
    new_w, new_h = dsize
    if len(img.shape) == 2:
        channels = 1
    else:
        channels = img.shape[2]
    result = np.zeros((new_h, new_w, channels), dtype=img.dtype)
    x_ratio = w / new_w
    y_ratio = h / new_h
    for i in range(new_h):
        for j in range(new_w):
            px = int(j * x_ratio)
            py = int(i * y_ratio)
            result[i, j] = img[py, px]
    return result


def nms_boxes(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> List[int]:
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep
