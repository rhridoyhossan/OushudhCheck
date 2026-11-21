from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

try:
    from backend.app.core.config import MAX_STRIPS_PER_IMAGE, YOLO_MODEL_PATH
    from backend.app.models.schemas import BoundingBox
except ImportError:
    from app.core.config import MAX_STRIPS_PER_IMAGE, YOLO_MODEL_PATH  # type: ignore
    from app.models.schemas import BoundingBox  # type: ignore


@dataclass
class DetectedStrip:
    strip_index: int
    crop: np.ndarray
    bbox: BoundingBox


class StripDetector:
    """
    Detects and crops 1..N medicine blister strips from an input image.
    - Uses YOLO ONNX model (`onnxruntime` or `ultralytics`) when weights exist at `YOLO_MODEL_PATH`.
    - Falls back to OpenCV multi-strip contour segmentation (or full-image crop) when no YOLO model is present,
      and applies CLAHE + glare reduction for reflective aluminum blister foil.
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self.model_path = Path(model_path) if model_path else YOLO_MODEL_PATH
        self._ort_session = None
        self._ultralytics_model = None
        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path or not self.model_path.exists():
            return

        try:
            from ultralytics import YOLO  # type: ignore

            self._ultralytics_model = YOLO(str(self.model_path))
            return
        except Exception:
            pass

        if self.model_path.suffix.lower() == ".onnx":
            try:
                import onnxruntime as ort  # type: ignore

                self._ort_session = ort.InferenceSession(
                    str(self.model_path),
                    providers=["CPUExecutionProvider"],
                )
            except Exception:
                self._ort_session = None

    @staticmethod
    def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
        """
        Enhances contrast on shiny/reflective medicine blister strips using CLAHE on the L-channel
        and mild sharpening so tiny foil text ('500 mg', '1000 mg', brand names) is clear for OCR.
        """
        if cv2 is None or image is None or image.size == 0:
            return image

        if len(image.shape) == 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            bgr = image.copy()

        h, w = bgr.shape[:2]
        if max(h, w) < 640:
            scale = 640.0 / max(h, w)
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_chan)
        merged = cv2.merge((l_enhanced, a_chan, b_chan))
        enhanced_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        gaussian = cv2.GaussianBlur(enhanced_bgr, (0, 0), 1.2)
        sharpened = cv2.addWeighted(enhanced_bgr, 1.35, gaussian, -0.35, 0)
        return sharpened

    def detect_strips(self, image: np.ndarray) -> List[DetectedStrip]:
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]

        if self._ultralytics_model is not None:
            boxes = self._detect_with_ultralytics(image)
            if boxes:
                return self._crops_from_boxes(image, boxes)

        if self._ort_session is not None and cv2 is not None:
            boxes = self._detect_with_onnx(image)
            if boxes:
                return self._crops_from_boxes(image, boxes)

        if cv2 is not None:
            cv_boxes = self._detect_with_opencv_contours(image)
            if cv_boxes:
                return self._crops_from_boxes(image, cv_boxes)

        return [
            DetectedStrip(
                strip_index=1,
                crop=image,
                bbox=BoundingBox(x1=0, y1=0, x2=w, y2=h, confidence=1.0),
            )
        ]

    def _detect_with_ultralytics(self, image: np.ndarray) -> List[BoundingBox]:
        results = self._ultralytics_model.predict(image, conf=0.25, iou=0.45, verbose=False)
        boxes: List[BoundingBox] = []
        if not results:
            return boxes
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                conf = float(b.conf[0].item())
                boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=round(conf, 3)))
        boxes.sort(key=lambda b: b.confidence, reverse=True)
        return boxes[:MAX_STRIPS_PER_IMAGE]

    def _detect_with_onnx(self, image: np.ndarray) -> List[BoundingBox]:
        h_orig, w_orig = image.shape[:2]
        input_name = self._ort_session.get_inputs()[0].name
        img_resized = cv2.resize(image, (640, 640))
        blob = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

        outputs = self._ort_session.run(None, {input_name: blob})
        preds = np.squeeze(outputs[0])
        if preds.ndim != 2:
            return []

        if preds.shape[0] < preds.shape[1]:
            preds = preds.T

        raw_boxes: List[List[int]] = []
        scores: List[float] = []
        x_scale = w_orig / 640.0
        y_scale = h_orig / 640.0

        for row in preds:
            conf = float(np.max(row[4:]))
            if conf < 0.25:
                continue
            cx, cy, bw, bh = row[:4]
            x1 = max(0, int((cx - bw / 2) * x_scale))
            y1 = max(0, int((cy - bh / 2) * y_scale))
            x2 = min(w_orig, int((cx + bw / 2) * x_scale))
            y2 = min(h_orig, int((cy + bh / 2) * y_scale))
            raw_boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(conf)

        if not raw_boxes:
            return []

        indices = cv2.dnn.NMSBoxes(raw_boxes, scores, score_threshold=0.25, nms_threshold=0.45)
        boxes: List[BoundingBox] = []
        for idx in np.array(indices).flatten():
            x, y, bw, bh = raw_boxes[int(idx)]
            boxes.append(
                BoundingBox(
                    x1=x,
                    y1=y,
                    x2=x + bw,
                    y2=y + bh,
                    confidence=round(scores[int(idx)], 3),
                )
            )
        return boxes[:MAX_STRIPS_PER_IMAGE]

    def _detect_with_opencv_contours(self, image: np.ndarray) -> List[BoundingBox]:
        h, w = image.shape[:2]
        img_area = float(h * w)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 140)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate_boxes: List[Tuple[float, BoundingBox]] = []

        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            box_area = float(bw * bh)
            if box_area < 0.06 * img_area or box_area > 0.85 * img_area:
                continue
            aspect = max(bw, bh) / max(1.0, float(min(bw, bh)))
            if aspect > 6.0:
                continue
            pad_x = int(bw * 0.03)
            pad_y = int(bh * 0.03)
            bbox = BoundingBox(
                x1=max(0, x - pad_x),
                y1=max(0, y - pad_y),
                x2=min(w, x + bw + pad_x),
                y2=min(h, y + bh + pad_y),
                confidence=round(min(0.95, 0.5 + (box_area / img_area)), 3),
            )
            candidate_boxes.append((box_area, bbox))

        if len(candidate_boxes) >= 2:
            candidate_boxes.sort(key=lambda item: item[0], reverse=True)
            selected = [b for _, b in candidate_boxes[:MAX_STRIPS_PER_IMAGE]]
            selected.sort(key=lambda b: (b.y1, b.x1))
            return selected

        return [BoundingBox(x1=0, y1=0, x2=w, y2=h, confidence=1.0)]

    @staticmethod
    def _crops_from_boxes(image: np.ndarray, boxes: List[BoundingBox]) -> List[DetectedStrip]:
        h, w = image.shape[:2]
        strips: List[DetectedStrip] = []
        for idx, b in enumerate(boxes, start=1):
            x1 = max(0, min(w - 1, b.x1))
            y1 = max(0, min(h - 1, b.y1))
            x2 = max(x1 + 1, min(w, b.x2))
            y2 = max(y1 + 1, min(h, b.y2))
            crop = image[y1:y2, x1:x2].copy()
            strips.append(DetectedStrip(strip_index=idx, crop=crop, bbox=b))
        return strips
