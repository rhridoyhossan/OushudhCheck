from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import time
import numpy as np

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

try:
    from backend.app.core.config import (
        YOLO_CONF_THRESHOLD,
        YOLO_IMG_SIZE,
        YOLO_IOU_THRESHOLD,
        YOLO_MAX_PROPOSALS,
        YOLO_MODEL_PATH,
        YOLO_ORT_THREADS,
        YOLO_PIPELINE_MODE,
        YOLO_REQUIRE_REAL_MODEL,
    )
    from backend.app.models.schemas import BoundingBox
except ImportError:
    from app.core.config import (  # type: ignore
        YOLO_CONF_THRESHOLD,
        YOLO_IMG_SIZE,
        YOLO_IOU_THRESHOLD,
        YOLO_MAX_PROPOSALS,
        YOLO_MODEL_PATH,
        YOLO_ORT_THREADS,
        YOLO_PIPELINE_MODE,
        YOLO_REQUIRE_REAL_MODEL,
    )
    from app.models.schemas import BoundingBox  # type: ignore

logger = logging.getLogger("oushudhcheck.yolo")

MIN_REAL_ONNX_BYTES = 100_000  # Real YOLO26n/s ONNX models are ~9–40 MB; reject 148 B stubs / LFS pointers


@dataclass(frozen=True)
class DetectedStrip:
    strip_index: int
    crop: np.ndarray
    bbox: BoundingBox


@dataclass(frozen=True)
class DetectionResult:
    strips: List[DetectedStrip] = field(default_factory=list)
    boxes: List[BoundingBox] = field(default_factory=list)
    inference_ms: int = 0
    backend: str = "placeholder_stub"
    is_real_model: bool = False
    model_label: str = "YOLO26 ONNX (Placeholder Stub -> OCR Fallback)"
    raw_count: int = 0


def _to_bgr3(image: np.ndarray) -> np.ndarray:
    """Safely normalize grayscale or BGRA images to 3-channel BGR."""
    if cv2 is None:
        return image
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def letterbox_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, float, float]:
    """
    Aspect-ratio-preserving Ultralytics letterbox resize.
    Returns `(padded_bgr, scale, pad_left, pad_top)`.
    """
    bgr = _to_bgr3(image)
    h, w = bgr.shape[:2]
    target_h, target_w = int(target_size[0]), int(target_size[1])

    scale = min(target_h / float(max(1, h)), target_w / float(max(1, w)))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR) if (new_w != w or new_h != h) else bgr

    dw = (target_w - new_w) / 2.0
    dh = (target_h - new_h) / 2.0
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    if padded.shape[0] != target_h or padded.shape[1] != target_w:
        padded = cv2.resize(padded, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return padded, scale, float(left), float(top)


def unletterbox_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    scale: float,
    pad_left: float,
    pad_top: float,
    orig_w: int,
    orig_h: int,
) -> Optional[Tuple[int, int, int, int]]:
    """Maps letterboxed xyxy coordinates back into clamped original image pixel coordinates."""
    inv = 1.0 / max(scale, 1e-6)
    ox1 = int(round(max(0.0, min(float(orig_w), (x1 - pad_left) * inv))))
    oy1 = int(round(max(0.0, min(float(orig_h), (y1 - pad_top) * inv))))
    ox2 = int(round(max(0.0, min(float(orig_w), (x2 - pad_left) * inv))))
    oy2 = int(round(max(0.0, min(float(orig_h), (y2 - pad_top) * inv))))
    if ox2 - ox1 < 8 or oy2 - oy1 < 8:
        return None
    return ox1, oy1, ox2, oy2


def _box_ioa(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection over min(area_a, area_b) to suppress nested duplicate boxes."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(1, (a.x2 - a.x1) * (a.y2 - a.y1))
    area_b = max(1, (b.x2 - b.x1) * (b.y2 - b.y1))
    return inter / float(min(area_a, area_b))


def _filter_and_sort_boxes(
    boxes: List[BoundingBox],
    orig_w: int,
    orig_h: int,
    max_proposals: int,
) -> List[BoundingBox]:
    """
    Sorts by confidence descending, suppresses nested duplicates (IoA > 0.85),
    drops degenerate whole-table false boxes when real strip boxes exist,
    truncates to `max_proposals`, and returns reading-order sorted boxes.
    """
    if not boxes:
        return []

    img_area = float(max(1, orig_w * orig_h))
    ranked = sorted(boxes, key=lambda b: b.confidence, reverse=True)

    # Drop degenerate full-table boxes (>94% of image) if smaller strip boxes were detected
    if len(ranked) > 1:
        non_full = [b for b in ranked if ((b.x2 - b.x1) * (b.y2 - b.y1)) / img_area <= 0.94]
        if non_full:
            ranked = non_full

    kept: List[BoundingBox] = []
    for cand in ranked:
        if any(_box_ioa(cand, prev) > 0.85 for prev in kept):
            continue
        kept.append(cand)
        if len(kept) >= max_proposals:
            break

    kept.sort(key=lambda b: (b.y1, b.x1))
    return kept


def decode_onnx_predictions(
    raw_output: np.ndarray,
    scale: float,
    pad_left: float,
    pad_top: float,
    orig_w: int,
    orig_h: int,
    conf_threshold: float = YOLO_CONF_THRESHOLD,
    iou_threshold: float = YOLO_IOU_THRESHOLD,
    max_proposals: int = YOLO_MAX_PROPOSALS,
) -> Tuple[List[BoundingBox], str]:
    """
    Schema-aware ONNX decoder supporting BOTH:
      1. Native Ultralytics YOLO26 end-to-end NMS-free output: `(1, N, 6)` -> `[x1, y1, x2, y2, conf, class_id]`
      2. Legacy dense YOLO head output: `(1, 4+nc, M)` or `(1, M, 4+nc)` -> `[cx, cy, w, h, cls_scores...]` + NMS
    Returns `(boxes, output_format)`.
    """
    arr = np.asarray(raw_output)
    if arr.ndim != 3 or arr.shape[0] < 1:
        return [], "identity_stub"

    # Case 1: Native YOLO26 end-to-end NMS-free `(1, 300, 6)` (or `(1, max_det, 6)`)
    if arr.shape[2] == 6:
        preds = arr[0]  # shape (N, 6): never collapses when N == 1
        confs = preds[:, 4]
        cls_ids = np.rint(preds[:, 5]).astype(np.int32)
        mask = (confs >= conf_threshold) & (cls_ids == 0)
        selected = preds[mask]

        boxes: List[BoundingBox] = []
        for row in selected:
            coords = unletterbox_xyxy(
                float(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                scale=scale,
                pad_left=pad_left,
                pad_top=pad_top,
                orig_w=orig_w,
                orig_h=orig_h,
            )
            if coords is None:
                continue
            x1, y1, x2, y2 = coords
            boxes.append(
                BoundingBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=round(float(row[4]), 3),
                )
            )
        return _filter_and_sort_boxes(boxes, orig_w, orig_h, max_proposals), "end2end_xyxy6"

    # Case 2: Legacy dense YOLOv8/v11/v26(end2end=False) `(1, 4+nc, 8400)` or `(1, 8400, 4+nc)`
    preds2d = arr[0]
    if preds2d.shape[0] < preds2d.shape[1] and preds2d.shape[0] <= 100:
        preds2d = preds2d.T  # -> (8400, 4+nc)
    if preds2d.shape[1] < 5:
        return [], "unknown"

    class_scores = preds2d[:, 4:]
    max_scores = np.max(class_scores, axis=1)
    mask = max_scores >= conf_threshold
    filtered = preds2d[mask]
    filtered_scores = max_scores[mask]
    if filtered.shape[0] == 0:
        return [], "dense_xywh"

    raw_xywh: List[List[int]] = []
    score_list: List[float] = []
    for row, sc in zip(filtered, filtered_scores):
        cx, cy, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        coords = unletterbox_xyxy(
            cx - bw / 2.0,
            cy - bh / 2.0,
            cx + bw / 2.0,
            cy + bh / 2.0,
            scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
            orig_w=orig_w,
            orig_h=orig_h,
        )
        if coords is None:
            continue
        x1, y1, x2, y2 = coords
        raw_xywh.append([x1, y1, x2 - x1, y2 - y1])
        score_list.append(float(sc))

    if not raw_xywh:
        return [], "dense_xywh"

    boxes_out: List[BoundingBox] = []
    if cv2 is not None:
        indices = cv2.dnn.NMSBoxes(raw_xywh, score_list, score_threshold=conf_threshold, nms_threshold=iou_threshold)
        for idx in np.array(indices).flatten():
            x, y, bw_i, bh_i = raw_xywh[int(idx)]
            boxes_out.append(
                BoundingBox(
                    x1=x,
                    y1=y,
                    x2=x + bw_i,
                    y2=y + bh_i,
                    confidence=round(score_list[int(idx)], 3),
                )
            )
    else:
        for (x, y, bw_i, bh_i), sc in zip(raw_xywh, score_list):
            boxes_out.append(BoundingBox(x1=x, y1=y, x2=x + bw_i, y2=y + bh_i, confidence=round(sc, 3)))

    return _filter_and_sort_boxes(boxes_out, orig_w, orig_h, max_proposals), "dense_xywh"


def preprocess_for_ocr_with_scale(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Enhances contrast on shiny/reflective medicine blister strips using CLAHE on the L-channel
    and mild sharpening. Returns `(enhanced_bgr, upscale_factor)` so callers can map crop-space
    OCR bounding boxes back into original pixel coordinates without drift.
    """
    if cv2 is None or image is None or image.size == 0:
        return image, 1.0

    bgr = _to_bgr3(image).copy()
    h, w = bgr.shape[:2]
    scale = 1.0
    if max(h, w) < 640 and max(h, w) > 0:
        scale = 640.0 / float(max(h, w))
        bgr = cv2.resize(bgr, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_CUBIC)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_chan)
    merged = cv2.merge((l_enhanced, a_chan, b_chan))
    enhanced_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    gaussian = cv2.GaussianBlur(enhanced_bgr, (0, 0), 1.2)
    sharpened = cv2.addWeighted(enhanced_bgr, 1.35, gaussian, -0.35, 0)
    return sharpened, scale


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    enhanced, _ = preprocess_for_ocr_with_scale(image)
    return enhanced


class StripDetector:
    """
    Stateless-per-call YOLO26s/n ONNX Blister-Strip Detector + Reflective Foil CLAHE Preprocessor.
    - Inspects ONNX file integrity (rejecting 148 B Identity stubs and Git LFS text pointers).
    - Uses aspect-preserving letterbox resize + native YOLO26 `(1, 300, 6)` NMS-free decoding.
    - Caps ONNXRuntime CPU threads and warms up once at startup.
    - Never fabricates fake Canny contours as YOLO detections; callers own explicit hybrid fallback.
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self.model_path = Path(model_path).resolve() if model_path else YOLO_MODEL_PATH
        self._ort_session = None
        self._ultralytics_model = None
        self._input_name: str = "images"
        self._input_hw: Tuple[int, int] = (YOLO_IMG_SIZE, YOLO_IMG_SIZE)
        self._input_shape: List[Any] = [1, 3, YOLO_IMG_SIZE, YOLO_IMG_SIZE]
        self._output_shape: List[Any] = []
        self._output_format: str = "placeholder_stub"
        self._is_real_model: bool = False
        self._file_size_bytes: int = 0
        self._sha256_prefix: str = ""
        self._variant_name: str = "YOLO26s ONNX"
        self._ort_version: str = ""
        self._load_model()

    @property
    def is_real_model(self) -> bool:
        return self._is_real_model

    @property
    def model_label(self) -> str:
        if self._is_real_model:
            return f"{self._variant_name} (CPUExecutionProvider)"
        return "YOLO26 ONNX (Placeholder Stub -> OCR Fallback)"

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "modelLabel": self.model_label,
            "variant": self._variant_name,
            "fileName": self.model_path.name,
            "fileSizeMb": round(self._file_size_bytes / (1024 * 1024), 2),
            "sha256Prefix": self._sha256_prefix,
            "isRealModel": self._is_real_model,
            "status": "ready_yolo26_onnx" if self._is_real_model else "degraded_placeholder_stub",
            "outputFormat": self._output_format,
            "inputShape": self._input_shape,
            "outputShape": self._output_shape,
            "provider": "CPUExecutionProvider",
            "ortVersion": self._ort_version,
            "ortIntraThreads": YOLO_ORT_THREADS,
            "defaultConfThreshold": YOLO_CONF_THRESHOLD,
            "defaultPipelineMode": YOLO_PIPELINE_MODE,
        }

    def _load_model(self) -> None:
        if not self.model_path.exists():
            msg = f"[DEV-YOLO] Model file not found at {self.model_path}; running in OCR-fallback mode."
            if YOLO_REQUIRE_REAL_MODEL:
                raise RuntimeError(msg)
            logger.warning(msg)
            return

        self._file_size_bytes = self.model_path.stat().st_size

        # Reject Git LFS pointer files and 148-byte placeholder stubs explicitly
        try:
            header = self.model_path.read_bytes()[:64]
            if header.startswith(b"version https://git-lfs.github.com"):
                msg = f"[DEV-YOLO] {self.model_path} is a Git LFS pointer ({self._file_size_bytes} B), not real ONNX weights!"
                if YOLO_REQUIRE_REAL_MODEL:
                    raise RuntimeError(msg)
                logger.error(msg)
                return
        except OSError as exc:
            logger.warning("[DEV-YOLO] Unable to read %s: %s", self.model_path, exc)
            return

        if self._file_size_bytes < MIN_REAL_ONNX_BYTES:
            msg = (
                f"[DEV-YOLO] {self.model_path.name} is a {self._file_size_bytes}-byte placeholder stub "
                "(not trained YOLO26 weights). Falling back to RapidOCR spatial clustering until trained "
                "yolo_strips.onnx (~10–40 MB) is placed in data/processed/."
            )
            if YOLO_REQUIRE_REAL_MODEL:
                raise RuntimeError(msg)
            logger.warning(msg)
            self._output_format = "placeholder_stub"
            self._is_real_model = False
            return

        # Compute short sha256 fingerprint for observability
        h = hashlib.sha256()
        with self.model_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        self._sha256_prefix = h.hexdigest()[:12]

        # Infer variant label from file size (~9-15 MB = YOLO26n, ~25-45 MB = YOLO26s, >45 MB = YOLO26m)
        size_mb = self._file_size_bytes / (1024 * 1024)
        if size_mb < 16.0:
            self._variant_name = "YOLO26n ONNX"
        elif size_mb < 50.0:
            self._variant_name = "YOLO26s ONNX"
        else:
            self._variant_name = "YOLO26m ONNX"

        if self.model_path.suffix.lower() == ".onnx":
            try:
                import onnxruntime as ort  # type: ignore

                self._ort_version = getattr(ort, "__version__", "")
                sess_opts = ort.SessionOptions()
                sess_opts.intra_op_num_threads = YOLO_ORT_THREADS
                sess_opts.inter_op_num_threads = 1
                sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                session = ort.InferenceSession(
                    str(self.model_path),
                    sess_options=sess_opts,
                    providers=["CPUExecutionProvider"],
                )
                in_meta = session.get_inputs()[0]
                out_meta = session.get_outputs()[0]
                self._input_name = in_meta.name
                self._input_shape = list(in_meta.shape)
                self._output_shape = list(out_meta.shape)

                if (
                    len(self._input_shape) == 4
                    and isinstance(self._input_shape[2], int)
                    and isinstance(self._input_shape[3], int)
                    and self._input_shape[2] > 0
                    and self._input_shape[3] > 0
                ):
                    self._input_hw = (int(self._input_shape[2]), int(self._input_shape[3]))

                # Warmup pass once on startup so judges never pay cold-start latency
                dummy = np.zeros((1, 3, self._input_hw[0], self._input_hw[1]), dtype=np.float32)
                warm_out = session.run(None, {self._input_name: dummy})[0]
                if warm_out.ndim == 3 and warm_out.shape[2] == 6:
                    self._output_format = "end2end_xyxy6"
                elif warm_out.ndim == 3:
                    self._output_format = "dense_xywh"
                else:
                    msg = f"[DEV-YOLO] Unexpected ONNX output shape {warm_out.shape}; refusing to treat as real YOLO26 detector."
                    if YOLO_REQUIRE_REAL_MODEL:
                        raise RuntimeError(msg)
                    logger.error(msg)
                    return

                self._ort_session = session
                self._is_real_model = True
                logger.info(
                    "[DEV-YOLO] Loaded real %s (%s, %.2f MB, sha256=%s, in=%s, out=%s, format=%s, threads=%d)",
                    self._variant_name,
                    self.model_path.name,
                    size_mb,
                    self._sha256_prefix,
                    self._input_shape,
                    list(warm_out.shape),
                    self._output_format,
                    YOLO_ORT_THREADS,
                )
                return
            except Exception as exc:
                msg = f"[DEV-YOLO] ONNXRuntime load error for {self.model_path}: {exc}"
                if YOLO_REQUIRE_REAL_MODEL:
                    raise RuntimeError(msg) from exc
                logger.error(msg)
                self._ort_session = None
                self._is_real_model = False
                return

        # Optional `.pt` local fallback if Ultralytics is installed in dev
        try:
            from ultralytics import YOLO  # type: ignore

            self._ultralytics_model = YOLO(str(self.model_path))
            self._output_format = "ultralytics_pt"
            self._is_real_model = True
            logger.info("[DEV-YOLO] Loaded Ultralytics YOLO model (%s)", self.model_path.name)
        except Exception as exc:
            logger.warning("[DEV-YOLO] Could not load %s via Ultralytics: %s", self.model_path.name, exc)

    @staticmethod
    def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
        return preprocess_for_ocr(image)

    @staticmethod
    def preprocess_for_ocr_with_scale(image: np.ndarray) -> Tuple[np.ndarray, float]:
        return preprocess_for_ocr_with_scale(image)

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        max_proposals: Optional[int] = None,
    ) -> DetectionResult:
        """
        Stateless per-call strip detector.
        Returns real YOLO boxes when real weights are loaded; returns `boxes=[]` with
        honest backend metadata when placeholder stub is active or no strips pass threshold.
        """
        t0 = time.perf_counter()
        if image is None or image.size == 0:
            return DetectionResult(
                strips=[],
                boxes=[],
                inference_ms=0,
                backend="disabled",
                is_real_model=self._is_real_model,
                model_label=self.model_label,
                raw_count=0,
            )

        if not self._is_real_model:
            return DetectionResult(
                strips=[],
                boxes=[],
                inference_ms=0,
                backend="placeholder_stub",
                is_real_model=False,
                model_label=self.model_label,
                raw_count=0,
            )

        conf = YOLO_CONF_THRESHOLD if conf_threshold is None else max(0.05, min(0.95, float(conf_threshold)))
        iou = YOLO_IOU_THRESHOLD if iou_threshold is None else max(0.10, min(0.95, float(iou_threshold)))
        cap = YOLO_MAX_PROPOSALS if max_proposals is None else max(1, int(max_proposals))

        try:
            if self._ort_session is not None and cv2 is not None:
                boxes = self._detect_with_onnx(image, conf_threshold=conf, iou_threshold=iou, max_proposals=cap)
                elapsed_ms = max(1, int((time.perf_counter() - t0) * 1000))
                strips = self._crops_from_boxes(image, boxes)
                logger.info(
                    "[DEV-YOLO] %s detected %d strip box(es) in %dms (conf>=%.2f)",
                    self._variant_name,
                    len(boxes),
                    elapsed_ms,
                    conf,
                )
                return DetectionResult(
                    strips=strips,
                    boxes=boxes,
                    inference_ms=elapsed_ms,
                    backend="yolo26_onnx",
                    is_real_model=True,
                    model_label=self.model_label,
                    raw_count=len(boxes),
                )

            if self._ultralytics_model is not None:
                boxes = self._detect_with_ultralytics(image, conf_threshold=conf, iou_threshold=iou, max_proposals=cap)
                elapsed_ms = max(1, int((time.perf_counter() - t0) * 1000))
                strips = self._crops_from_boxes(image, boxes)
                return DetectionResult(
                    strips=strips,
                    boxes=boxes,
                    inference_ms=elapsed_ms,
                    backend="ultralytics",
                    is_real_model=True,
                    model_label=self.model_label,
                    raw_count=len(boxes),
                )
        except Exception as exc:
            elapsed_ms = max(1, int((time.perf_counter() - t0) * 1000))
            logger.exception("[DEV-YOLO] Inference failed (%s); failing open to OCR fallback.", exc)
            return DetectionResult(
                strips=[],
                boxes=[],
                inference_ms=elapsed_ms,
                backend="error_fallback",
                is_real_model=self._is_real_model,
                model_label=self.model_label,
                raw_count=0,
            )

        return DetectionResult(
            strips=[],
            boxes=[],
            inference_ms=0,
            backend="disabled",
            is_real_model=False,
            model_label=self.model_label,
            raw_count=0,
        )

    def detect_strips(
        self,
        image: np.ndarray,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> List[DetectedStrip]:
        """Backward-compatible wrapper returning `List[DetectedStrip]`."""
        return self.detect(image, conf_threshold=conf_threshold, iou_threshold=iou_threshold).strips

    def _detect_with_ultralytics(
        self,
        image: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
        max_proposals: int,
    ) -> List[BoundingBox]:
        h_orig, w_orig = image.shape[:2]
        results = self._ultralytics_model.predict(image, conf=conf_threshold, iou=iou_threshold, verbose=False)
        boxes: List[BoundingBox] = []
        if not results:
            return boxes
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = [int(round(v)) for v in b.xyxy[0].tolist()]
                conf = float(b.conf[0].item())
                boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=round(conf, 3)))
        return _filter_and_sort_boxes(boxes, w_orig, h_orig, max_proposals)

    def _detect_with_onnx(
        self,
        image: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
        max_proposals: int,
    ) -> List[BoundingBox]:
        h_orig, w_orig = image.shape[:2]
        padded, scale, pad_left, pad_top = letterbox_image(image, target_size=self._input_hw)
        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(np.transpose(blob, (2, 0, 1))[np.newaxis, ...])

        outputs = self._ort_session.run(None, {self._input_name: blob})
        boxes, _ = decode_onnx_predictions(
            outputs[0],
            scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
            orig_w=w_orig,
            orig_h=h_orig,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_proposals=max_proposals,
        )
        return boxes

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
