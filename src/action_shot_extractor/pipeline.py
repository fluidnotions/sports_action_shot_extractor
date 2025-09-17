"""Core pipeline orchestration for FrameScout."""

from __future__ import annotations

import logging
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .utils import (
    bgr_to_hue_deg,
    color_match_ratio,
    extract_frames,
    find_ffmpeg,
    hex_to_bgr,
    is_sharp,
)
from .player_matcher import create_player_matcher

logger = logging.getLogger(__name__)

__all__ = [
    "FrameResult",
    "RunSummary",
    "ensure_device",
    "load_yolo_model",
    "load_segformer_model",
    "detect_objects",
    "segment_frame",
    "frame_is_hit",
    "run_pipeline",
]


@dataclass
class FrameResult:
    """Result metadata for an analysed frame."""

    index: int
    frame_path: Path
    sharpness: float
    color_ratio: float
    detection_count: int
    mask_ratio: float
    hit: bool
    export_path: Optional[Path] = None


@dataclass
class RunSummary:
    """High-level summary returned by :func:`run_pipeline`."""

    video: Path
    output_dir: Path
    frames_dir: Path
    detection_method: str
    detection_display: str
    target_hue: float
    ffmpeg_path: Path
    device: str
    frames_processed: int
    hits: List[FrameResult]
    dry_run: bool = False


def ensure_device(device: Optional[str] = None) -> str:
    """Resolve the requested device string."""

    requested = (device or "cpu").strip().lower()
    if requested in {"", "cpu"}:
        return "cpu"

    if requested == "auto":
        try:
            import torch  # type: ignore
        except ImportError:
            return "cpu"
        if torch.cuda.is_available():  # type: ignore[attr-defined]
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return "mps"
        return "cpu"

    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover - executed only when torch missing
        raise RuntimeError("PyTorch is required for non-CPU execution") from exc

    if requested == "cuda":
        if not torch.cuda.is_available():  # type: ignore[attr-defined]
            raise RuntimeError("CUDA device requested but CUDA is not available")
        return "cuda"
    if requested.startswith("cuda:"):
        if not torch.cuda.is_available():  # type: ignore[attr-defined]
            raise RuntimeError("CUDA device requested but CUDA is not available")
        index = int(requested.split(":", 1)[1])
        device_count = torch.cuda.device_count()
        if index >= device_count:
            raise RuntimeError(
                f"CUDA device index {index} requested but only {device_count} device(s) present"
            )
        return f"cuda:{index}"
    if requested == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():  # type: ignore[attr-defined]
            raise RuntimeError("Apple MPS backend is not available on this system")
        return "mps"

    return requested


def load_yolo_model(device: str):
    """Load the YOLOv8s model using Ultralytics."""

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:  # pragma: no cover - heavy optional dependency
        raise RuntimeError("Ultralytics package is required for detection") from exc

    model = YOLO("yolov8s.pt")
    try:
        model.to(device)
    except AttributeError:
        logger.debug("YOLO model does not implement .to(); relying on default device")
    return model


def load_segformer_model(device: str):
    """Load the SegFormer model and image processor from HuggingFace."""

    try:
        from transformers import (  # type: ignore
            AutoImageProcessor,
            SegformerForSemanticSegmentation,
        )
    except ImportError as exc:  # pragma: no cover - heavy optional dependency
        raise RuntimeError("Transformers package is required for segmentation") from exc

    processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

    try:
        import torch  # type: ignore
    except ImportError:
        if device not in {"cpu", ""}:
            raise RuntimeError("PyTorch is required for GPU inference")
    else:  # pragma: no cover - depends on torch availability
        model.to(device)
    model.eval()
    return processor, model


def detect_objects(
    model: Any,
    frame: np.ndarray,
    *,
    conf: float,
    iou: float,
    imgsz: int,
) -> List[Dict[str, Any]]:
    """Run YOLO detection and normalise the results."""

    if model is None:
        return []

    predictions = model.predict(source=frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    detections: List[Dict[str, Any]] = []
    for prediction in predictions:
        boxes = getattr(prediction, "boxes", None)
        if boxes is None:
            continue
        xyxy = getattr(boxes, "xyxy", None)
        confs = getattr(boxes, "conf", None)
        classes = getattr(boxes, "cls", None)

        if xyxy is None:
            continue
        coords = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else xyxy
        if confs is not None:
            conf_values = confs.cpu().numpy() if hasattr(confs, "cpu") else confs
        else:
            conf_values = [conf] * len(coords)
        if classes is not None:
            class_values = classes.cpu().numpy() if hasattr(classes, "cpu") else classes
        else:
            class_values = [0] * len(coords)

        for bbox, confidence, class_id in zip(coords, conf_values, class_values):
            if hasattr(bbox, "tolist"):
                bbox_values = bbox.tolist()
            else:
                bbox_values = bbox
            detections.append(
                {
                    "bbox": tuple(float(v) for v in bbox_values),
                    "confidence": float(confidence),
                    "class_id": int(class_id),
                }
            )
    return detections


def segment_frame(processor, model, frame: np.ndarray, device: str):
    """Generate a segmentation mask for a given frame."""

    if processor is None or model is None:
        return None

    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover - requires torch
        raise RuntimeError("PyTorch is required for segmentation") from exc

    rgb_frame = frame[:, :, ::-1]
    inputs = processor(images=rgb_frame, return_tensors="pt")
    if device != "cpu":
        inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        logits = torch.nn.functional.interpolate(
            logits,
            size=rgb_frame.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        mask = logits.argmax(dim=1).squeeze(0).to("cpu").numpy()
    return mask


def mask_focus_ratio(mask: Optional[np.ndarray], detections: Sequence[Dict[str, Any]], proximity: int) -> float:
    """Compute the ratio of mask pixels overlapping detection boxes."""

    if mask is None or mask.size == 0 or not detections:
        return 0.0
    mask_array = np.asarray(mask)
    if mask_array.ndim != 2:
        mask_array = mask_array.squeeze()
    if mask_array.ndim != 2:
        return 0.0
    height, width = mask_array.shape
    focus = np.zeros((height, width), dtype=bool)
    for det in detections:
        bbox = det.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = bbox[:4]
        x1 = max(int(math.floor(x1)) - proximity, 0)
        y1 = max(int(math.floor(y1)) - proximity, 0)
        x2 = min(int(math.ceil(x2)) + proximity, width)
        y2 = min(int(math.ceil(y2)) + proximity, height)
        if x2 <= x1 or y2 <= y1:
            continue
        focus[y1:y2, x1:x2] = True
    mask_pixels = mask_array > 0
    if not np.any(mask_pixels):
        return 0.0
    overlap = np.logical_and(mask_pixels, focus)
    if not np.any(overlap):
        return 0.0
    return float(overlap.sum()) / float(mask_pixels.sum())


def frame_is_hit(
    *,
    sharpness: float,
    color_ratio: float,
    detection_count: int,
    mask_ratio: float,
    blur_threshold: float,
    keep_min_pct: float,
) -> bool:
    """Decide whether a frame qualifies as a hit."""

    if sharpness < blur_threshold:
        return False
    if color_ratio < keep_min_pct:
        return False
    if detection_count <= 0:
        return False
    if mask_ratio < max(keep_min_pct * 0.5, 1e-3):
        return False
    return True


def frame_is_hit_with_player(
    *,
    sharpness: float,
    color_ratio: float,
    player_confidence: float,
    detection_count: int,
    mask_ratio: float,
    blur_threshold: float,
    keep_min_pct: float,
    detection_method: str,
) -> bool:
    """Decide whether a frame qualifies as a hit, considering player detection."""

    # Basic quality checks
    if sharpness < blur_threshold:
        return False
    if detection_count <= 0:
        return False
    if mask_ratio < max(keep_min_pct * 0.5, 1e-3):
        return False

    # Player-specific checks based on detection method
    if detection_method == "color":
        # For color detection, use both traditional color ratio and player confidence
        if color_ratio < keep_min_pct and player_confidence < 0.3:
            return False
    elif detection_method == "reference":
        # For reference detection, rely primarily on player confidence
        if player_confidence < 0.3:
            return False

    return True


def process_single_frame(
    frame_path: Path,
    index: int,
    player_matcher,
    yolo_model,
    processor,
    segformer_model,
    detection_method: str,
    target_hue: float,
    blur_thr: float,
    hue_tol: float,
    sat_min: float,
    val_min: float,
    conf: float,
    iou: float,
    imgsz: int,
    proximity: int,
    keep_min_pct: float,
    resolved_device: str,
) -> FrameResult:
    """Process a single frame and return the result."""

    frame = load_frame(frame_path)
    _, sharpness_value = is_sharp(frame, threshold=blur_thr)

    # Use player matcher for detection
    player_result = player_matcher.find_player_in_frame(frame)
    player_confidence = player_result['confidence'] if player_result['found'] else 0.0

    # For backward compatibility, also compute color ratio if using color method
    if detection_method == "color":
        color_ratio = color_match_ratio(frame, target_hue, hue_tol, sat_min=sat_min, val_min=val_min)
    else:
        color_ratio = player_confidence  # Use player confidence as color ratio equivalent

    detections = detect_objects(yolo_model, frame, conf=conf, iou=iou, imgsz=imgsz)
    mask = segment_frame(processor, segformer_model, frame, resolved_device)
    mask_ratio = mask_focus_ratio(mask, detections, proximity)

    # Modified hit detection to consider player matching
    hit = frame_is_hit_with_player(
        sharpness=sharpness_value,
        color_ratio=color_ratio,
        player_confidence=player_confidence,
        detection_count=len(detections),
        mask_ratio=mask_ratio,
        blur_threshold=blur_thr,
        keep_min_pct=keep_min_pct,
        detection_method=detection_method,
    )

    return FrameResult(
        index=index,
        frame_path=frame_path,
        sharpness=sharpness_value,
        color_ratio=color_ratio,
        detection_count=len(detections),
        mask_ratio=mask_ratio,
        hit=hit,
        export_path=None,  # Will be set later if it's a hit
    )


def load_frame(frame_path: Path) -> np.ndarray:
    """Load an image into a BGR numpy array."""

    try:
        import cv2  # type: ignore
    except ImportError:
        cv2 = None
    if cv2 is not None:
        frame = cv2.imread(str(frame_path))
        if frame is not None:
            return frame

    from PIL import Image

    with Image.open(frame_path) as image:
        return np.array(image.convert("RGB"))[:, :, ::-1]


def run_pipeline(
    *,
    video: Path | str,
    detection_method: str,
    detection_config: Dict[str, Any],
    output_dir: Optional[Path | str] = None,
    fps: Optional[float] = None,
    conf: float = 0.35,
    iou: float = 0.45,
    imgsz: int = 640,
    blur_thr: float = 150.0,
    hue_tol: float = 15.0,
    sat_min: float = 0.2,
    val_min: float = 0.2,
    proximity: int = 25,
    keep_min_pct: float = 0.05,
    device: Optional[str] = "cpu",
    tempdir: Optional[Path | str] = None,
    max_frames: Optional[int] = None,
    max_workers: Optional[int] = None,
    dry_run: bool = False,
) -> RunSummary:
    """Execute the end-to-end frame scouting pipeline."""

    video_path = Path(video).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if fps is not None and fps <= 0:
        raise ValueError("fps must be a positive value")
    if not (0.0 < conf <= 1.0):
        raise ValueError("conf must be in the (0, 1] range")
    if not (0.0 < iou <= 1.0):
        raise ValueError("iou must be in the (0, 1] range")
    if imgsz <= 0:
        raise ValueError("imgsz must be a positive integer")
    if blur_thr < 0:
        raise ValueError("blur_thr must be non-negative")
    if hue_tol < 0:
        raise ValueError("hue_tol must be non-negative")
    if not (0.0 <= sat_min <= 1.0):
        raise ValueError("sat_min must be between 0 and 1")
    if not (0.0 <= val_min <= 1.0):
        raise ValueError("val_min must be between 0 and 1")
    if keep_min_pct < 0:
        raise ValueError("keep_min_pct must be non-negative")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be a positive integer")

    hits_dir = Path(output_dir) if output_dir else video_path.parent / f"{video_path.stem}_framescout"
    hits_dir = hits_dir.expanduser().resolve()
    hits_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = Path(tempdir) if tempdir else hits_dir / "_frames"
    frames_dir = frames_dir.expanduser().resolve()
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Initialize player matcher based on detection method
    if detection_method == "color":
        hex_color = detection_config["hex_color"]
        player_matcher = create_player_matcher("color", hex_color=hex_color)
        target_bgr = hex_to_bgr(hex_color)
        target_hue = bgr_to_hue_deg(*target_bgr)
        detection_display = hex_color
    elif detection_method == "reference":
        reference_dir = detection_config["reference_dir"]
        player_matcher = create_player_matcher("reference", reference_dir=str(reference_dir))
        target_hue = 0.0  # Not used for reference matching
        detection_display = f"Reference frames from {reference_dir.name}"
    else:
        raise ValueError(f"Unknown detection method: {detection_method}")

    ffmpeg_path = find_ffmpeg()
    resolved_device = ensure_device(device)

    summary = RunSummary(
        video=video_path,
        output_dir=hits_dir,
        frames_dir=frames_dir,
        detection_method=detection_method,
        detection_display=detection_display,
        target_hue=target_hue,
        ffmpeg_path=ffmpeg_path,
        device=resolved_device,
        frames_processed=0,
        hits=[],
        dry_run=dry_run,
    )

    if dry_run:
        logger.info("Dry run requested; skipping heavy computation")
        return summary

    frame_paths = extract_frames(
        video_path,
        frames_dir,
        fps=fps,
        max_frames=max_frames,
        ffmpeg_path=ffmpeg_path,
        dry_run=False,
    )

    if not frame_paths:
        logger.warning("No frames were extracted from %s", video_path)
        return summary

    yolo_model = load_yolo_model(resolved_device)
    processor, segformer_model = load_segformer_model(resolved_device)

    # Determine optimal number of workers
    if max_workers is None:
        import os
        max_workers = min(4, os.cpu_count() or 1)

    logger.info(f"Processing {len(frame_paths)} frames using {max_workers} workers")

    # Process frames in parallel
    results: List[FrameResult] = []

    if max_workers == 1:
        # Sequential processing (fallback)
        for index, frame_path in enumerate(frame_paths):
            result = process_single_frame(
                frame_path=frame_path,
                index=index,
                player_matcher=player_matcher,
                yolo_model=yolo_model,
                processor=processor,
                segformer_model=segformer_model,
                detection_method=detection_method,
                target_hue=target_hue,
                blur_thr=blur_thr,
                hue_tol=hue_tol,
                sat_min=sat_min,
                val_min=val_min,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                proximity=proximity,
                keep_min_pct=keep_min_pct,
                resolved_device=resolved_device,
            )
            results.append(result)
    else:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all frame processing jobs
            future_to_frame = {
                executor.submit(
                    process_single_frame,
                    frame_path=frame_path,
                    index=index,
                    player_matcher=player_matcher,
                    yolo_model=yolo_model,
                    processor=processor,
                    segformer_model=segformer_model,
                    detection_method=detection_method,
                    target_hue=target_hue,
                    blur_thr=blur_thr,
                    hue_tol=hue_tol,
                    sat_min=sat_min,
                    val_min=val_min,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    proximity=proximity,
                    keep_min_pct=keep_min_pct,
                    resolved_device=resolved_device,
                ): (index, frame_path) for index, frame_path in enumerate(frame_paths)
            }

            # Collect results as they complete
            temp_results = {}
            for future in as_completed(future_to_frame):
                index, frame_path = future_to_frame[future]
                try:
                    result = future.result()
                    temp_results[index] = result
                except Exception as exc:
                    logger.error(f"Frame {index} ({frame_path}) generated an exception: {exc}")
                    # Create a failed result
                    temp_results[index] = FrameResult(
                        index=index,
                        frame_path=frame_path,
                        sharpness=0.0,
                        color_ratio=0.0,
                        detection_count=0,
                        mask_ratio=0.0,
                        hit=False,
                        export_path=None,
                    )

            # Sort results by index to maintain order
            results = [temp_results[i] for i in sorted(temp_results.keys())]

    # Process hits (copy files if needed)
    for result in results:
        if result.hit:
            export_path = hits_dir / result.frame_path.name
            if export_path != result.frame_path:
                shutil.copy2(result.frame_path, export_path)
            result.export_path = export_path

    summary.frames_processed = len(frame_paths)
    summary.hits = [result for result in results if result.hit]
    summary.dry_run = False
    return summary
