"""Utility helpers for FrameScout."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "find_ffmpeg",
    "extract_frames",
    "is_sharp",
    "hex_to_bgr",
    "bgr_to_hex",
    "bgr_to_hsv",
    "bgr_to_hue_deg",
    "hue_distance",
    "matches_hue",
    "color_match_ratio",
]


def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Convert a hex color string into an OpenCV-friendly BGR tuple."""

    value = hex_color.strip().lower().lstrip("#")
    if value.startswith("0x"):
        value = value[2:]
    if len(value) != 6 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"Invalid hexadecimal color: {hex_color!r}")
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return (b, g, r)


def bgr_to_hex(b: int, g: int, r: int) -> str:
    """Convert BGR values back into a hex color string."""

    return f"#{r:02X}{g:02X}{b:02X}"


def bgr_to_hsv(b: int, g: int, r: int) -> Tuple[float, float, float]:
    """Convert a BGR triplet into HSV values."""

    rf = r / 255.0
    gf = g / 255.0
    bf = b / 255.0
    maxc = max(rf, gf, bf)
    minc = min(rf, gf, bf)
    delta = maxc - minc

    if delta == 0:
        hue = 0.0
    elif maxc == rf:
        hue = (60 * ((gf - bf) / delta) + 360) % 360
    elif maxc == gf:
        hue = (60 * ((bf - rf) / delta) + 120) % 360
    else:
        hue = (60 * ((rf - gf) / delta) + 240) % 360

    saturation = 0.0 if maxc == 0 else delta / maxc
    value = maxc
    return hue, saturation, value


def bgr_to_hue_deg(b: int, g: int, r: int) -> float:
    """Return the hue (in degrees) for a BGR pixel."""

    hue, _, _ = bgr_to_hsv(b, g, r)
    return hue


def hue_distance(a: float, b: float) -> float:
    """Return the wrapped distance between two hue angles."""

    diff = abs((a % 360) - (b % 360))
    return min(diff, 360.0 - diff)


def matches_hue(
    bgr: Sequence[int],
    target_hue: float,
    tolerance: float,
    *,
    sat_min: float = 0.0,
    val_min: float = 0.0,
) -> bool:
    """Check if a BGR pixel is within the desired hue tolerance."""

    if len(bgr) < 3:
        raise ValueError("Expected a 3-channel color for BGR input")
    hue, saturation, value = bgr_to_hsv(int(bgr[0]), int(bgr[1]), int(bgr[2]))
    if value < val_min or saturation < sat_min:
        return False
    return hue_distance(hue, target_hue) <= tolerance


def color_match_ratio(
    image: np.ndarray,
    target_hue: float,
    hue_tolerance: float,
    *,
    sat_min: float = 0.2,
    val_min: float = 0.2,
) -> float:
    """Return the ratio of pixels that match the requested color constraints."""

    if image is None:
        return 0.0
    array = np.asarray(image)
    if array.ndim == 2:
        # Grayscale image – no color content.
        return 0.0
    if array.shape[-1] < 3:
        raise ValueError("Color matching expects an array with at least 3 channels")

    flat = array.reshape(-1, array.shape[-1])
    total = flat.shape[0]
    if total == 0:
        return 0.0

    matches = 0
    for b, g, r, *_ in flat:
        if matches_hue((int(b), int(g), int(r)), target_hue, hue_tolerance, sat_min=sat_min, val_min=val_min):
            matches += 1
    return matches / float(total)


def is_sharp(image: np.ndarray, threshold: float = 100.0) -> Tuple[bool, float]:
    """Estimate if an image is sharp using the gradient energy heuristic."""

    if image is None:
        return False, 0.0
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 3:
        array = array.mean(axis=2)
    if array.size == 0:
        return False, 0.0

    gradients = np.gradient(array)
    if isinstance(gradients, (list, tuple)):
        energy = sum(float(np.var(g)) for g in gradients)
    else:
        energy = float(np.var(gradients))

    if not math.isfinite(energy):
        energy = 0.0
    return energy >= threshold, energy


def find_ffmpeg(additional_paths: Optional[Iterable[Path | str]] = None) -> Path:
    """Locate an ffmpeg executable across common installation locations."""

    candidates: List[Path] = []
    env_path = os.environ.get("FFMPEG_PATH")
    if env_path:
        candidates.append(Path(env_path))

    for name in ("ffmpeg", "ffmpeg.exe"):
        resolved = shutil.which(name)
        if resolved:
            return Path(resolved)

    if additional_paths:
        candidates.extend(Path(path) for path in additional_paths)

    if os.name == "nt":
        program_files = os.environ.get("ProgramFiles")
        program_files_x86 = os.environ.get("ProgramFiles(x86)")
        for root in filter(None, (program_files, program_files_x86)):
            for sub in ("ffmpeg", "FFmpeg"):
                candidates.append(Path(root) / sub / "bin" / "ffmpeg.exe")
    else:
        for prefix in ("/usr/bin", "/usr/local/bin", "/opt/homebrew/bin"):
            candidates.append(Path(prefix) / "ffmpeg")

    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return path

    raise FileNotFoundError("ffmpeg executable was not found on the system path")


def extract_frames(
    video_path: Path,
    output_dir: Path,
    *,
    fps: Optional[float] = None,
    max_frames: Optional[int] = None,
    ffmpeg_path: Optional[Path] = None,
    dry_run: bool = False,
) -> List[Path]:
    """Extract frames from a video using OpenCV or ffmpeg."""

    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        return []

    if ffmpeg_path is None:
        try:
            ffmpeg_path = find_ffmpeg()
        except FileNotFoundError:
            ffmpeg_path = None

    cv2 = _try_import_cv2()
    if cv2 is not None:
        return _extract_with_cv2(cv2, video_path, output_dir, fps=fps, max_frames=max_frames)

    if ffmpeg_path is None:
        raise RuntimeError("OpenCV and ffmpeg are unavailable for frame extraction")

    return _extract_with_ffmpeg(ffmpeg_path, video_path, output_dir, fps=fps, max_frames=max_frames)


def _try_import_cv2():
    try:
        import cv2  # type: ignore

        return cv2
    except ImportError:
        return None


def _extract_with_cv2(cv2, video_path: Path, output_dir: Path, *, fps: Optional[float], max_frames: Optional[int]) -> List[Path]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_interval = 1
    if fps and original_fps:
        frame_interval = max(int(round(original_fps / fps)), 1)

    extracted: List[Path] = []
    index = 0
    written = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        if index % frame_interval == 0:
            frame_path = output_dir / f"frame_{index:06d}.jpg"
            if not cv2.imwrite(str(frame_path), frame):
                cap.release()
                raise RuntimeError(f"Failed to write frame to {frame_path}")
            extracted.append(frame_path)
            written += 1
            if max_frames and written >= max_frames:
                break
        index += 1

    cap.release()
    return extracted


def _extract_with_ffmpeg(
    ffmpeg_path: Path,
    video_path: Path,
    output_dir: Path,
    *,
    fps: Optional[float],
    max_frames: Optional[int],
) -> List[Path]:
    template = output_dir / "frame_%06d.jpg"
    command = [str(ffmpeg_path), "-y", "-i", str(video_path)]
    if fps:
        command.extend(["-vf", f"fps={fps}"])
    if max_frames:
        command.extend(["-vframes", str(max_frames)])
    command.append(str(template))

    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed with exit code "
            f"{result.returncode}: {result.stderr.decode(errors='ignore')}"
        )

    return sorted(output_dir.glob("frame_*.jpg"))
