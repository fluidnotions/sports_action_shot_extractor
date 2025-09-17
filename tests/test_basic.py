"""Smoke tests for the Action Shot Extractor project."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from action_shot_extractor.cli import main as cli_main
from action_shot_extractor.utils import hex_to_bgr, hue_distance, is_sharp
from action_shot_extractor.player_matcher import create_player_matcher
from action_shot_extractor.pipeline import run_pipeline, FrameResult, ensure_device


def test_hex_to_bgr_parses_color():
    """Hex strings should convert to BGR tuples."""

    assert hex_to_bgr("#FF0000") == (0, 0, 255)
    assert hex_to_bgr("00FF00") == (0, 255, 0)
    assert hex_to_bgr("#1a2b3c") == (0x3C, 0x2B, 0x1A)


def test_hue_distance_wraps_correctly():
    """Hue distance must wrap around the 0/360 boundary."""

    assert pytest.approx(20.0, rel=1e-4) == hue_distance(350, 10)
    assert pytest.approx(0.0, rel=1e-4) == hue_distance(45, 45)


def test_is_sharp_discriminates_textures():
    """High frequency content should register as sharp."""

    sharp = np.zeros((64, 64), dtype=np.uint8)
    sharp[:, ::2] = 255
    smooth = np.full((64, 64), 127, dtype=np.uint8)

    sharp_flag, sharp_score = is_sharp(sharp, threshold=1.0)
    smooth_flag, smooth_score = is_sharp(smooth, threshold=1.0)

    assert sharp_flag
    assert not smooth_flag
    assert sharp_score > smooth_score


@pytest.mark.skipif(
    any(importlib.util.find_spec(pkg) is None for pkg in ("torch", "ultralytics", "transformers")),
    reason="Heavy optional dependencies are not installed.",
)
def test_cli_dry_run(tmp_path: Path):
    """The CLI should support dry-run execution for validation."""

    from action_shot_extractor.utils import find_ffmpeg

    try:
        find_ffmpeg()
    except FileNotFoundError:
        pytest.skip("ffmpeg is not available on the test system")

    video = tmp_path / "dummy.mp4"
    video.write_bytes(b"00")
    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        [
            "--video",
            str(video),
            "--hex",
            "#E10600",
            "--out",
            str(out_dir),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Dry run successful" in result.output


def test_player_matcher_creation():
    """Player matcher can be created with different methods."""

    # Test color-based matcher
    color_matcher = create_player_matcher("color", hex_color="#FF0000")
    assert color_matcher.method == "color"
    assert "target_hue" in color_matcher.matcher_data

    # Test invalid method
    with pytest.raises(ValueError, match="Unknown method"):
        create_player_matcher("invalid_method")


def test_device_resolution():
    """Device resolution should handle various inputs correctly."""

    # Test CPU default
    assert ensure_device("cpu") == "cpu"
    assert ensure_device("") == "cpu"
    assert ensure_device(None) == "cpu"

    # Test auto detection (should not raise)
    device = ensure_device("auto")
    assert device in ["cpu", "cuda", "mps", "cuda:0"]


def test_frame_result_creation():
    """FrameResult objects can be created with expected fields."""

    result = FrameResult(
        index=1,
        frame_path=Path("test.jpg"),
        sharpness=100.0,
        color_ratio=0.5,
        detection_count=2,
        mask_ratio=0.3,
        hit=True
    )

    assert result.index == 1
    assert result.frame_path == Path("test.jpg")
    assert result.sharpness == 100.0
    assert result.color_ratio == 0.5
    assert result.detection_count == 2
    assert result.mask_ratio == 0.3
    assert result.hit is True
    assert result.export_path is None


@pytest.mark.skipif(
    any(importlib.util.find_spec(pkg) is None for pkg in ("cv2", "numpy")),
    reason="OpenCV or NumPy not available.",
)
def test_color_player_matching():
    """Color-based player matching should work with synthetic data."""

    # Create a simple test frame with red pixels
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[40:60, 40:60] = [0, 0, 255]  # Red square in BGR

    # Create color matcher for red
    matcher = create_player_matcher("color", hex_color="#FF0000")

    # Find player in frame
    result = matcher.find_player_in_frame(frame)

    # Should find the red square
    assert result["found"] is True
    assert result["confidence"] > 0.0
    assert result["bbox"] is not None
    assert "mask" in result["method_data"]


@pytest.mark.skipif(
    any(importlib.util.find_spec(pkg) is None for pkg in ("torch", "ultralytics", "transformers")),
    reason="Heavy optional dependencies are not installed.",
)
def test_pipeline_with_reference_frames(tmp_path: Path):
    """Pipeline should accept reference frame detection method."""

    # Create dummy reference frames
    ref_dir = tmp_path / "references"
    ref_dir.mkdir()

    # Create minimal reference images (just to test config validation)
    for ref_name in ["front.jpg", "side.jpg", "back.jpg"]:
        ref_path = ref_dir / ref_name
        # Create a small dummy image
        dummy_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        import cv2
        cv2.imwrite(str(ref_path), dummy_image)

    # Test that reference detection config is accepted
    detection_config = {"reference_dir": ref_dir}

    # This should not raise an error during initialization
    matcher = create_player_matcher("reference", reference_dir=str(ref_dir))
    assert matcher.method == "reference"
    assert "reference_features" in matcher.matcher_data
