"""Smoke tests for the FrameScout project."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from framescout.cli import main as cli_main
from framescout.utils import hex_to_bgr, hue_distance, is_sharp


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

    from framescout.utils import find_ffmpeg

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
