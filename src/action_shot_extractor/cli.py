"""Command line interface for FrameScout."""

from __future__ import annotations

from pathlib import Path

import click
from colorama import Fore, Style, init as colorama_init

from .pipeline import run_pipeline

colorama_init(autoreset=True)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--video",
    "video_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the input video file.",
)
@click.option(
    "--hex",
    "hex_color",
    type=str,
    help="Target color in hex format (e.g. #FF0000). Required for color-based detection.",
)
@click.option(
    "--reference-frames",
    "reference_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory containing front.jpg, side.jpg, back.jpg for feature-based detection.",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory where hit frames will be written.",
)
@click.option("--fps", type=float, default=None, help="Override frame extraction rate.")
@click.option("--conf", type=float, default=0.35, show_default=True, help="YOLO confidence threshold.")
@click.option("--iou", type=float, default=0.45, show_default=True, help="YOLO IoU threshold.")
@click.option("--imgsz", type=int, default=640, show_default=True, help="Inference image size for YOLO.")
@click.option("--blur-thr", type=float, default=150.0, show_default=True, help="Sharpness threshold.")
@click.option("--hue-tol", type=float, default=15.0, show_default=True, help="Hue tolerance in degrees.")
@click.option("--sat-min", type=float, default=0.2, show_default=True, help="Minimum saturation for a pixel to count.")
@click.option("--val-min", type=float, default=0.2, show_default=True, help="Minimum value/brightness for a pixel to count.")
@click.option("--proximity", type=int, default=25, show_default=True, help="Padding around detections when scoring masks.")
@click.option("--keep-min-pct", type=float, default=0.05, show_default=True, help="Minimum color ratio for a frame to qualify.")
@click.option("--device", type=str, default="cpu", show_default=True, help="Torch device to run inference on.")
@click.option(
    "--tempdir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Optional working directory for raw frames.",
)
@click.option("--max-frames", type=int, default=None, help="Limit the number of frames to inspect.")
@click.option("--workers", type=int, default=None, help="Number of parallel workers (default: auto-detect, max 4).")
@click.option("--dry-run", is_flag=True, help="Validate configuration without extracting frames.")
def main(
    *,
    video_path: Path,
    hex_color: str | None,
    reference_dir: Path | None,
    out_dir: Path | None,
    fps: float | None,
    conf: float,
    iou: float,
    imgsz: int,
    blur_thr: float,
    hue_tol: float,
    sat_min: float,
    val_min: float,
    proximity: int,
    keep_min_pct: float,
    device: str,
    tempdir: Path | None,
    max_frames: int | None,
    workers: int | None,
    dry_run: bool,
) -> None:
    """Entry point for Action Shot Extractor CLI."""

    # Validate detection method
    if not hex_color and not reference_dir:
        raise click.BadParameter(
            "Must specify either --hex for color-based detection or --reference-frames for feature-based detection."
        )

    if hex_color and reference_dir:
        raise click.BadParameter(
            "Cannot use both --hex and --reference-frames. Choose one detection method."
        )

    # Determine detection method
    if hex_color:
        detection_method = "color"
        detection_config = {"hex_color": hex_color}
    else:
        detection_method = "reference"
        detection_config = {"reference_dir": reference_dir}

    try:
        summary = run_pipeline(
            video=video_path,
            detection_method=detection_method,
            detection_config=detection_config,
            output_dir=out_dir,
            fps=fps,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            blur_thr=blur_thr,
            hue_tol=hue_tol,
            sat_min=sat_min,
            val_min=val_min,
            proximity=proximity,
            keep_min_pct=keep_min_pct,
            device=device,
            tempdir=tempdir,
            max_frames=max_frames,
            max_workers=workers,
            dry_run=dry_run,
        )
    except click.ClickException:
        raise
    except Exception as exc:
        click.echo(f"{Fore.RED}{Style.BRIGHT}Error:{Style.RESET_ALL} {exc}", err=True)
        raise click.ClickException(str(exc)) from exc

    if summary.dry_run:
        click.echo(
            (
                f"{Fore.YELLOW}{Style.BRIGHT}Dry run successful!{Style.RESET_ALL}\n"
                f"  ffmpeg: {summary.ffmpeg_path}\n"
                f"  device: {summary.device}\n"
                f"  detection: {summary.detection_display}"
            )
        )
        return

    click.echo(
        f"{Fore.GREEN}{Style.BRIGHT}Processed {summary.frames_processed} frame(s).{Style.RESET_ALL}"
    )
    click.echo(
        f"{Fore.GREEN}{len(summary.hits)} hit(s) saved to {summary.output_dir}{Style.RESET_ALL}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
