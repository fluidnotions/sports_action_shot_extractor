"""Action Shot Extractor package exports."""

from .pipeline import FrameResult, RunSummary, run_pipeline
from .utils import hex_to_bgr, hue_distance

__all__ = [
    "__version__",
    "FrameResult",
    "RunSummary",
    "run_pipeline",
    "hex_to_bgr",
    "hue_distance",
]

__version__ = "0.1.0"
