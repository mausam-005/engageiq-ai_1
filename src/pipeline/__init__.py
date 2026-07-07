"""Pipeline package - Capture and preprocessing utilities."""

from src.pipeline.capture import WebcamCapture
from src.pipeline.preprocessor import FramePreprocessor

__all__ = ["WebcamCapture", "FramePreprocessor"]
