"""Unit tests for the FramePreprocessor class."""

import numpy as np
import pytest

from src.pipeline.preprocessor import FramePreprocessor


def test_preprocessor_initialization():
    """Test initializing preprocessor with custom and default parameters."""
    prep_default = FramePreprocessor()
    assert prep_default.target_size == (640, 480)
    assert prep_default.normalize is True

    prep_custom = FramePreprocessor(target_size=(224, 224), normalize=False)
    assert prep_custom.target_size == (224, 224)
    assert prep_custom.normalize is False


def test_preprocessor_shape_and_color():
    """Test preprocessing output shape and color conversion."""
    # Create a dummy BGR frame (height=1080, width=1920)
    # Put a specific color value (e.g. blue color) in BGR: B=255, G=0, R=0
    raw_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    raw_frame[:, :, 0] = 255  # Blue channel in BGR

    prep = FramePreprocessor(target_size=(640, 480), normalize=False)
    processed = prep.process(raw_frame)

    # Output shape should be (height, width, channels)
    assert processed.shape == (480, 640, 3)
    assert processed.dtype == np.uint8

    # The color space conversion should swap blue (BGR channel 0) to red (RGB channel 0)
    # The output is RGB, so our blue BGR pixels (255, 0, 0) should become (0, 0, 255)
    # Check pixel value at 0, 0
    assert processed[0, 0, 0] == 0  # Red
    assert processed[0, 0, 1] == 0  # Green
    assert processed[0, 0, 2] == 255  # Blue


def test_preprocessor_normalization():
    """Test normalising image pixels to float32 [0.0, 1.0] scale."""
    raw_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    prep = FramePreprocessor(target_size=(640, 480), normalize=True)
    processed = prep.process(raw_frame)

    assert processed.shape == (480, 640, 3)
    assert processed.dtype == np.float32
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0


def test_preprocessor_invalid_input():
    """Test that preprocessor raises ValueError when input is None."""
    prep = FramePreprocessor()
    with pytest.raises(ValueError, match="Input frame cannot be None"):
        prep.process(None)  # type: ignore
