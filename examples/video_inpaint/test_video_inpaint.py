"""Unit tests for examples/video_inpaint."""

import numpy as np

from examples.video_inpaint.video_inpaint_pipeline import load_synthetic_video, VideoInpaintPipeline


def test_video_inpaint_runs_and_returns_same_length(tmp_path):
    frames, masks = load_synthetic_video(n_frames=4, size=(64, 64))
    pipe = VideoInpaintPipeline.from_pretrained(None, use_flow=False)
    result = pipe(frames, masks, prompt="test")
    assert len(result.frames) == len(frames)
    for f in result.frames:
        assert isinstance(f, np.ndarray)
        assert f.shape[2] == 3
