"""Minimal VideoInpaintPipeline prototype.

This script provides a small wrapper around an image inpainting pipeline (if
available) and a fast mock fallback so the example can run quickly in tests.

Goals / API
- Provide a minimal, testable API that mirrors Diffusers pipelines so reviewers
    can see how VideoInpaintPipeline would be instantiated and used.

Factory:
    VideoInpaintPipeline.from_pretrained(model_id_or_path: str, use_flow: bool = True, **kwargs) -> VideoInpaintPipeline

Call:
    result = pipe(frames: List[np.ndarray], masks: Optional[List[np.ndarray]] = None, prompt: Optional[str] = None, **kwargs)
    -> returns VideoResult(frames: List[np.ndarray], fps: float)

Acceptance criteria (for a PR based on this prototype):
- A fast unit test using synthetic frames that runs in CI without model downloads.
- A factory method that uses a real StableDiffusionInpaintPipeline when diffusers
    are installed and otherwise falls back to the mock pipeline.
- Documentation of API and a short example showing how to run the pipeline.

This is a prototype aimed at examples/PRs; it's not production-ready.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    import imageio
except Exception:
    imageio = None


@dataclass
class VideoResult:
    frames: list[np.ndarray]
    fps: float = 10.0

    def save(self, path: str):
        if imageio is None:
            raise RuntimeError("imageio is required to save videos")
        imageio.mimsave(path, self.frames, fps=self.fps)


class MockInpaintPipeline:
    """A tiny mock inpainting pipeline that slightly blurs masked areas.

    This allows tests to run without downloading large models.
    """

    def __init__(self):
        pass

    def __call__(self, image: np.ndarray, mask: np.ndarray | None, prompt: str | None = None):
        out = image.copy()
        if mask is not None:
            # Simple effect: blur masked area
            k = 5
            blurred = cv2.GaussianBlur(out, (k, k), 0) if cv2 is not None else out
            m = (mask > 127).astype(np.uint8)[..., None]
            out = out * (1 - m) + blurred * m
            out = out.astype(np.uint8)
        return out


class VideoInpaintPipeline:
    def __init__(self, inpaint_pipe=None, use_flow: bool = True):
        """Create a VideoInpaintPipeline.

        Args:
            inpaint_pipe: callable(image, mask, prompt) -> image. If None, a
                small mock implementation is used so tests run quickly.
            use_flow: whether to use optical flow warping (requires OpenCV).
        """
        self.inpaint_pipe = inpaint_pipe or MockInpaintPipeline()
        self.use_flow = use_flow and (cv2 is not None)

    @classmethod
    def from_pretrained(cls, model_id_or_path: str | None = None, use_flow: bool = True, **kwargs: Any):
        """Factory that mirrors `diffusers` pipelines.

        If Diffusers and a StableDiffusionInpaintPipeline are available, this
        method will load the real image inpainting pipeline. Otherwise it will
        return a pipeline using the internal mock inpainting backend.

        Args:
            model_id_or_path: model identifier passed to the underlying image
                inpainting pipeline. Ignored by the mock backend.
            use_flow: whether to enable optical-flow warping.
            **kwargs: forwarded to the underlying from_pretrained call when
                diffusers is available.
        """
        try:
            # Lazy import to avoid heavy dependencies when not needed.
            from diffusers import StableDiffusionInpaintPipeline

            inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id_or_path, **kwargs)
            return cls(inpaint_pipe=inpaint_pipe, use_flow=use_flow)
        except Exception:
            # Fall back to the lightweight mock.
            print("diffusers StableDiffusionInpaintPipeline not available; using mock inpaint backend")
            return cls(inpaint_pipe=MockInpaintPipeline(), use_flow=use_flow)

    def _flow_warp(self, prev_frame: np.ndarray, curr_frame: np.ndarray, prev_latent: np.ndarray) -> np.ndarray:
        """Warp prev_latent toward curr_frame using Farneback optical flow between frames.

        prev_latent is an ndarray (H, W, C) matching frame size. We'll compute flow
        at frame resolution and remap prev_latent accordingly.
        """
        if not self.use_flow:
            return prev_latent
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        h, w = prev_latent.shape[:2]
        # flow is same size as frames; build map and remap
        flow_map = np.zeros((h, w, 2), dtype=np.float32)
        flow_map[..., 0] = np.arange(w)
        flow_map[..., 1] = np.arange(h)[:, None]
        flow_map += flow
        remapped = cv2.remap(prev_latent.astype(np.float32), flow_map[..., 0], flow_map[..., 1], cv2.INTER_LINEAR)
        return remapped.astype(prev_latent.dtype)

    def __call__(self, frames: list[np.ndarray], masks: list[np.ndarray] | None = None, prompt: str | None = None, **kwargs) -> VideoResult:
        n = len(frames)
        masks = masks or [None] * n
        out_frames = []
        prev_latent = None
        prev_frame: np.ndarray | None = None
        for i in range(n):
            frame = frames[i]
            mask = masks[i]
            if prev_latent is not None:
                # warp prev_latent to current frame as init (mocked)
                assert prev_frame is not None
                warped = self._flow_warp(prev_frame, frame, prev_latent)
                # In a real implementation we'd pass warped latents into the denoiser.
                # Here, we just blend warped and frame slightly to simulate reuse.
                init = (frame.astype(np.float32) * 0.7 + warped.astype(np.float32) * 0.3).astype(np.uint8)
            else:
                init = frame
            out = self.inpaint_pipe(init, mask, prompt)
            out_frames.append(out)
            prev_frame = frame
            prev_latent = out
        return VideoResult(out_frames)


def load_synthetic_video(n_frames=6, size=(128, 128)):
    frames = []
    masks = []
    for i in range(n_frames):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        # moving square
        x = int((i / n_frames) * (size[0] - 32))
        frame[32:64, x:x + 32] = [int(255 * (i / n_frames)), 50, 200]
        mask = np.zeros((size[1], size[0]), dtype=np.uint8)
        mask[32:64, x:x + 32] = 255
        frames.append(frame)
        masks.append(mask)
    return frames, masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="out.mp4")
    parser.add_argument("--use-flow", action="store_true")
    args = parser.parse_args()

    frames, masks = load_synthetic_video()
    pipe = VideoInpaintPipeline(use_flow=args.use_flow)
    result = pipe(frames, masks, prompt="remove square")
    tmp = tempfile.mkdtemp()
    out_path = os.path.join(tmp, args.out)
    result.save(out_path)
    print("Saved result to", out_path)


if __name__ == "__main__":
    main()
