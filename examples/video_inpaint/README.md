Video inpainting prototype
===========================

This folder contains a small prototype `VideoInpaintPipeline` that wraps an image inpainting pipeline (StableDiffusionInpaintPipeline from diffusers) and adds minimal scaffolding for temporal processing.

Purpose
- Provide a minimal, GPU-friendly API shape for video inpainting.
- Include a fallback mock implementation so the example and unit test run quickly without heavy model downloads.

Files
- `video_inpaint_pipeline.py`: the prototype pipeline + CLI.
- `test_video_inpaint.py`: a lightweight pytest that runs with the mock backend.
- `requirements.txt`: optional dependencies (diffusers, transformers, torch, opencv-python, imageio-ffmpeg).

Usage
1. (Optional) Install dependencies for real-model runs:

```powershell
python -m pip install -r examples\video_inpaint\requirements.txt
```

API contract
- Factory: `VideoInpaintPipeline.from_pretrained(model_id_or_path: str | None = None, use_flow: bool = True, **kwargs) -> VideoInpaintPipeline`.
- Call: `result = pipe(frames: List[np.ndarray], masks: Optional[List[np.ndarray]] = None, prompt: Optional[str] = None, **kwargs)` returning `VideoResult(frames: List[np.ndarray], fps: float)`.

Acceptance criteria for PRs based on this prototype
- Fast unit test that uses synthetic frames and a mock backend so CI doesn't download models.
- Factory that loads `StableDiffusionInpaintPipeline` when `diffusers` is installed; otherwise falls back to the mock backend.
- Documentation (README and example notebook) and optional integration tests gated behind extras.

2. Run the CLI (with a real model if installed) or test with the mock backend:

```powershell
PYTHONPATH=src python examples\\video_inpaint\\video_inpaint_pipeline.py --out out.mp4
```

Notes
- Tests use a mock backend. To run the example script directly, set PYTHONPATH to src.
Notes
- This is a minimal prototype to demonstrate API design and a working unit test. For production-quality video inpainting plug in a high-quality optical-flow method (RAFT/GMFlow) and the full Diffusers StableDiffusionInpaintPipeline.
