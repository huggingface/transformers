Title: Add VideoInpaintPipeline — temporally coherent video inpainting (prototype)

Motivation
----------
The current community approach of calling image inpainting per-frame produces temporal flicker and poor GPU utilization. This PR introduces a VideoInpaintPipeline that performs temporally coherent video inpainting by reusing latents across frames and optionally warping them with optical flow.

What this PR contains
---------------------
- A prototype `VideoInpaintPipeline` under `examples/video_inpaint/` with a mock inpainting backend and optional Farneback optical-flow-based latent warping for fast tests and demos.
- A fast unit test using synthetic frames (no model downloads).
- README with API contract, usage, and acceptance criteria.
- `requirements.txt` listing optional heavy dependencies for real-model runs.

API (example)
-------------
```py
pipe = VideoInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", use_flow=True, torch_dtype=torch.float16)
result = pipe(frames=frames_list, masks=masks_list, prompt="replace background with a snowy mountain", num_inference_steps=10)
result.save("out.mp4")
```

Acceptance criteria
-------------------
- Fast unit test that uses synthetic frames and a mock backend (CI safe).
- Factory that loads `StableDiffusionInpaintPipeline` when `diffusers` is installed; otherwise falls back to the mock backend.
- Documentation (README and example notebook) and optional integration tests gated behind extras.

Notes & follow-ups
------------------
- This is a prototype; a production-ready implementation should reuse diffusion latents and scheduler state across frames, provide RAFT/GMFlow hooks, and include performance optimizations. Those will be submitted in follow-up PRs.
