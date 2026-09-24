# TPU test suite: progress and plan of action

Status as of 2026-09-24. Build: `torch_tpu 0.1.1.dev20260923101118`, `libtpu 0.0.49`, `torch 2.11.0+cpu`,
TPU v6e with 8 chips, branch `tpu-tests`. Results are in `hf-gcp-tpu-internal/transformers_daily_ci`
(`2026-09-23/` fast run, `2026-09-24/` slow run); the details are in `conclusion.md`.

| Run | Passed | Failed | Skipped | Errors |
|---|---|---|---|---|
| fast | 3810 | 304 | 4070 | 0 |
| slow (`RUN_SLOW=1`) | 4224 | 402 | 3558 | 0 |

Legend: `[ ]` to do, `[~]` in progress, `[x]` done.

## 0. Failures from torch_tpu bugs already reported

Nothing to do but wait. Each has a probe in `utils/tpu_ci/backend_gaps.py` (all 15 still reproduce on
this build), so a fix shows up as a probe passing.

| torch_tpu issue | Gap | Slow-run failures |
|---|---|---|
| #4148 | attention and matmuls run in bfloat16 | 132 (some only suspected, see §4) |
| #4158 | `linalg_cholesky_ex` not implemented | 69 |
| #4147 | SDPA refuses an `attn_mask` that requires grad | 38 |
| #4166 | a second process cannot open a held device, aborts | 37 |
| #4164 | `Tensor.to(<int>)` ignores the index | 25 |
| #4161 | moving a module unties tied parameters | 6 |
| #4160 | `x[:, :, mask]` boolean indexing | 5 |
| #4141 | `avg_pool1d` has no gradient | 3 |
| #4159 | `unique(dim=...)` not implemented | 2 |
| #4163 | dynamo backend guards on storage offset | 2 |
| #4146 | a fully masked attention row leaks | 1 |
| #4142 | `multinomial` ignores the seed | 1 |
| #4143, #4144 | `ctc_loss` and same-size `interpolate` abort | 0 (still skipped, they would kill the process) |

- [ ] When one of these is fixed: its probe passes → re-run the models it affects, and remove the
  workaround if there is one (`ctc_loss`, `interpolate`).

## 1. Unreported torch_tpu bugs

All of these were reproduced standalone, without transformers, on this build. The write-ups for the
reporting agent are in `TORCH_TPU_BUGS.md`.

- [x] **B1 — antialiased upsampling is not implemented.** `interpolate(..., antialias=True)` raises
  for both `bicubic` (`aten::_upsample_bicubic2d_aa.out`) and `bilinear` (`_upsample_bilinear2d_aa`);
  without `antialias` both work. 2 failures (`test_can_compile_fast_video_processor` in internvl and
  smolvlm), and it breaks every fast image/video processor that resizes with antialiasing on TPU.
- [x] **B2 — `torch.compile(mode=...)` fails on TPU tensors.** The TPU backend does not accept `mode`:
  `mode="reduce-overhead"` and `"max-autotune"` raise `TypeError: Unexpected keyword arguments`, both
  with an explicit `backend="tpu"` and through torch_tpu's default `_default_backend_selector` (which
  forwards it). `mode="default"`, `options=` and CPU tensors are fine.
  1 failure (`LlamaModelTest::test_torch_compile_for_training`); `CompileConfig` defaults to
  `reduce-overhead`, so user code hits this too.
- [x] **B3 — `torch.compile(backend="inductor")` fails on TPU tensors** with a binding mismatch:
  `torch_tpu/_internal/pallas/pallas.py:713` passes `serialized_mlir_module=lowered.mlir_module_serialized`,
  which is `bytes`, while the `register_custom_kernel` binding only accepts `str`. No failure in the
  report, because the branch compiles with the backend the model picks for the device
  (`model._default_compile_config()`).
- [x] **B4 — multi-process single-host `tpu_dist` cannot be started with standard tooling.** 10 failures
  (the FSDP and TP tests, `missing required environment variables for distributed training`). Found:
  - the three variables it needs (`TORCH_TPU_SLICEBUILDER_ADDRESSES`, `TORCH_TPU_TOPOLOGY`,
    `TORCH_TPU_XPROF_SESSION_ID`) are only set by the private
    `torch_tpu._internal.distributed.launchers.environment.set_tpu_launch_env()`, called in the parent.
    The error suggests torchrun, but torchrun's ranks do not get them;
  - without them, `mp.spawn` ranks abort (`Check failed: client != nullptr PjRtClient is null after
    initialization`) instead of raising;
  - `set_tpu_launch_env` only accepts 1, 4 or 8 ranks on v6e, so the 2-rank TP and FSDP tests cannot run;
  - each rank sees its chip as `tpu:0`, so the usual `set_device(local_rank)` raises
    `ValueError: Cannot set TPU device to index 1, current process is bound to device index 0`.
  What works: `set_tpu_launch_env(nproc_per_node=4)` in the parent, `RANK`/`LOCAL_RANK`/`WORLD_SIZE` in
  each rank's environment, and no `set_device`. Four ranks then all-reduce correctly.
- [ ] Hand `TORCH_TPU_BUGS.md` to the reporting agent. Once the issues exist: add a probe for each
  to `backend_gaps.py`, and put the issue numbers in §0.

## 2. Changes needed in transformers or accelerate

- [ ] **T1 — `device_map="auto"` on TPU (accelerate).** accelerate enumerates devices with a fixed
  chain of backends and finds none on TPU, so the model stays on CPU while the inputs go to TPU:
  `tensor is expected to be on tpu, got cpu`. 6 failures (llama 7B logits ×2, qwen2_5_omni ×3,
  `ViTModelIntegrationTest::test_inference_fp16`). Plan: find what `torch.tpu` exposes for per-device
  memory (`mem_get_info` / `memory_stats` equivalents), then add TPU to accelerate's device enumeration
  and `get_max_memory`, as a PR to accelerate. Spreading over chips also needs #4164 fixed.
- [ ] **T2 — TPU reference values for integration tests.** 25 tests have no `Expectations` entry TPU
  can use (`No matching expectation found for ('tpu', None, None)`). 27 more produce text or logits that
  differ from the values recorded on other hardware. Plan, per model:
  1. check the difference is precision (§4 V2), not a real bug;
  2. add `("tpu", None)` entries from the TPU output, the way `xpu` and `rocm` have their own;
  3. hold back the tests that sample (`GPT2ModelLanguageGenerationTest::test_gpt2_sample`, Whisper
     temperature fallback) until #4142 is fixed, since their output cannot be pinned.
  Could go upstream one model at a time, once TPU support is public.
- [ ] **T3 — Whisper speculative-decoding test bug (hardware-neutral).**
  `WhisperModelIntegrationTests::test_speculative_decoding_{distil,non_distil}` load the model in
  float16 only on CUDA/XPU, but always cast `input_features` to float16, so any other accelerator fails
  with `Input type (c10::Half) and bias type (float) should be the same`. Fix: cast the inputs to the
  dtype the model was loaded in. Small upstream PR; check for an existing one first.
- [ ] **T4 — distributed test mixins (TP and FSDP) on TPU.** Once B4 has an answer (a public launch-env
  API, and 2-rank sub-slices or not), adapt `_init_distributed` / `_fsdp_global_wrapper`: set up the
  launch env in the parent, set `RANK`/`LOCAL_RANK`/`WORLD_SIZE` per rank, don't `set_device(rank)` on
  TPU, and use a world size the topology supports. Even then, in the test suite the pytest parent
  already holds the chips (#4166). Investigate whether a separate, distributed-only pytest session can
  avoid opening the device in the parent.
- [ ] **T5 — gated checkpoint.** The four CSM integration tests need the token running the suite to
  have access to `sesame/csm-1b` (request it on the Hub). Not a code change.
- [ ] **T6 — fold the failure attribution into the repo tooling.** The per-cause table in
  `tpu_report.md` came from a one-off script. Move its rules into `TRIAGE_RULES` in
  `utils/tpu_ci/make_model_results.py`, so `model_results.md` buckets every failure on each run, and
  keep `needs triage` for anything new.

Still pending from the original plan, not failures:
- [ ] gate `require_torch_large_accelerator` and `get_accelerator_total_memory_gib` on TPU as well (32 GiB
  HBM per v6e chip), which unlocks the tests currently skipped for CUDA/XPU only;
- [ ] upstream the hardware-neutral Phase 0 commits as separate PRs (coordinate on issues first);
- [ ] dashboard fork with a TPU column (Phase 4); the dataset is private, so the fork needs a token or a
  public dataset;
- [ ] `single` column: blocked on `TPU_VISIBLE_DEVICES` not restricting `device_count()`.

## 3. Tests that take a very long time, reason unknown

- [ ] **S1 — Whisper long-form generation.** Three 8-clip long-form tests with beam search or temperature
  fallback were failed by the one-hour timeout (`test_whisper_empty_longform`,
  `test_whisper_longform_multi_batch_hard_prev_cond`, `test_whisper_longform_no_speech_detection`).
  The whisper test file takes 5 h 53 min overall, and the pytest process's host memory kept growing
  (from 54 to about 150 GB). Plan:
  1. standalone repro with `openai/whisper-tiny` on one long clip: time per decoding step on TPU vs
     CPU, and whether it stays flat or grows with the sequence length;
  2. check whether each new shape (growing KV cache, beam reordering) triggers a compilation. Use the
     runtime's cache statistics or `TORCH_TPU_DEBUG`, and host memory per `generate` call;
  3. compare with `cache_implementation="static"` (fixed shapes);
  4. if it is per-shape recompilation or unbounded cache growth, write it up for `TORCH_TPU_BUGS.md`.
- [ ] **S2 — `ViTModelTest::test_model_outputs_equivalence` takes 22 min** (a fast test; bert's takes
  several minutes too). Same measurement as S1: it runs many small forward passes with different
  configurations, which fits per-shape compilation.
- [ ] **S3 — mistral3 under `accelerate.cpu_offload`.** `test_mistral3_integration_batched_generate`
  takes 18 min. In the fourth test on the same model, the host-to-device copy fails with
  `RuntimeBufferAllocationFailure`. Not a plain leak: allocating and freeing 10 GB buffers, and
  streaming 40 GB of 1 GB layers through a 32 GB chip twice, both work on this build. Plan: run the
  multi-image test alone, log device memory between the four tests, and check whether accelerate's hooks
  leave tensors on the chip (buffers are not offloaded by default).

## 4. Failures suspected to be a filed bug, to verify

- [ ] **V1 — `CLIPTextModelTest::test_eager_matches_sdpa_inference_08_fp32_pad_left_sdpa_kernels`**
  differs by a mean relative 0.49, far beyond bfloat16 rounding. Suspect #4146: left padding makes
  fully masked attention rows. Verify by comparing only the non-padded positions, and by checking the
  test's mask for all-False rows.
- [ ] **V2 — slow-run failures counted as precision (#4148) by pattern, not by test.** 19 are new in the
  slow run: speecht5 batch-vs-single spectrogram lengths, whisper
  `test_longform_generate_multi_batch` diverging at token 207, `wav2vec2` batching equivalence,
  eager/SDPA float32 variants in CLIP, the CLIP/DETR/ViT/table-transformer integration logits. Verify:
  re-run them with eager attention and `torch.set_float32_matmul_precision("highest")`. What passes
  then is precision; what still fails is a real bug and moves to §1 or §2.
- [ ] **V3 — the 27 integration outputs that differ from the reference.** Same check as V2, before
  T2 records TPU values for them.
- [ ] **V4 — `test_model_parallelism` `TypeError` inside accelerate (25).** Traced to #4164: before
  dispatch, the shards meant for chip 1 are on `tpu:0`. Re-check when #4164 is fixed.
- [ ] **V5 — Whisper temperature-fallback tests.** Fallback samples, so they may be #4142 (unseeded
  `multinomial`). Verify by running one twice with the same seed and comparing.
- [ ] **V6 — two borderline tests flip between runs** (`CLIPVisionModelTest::test_eager_matches_sdpa_inference_00_fp16_pad_left_sdpa_kernels`,
  `DetrModelTest::test_batching_equivalence`). Expected under #4148; just track them.

## Log

- 2026-09-24: B1–B4 reproduced standalone and written up in `TORCH_TPU_BUGS.md`. S3: three device
  memory probes pass on this build — 8 rounds of 10 GB allocate/free, 40 GB of 1 GB layers streamed
  through the chip twice, and 100 round trips of a 0.5 GB weight to the chip and back to `meta` — so
  the mistral3 failure is not a simple leak.
