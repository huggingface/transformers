## Branch cleanup
- I reverted 23 workaround commits, then rebased to drop each original together with its revert. That took the branch from 53 to 30 commits. The final tree is byte-identical to the state after the reverts.
- Force-pushed as `tpu-tests` @ `b3bb511880`. The old head is kept as the local tag `tpu-tests-before-cleanup` in case you need it.
- **Kept, because the backend kills the whole pytest process:** the `ctc_loss` skip (torch_tpu#4143) and the same-size `interpolate` skip in `get_rel_pos` (torch_tpu#4144). Both still abort on today's `torch_tpu 0.1.1.dev20260923101118`.
- **Kept, because they aren't reported bugs:** the FlexAttention skip, compiling with the model's own backend, the RoPE epsilon slack, the SDPA difference reported in float32, and binding the accelerator before creating the process group.
- **Removed:** the workarounds for torch_tpu#4158, #4161, #4148, #4160, #4159, #4141, #4142, #4146, #4147 and #4163. I also removed the skips for torch_tpu#4164 (device map) and #4166 (subprocess and spawned-rank tests). For those two the child process dies, but the test suite keeps running, so by your rule they go.
- I kept the probes in `utils/tpu_ci/backend_gaps.py`, since they are neither tests nor workarounds. Two probe commit messages (`e557d2d104`, `34143bac58`) still mention workarounds that are now gone.

## Fast test run
Fast tests only (`RUN_SLOW` off), all 27 `IMPORTANT_MODELS`, on 8 × TPU v6e:

| Passed | Failed | Skipped | Errors | Pass rate over attempted |
|---|---|---|---|---|
| 3810 | 304 | 4070 | 0 | 92.6% |

- Every model produced a report.
- Only 44 skips come from TPU gates still on the branch (34 FlexAttention, 10 CTC loss); the rest are the usual ones (slow, flash-attn, deepspeed and so on).
- Every failure is attributed to a known gap. The main ones: bf16 precision 113 (torch_tpu#4148), Cholesky 69 (#4158), the subprocess-can't-open-device abort 37 (#4166), the differentiable `attn_mask` 30 (#4147), device index 25 (#4164).
- **Two new findings on this build:**
  - The FSDP and TP tests (10) now fail earlier than before. `tpu_dist` refuses to initialise without `TORCH_TPU_SLICEBUILDER_ADDRESSES` and `TORCH_TPU_TOPOLOGY`, which only a launcher like torchrun sets.
  - `test_model_parallelism` fails inside accelerate with a `TypeError`. Shards meant for chip 1 stay on `tpu:0` (torch_tpu#4164), and accelerate then chokes while re-placing them.

## Slow test run
`RUN_SLOW=1`, all 27 `IMPORTANT_MODELS`, same machine and build, 2026-09-23 20:38 → 2026-09-24 08:56 UTC:

| Passed | Failed | Skipped | Errors | Pass rate over attempted |
|---|---|---|---|---|
| 4224 | 402 | 3558 | 0 | 91.3% |

- Every model produced a report and no pytest process crashed. The same 44 skips come from TPU gates on the branch.
- Every fast-run failure shows up again, except two borderline precision comparisons that passed this time (`CLIPVisionModelTest::test_eager_matches_sdpa_inference_00_fp16_pad_left_sdpa_kernels`, `DetrModelTest::test_batching_equivalence`), so those flip from run to run.
- The slow run has 98 more failures than the fast run (100 new, 2 gone). Most are what the plan expected from integration tests: 25 have no TPU entry in their `Expectations` (`No matching expectation found for ('tpu', None, None)`), and 27 produce text or logits that differ from the values recorded on other hardware. They need TPU reference values before they say anything about TPU. The known gaps account for most of the rest: bf16 precision (+19), the differentiable `attn_mask` (+8 in T5's export and integration tests).
- One commit added: `474326f524 fix: fail a test that never finishes instead of waiting on it`. There was no crash, but `WhisperModelIntegrationTests::test_whisper_empty_longform` was still running after an hour and would have held up the run. `run_tpu_tests.sh` now passes `--timeout=${TEST_TIMEOUT:-3600}` to pytest, so such a test fails and is counted. Three whisper tests hit it.
- Every failure is attributed; the per-cause table is in `tpu_report.md` in the dataset.

### New, unexpected failures
- **`aten::_upsample_bicubic2d_aa.out` is not implemented for TPU** (2). `test_can_compile_fast_video_processor` in internvl and smolvlm: the fast video processors resize with antialiased bicubic. Not filed yet.
- **The TPU dynamo backend rejects `mode=`** (1). `LlamaModelTest::test_torch_compile_for_training` compiles with `mode="reduce-overhead"`, and `_default_backend_selector` raises `TypeError: Unexpected keyword arguments: {'mode': 'reduce-overhead'}`. Not filed yet.
- **Device memory runs out under `accelerate.cpu_offload`** (1). The mistral3 integration tests keep Mistral-Small-24B on CPU and stream it to the chip layer by layer. Three tests on that model ran, but in `test_mistral3_integration_batched_generate_multi_image` the host-to-device copy fails with `RuntimeBufferAllocationFailure`. Offloaded weights may not be released from the chip; I haven't verified that.
- **`device_map="auto"` leaves the model on CPU** (6). llama's two 7B logits tests, three qwen2_5_omni integration tests and `ViTModelIntegrationTest::test_inference_fp16` fail with `tensor is expected to be on tpu, got cpu`. accelerate finds no TPU to map to, which is a separate problem from the ignored device index (torch_tpu#4164).
- **Whisper long-form generation is very slow on TPU.** The 8-clip long-form tests with beam search or temperature fallback (`test_whisper_empty_longform`, `test_whisper_longform_multi_batch_hard_prev_cond`, `test_whisper_longform_no_speech_detection`) had not finished after an hour. The whisper file alone takes 5 h 53 min, 3 h of it in those three timeouts.
- **Test bug, not TPU-specific** (2). `WhisperModelIntegrationTests::test_speculative_decoding_{distil,non_distil}` load the model in float16 only when CUDA or XPU is available, but always cast the inputs to float16. On any other device that fails with `Input type (c10::Half) and bias type (float) should be the same`. A candidate upstream fix.
- **Suspicious, not verified:** `CLIPTextModelTest::test_eager_matches_sdpa_inference_08_fp32_pad_left_sdpa_kernels` differs by a mean relative 0.49, far beyond bf16 rounding. It is counted under precision, but left padding creates fully masked rows, so this is more likely the masked-row leak (torch_tpu#4146).
- **Environment:** the four CSM integration tests need access to the gated `sesame/csm-1b` for the token running the suite.

## Uploaded
In `2026-09-23/ci_results_run_models_gpu/` (fast run) and `2026-09-24/ci_results_run_models_gpu/` (slow run):
- `model_results.json`, the file the dashboard reads. I checked that the Hub copy matches the local file.
- `model_results.md`, the generated summary.
- `tpu_report.md`, a table attributing each failure to its gap. To respect the no-private-repo rule it names gaps by their probe names, not torch_tpu issue numbers.

The script that attributes failures is a one-off in the scratchpad; I didn't add those rules to `make_model_results.py`.

## Environment fixes
- The container's `/etc/hosts` had lost its `localhost` line. That's likely from this morning's hostname change. pytest couldn't start, so the first launch died. I restored the loopback entries; the original is backed up as `etc_hosts.orig` in the session scratchpad.
- The venv only had torch and torch_tpu. I installed the testing and media extras (timm, librosa, av and others) the same way the previous session did, and restarted the run so the numbers are comparable.
- The slow tests also need `torchcodec` for their audio and video datasets. The PyPI wheels are CUDA builds (`libnvrtc.so.13` missing), so it has to come from the PyTorch CPU index. It also needs FFmpeg's shared libraries and `libpython3.13.so`, which the image lacked; I installed `ffmpeg` and `libpython3.13` with apt. The six models that had failed on it (gemma3n, internvl, qwen2_5_vl, smolvlm, speecht5, whisper) were re-run, and the slow numbers above use those re-runs.

## Running the suite

### Environment

A fresh TPU machine's `.venv` only has `torch`, `torch_tpu` and `libtpu`. From the repo root:

```bash
source .venv/bin/activate
uv pip install -e ".[testing]" slack_sdk
# Pin torch, or the resolver silently replaces the TPU-compatible build.
uv pip install --index-url https://download.pytorch.org/whl/cpu torchvision torchaudio \
    "torch==$(python -c 'import torch; print(torch.__version__.split("+")[0])')"
uv pip install librosa av timm sentencepiece protobuf num2words
getent hosts localhost   # must resolve, or pytest dies before collecting anything
```

For the slow tests, also `torchcodec`. It needs FFmpeg's shared libraries and `libpython`, and the
PyPI wheels are CUDA builds, so take the CPU one:

```bash
apt-get install -y ffmpeg libpython3.13
uv pip install --index-url https://download.pytorch.org/whl/cpu torchcodec \
    "torch==$(python -c 'import torch; print(torch.__version__.split("+")[0])')"
```

Leave out `pyctcdecode` (pins numpy < 2) and `phonemizer` (needs espeak). Nothing else may hold a
TPU chip while the suite runs: only one process at a time can open a chip.

### Fast tests

```bash
source .venv/bin/activate
bash utils/tpu_ci/run_tpu_tests.sh
```

This runs every `IMPORTANT_MODELS` directory in its own pytest process and writes `reports/`,
`model_results.json` and `model_results.md`. On a TPU v6e with 8 chips it took **1 h 56 min**
(2026-09-23, 18:14 → 20:10 UTC). The slowest models are `vit` (23 min, most of it
`test_model_outputs_equivalence`) and `whisper` (14 min); most others take 2–5 min.

### Slow tests

```bash
source .venv/bin/activate
RUN_SLOW=1 bash utils/tpu_ci/run_tpu_tests.sh
# optionally with a throwaway hub cache for the downloaded checkpoints:
RUN_SLOW=1 TMP_CACHE=/mnt/cache/tmp bash utils/tpu_ci/run_tpu_tests.sh
```

This runs the fast tests too. It took **about 9.3 h** on the same machine: the per-model pytest
times add up to 9.2 h, with the checkpoints already in the shared hub cache (`HF_HUB_CACHE`).
`whisper` alone takes 5 h 53 min, 3 h of it in three long-form tests that hit the per-test timeout.
Next are `mistral3` (34 min, a 24B checkpoint streamed through `cpu_offload`) and `vit` (23 min).
Everything except whisper takes about 3.3 h.

A single test that runs longer than `TEST_TIMEOUT` seconds (default 3600) is failed so the run moves
on; the slowest tests that do finish take about 22 minutes.

## Uploading

Results (what the dashboard reads), from the reports of a finished run:

```bash
python utils/tpu_ci/make_model_results.py reports/ --summary model_results.md --upload \
    --repo-id hf-gcp-tpu-internal/transformers_daily_ci --date "$(date -u +%F)"
```

Or run and upload in one go:

```bash
UPLOAD=1 RESULTS_REPO_ID=hf-gcp-tpu-internal/transformers_daily_ci bash utils/tpu_ci/run_tpu_tests.sh
```

Summaries, next to the results:

```bash
DAY="$(date -u +%F)"
hf upload hf-gcp-tpu-internal/transformers_daily_ci model_results.md \
    "$DAY/ci_results_run_models_gpu/model_results.md" --repo-type dataset
hf upload hf-gcp-tpu-internal/transformers_daily_ci tpu_report.md \
    "$DAY/ci_results_run_models_gpu/tpu_report.md" --repo-type dataset
```

`--date` defaults to today (UTC); pass the day the run started if it crosses midnight, and use the
same day for the summaries. `tpu_report.md` was written by the one-off attribution script, which is
not in the repo.
