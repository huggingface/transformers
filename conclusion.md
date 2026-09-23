## Branch cleanup
- I reverted 23 workaround commits, then rebased to drop each original together with its revert. That took the branch from 53 to 30 commits. The final tree is byte-identical to the state after the reverts.
- Force-pushed as `tpu-tests` @ `b3bb511880`. The old head is kept as the local tag `tpu-tests-before-cleanup` in case you need it.
- **Kept, because the backend kills the whole pytest process:** the `ctc_loss` skip (torch_tpu#4143) and the same-size `interpolate` skip in `get_rel_pos` (torch_tpu#4144). Both still abort on today's `torch_tpu 0.1.1.dev20260923101118`.
- **Kept, because they aren't reported bugs:** the FlexAttention skip, compiling with the model's own backend, the RoPE epsilon slack, the SDPA difference reported in float32, and binding the accelerator before creating the process group.
- **Removed:** the workarounds for torch_tpu#4158, #4161, #4148, #4160, #4159, #4141, #4142, #4146, #4147 and #4163. I also removed the skips for torch_tpu#4164 (device map) and #4166 (subprocess and spawned-rank tests). For those two the child process dies, but the test suite keeps running, so by your rule they go.
- I kept the probes in `utils/tpu_ci/backend_gaps.py`, since they are neither tests nor workarounds. Two probe commit messages (`e557d2d104`, `34143bac58`) still mention workarounds that are now gone.

## Test run
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

## Uploaded
In `2026-09-23/ci_results_run_models_gpu/`:
- `model_results.json`, the file the dashboard reads. I checked that the Hub copy matches the local file.
- `model_results.md`, the generated summary.
- `tpu_report.md`, a table attributing each failure to its gap. To respect the no-private-repo rule it names gaps by their probe names, not torch_tpu issue numbers.

The script that attributes failures is a one-off in the scratchpad; I didn't add those rules to `make_model_results.py`.

## Environment fixes
- The container's `/etc/hosts` had lost its `localhost` line. That's likely from this morning's hostname change. pytest couldn't start, so the first launch died. I restored the loopback entries; the original is backed up as `etc_hosts.orig` in the session scratchpad.
- The venv only had torch and torch_tpu. I installed the testing and media extras (timm, librosa, av and others) the same way the previous session did, and restarted the run so the numbers are comparable.

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

**Duration not measured yet:** the slow tests have never been run on TPU. They download real
checkpoints, so expect them to take much longer than the fast run.

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
