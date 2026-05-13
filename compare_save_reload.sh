#!/bin/bash
# Compare the loss/grad_norm trajectory of `train_save_reload.py` with and without
# the mid-training save/reload. They should match step-for-step.
#
# Each mode writes to its own dir (`./checkpoints_baseline/`, `./checkpoints_save_reload/`)
# so the artifacts are inspectable side-by-side after the run.
set -euo pipefail

NPROC="${NPROC:-4}"
BASELINE_LOG="${BASELINE_LOG:-baseline.log}"
SAVE_RELOAD_LOG="${SAVE_RELOAD_LOG:-save_reload.log}"
DIFF_LOG="${DIFF_LOG:-save_reload_diff.log}"

rm -rf ./checkpoints_baseline ./checkpoints_save_reload

# Pull out the per-step training lines and the post-reload generation lines so
# the diff is mechanical and covers both the training trajectory and the
# generated tokens/text.
filter_steps() { grep -E '^step |^# gen '; }

echo "=== Run 1/2: --mode baseline ==="
torchrun --nproc_per_node="$NPROC" train_save_reload.py --mode baseline 2>&1 \
    | tee /dev/stderr | filter_steps > "$BASELINE_LOG"

echo
echo "=== Run 2/2: --mode save_reload ==="
torchrun --nproc_per_node="$NPROC" train_save_reload.py --mode save_reload 2>&1 \
    | tee /dev/stderr | filter_steps > "$SAVE_RELOAD_LOG"

echo
echo "=== Diff (baseline vs save_reload) ==="
git diff --no-index --color --word-diff=color "$BASELINE_LOG" "$SAVE_RELOAD_LOG" | tee "$DIFF_LOG" || true
echo "Diff written to $DIFF_LOG"
