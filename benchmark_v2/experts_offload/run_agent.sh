#!/usr/bin/env bash
# Autonomous pi agent on an HF Job: works on transformers@stream-experts toward TASK.md, saves patches + logs to the bucket.
set -uo pipefail
KIT=/mnt/agent/kit
OUT=/mnt/agent/run-$(date +%Y%m%d-%H%M%S)
MODEL=${MODEL:-moonshotai/Kimi-K2.7-Code}
HOURS=${HOURS:-7}
mkdir -p "$OUT"
exec > >(tee -a "$OUT/runner.log") 2>&1
echo "[setup] model=$MODEL hours=$HOURS out=$OUT"

curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
export PATH=$HOME/.local/bin:$PATH
npm install -g @earendil-works/pi-coding-agent >/dev/null 2>&1
git clone -q -b stream-experts https://github.com/huggingface/transformers /work/transformers
cd /work/transformers
uv venv -q -p 3.12 .venv && source .venv/bin/activate
uv pip install -q -e . torch numpy accelerate kernels triton pytest "huggingface_hub[hf_xet]" \
    /mnt/wheels/safetensors-0.9.0.dev0-cp311-abi3-manylinux_2_34_x86_64.whl
cp $KIT/TASK.md $KIT/*.py .
[ -f BENCH_LOG.md ] || printf "# BENCH_LOG (A100 agent)\n\n" > BENCH_LOG.md
git config user.email agent@jobs.local && git config user.name "pi agent"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "[setup] done, starting agent"

deadline=$(( $(date +%s) + HOURS * 3600 ))
i=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  i=$((i + 1))
  echo "[iter $i] $(date -u +%H:%M:%S), $(( (deadline - $(date +%s)) / 60 )) min left"
  if [ "$i" -eq 1 ]; then
    prompt="Read TASK.md and do the task. Start by measuring the baseline."
    cont=""
  else
    prompt="Continue toward the goal in TASK.md. Re-read BENCH_LOG.md for where you are. Time left: $(( (deadline - $(date +%s)) / 60 )) minutes."
    cont="-c"
  fi
  timeout $(( deadline - $(date +%s) )) pi -p $cont --approve --provider huggingface --model "$MODEL" --thinking high "$prompt" \
    > "$OUT/iter_$i.txt" 2>&1
  tail -5 "$OUT/iter_$i.txt"
  # everything the agent did so far, against the branch it started from
  git add -A >/dev/null 2>&1
  git diff --cached origin/stream-experts -- . ':!*.pyc' > "$OUT/cumulative.patch"
  git log --oneline origin/stream-experts..HEAD > "$OUT/commits.txt"
  cp BENCH_LOG.md "$OUT/BENCH_LOG.md"
  cp -r ~/.pi/agent/sessions "$OUT/" 2>/dev/null
  git reset -q  # keep the agent's index as it was
done
echo "[done] $i iterations"
