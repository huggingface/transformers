<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# SkyPilot

SkyPilot is a framework for running AI/ML jobs on any cloud with one interface. This guide shows how to launch Transformers training and inference workloads on managed GPU instances via SkyPilot, including single-node and multi-node (distributed) runs.

References: see the feature request in [Issue #40179](https://github.com/huggingface/transformers/issues/40179).

### Prerequisites
- Python 3.9+
- A cloud account (AWS, GCP, Azure, OCI, etc.) configured per SkyPilot docs
- Install SkyPilot:
```bash
pip install skypilot[aws,gcp,azure]  # pick your providers
```
- Optional: install Deepspeed/Accelerate depending on your launcher:
```bash
pip install deepspeed accelerate
```

### Single-node GPU: fine-tune a text classification model
Create `train_glue.sky.yaml`:
```yaml
name: hf-train-glue
resources:
  accelerators: A10:1   # or A100:1, L4:1, etc.
  use_spot: true        # optional
file_mounts:
  /workspace:
    source: .
    mode: MOUNT
setup: |
  conda activate base || true
  pip install -U pip
  pip install -e /workspace[torch]  # if running from a local clone; otherwise: pip install transformers datasets
  pip install datasets evaluate torch
run: |
  cd /workspace
  python examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path bert-base-uncased \
    --task_name mrpc \
    --do_train --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --output_dir /tmp/hf-glue-output \
    --overwrite_output_dir
```
Launch:
```bash
sky launch -c hf-train ./train_glue.sky.yaml
# stream logs
sky logs hf-train -f
```

### Single-node inference: serve a model
Create `serve.sky.yaml`:
```yaml
name: hf-serve
resources:
  accelerators: A10:1
setup: |
  pip install -U pip
  pip install transformers fastapi uvicorn torch
run: |
  cat > app.py << 'PY'
import torch
from fastapi import FastAPI
from transformers import pipeline
app = FastAPI()
pipe = pipeline("text-generation", model="gpt2", device=0 if torch.cuda.is_available() else -1)
@app.get("/generate")
def generate(q: str = "hello world"):
    out = pipe(q, max_new_tokens=50)
    return {"output": out[0]["generated_text"]}
PY
  uvicorn app:app --host 0.0.0.0 --port 8000
```
Launch and open the service:
```bash
sky launch -c hf-serve ./serve.sky.yaml
sky status hf-serve --endpoint
```

### Multi-node distributed training (torchrun / DeepSpeed)
Below are two approaches commonly used in our docs. SkyPilot provisions multiple nodes and sets SSH connectivity; we use a small launcher script to derive environment variables and invoke torch.distributed or Deepspeed across nodes.

Create `train_multi.sky.yaml`:
```yaml
name: hf-train-multi
resources:
  accelerators: A100:8
  num_nodes: 2
file_mounts:
  /workspace:
    source: .
    mode: MOUNT
setup: |
  pip install -U pip
  pip install -e /workspace[torch]  # or: pip install transformers datasets
  pip install torch accelerate deepspeed
run: |
  cat > launch.sh << 'SH'
#!/usr/bin/env bash
set -euo pipefail
# Discover host list from SkyPilot
IFS=',' read -ra HOSTS <<< "$(sky show ips --internal | tr '\n' ',' | sed 's/,$//')"
MASTER_ADDR=${HOSTS[0]}
NNODES=${#HOSTS[@]}
GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
# Torch distributed example (replace with your script/args)
python -m torch.distributed.run \
  --nproc_per_node ${GPUS_PER_NODE} \
  --nnodes ${NNODES} \
  --master_addr ${MASTER_ADDR} \
  --master_port 29500 \
  examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path bert-base-uncased \
    --task_name mrpc \
    --do_train --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --output_dir /tmp/hf-glue-multi \
    --overwrite_output_dir
SH
  chmod +x launch.sh
  ./launch.sh
```
Launch:
```bash
sky launch -c hf-train-multi ./train_multi.sky.yaml
```

To switch to DeepSpeed, replace the `python -m torch.distributed.run ...` block with:
```bash
deepspeed --num_nodes ${NNODES} --num_gpus ${GPUS_PER_NODE} \
  examples/pytorch/text-classification/run_glue.py \
    --deepspeed ds_config.json \
    ... (same args as above)
```

### Tips
- Storage: use `file_mounts` or SkyPilot Storage for datasets and checkpoints; or mount an object store.
- Fault tolerance: enable `use_spot: true` and set up resumption (e.g., save checkpoints to durable storage).
- Multi-cloud capacity: pass `--cloud aws,gcp,azure` to chase availability.
- Networking: SkyPilot creates an internal overlay; use `sky ssh hf-train-multi` for debugging.

### Troubleshooting
- CUDA not found: ensure driver/toolkit available on the base image; SkyPilot defaults are GPU-ready on major clouds.
- Port not reachable: open service via `sky status <cluster> --endpoint` or map ports with `--ports` in the YAML.
- Slow installs: bake a custom image with preinstalled dependencies using SkyPilot image builder. 