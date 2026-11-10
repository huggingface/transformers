<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# NVIDIA GPU

NVIDIA GPUs run PyTorch through CUDA, with the device string `cuda`. Most attention kernels and serving optimizations in Transformers support CUDA, and some of them, like FlashAttention-3 and FlashAttention-4, only run on CUDA.

## Installation

Install the CUDA drivers for your GPU, then install PyTorch and Transformers. See the [installation guide](./installation) for NVIDIA Spark (ARM64) devices, which need PyTorch from NVIDIA's package index.

```bash
pip install "transformers[torch]"
```

Check that PyTorch can see the GPUs.

```py
import torch

print(torch.cuda.is_available(), torch.cuda.device_count())
```

It prints `True` and the number of GPUs, for example `True 4`.

## Select and order GPUs

Use the `CUDA_VISIBLE_DEVICES` environment variable to select GPUs and set their order. Set it on the same command line as the run, because an exported variable you forget about can silently send a later run to the wrong GPUs.

For example, to select GPUs 0 and 2 out of four:

```cli
CUDA_VISIBLE_DEVICES=0,2 torchrun trainer-program.py ...
```

PyTorch sees only GPUs 0 and 2, which are mapped to `cuda:0` and `cuda:1`. To reverse the order (use GPU 2 as `cuda:0` and GPU 0 as `cuda:1`):

```cli
CUDA_VISIBLE_DEVICES=2,0 torchrun trainer-program.py ...
```

To run without any GPUs:

```cli
CUDA_VISIBLE_DEVICES= python trainer-program.py ...
```

CUDA numbers GPUs fastest first by default, which may not match `nvidia-smi`. Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` so the indices in `CUDA_VISIBLE_DEVICES` match `nvidia-smi`.

```cli
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2 torchrun trainer-program.py ...
```

## Choose an attention backend

You set the attention backend with `attn_implementation` when you load a model, and it applies to both inference and training. Models use [SDPA](./attention_interface) by default. NVIDIA GPUs also support three FlashAttention versions. When the FlashAttention package isn't installed, Transformers loads a prebuilt kernel from the Hub instead, which needs the `kernels` extra (`pip install "transformers[kernels]"`).

| `attn_implementation` | GPUs | Hub kernel when the package isn't installed |
|---|---|---|
| `"flash_attention_2"` | Ampere or newer | `kernels-community/flash-attn2` |
| `"flash_attention_3"` | Ampere or newer | `kernels-community/vllm-flash-attn3` |
| `"flash_attention_4"` | Hopper or newer | `kernels-community/flash-attn4` |

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    dtype="bfloat16",
    device_map="cuda",
    attn_implementation="flash_attention_2",
)
```

## Run inference

Pass `device_map="auto"` to [`~PreTrainedModel.from_pretrained`] to split a model that doesn't fit on one GPU across the available GPUs, or `device_map="cuda"` to load it on a single GPU. [`pipeline`] picks a GPU automatically when you don't pass `device`.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype="bfloat16", device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

inputs = tokenizer("NVIDIA GPUs are", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

[Continuous batching](./continuous_batching) supports CUDA graphs and asynchronous batching on NVIDIA GPUs, and decides whether to use them automatically.

## Use Hub kernels

Set `use_kernels=True` in [`~PreTrainedModel.from_pretrained`] to replace supported layers with optimized kernels from the Hub. NVIDIA GPUs have the widest coverage, including activations, RMSNorm, rotary embeddings, MegaBlocks mixture-of-experts layers, and Mamba state-space ops. Install the `kernels` extra to use them.

```py
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", dtype="bfloat16", device_map="cuda", use_kernels=True
)
```

Hub kernels apply to training too. Calling `model.train()` or `model.eval()` swaps in the kernels for that mode. To choose a kernel yourself, pass a [`KernelConfig`] and use `"cuda"` as the device key for NVIDIA-specific mappings. See [Loading kernels](./kernel_doc/loading_kernels) for the full mapping.

## Quantize models

Quantization lets a larger model fit in GPU memory. Most quantization methods in Transformers support CUDA. Check the CUDA column in the [quantization overview](./quantization/overview) to compare them.

To fine-tune a model that doesn't fit in GPU memory, load it in 4-bit with [bitsandbytes](./quantization/bitsandbytes) and train LoRA adapters with [PEFT](./peft).

## Train with Trainer

[`TrainingArguments`] detects NVIDIA GPUs and sets the device to `cuda`, so [`Trainer`] trains on them without any device settings. Use bf16 mixed precision on Ampere or newer GPUs, and fp16 on older ones.

```py
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./outputs",
    bf16=True,
)
print(training_args.device)
```

It prints `cuda:0` when `Trainer` will train on a single GPU.

### Distributed training

Multi-GPU training uses the [NCCL](https://developer.nvidia.com/nccl) collective backend. Launch one process per GPU with `torchrun` or `accelerate launch`.

```bash
torchrun --nproc_per_node=4 train.py
```

See [DDP](./ddp), [FSDP2](./fsdp), [DeepSpeed ZeRO](./deepspeed), and [Tensor parallelism](./tensor_parallelism) to choose a parallelism strategy.

## Next steps

- Read the PyTorch [CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html) notes for how PyTorch manages devices, streams, and memory.
- Read [GPU memory usage](./model_memory_anatomy) to understand where GPU memory goes during training.
- Read [Attention backends](./attention_interface) to learn how `attn_implementation` selects an attention function.
