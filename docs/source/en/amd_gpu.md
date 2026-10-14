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

# AMD GPU

[AMD Instinct](https://www.amd.com/en/products/accelerators/instinct.html) (MI200, MI300, MI350 series) and [Radeon](https://www.amd.com/en/products/graphics/desktops/radeon.html) GPUs run PyTorch through [ROCm](https://rocm.docs.amd.com/), AMD's GPU software stack. The ROCm build of PyTorch uses [HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/what_is_hip.html), AMD’s CUDA-like programming interface, to expose AMD GPUs through the `torch.cuda` API. The device string is `cuda`, so code written for NVIDIA GPUs runs unchanged. Transformers checks `torch.version.hip` at runtime to tell the two apart and routes attention and Hub kernels to AMD-specific builds.

## Installation

Install ROCm from the [huggingface/transformers-pytorch-amd-gpu](https://hub.docker.com/r/huggingface/transformers-pytorch-amd-gpu) Docker image. It ships the ROCm build of PyTorch, a `flash-attn` build for MI300 and MI350 GPUs (`gfx942` and `gfx950`), and Transformers installed from source.

```bash
docker run -it --device=/dev/kfd --device=/dev/dri --group-add video \
    --ipc=host --shm-size 16G \
    huggingface/transformers-pytorch-amd-gpu
```

`--device=/dev/kfd` and `--device=/dev/dri` expose the AMD GPUs to the container. Install [kernels](https://github.com/huggingface/kernels) inside the container to [use Hub kernels](#use-hub-kernels).

```bash
pip install "transformers[kernels]"
```

To install on your own system instead, select ROCm in the [PyTorch install selector](https://pytorch.org/get-started/locally/) and make sure the ROCm version of the wheel is supported by your driver.

Check that PyTorch sees the AMD GPUs.

```py
import torch

print(torch.version.hip, torch.cuda.is_available(), torch.cuda.device_count())
print(torch.cuda.get_device_properties(0).gcnArchName)
```

```txt
7.2.53211 True 2
gfx942:sramecc+:xnack-
```

`torch.version.hip` is `None` on a CUDA build of PyTorch. `gcnArchName` reports the GPU architecture, for example `gfx942` for MI300, `gfx950` for MI350, and `gfx1151` for Ryzen AI Max (Strix Halo).

## Select and order GPUs

Use `HIP_VISIBLE_DEVICES` to select GPUs and set their order. Set it on the same command line as the run.

```cli
HIP_VISIBLE_DEVICES=0,2 torchrun trainer-program.py ...
```

PyTorch sees only GPUs 0 and 2, which are mapped to `cuda:0` and `cuda:1`. `ROCR_VISIBLE_DEVICES` filters GPUs at a lower level of the stack and is commonly set by cluster schedulers. If both are set, `HIP_VISIBLE_DEVICES` indexes into the GPUs left visible by `ROCR_VISIBLE_DEVICES`.

Use `rocm-smi` or `amd-smi` to list the GPUs and their indices.

## Run inference

[`Pipeline`] and `device_map="auto"` detect AMD GPUs and place the model on `cuda` automatically. You can also pass `device_map="cuda"` or `device_map="cuda:1"` explicitly.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", dtype="bfloat16", device_map="auto"
)

inputs = tokenizer("The French Bread Law states", return_tensors="pt").to(model.device)
output = model.generate(**inputs, do_sample=False, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

```txt
The French Bread Law states that the French government must provide a certain amount of bread for the citizens of the country. The law is passed by the French parliament. The French parliament is a body of the French government. The French government is a body of the French state. The
```

## Choose an attention backend

Set the attention backend with `attn_implementation` when you load a model, and it applies to both inference and training. SDPA is the default and works on every AMD GPU supported by PyTorch. Two FlashAttention options are also available.

| `attn_implementation` | Requires | Notes |
|---|---|---|
| `"sdpa"` | – | Default. |
| `"flash_attention_2"` | [flash-attn](https://github.com/Dao-AILab/flash-attention#amd-rocm-support) built for ROCm, or [kernels](https://github.com/huggingface/kernels) | Uses the `flash-attn` package when it is installed. Its default Composable Kernel (CK) backend supports MI200, MI300, MI355, and RDNA 3/4 GPUs, and the [Docker image](#installation) includes it on MI300 or MI350. Otherwise, loads [kernels-community/aiter-flash-attn](https://huggingface.co/kernels-community/aiter-flash-attn) from the Hub. |
| `"flash_attention_3"` | [kernels](https://github.com/huggingface/kernels) | Loads [kernels-community/aiter-flash-attn](https://huggingface.co/kernels-community/aiter-flash-attn), a prebuilt [AITER](https://github.com/ROCm/aiter) FlashAttention kernel. No compilation required. |

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    dtype="bfloat16",
    device_map="auto",
    attn_implementation="flash_attention_3",
)
```

On ROCm, `flash_attention_3` resolves to the AITER kernel and logs the following. The message refers to the FlashAttention-3 package, so it appears even when `flash_attention_2` is installed.

```txt
You do not have `flash_attn` installed, using `kernels-community/aiter-flash-attn` from the `kernels` library instead!
```

You can also pass the kernel repository directly with `attn_implementation="kernels-community/aiter-flash-attn"`. See [Attention backends](./attention_interface) for how backends are selected.

## Use Hub kernels

Set `use_kernels=True` in [`~PreTrainedModel.from_pretrained`] to replace supported layers with optimized kernels from the Hub. Transformers detects ROCm and picks the AMD build for each layer, such as `kernels-community/aiter-rope` for rotary embeddings and `kernels-community/liger-kernels` for RMSNorm. When you write a custom [`KernelConfig`], use `"rocm"` as the device key for AMD-specific mappings.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", dtype="bfloat16", device_map="auto", use_kernels=True
)
```

Hub kernels mostly speed up prefill and large batches. Benchmark your own workload before enabling them for small-batch decoding. See [Loading kernels](./kernel_doc/loading_kernels) for the full mapping.

## Quantize models

Methods supported on ROCm include [AWQ](./quantization/awq), [GPT-QModel](./quantization/gptq), [compressed-tensors](./quantization/compressed_tensors), and AMD's [Quark](./quantization/quark). [bitsandbytes](./quantization/bitsandbytes) has partial support. MI300 and newer GPUs support FP8 natively. See the [compatibility table](./quantization/overview) for the full list.

## Train with Trainer

[`TrainingArguments`] detects AMD GPUs and sets the device to `cuda` automatically.

```py
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./outputs",
    bf16=True,
)
```

### Distributed training

The ROCm build of PyTorch ships RCCL, AMD's implementation of the NCCL API, so the `nccl` backend, `torchrun`, [FSDP](./fsdp), [DeepSpeed](./main_classes/deepspeed), and the Transformers distributed API work without changes.

```py
import torch
from transformers import AutoModelForCausalLM, DistributedConfig

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B",
    dtype=torch.bfloat16,
    distributed_config=DistributedConfig(tp_size=2, fsdp_size=4),
)
```

```bash
torchrun --nproc_per_node=8 train.py
```

See [Tensor parallelism](./tensor_parallelism), [Expert parallelism](./expert_parallelism), and [FSDP](./fsdp) for the available options.

## Limitations

- `CodeCarbonCallback` isn't compatible with ROCm. Set `report_to` in [`TrainingArguments`] to exclude `codecarbon`.
- Quantization methods marked 🔴 for ROCm in the [compatibility table](./quantization/overview) rely on CUDA-only kernels and fail to load on AMD GPUs.

## Next steps

- Read the [ROCm documentation](https://rocm.docs.amd.com/) for driver and PyTorch setup.
- Browse the ROCm builds in the [kernels-community](https://huggingface.co/kernels-community) organization.
- Read [Loading kernels](./kernel_doc/loading_kernels) to customize which kernels replace which layers.
