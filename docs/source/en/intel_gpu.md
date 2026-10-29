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

# Intel GPU

Intel GPUs run PyTorch through the XPU backend, with the device string `xpu`. PyTorch's XPU build comes from a separate package index, so the install command below adds it as an extra index.

## Installation

Install the [Intel GPU drivers for PyTorch](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-15.html), then run `xpu-smi` to [check that your system detects the GPU](https://dgpu-docs.intel.com/driver/verification.html).

Install PyTorch from the XPU index along with Transformers.

```bash
pip install "transformers[torch]" --extra-index-url https://download.pytorch.org/whl/xpu
```

Check that PyTorch can see the GPUs.

```py
import torch

print(torch.xpu.is_available(), torch.xpu.device_count())
```

It prints `True` and the number of GPUs, for example `True 2`.

## Select and order XPUs

Use the `ZE_AFFINITY_MASK` environment variable to select XPUs. Set it on the same command line as the run, because an exported variable you forget about can silently send a later run to the wrong XPUs.

For example, to select XPUs 0 and 2 out of four:

```cli
ZE_AFFINITY_MASK=0,2 torchrun trainer-program.py ...
```

PyTorch sees only XPUs 0 and 2. The order of the values in `ZE_AFFINITY_MASK` doesn't change the order PyTorch enumerates them in. To get a consistent order across runs, also set `ZE_ENABLE_PCI_ID_DEVICE_ORDER=1`, which reports devices from lowest to highest PCI bus ID.

```cli
ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 ZE_AFFINITY_MASK=0,2 torchrun trainer-program.py ...
```

See the [Level Zero specification](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/PROG.html) for more on device enumeration.

## Choose an attention backend

You set the attention backend with `attn_implementation` when you load a model, and it applies to both inference and training. Models use [SDPA](./attention_interface) by default. With PyTorch 2.8 or later, SDPA on XPU handles grouped-query attention directly, without repeating the key and value heads first.

`attn_implementation="flash_attention_2"` loads [kernels-community/flash-attn2](https://huggingface.co/kernels-community/flash-attn2) from the Hub, which ships XPU builds. Install the `kernels` extra to use it. FlashAttention-3 and FlashAttention-4 only run on NVIDIA GPUs.

```bash
pip install "transformers[kernels]"
```

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    dtype="bfloat16",
    device_map="xpu",
    attn_implementation="flash_attention_2",
)
```

## Run inference

Pass `device_map="auto"` to [`~PreTrainedModel.from_pretrained`] to split a model that doesn't fit on one XPU across the available XPUs, or `device_map="xpu"` to load it on a single XPU. [`pipeline`] picks an XPU automatically when you don't pass `device`.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype="bfloat16", device_map="xpu")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

inputs = tokenizer("Intel GPUs are", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

[Continuous batching](./continuous_batching) supports XPU graphs and asynchronous batching on Intel GPUs, and decides whether to use them automatically.

## Use Hub kernels

Set `use_kernels=True` in [`~PreTrainedModel.from_pretrained`] to replace supported layers with optimized kernels from the Hub. Intel GPUs cover most of the same layers as NVIDIA GPUs, such as activations, RMSNorm, and rotary embeddings. Install the `kernels` extra to use them.

```py
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", dtype="bfloat16", device_map="xpu", use_kernels=True
)
```

Hub kernels apply to training too. Calling `model.train()` or `model.eval()` swaps in the kernels for that mode. To choose a kernel yourself, pass a [`KernelConfig`] and use `"xpu"` as the device key for XPU-specific mappings. See [Loading kernels](./kernel_doc/loading_kernels) for the full mapping.

## Quantize models

Quantization lets a larger model fit in GPU memory. Check the Intel GPU column in the [quantization overview](./quantization/overview) for the methods that run on XPU.

To fine-tune a model that doesn't fit in GPU memory, load it in 4-bit with [bitsandbytes](./quantization/bitsandbytes) and train LoRA adapters with [PEFT](./peft).

## Train with Trainer

[`TrainingArguments`] detects Intel GPUs and sets the device to `xpu`, so [`Trainer`] trains on them without any device settings. `TrainingArguments` checks bf16 support with `torch.xpu.is_bf16_supported()` and raises a `ValueError` if the GPU doesn't support it.

```py
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./outputs",
    bf16=True,
)
print(training_args.device)
```

It prints `xpu:0` when `Trainer` will train on a single XPU.

### Distributed training

Distributed training uses the `xccl` collective backend. Launch one process per XPU with `torchrun` or `accelerate launch`.

```bash
torchrun --nproc_per_node=2 train.py
```

See [DDP](./ddp), [FSDP2](./fsdp), [DeepSpeed ZeRO](./deepspeed), and [Tensor parallelism](./tensor_parallelism) to choose a parallelism strategy.

## Next steps

- Read the PyTorch [getting started on Intel GPU](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html) guide for supported hardware and platforms.
- Read [Attention backends](./attention_interface) to learn how `attn_implementation` selects an attention function.
