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

# Google TPU

[Google Cloud TPUs](https://cloud.google.com/tpu) run PyTorch natively through the [torch_tpu](https://github.com/google-pytorch/torch_tpu) package, which registers the `tpu` device. `torch_tpu` compiles the operations it runs for each input shape it sees, so keeping input shapes static avoids recompiling.

## Installation

Install `torch_tpu` by following the instructions in the [torch_tpu repository](https://github.com/google-pytorch/torch_tpu). `torch_tpu` loads automatically when you import `torch`. Check that PyTorch sees the TPU chips.

```py
import torch

print(torch.accelerator.is_available(), torch.accelerator.device_count())
```

It prints `True` and the number of TPU chips, for example `True 4`.

## Run inference

`device_map="auto"` doesn't detect TPUs. Automatic placement relies on [Accelerate](https://huggingface.co/docs/accelerate/index) to list the available devices, and Accelerate doesn't list TPU devices, so the model silently loads on CPU. Pass `device_map="tpu"` to [`~PreTrainedModel.from_pretrained`] explicitly, or `device="tpu"` to [`Pipeline`].

Pass the generation inputs on CPU instead of calling `.to(model.device)`. [`~GenerationMixin.generate`] moves only the tensors that `forward` consumes to the TPU. The tensors that grow each step (output sequence, `attention_mask`) stay on CPU, so the compiled graph stays stable. `generate` returns the output on the same device as the inputs. See [Keep generation tensors on CPU](./kv_cache#keep-generation-tensors-on-cpu) for more details. Models use SDPA by default on TPU.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", dtype="bfloat16", device_map="tpu"
)

# leave the inputs on CPU
inputs = tokenizer("The French Bread Law states", return_tensors="pt")
output = model.generate(
    **inputs, do_sample=True, max_new_tokens=50, cache_implementation="static"
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

```txt
The French Bread Law states that a farmer can sell their harvest of bread at a certain price. The price is determined by the market price and the price of the bread, and the price is set by the farmers' decision-making committee. What are the implications of this law on
```

With a static cache, [`~GenerationMixin.generate`] compiles the forward pass with `torch.compile` and the `tpu` backend automatically. Pass `disable_compile=True` to `generate` to opt out.

### Tensor parallelism

`torch_tpu` runs one process per TPU chip, and each process sees its chip as `tpu:0`. Create a [`DistributedConfig`] with the tensor parallelism size and pass it to [`~PreTrainedModel.from_pretrained`]. Transformers shards the model at load time.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DistributedConfig

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    dtype=torch.bfloat16,
    distributed_config=DistributedConfig(tp_size=4),
)

# leave the inputs on CPU
inputs = tokenizer("The French Bread Law states", return_tensors="pt")
output = model.generate(
    **inputs, do_sample=False, max_new_tokens=50, cache_implementation="static"
)
if torch.distributed.get_rank() == 0:
    print(tokenizer.decode(output[0], skip_special_tokens=True))
```

`torch_tpu` needs some environment variables set before the processes start, which `torchrun` doesn't do on its own. Call `torch.tpu.distributed.set_tpu_launch_env()` and start `torchrun` from Python.

```py
# launch.py
import sys

import torch
from torch.distributed.run import main

torch.tpu.distributed.set_tpu_launch_env()
main(sys.argv[1:])
```

Pass the usual `torchrun` arguments to the launcher.

```bash
python launch.py --nproc_per_node=4 generate.py
```

Some models need adapting before they can be sharded across TPU chips.

Models whose number of key-value heads isn't divisible by `tp_size` can't shard their attention layers, and fail in the forward pass with a shape error. Replicate the attention projections on every chip instead, by overriding their rules in `tp_plan`. Rules you don't override keep the model's default plan. For example, [GLM-4-9B-0414](https://huggingface.co/zai-org/GLM-4-9B-0414) has 2 key-value heads, which can't be split across 4 chips.

```py
model = AutoModelForCausalLM.from_pretrained(
    "zai-org/GLM-4-9B-0414",
    dtype=torch.bfloat16,
    distributed_config=DistributedConfig(
        tp_size=4,
        tp_plan={
            "layers.*.self_attn.q_proj": "replicated_with_grad_allreduce",
            "layers.*.self_attn.k_proj": "replicated_with_grad_allreduce",
            "layers.*.self_attn.v_proj": "replicated_with_grad_allreduce",
            "layers.*.self_attn.o_proj": "replicated_with_grad_allreduce",
        },
    ),
)
```

Models with tied weights, such as Qwen3-0.6B, fail to load with `tp_size` larger than 1.

## Use Hub kernels

The [kernels](https://github.com/huggingface/kernels) library supports TPU since v0.17. TPU kernels wrap JAX and [Pallas](https://docs.jax.dev/en/latest/pallas/index.html) code, so install `jax` alongside `kernels`.

```bash
pip install "transformers[kernels]" jax
```

See [Loading kernels](./kernel_doc/loading_kernels) for how to load a kernel from the Hub.

## Limitations

- `device_map="auto"` loads the model on CPU instead of TPU. Pass `device_map="tpu"` explicitly.
- Dynamic shapes trigger recompilation. Generation slows down when input shapes change from step to step. Use a static cache for faster generation.
- Each process can only access one TPU chip. Use one process per chip to run on several chips.
- Transformers integrations for training and quantization on TPU are still in progress, so some features might not work yet. Please [open an issue](https://github.com/huggingface/transformers/issues) if you run into a problem.
