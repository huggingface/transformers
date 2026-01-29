<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Overview

Transformers provides multiple inference optimization techniques to make models fast, affordable, and accessible. Options include alternative attention mechanisms for reduced memory traffic, code compilation for faster execution, and optimized kernels for throughput. Stack these techniques for maximum performance.

> [!NOTE]
> Memory and speed are closely related but not the same. Shrinking your memory footprint makes a model "faster" because there is less data to move around. Pure speed optimizations don't always reduce memory and sometimes increase usage. Choose the appropriate optimization based on your use case and hardware.

Use the table below to pick an optimization technique.

| Technique | Speed | Memory |
|---|:---:|:---:|
| [Compilation](#compilation) | ✅ | |
| [Attention backends](#attention-backends) | ✅ | ✅ |
| [Kernels](#kernels) | ✅ | ✅ |
| [Quantization](#quantization) | ✅ | ✅ |
| [Caching](#caching) | ✅ | ✅ |
| [Parallelism](#parallelism) | ✅ | |
| [Continuous batching](#continuous-batching) | ✅ | |

This guide gives you a quick start on Transformers optimizations.

## Compilation

[torch.compile](./perf_torch_compile) reduces Python overhead, fuses operations, and creates kernels tuned for your shapes and hardware. The first run warms it up and subsequent runs use the faster compiled path.

Pass a [fixed size cache](./kv_cache#fixed-size-cache) to [`~GenerationMixin.generate`] to trigger `torch.compile` automatically.

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float16, device_map="auto")
input = tokenizer("The French Bread Law states", return_tensors="pt").to(model.device)

output = model.generate(**input, do_sample=False, max_new_tokens=20, cache_implementation="static")
tokenizer.batch_decode(output, skip_special_tokens=True)[0]
```

> [!WARNING]
> Avoid calling `torch.compile(model)` outside of [`~GenerationMixin.generate`] to prevent the model from recompiling every step.

## Attention backends

Alternative [attention backends](./attention_interface) lower memory traffic. For example, FlashAttention tiles attention computations and avoids large intermediate tensors to reduce memory footprint.

Set `attn_implementation` in [`~PreTrainedModel.from_pretrained`] to load an optimized attention backend.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", attn_implementation="flash_attention_2")
```

## Kernels

Kernels fuse operations to boost throughput and reduce memory usage. The [Kernels](https://huggingface.co/docs/kernels/en/index) library loads optimized compute kernels from the [Hub](https://huggingface.co/kernels-community) in a flexible and version-safe way.

The example below loads an optimized FlashAttention-2 kernel without installing the package.

```py
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", attn_implementation="kernels-community/flash-attn2"
)
```

## Quantization

[Quantization](./quantization/overview) shrinks the size of every parameter which lowers memory footprint and increases speed because you can do more operations.

Pass a quantization config to the `quantization_config` argument in [`~PreTrainedModel.from_pretrained`]. Each quantization backend has a different config with different arguments. The example below quantizes a model to 4-bits and configures the computation dtype with the [bitsandbytes](./quantization/bitsandbytes) backend.

```py
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

model = AutoModelForCausalLM.from_pretrained(
    "allenai/Olmo-3-7B-Think", quantization_config=bnb_config
)
```

## Caching

[Caching](./kv_cache) speeds up generation by reusing past keys and values instead of recomputing them for every token. To offset and reduce the memory cost of storing past keys and values, Transformers
supports offloading the cache to the CPU. Only the current layer remains on the GPU.

Use the `cache_implementation` argument in [`~GenerationMixin.generate`] to set a cache strategy.

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", attn_implementation="kernels-community/flash-attn2"
)
inputs = tokenizer("The Le Décret Pain states that a baguette must,", return_tensors="pt")
outputs = model.generate(**inputs, do_sample=False, max_new_tokens=50, cache_implementation="offloaded")
```

## Parallelism

[Parallelism](./perf_infer_gpu_multi) distributes a model across devices so models too big for one device run fast. This approach uses more memory due to sharding overhead and communication to sync results.

[Tensor parallelism](./perf_infer_gpu_multi) splits a model layer across devices. Set `tp_plan="auto"` in [`~PreTrainedModel.from_pretrained`] to enable it.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", tp_plan="auto")
print(model._tp_plan)
```

## Continuous batching

[Continuous batching](./continuous_batching) maximizes throughput by keeping the GPU busy with dynamic scheduling and chunked prefill. [Serving](./serve-cli/serving) applications use it to process multiple incoming requests concurrently.

Use [`~ContinuousMixin.generate_batch`] to enable continuous batching.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    attn_implementation="paged|sdpa",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

prompts = [
    "The Le Décret Pain states that a baguette must",
    "Explain gravity in one sentence.",
    "Name the capital of France.",
]
inputs = [tokenizer.encode(p) for p in prompts]

generation_config = GenerationConfig(
    max_new_tokens=32,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    do_sample=False,
    max_batch_tokens=512,
)

outputs = model.generate_batch(
    inputs=inputs,
    generation_config=generation_config,
)

for request_id, output in outputs.items():
    text = tokenizer.decode(output.generated_tokens, skip_special_tokens=True)
    print(f"[{request_id}] {text}")
```
