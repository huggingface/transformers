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

# Compile a model

Quantization saves memory, but dequantizing weights inside every matmul can make generation slower than in bf16. [torch.compile](../perf_torch_compile) fuses that work into optimized kernels and recovers much of the speed.

## Compile automatically in generate

[`~GenerationMixin.generate`] compiles the model's forward pass automatically when all of these hold.

- The quantization method supports compilation ([compressed-tensors](./compressed_tensors), [Fine-grained FP8](./finegrained_fp8), [GGUF](./gguf), [NVFP4](./nvfp4), and [torchao](./torchao)).
- The model uses a cache with a fixed size, like the [static cache](../kv_cache#fixed-size-cache).
- The model runs on CUDA, XPU, Neuron, or TPU.
- No part of the model is offloaded to the CPU or disk.

Pass `cache_implementation="static"` to [`~GenerationMixin.generate`]. The example below uses [torchao](./torchao) int8 weight-only quantization.

```py
import torch
from torchao.quantization import Int8WeightOnlyConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

model_id = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=TorchAoConfig(quant_type=Int8WeightOnlyConfig()),
    device_map="cuda",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=32, cache_implementation="static")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

The first calls are slow while the model compiles. Later calls with the same shapes reuse the compiled kernels.

Pass `disable_compile=True` to [`~GenerationMixin.generate`] to turn automatic compilation off.
