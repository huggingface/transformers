<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Pruna

[Pruna](https://www.pruna.ai/) is an optimization framework for LLMs. It applies quantization methods (HIGGS, HQQ, GPTQ, AWQ), caching, factorization, and torch_compile to reduce memory usage and increase inference speed.

Install Pruna with `pip install pruna` before optimizing models.

```py
from pruna import smash, SmashConfig
from transformers import pipeline

pipe = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M-Instruct")

smash_config = SmashConfig(
    {
        "hqq": {"weight_bits": 4, "compute_dtype": "torch.bfloat16"},
        "torch_compile": {"fullgraph": True, "dynamic": True, "mode": "max-autotune"}
    },
    device="cuda"
)

smashed_model = smash(model=pipe.model, smash_config=smash_config)

messages = [{"role": "user", "content": "Who are you?"}]
pipe(messages, max_new_tokens=100)
```

## Transformers integration

Pruna loads Transformers models and wraps them in optimized inference handlers.

1. `smash` accepts any Transformers model loaded with [`AutoModel.from_pretrained`] or pipeline. Pruna wraps the model in a `PrunaModel` class that tracks the applied optimizations.

2. `SmashConfig` defines optimization algorithms. Each algorithm has parameters to control compression (bits, group size) or compilation (backend, mode). Pruna applies algorithms sequentially based on the config order.

3. Quantization methods like HQQ and GPTQ reduce weights to int4 or int8. Caching techniques reuse attention states to skip redundant computations. Factorization fuses QKV matrices for attention layers. Torch_compile optimizes the model graph for target hardware.

4. Save optimized models with `smashed_model.save_pretrained()` or push to the Hub with `smashed_model.push_to_hub()`. Some optimizations like torch_compile are device-specific and reapply when loading on different hardware.

## Resources

- [Pruna docs](https://docs.pruna.ai/en/stable/) for optimization algorithms and parameters
- [LLM optimization tutorial](https://docs.pruna.ai/en/stable/docs_pruna/tutorials/llms.html) with evaluation examples
- [Compression methods](https://docs.pruna.ai/en/stable/compression.html) reference
