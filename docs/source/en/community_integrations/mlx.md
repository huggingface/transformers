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

# MLX

[MLX](https://ml-explore.github.io/mlx/build/html/index.html) is an array framework for machine learning on Apple silicon. Arrays stay in shared memory. This avoids data copies between CPU and GPU. Native [safetensors](https://huggingface.co/docs/safetensors/en/index) support lets you run Transformers models directly on Apple silicon.

Install the [mlx-lm](https://github.com/ml-explore/mlx-lm) library.

```bash
pip install mlx-lm transformers
```

Load any Transformers model from the Hub and run inference on Apple silicon.

```py
from mlx_lm import load, generate

model, tokenizer = load("openai/gpt-oss-20b")
output = generate(
    model,
    tokenizer,
    prompt="The capital of France is",
    max_tokens=100,
)
print(output)
```

## Transformers integration

1. The [mlx_lm.load](https://github.com/ml-explore/mlx-lm?tab=readme-ov-file#python-api) function loads safetensor weights and returns a model and tokenizer.
2. Internally, MLX loads weight arrays keyed by tensor names and maps them into the parameter tree of an MLX [nn.Module](https://ml-explore.github.io/mlx/build/html/python/nn/module.html#), matching how Transformers checkpoints are organized.

## Resources

- [MLX](https://ml-explore.github.io/mlx/build/html/index.html) documentation
- [mlx-lm](https://github.com/ml-explore/mlx-lm) repository