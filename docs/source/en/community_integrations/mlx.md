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

[MLX](https://ml-explore.github.io/mlx/build/html/index.html) is an array framework for machine learning on Apple silicon that also works with CUDA. On Apple silicon, arrays stay in shared memory to avoid data copies between CPU and GPU. Lazy computation enables graph manipulation and optimizations. Native [safetensors](https://huggingface.co/docs/safetensors/en/index) support means Transformers language models run directly on MLX.

Install the [mlx-lm](https://github.com/ml-explore/mlx-lm) library.

```bash
pip install mlx-lm transformers
```

Load any Transformers language model from the Hub as long as the model architecture is [supported](https://huggingface.co/mlx-community/models). No weight conversion is required.

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

- [mlx_lm.load](https://github.com/ml-explore/mlx-lm?tab=readme-ov-file#python-api) loads safetensor weights and returns a model and tokenizer.
- MLX loads weight arrays keyed by tensor names and maps them into an MLX [nn.Module](https://ml-explore.github.io/mlx/build/html/python/nn/module.html#) parameter tree. This matches how Transformers checkpoints are organized.

> [!TIP]
> The MLX Transformers integration is bidirectional. Transformers can also load and run MLX weights from the Hub.

## Resources

- [MLX](https://ml-explore.github.io/mlx/build/html/index.html) documentation
- [mlx-lm](https://github.com/ml-explore/mlx-lm) repository containing MLX LLM implementations
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) community library with VLM implementations