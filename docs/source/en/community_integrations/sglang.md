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

# SGLang

[SGLang](https://docs.sglang.ai) is a low-latency, high-throughput inference engine for large language models (LLMs). It also includes a frontend language for building agentic workflows.

Set `model_impl="transformers"` to load a Transformers modeling backend.

```py
import sglang as sgl

llm = sgl.Engine("meta-llama/Llama-3.2-1B-Instruct", model_impl="transformers")
print(llm.generate(["The capital of France is"], {"max_new_tokens": 20})[0])
```

Pass `--model-impl transformers` to the `sglang.launch_server` command for online serving.

```bash
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --model-impl transformers \
  --host 0.0.0.0 \
  --port 30000
```

## Transformers integration

Setting `model_impl="transformers"` tells SGLang to skip its native model matching and use the Transformers model directly.

1. [`PreTrainedConfig.from_pretrained`] loads the model's `config.json` from the Hub or your Hugging Face cache.
2. [`AutoModel.from_config`] resolves the model class based on the config.
3. During loading, `_attn_implementation` is set to `"sglang"`. This routes attention calls through SGLang's RadixAttention kernels.
4. SGLang's parallel linear class replaces linear layers to support tensor parallelism.
5. The [load_weights](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/transformers.py#L277) function populates the model with weights from safetensors files.

The model benefits from all SGLang optimizations while using the Transformers model structure.

> [!WARNING]
> Compatible models require `_supports_attention_backend=True` so SGLang can control attention execution. See the [Building a compatible model backend for inference](./transformers_as_backend#model-implementation) guide for details.

## Resources

- [SGLang docs](https://docs.sglang.ai/supported_models/transformers_fallback.html) has more usage examples and tips for using Transformers as a backend.
- [Transformers backend integration in SGLang](https://huggingface.co/blog/transformers-backend-sglang) blog post explains what this integration enables.