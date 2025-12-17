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

# vLLM

[vLLM](https://github.com/vllm-project/vllm) is a high-throughput inference engine for serving LLMs at scale. It continuously batches requests and keeps KV cache memory compact with PagedAttention.

Set `model_impl="transformers"` to load a Transformers modeling backend.

```py
from vllm import LLM

llm = LLM(model="meta-llama/Llama-3.2-1B", model_impl="transformers")
print(llm.generate(["The capital of France is"]))
```

Pass `--model-impl transformers` to the `vllm serve` command for online serving.

```bash
vllm serve meta-llama/Llama-3.2-1B \
    --task generate \
    --model-impl transformers
```

vLLM fetches `config.json` from the Hub or your Hugging Face cache and reads `model_type` to load the config class. If the model type isn't in vLLM's internal registry, it falls back to [`AutoConfig.from_pretrained`]. With `model_impl="transformers"`, vLLM passes the config to [`~AutoModelForCausalLM.from_pretrained`] to instantiate the model instead of using vLLM's native model classes.

[`AutoTokenizer.from_pretrained`] loads tokenizer files. Some tokenizer internals are cached to reduce repeated overhead during inference. Model weights download from the Hub in safetensors format.

Transformers handles the model and forward pass. vLLM provides the high-throughput serving stack.

## Resources

- Refer to the [vLLM docs](https://docs.vllm.ai/en/latest/models/supported_models.html#transformers) for more usage examples and tips for using a Transformers model as the backend.
- Learn more about how vLLM integrates with Transformers in the [Integration with Hugging Face](https://docs.vllm.ai/en/latest/design/huggingface_integration/) doc.