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
*This model was published in HF papers on 2026-07-27 and contributed to Hugging Face Transformers on 2026-11-25.*

# Nanbeige4.2

[Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B) is a looped language model from [Nanbeige Lab](https://huggingface.co/Nanbeige), described in its [technical report](https://huggingface.co/papers/2607.22083).
The released checkpoint runs the same 22 [Llama-style](./llama) decoder layers twice per forward pass, with shared weights.
The output RMSNorm runs after each pass. Each pass gets its own key-value cache entries, so the released configuration
uses 44 cache slots across both passes.

This model was contributed by [Sushmanth Reddy](https://github.com/sushmanthreddy).

## Usage

Load the tokenizer and model with the auto classes. This example disables thinking in the chat template to request a short response.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "Nanbeige/Nanbeige4.2-3B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", dtype="auto")

inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is the capital of France? Answer with the city name."}],
    add_generation_prompt=True,
    enable_thinking=False,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(tokenizer.decode(outputs[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
```

## Notes

- The chat template enables thinking by default. In multi-turn conversations, set `preserve_thinking=True` to keep
  reasoning from earlier assistant turns in the prompt.
- `output_hidden_states` and `output_attentions` return outputs from the final decoder pass.
- Continuous batching (`generate_batch`) is currently unsupported.

## NanbeigeConfig

[[autodoc]] NanbeigeConfig

## NanbeigeModel

[[autodoc]] NanbeigeModel
    - forward

## NanbeigeForCausalLM

[[autodoc]] NanbeigeForCausalLM
    - forward
