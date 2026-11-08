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
*This model was contributed to Hugging Face Transformers on 2026-11-08.*
<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# TeleChat3

[TeleChat3](https://huggingface.co/Tele-AI/TeleChat3-36B-Thinking) is a family of 36B reasoning language models developed by [China Telecom Artificial Intelligence Technology](https://huggingface.co/Tele-AI), including the general-purpose TeleChat3-36B-Thinking and the code-focused TeleChat3-Coder-36B-Thinking. Its architecture is the same as [Llama](./llama).

The original checkpoints use a custom `"telechat3-yarn"` RoPE type, which is YaRN with 0.07 instead of 0.1 as the coefficient of its default attention factor. [`TeleChat3Config`] converts it to `"yarn"` with an explicit `attention_factor` when the config is loaded.

The tokenizer class named in the checkpoints' `tokenizer_config.json` only exists in their remote code, so loading the tokenizer requires `trust_remote_code=True`.

The example below demonstrates how to generate text with the [`AutoModelForCausalLM`] class.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("Tele-AI/TeleChat3-36B-Thinking", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Tele-AI/TeleChat3-36B-Thinking", device_map="auto")

messages = [{"role": "user", "content": "Explain the Pythagorean theorem in one sentence."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
output = model.generate(**inputs.to(model.device), max_new_tokens=256)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True))
```

## TeleChat3Config

[[autodoc]] TeleChat3Config

## TeleChat3Model

[[autodoc]] TeleChat3Model
    - forward

## TeleChat3ForCausalLM

[[autodoc]] TeleChat3ForCausalLM
    - forward
