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
*This model was contributed to Hugging Face Transformers on 2026-11-10.*
<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# TeleChat

[TeleChat](https://huggingface.co/Tele-AI) is a family of language models developed by [China Telecom Artificial Intelligence Technology](https://huggingface.co/Tele-AI), covering TeleChat (1B to 12B), TeleChat2 (3B to 115B) and TeleChat2.5 (35B and 115B). TeleChat3 is a separate model, see [TeleChat3](./telechat3).

Its architecture is the same as [Llama](./llama), except that:

- only the attention output projection and the down projection of the MLP have biases;
- once the sequence grows beyond the training length (`original_max_position_embeddings` in `rope_parameters`), RoPE is rescaled with a stepwise NTK scaling of its base and the cos/sin are scaled by `0.1 * ln(seq_len / original_max_position_embeddings) + 1`.

> [!NOTE]
> TeleChat-52B uses a different architecture and isn't supported by this model.

The example below demonstrates how to generate text with the [`AutoModelForCausalLM`] class.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("Tele-AI/TeleChat2-3B")
model = AutoModelForCausalLM.from_pretrained("Tele-AI/TeleChat2-3B", device_map="auto")

messages = [{"role": "user", "content": "Explain the Pythagorean theorem in one sentence."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
output = model.generate(**inputs.to(model.device), max_new_tokens=64)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True))
```

## TeleChatConfig

[[autodoc]] TeleChatConfig

## TeleChatModel

[[autodoc]] TeleChatModel
    - forward

## TeleChatForCausalLM

[[autodoc]] TeleChatForCausalLM
    - forward
