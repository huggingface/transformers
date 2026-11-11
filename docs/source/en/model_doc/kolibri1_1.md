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
*This model was contributed to Hugging Face Transformers on 2026-11-11.*

# Kolibri 1.1

Kolibri 1.1 is Aleph Alpha's successor to [Kolibri 1](./kolibri1).
Comparison to Kolibri 1:
- It keeps the mixture-of-experts layers and the sliding-window attention layers of Kolibri 1.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("<checkpoint>")
model = AutoModelForCausalLM.from_pretrained("<checkpoint>", device_map="auto", attn_implementation="sdpa")

messages = [{"role": "user", "content": "Erkläre kurz, was ein Mixture-of-Experts-Modell ist."}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
```

## Kolibri1_1Config

[[autodoc]] Kolibri1_1Config

## Kolibri1_1Model

[[autodoc]] Kolibri1_1Model
    - forward

## Kolibri1_1ForCausalLM

[[autodoc]] Kolibri1_1ForCausalLM
    - forward
