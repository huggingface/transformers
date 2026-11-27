<!--Copyright 2026 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-11-26.*

# MiniCPM5-15B-A2.5B

MiniCPM5-15B-A2.5B is a decoder-only language model with 15 billion parameters and 2.5 billion active parameters per token. It uses gated grouped-query attention. The first decoder layer has a dense feed-forward network. Each subsequent layer uses a shared expert and selects 16 of 160 routed experts per token.

## Usage example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM5-15B-A2.5B")
model = AutoModelForCausalLM.from_pretrained("openbmb/MiniCPM5-15B-A2.5B", device_map="auto")

messages = [{"role": "user", "content": "What is 7 + 8? Answer with only the number."}]
inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=True, return_tensors="pt"
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=16, do_sample=False)
print(tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

## MiniCPM5MoeConfig

[[autodoc]] MiniCPM5MoeConfig

## MiniCPM5MoeModel

[[autodoc]] MiniCPM5MoeModel
    - forward

## MiniCPM5MoeForCausalLM

[[autodoc]] MiniCPM5MoeForCausalLM
    - forward
