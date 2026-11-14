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
*This model was contributed to Hugging Face Transformers on 2026-11-12.*
<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# InternLM3

[InternLM3](https://huggingface.co/internlm/internlm3-8b-instruct) is an 8B instruction-tuned language model developed by [Shanghai AI Laboratory](https://huggingface.co/internlm). Its architecture is the same as [Llama](./llama). Its config names the biases of the query, key and value projections `qkv_bias` and those of the attention output projection and MLP layers `bias`; [`InternLM3Config`] loads them as `attention_bias` and `mlp_bias`, and requires them to be equal, as they are in the released checkpoints.

The auto classes load it with the Llama implementation ([`LlamaModel`] and [`LlamaForCausalLM`]), so only [`InternLM3Config`] is specific to InternLM3.

The example below demonstrates how to generate text with the [`AutoModelForCausalLM`] class.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("internlm/internlm3-8b-instruct")
model = AutoModelForCausalLM.from_pretrained("internlm/internlm3-8b-instruct", device_map="auto")

messages = [{"role": "user", "content": "Explain the Pythagorean theorem in one sentence."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
output = model.generate(**inputs.to(model.device), max_new_tokens=64)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True))
```

## InternLM3Config

[[autodoc]] InternLM3Config
