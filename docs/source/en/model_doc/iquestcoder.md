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

# IQuestCoder

[IQuest-Coder-V1](https://huggingface.co/IQuestLab/IQuest-Coder-V1-40B-Instruct) is a family of code language models developed by [IQuest Lab](https://huggingface.co/IQuestLab), available in 7B, 14B and 40B sizes with base, instruct and thinking variants. Its architecture is the same as [Llama](./llama).

> [!NOTE]
> The looped `IQuest-Coder-V1-*-Loop-*` checkpoints use a different architecture and aren't supported by this model.

> [!WARNING]
> The tokenizer is built from the checkpoints' SentencePiece model. Many of its pieces share the same score, and SentencePiece breaks those ties by position, which the `tokenizers` library can't reproduce, so a small fraction of texts tokenize differently from the original remote code.

The auto classes load it with the Llama implementation ([`LlamaModel`] and [`LlamaForCausalLM`]), so only [`IQuestCoderConfig`] is specific to IQuestCoder.

The example below demonstrates how to generate text with the [`AutoModelForCausalLM`] class.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("IQuestLab/IQuest-Coder-V1-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("IQuestLab/IQuest-Coder-V1-7B-Instruct", device_map="auto")

messages = [{"role": "user", "content": "Write a Python function that checks whether a number is prime."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
output = model.generate(**inputs.to(model.device), max_new_tokens=128)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True))
```

## IQuestCoderConfig

[[autodoc]] IQuestCoderConfig
