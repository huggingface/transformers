<!--Copyright 2026 Sarvam AI and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-11-20.*
<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# SarvamMoE

SarvamMoE is the architecture of [Sarvam-30B](https://huggingface.co/sarvamai/sarvam-30b), a 30B parameter Mixture of Experts (MoE) language model developed by [Sarvam AI](https://www.sarvam.ai/). Its larger sibling, Sarvam-105B, uses [SarvamMLA](./sarvam_mla).

Key architectural features:

- **Grouped-query attention** with QK normalization and full rotary position embeddings.
- **Sparse Mixture of Experts**: 128 routed experts with 6 active per token, plus 1 shared expert. The first layer uses a dense MLP.
- **Sigmoid routing** with an expert bias correction and a routed scaling factor.

The architecture is the same as [GLM-4-MoE](./glm4_moe), with different parameter and config names. The auto classes load it with the GLM-4-MoE implementation ([`Glm4MoeModel`] and [`Glm4MoeForCausalLM`]), so only [`SarvamMoEConfig`] is specific to SarvamMoE. The released checkpoints are converted to the GLM-4-MoE names when loading and back when saving.

The example below demonstrates how to generate text with the [`AutoModelForCausalLM`] class.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("sarvamai/sarvam-30b")
model = AutoModelForCausalLM.from_pretrained("sarvamai/sarvam-30b", device_map="auto")

messages = [{"role": "user", "content": "Explain the Pythagorean theorem in one sentence."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
output = model.generate(**inputs.to(model.device), max_new_tokens=64)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True))
```

## SarvamMoEConfig

[[autodoc]] SarvamMoEConfig
