<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2024-04-22 and added to Hugging Face Transformers on 2024-10-04.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# PhiMoE

[Phi-3.5-MoE](https://huggingface.co/papers/2404.14219) s a 3.8 billion parameter language model trained on 3.3 trillion tokens, achieving competitive performance with models like GPT-3.5 and Mixtral 8x7B (69% on MMLU, 8.38 on MT-bench) while being small enough for mobile deployment. Its training data is a scaled-up, heavily filtered version of phi-2’s dataset, including publicly available web data and synthetic data, and the model is further aligned for safety and chat robustness. Larger models, phi-3-small and phi-3-medium, trained on 4.8 trillion tokens, reach higher benchmarks (75–78% MMLU, 8.7–8.9 MT-bench). The phi-3.5 series expands capabilities with multilingual, multimodal, and long-context support, including phi-3.5-MoE, a 16×3.8B Mixture-of-Experts model outperforming similar-scale open models, and phi-3.5-Vision, which handles both single- and multi-image plus text reasoning tasks.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="microsoft/Phi-3.5-MoE-instruct", dtype="auto",)
messages = [ 
    {"role": "system", "content": "You are a plant biologist."}, 
    {"role": "user", "content": "Can you explain how plants create energy?"}, 
    {"role": "assistant", "content": "Plants create energy through photosynthesis, which is a process that converts sunlight into chemical energy. During photosynthesis, plants use chlorophyll in their leaves to capture light energy from the sun. They combine this energy with carbon dioxide from the air and water from the soil to produce glucose (sugar) and oxygen. The glucose serves as the plant's food source and energy storage."}, 
    {"role": "user", "content": "What are the key components needed for photosynthesis?"}, 
] 
pipeline(messages)
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-MoE-instruct")
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-MoE-instruct", dtype="auto")

messages = [ 
    {"role": "system", "content": "You are a plant biologist."}, 
    {"role": "user", "content": "Can you explain how plants create energy?"}, 
    {"role": "assistant", "content": "Plants create energy through photosynthesis, which is a process that converts sunlight into chemical energy. During photosynthesis, plants use chlorophyll in their leaves to capture light energy from the sun. They combine this energy with carbon dioxide from the air and water from the soil to produce glucose (sugar) and oxygen. The glucose serves as the plant's food source and energy storage."}, 
    {"role": "user", "content": "What are the key components needed for photosynthesis?"}, 
] 

inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## Usage tips

- This model is very similar to Mixtral. The main difference is [`Phi3LongRoPEScaledRotaryEmbedding`], which extends the context of rotary embeddings.
- Query, key, and values are fused. The MLP's up and gate projection layers are also fused.
- The tokenizer is identical to [`LlamaTokenizer`], except for additional tokens.

## PhimoeConfig

[[autodoc]] PhimoeConfig

## PhimoeModel

[[autodoc]] PhimoeModel
    - forward

## PhimoeForCausalLM

[[autodoc]] PhimoeForCausalLM
    - forward
    - generate

## PhimoeForSequenceClassification

[[autodoc]] PhimoeForSequenceClassification
    - forward

