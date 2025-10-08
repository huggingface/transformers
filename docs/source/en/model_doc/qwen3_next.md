<!--Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2025-10-07.*

# Qwen3-Next

[Qwen3-Next](https://qwen.ai/blog?id=3425e8f58e31e252f5c53dd56ec47363045a3f6b&from=research.research-list) introduces a hybrid attention mechanism, a highly sparse Mixture-of-Experts (MoE) structure, multi-token prediction, and training-stability optimizations to improve efficiency in long-context and large-parameter settings. Its base model, Qwen3-Next-80B-A3B-Base, has 80 billion parameters but activates only 3 billion during inference, achieving comparable performance to the dense Qwen3-32B while using less than 10% of the training cost and delivering over 10× higher throughput for contexts beyond 32K tokens. Two post-trained versions, Qwen3-Next-80B-A3B-Instruct and Qwen3-Next-80B-A3B-Thinking, address RL stability and efficiency, excelling in ultra-long context tasks (up to 256K tokens) and complex reasoning, outperforming higher-cost models and approaching top-tier model performance. Overall, Qwen3-Next demonstrates extreme efficiency in both training and inference without sacrificing accuracy or reasoning capability.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="Qwen/Qwen3-Next-80B-A3B-Instruct", dtype="auto",)
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

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct", dtype="auto",)

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

## Qwen3NextConfig

[[autodoc]] Qwen3NextConfig

## Qwen3NextModel

[[autodoc]] Qwen3NextModel
    - forward

## Qwen3NextForCausalLM

[[autodoc]] Qwen3NextForCausalLM
    - forward

## Qwen3NextForSequenceClassification

[[autodoc]] Qwen3NextForSequenceClassification
    - forward

## Qwen3NextForQuestionAnswering

[[autodoc]] Qwen3NextForQuestionAnswering
    - forward

## Qwen3NextForTokenClassification

[[autodoc]] Qwen3NextForTokenClassification
    - forward
