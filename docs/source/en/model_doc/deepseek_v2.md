<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

*This model was released on 2024-05-07 and added to Hugging Face Transformers on 2025-07-09 and contributed by [VladOS95-cyber](https://github.com/VladOS95-cyber).*

# DeepSeek-V2

[DeepSeek-V2](https://huggingface.co/papers/2405.04434) is a Mixture-of-Experts (MoE) language model with 236B total parameters, where 21B are active per token, and supports a 128K token context length. It utilizes Multi-head Latent Attention (MLA) to compress the Key-Value (KV) cache and DeepSeekMoE for cost-effective training. Compared to DeepSeek 67B, DeepSeek-V2 offers superior performance, reduced training costs by 42.5%, decreased KV cache by 93.3%, and increased generation throughput by 5.76 times. Trained on an 8.1T token corpus and enhanced with Supervised Fine-Tuning and Reinforcement Learning, DeepSeek-V2 achieves top-tier performance with only 21B active parameters.

This model was contributed by [VladOS95-cyber](https://github.com/VladOS95-cyber).
The original code can be found [here](https://huggingface.co/deepseek-ai/DeepSeek-V2).

### Usage tips

The model uses Multi-head Latent Attention (MLA) and DeepSeekMoE architectures for efficient inference and cost-effective training. It employs an auxiliary-loss-free strategy for load balancing and multi-token prediction training objective. The model can be used for various language tasks after being pre-trained on 14.8 trillion tokens and going through Supervised Fine-Tuning and Reinforcement Learning stages.

## DeepseekV2Config
```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V2-Lite")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2-Lite")

inputs = tokenizer("Hello, my name is", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```


[[autodoc]] DeepseekV2Config

## DeepseekV2Model

[[autodoc]] DeepseekV2Model
    - forward

## DeepseekV2ForCausalLM

[[autodoc]] DeepseekV2ForCausalLM
    - forward

## DeepseekV2ForSequenceClassification

[[autodoc]] DeepseekV2ForSequenceClassification
    - forward

