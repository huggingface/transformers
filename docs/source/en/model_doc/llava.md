<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Llava

## Overview

The Llava model was proposed in [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) by Haotian Liu, Chunyuan Li, Qingyang Wu and Yong Jae Lee. It is an end-to-end trained large multimodal model that connects a vision encoder and LLM for general-purpose visual and language understanding.

The abstract from the paper is the following:

*Instruction tuning large language models (LLMs) using machine-generated instruction-following data has improved zero-shot capabilities on new tasks, but the idea is less explored in the multimodal field. In this paper, we present the first attempt to use language-only GPT-4 to generate multimodal language-image instruction-following data. By instruction tuning on such generated data, we introduce LLaVA: Large Language and Vision Assistant, an end-to-end trained large multimodal model that connects a vision encoder and LLM for general-purpose visual and language understanding.Our early experiments show that LLaVA demonstrates impressive multimodel chat abilities, sometimes exhibiting the behaviors of multimodal GPT-4 on unseen images/instructions, and yields a 85.1% relative score compared with GPT-4 on a synthetic multimodal instruction-following dataset. When fine-tuned on Science QA, the synergy of LLaVA and GPT-4 achieves a new state-of-the-art accuracy of 92.53%. We make GPT-4 generated visual instruction tuning data, our model and code base publicly available.*

Checkout all Llava models by the authors[here](https://huggingface.co/models?search=llava)
Checkout all HF friendly Llava models [here](https://huggingface.co/models?search=llava-hf)

Tips:

- Weights for the Llava models can be obtained from [here](https://huggingface.co/shauray/Llava-Llama-2-7B-hf/)
- The architecture is very similar to the first Llama, with the addition of Grouped Query Attention (GQA) following this [paper](https://arxiv.org/pdf/2305.13245.pdf)


Example:

```python
>>> from transformers import LlavaLlamaForCausalLM, LlamaProcessor

>>> processor = LlamaProcessor.from_pretrained("shauray/Llava-Llama-2-7B-hf")
>>> model = LlavaLlamaForCausalLM.from_pretrained("shauray/Llava-Llama-2-7B-hf")
```

Note that executing the script requires enough CPU RAM to host the whole model in float16 precision (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM). For the 13B model, it's thus 26GB of RAM needed.


This model was contributed by [Shauray Singh](https://huggingface.co/shauray) The original code of the authors can be found [here](https://github.com/haotian-liu/LLaVA).


## LlavaLlamaConfig

[[autodoc]] LlavaLlamaConfig
    - from_llava_llama_configs

## LlavaConfig

[[autodoc]] LlavaConfig

## LlamaConfig

[[autodoc]] LlamaConfig

## LlavaLlamaForCausalLM

[[autodoc]] LlavaLlamaForCausalLM
    - forward
    - generate

## LlavaProcessor

[[autodoc]] LlavaProcessor
