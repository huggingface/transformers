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

*Instruction tuning large language models (LLMs) using machine-generated instruction-following data has improved zero-shot capabilities on new tasks, but the idea is less explored in the multimodal field. In this paper, we present the first attempt to use language-only GPT-4 to generate multimodal language-image instruction-following data. By instruction tuning on such generated data, we introduce LLaVA: Large Language and Vision Assistant, an end-to-end trained large multimodal model that connects a vision encoder and LLM for general-purpose visual and language understanding. Our early experiments show that LLaVA demonstrates impressive multimodel chat abilities, sometimes exhibiting the behaviors of multimodal GPT-4 on unseen images/instructions, and yields a 85.1% relative score compared with GPT-4 on a synthetic multimodal instruction-following dataset. When fine-tuned on Science QA, the synergy of LLaVA and GPT-4 achieves a new state-of-the-art accuracy of 92.53%. We make GPT-4 generated visual instruction tuning data, our model and code base publicly available.*

Checkout all Llava models by the authors[here](https://huggingface.co/models?search=llava)
Checkout all HF friendly Llava models [here](https://huggingface.co/models?search=llava-hf)

Tips:

- Weights for the Llava 7B can be obtained from [here](https://huggingface.co/shauray/Llava-Llama-2-7B-hf/)
- Weights for the Llava 13B can be obtained from [here](https://huggingface.co/shauray/Llava-Llama-2-13B-hf/)
- The architecture is very similar to the first Llama, with the addition of Grouped Query Attention (GQA) following this [paper](https://arxiv.org/pdf/2305.13245.pdf)

Note that executing the script requires enough CPU RAM to host the whole model in float16 precision (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM). For the 13B model, it's thus 26GB of RAM needed.

```python
>>> from transformers import LlavaProcessor, LlavaForCausalLM
>>> from PIL import Image

>>> import requests
>>> import torch

>>> PATH_TO_CONVERTED_WEIGHTS = "shauray/Llava-Llama-2-7B-hf"

>>> model = LlavaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
>>> processor = LlavaProcessor.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)

>>> url = "https://llava-vl.github.io/static/images/view.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
>>> prompt = "How can you best describe this image?"

>>> inputs = processor(text=prompt, images=image, return_tensors="pt")

>>> generate_ids = model.generate(**inputs,
...     do_sample=True,
...     max_length=1024,
...     temperature=0.1,
...     top_p=0.9,
... )

>>> out = processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
```

This model was contributed by [Shauray Singh](https://huggingface.co/shauray) The original code of the authors can be found [here](https://github.com/haotian-liu/LLaVA).


## LlavaConfig

[[autodoc]] LlavaConfig
    - from_llava_configs

## LlavaVisionConfig

[[autodoc]] LlavaVisionConfig

## LlamaConfig

[[autodoc]] LlamaConfig

## LlavaForCausalLM

[[autodoc]] LlavaForCausalLM
    - forward
    - generate

## LlavaProcessor

[[autodoc]] LlavaProcessor
