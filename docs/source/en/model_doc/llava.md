<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->


# LLaVA

## Overview

The LLaVA model was proposed in [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) by Haotian Liu, Chunyuan Li, Qingyang Wu and Yong Jae Lee. It is an end-to-end trained large multimodal model that connects a vision encoder and LLM for general-purpose visual and language understanding.

The abstract from the paper is the following:

*Instruction tuning large language models (LLMs) using machine-generated instruction-following data has improved zero-shot capabilities on new tasks, but the idea is less explored in the multimodal field. In this paper, we present the first attempt to use language-only GPT-4 to generate multimodal language-image instruction-following data. By instruction tuning on such generated data, we introduce LLaVA: Large Language and Vision Assistant, an end-to-end trained large multimodal model that connects a vision encoder and LLM for general-purpose visual and language understanding. Our early experiments show that LLaVA demonstrates impressive multimodel chat abilities, sometimes exhibiting the behaviors of multimodal GPT-4 on unseen images/instructions, and yields a 85.1% relative score compared with GPT-4 on a synthetic multimodal instruction-following dataset. When fine-tuned on Science QA, the synergy of LLaVA and GPT-4 achieves a new state-of-the-art accuracy of 92.53%. We make GPT-4 generated visual instruction tuning data, our model and code base publicly available.*

Checkout all LLaVA models by the authors [here](https://huggingface.co/models?search=LLaVA)

Tips:

- Weights for the LLaVA v1.5 7B can be obtained from [liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b)
- Weights for the LLaVA v1.5 13B can be obtained from [liuhaotian/llava-v1.5-13b](https://huggingface.co/liuhaotian/llava-v1.5-13b)
- Weights for the LLaVA v1.5 7B (LoRa) can be obtained from [liuhaotian/llava-v1.5-7b-lora](https://huggingface.co/liuhaotian/llava-v1.5-7b-lora)
- Weights for the LLaVA v1.5 13B (LoRa) can be obtained from [liuhaotian/llava-v1.5-13b-lora](https://huggingface.co/liuhaotian/llava-v1.5-13b-lora)

The architecture is very similar to the first Llama, with the addition of Grouped Query Attention (GQA) following this [paper](https://arxiv.org/pdf/2305.13245.pdf)

Note that executing the script requires enough CPU RAM to host the whole model in float16 precision (even if the biggest versions come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM). For the 13B model, it's thus 26GB of RAM needed.

```python
>>> from transformers import LlaVaProcessor, LlaVaForCausalLM
>>> from PIL import Image

>>> import requests
>>> import torch

>>> checkpoint = "liuhaotian/llava-v1.5-7b"

>>> model = LlaVaForCausalLM.from_pretrained(checkpoint)
>>> processor = LlaVaProcessor.from_pretrained(checkpoint)

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

>>> output = processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
```

This model was contributed by [Matt Mazzola](https://huggingface.co/mattmazzola) The original code of the authors can be found [here](https://github.com/haotian-liu/LLaVA).

## LlaVaConfig

[[autodoc]] LlaVaConfig


## LlaVaTokenizer

[[autodoc]] LlaVaTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary


## LlaVaTokenizerFast

[[autodoc]] LlaVaTokenizerFast


## LlaVaModel

[[autodoc]] LlaVaModel
    - forward


## LlaVaForCausalLM

[[autodoc]] LlaVaForCausalLM
    - forward


## LlaVaForMaskedLM

[[autodoc]] LlaVaForMaskedLM
    - forward


## LlaVaForSequenceClassification

[[autodoc]] transformers.LlaVaForSequenceClassification
    - forward


## LlaVaForMultipleChoice

[[autodoc]] transformers.LlaVaForMultipleChoice
    - forward


## LlaVaForTokenClassification

[[autodoc]] transformers.LlaVaForTokenClassification
    - forward


## LlaVaForQuestionAnswering

[[autodoc]] LlaVaForQuestionAnswering
    - forward