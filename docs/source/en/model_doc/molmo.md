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

# Molmo

## Overview

The Molmo model was proposed in [Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models]([https://arxiv.org/abs/2409.17146]) by Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, Jiasen Lu, Taira Anderson, Erin Bransom, Kiana Ehsani, Huong Ngo, YenSung Chen, Ajay Patel, Mark Yatskar, Chris Callison-Burch, Andrew Head, Rose Hendrix, Favyen Bastani, Eli VanderBilt, Nathan Lambert, Yvonne Chou, Arnavi Chheda, Jenna Sparks, Sam Skjonsberg, Michael Schmitz, Aaron Sarnat, Byron Bischoff, Pete Walsh, Chris Newell, Piper Wolters, Tanmay Gupta, Kuo-Hao Zeng, Jon Borchardt, Dirk Groeneveld, Jen Dumas, Crystal Nam, Sophie Lebrecht, Caitlin Wittlif, Carissa Schoenick, Oscar Michel, Ranjay Krishna, Luca Weihs, Noah A. Smith, Hannaneh Hajishirzi, Ross Girshick, Ali Farhadi, Aniruddha Kembhavi.

Molmo, developed by AllenAI team, is an open-source multimodal AI model capable of processing text and images within a unified framework. It outperforms larger models in efficiency and accuracy, leveraging high-quality datasets like PixMo for tasks such as captioning, question answering, and visual pointing.

The abstract from the paper is the following:

*Today's most advanced multimodal models remain proprietary. The strongest open-weight models rely heavily on synthetic data from proprietary VLMs to achieve good performance, effectively distilling these closed models into open ones. As a result, the community is still missing foundational knowledge about how to build performant VLMs from scratch. We present Molmo, a new family of VLMs that are state-of-the-art in their class of openness. Our key innovation is a novel, highly detailed image caption dataset collected entirely from human annotators using speech-based descriptions. To enable a wide array of user interactions, we also introduce a diverse dataset mixture for fine-tuning that includes in-the-wild Q&A and innovative 2D pointing data. The success of our approach relies on careful choices for the model architecture details, a well-tuned training pipeline, and, most critically, the quality of our newly collected datasets, all of which will be released. The best-in-class 72B model within the Molmo family not only outperforms others in the class of open weight and data models but also compares favorably against proprietary systems like GPT-4o, Claude 3.5, and Gemini 1.5 on both academic benchmarks and human evaluation.
*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/molmo_arch.png"
alt="drawing" width="600"/>

<small> Molmo incorporates images by encoding various patches of the input image. Taken from the <a href="https://arxiv.org/abs/2409.17146">original paper.</a> </small>


Tips:

- We recommend calling `processor.tokenizer.padding_side = "left"` for batched generation because it leads to more accurate results.


This model was contributed by [Molbap](https://huggingface.co/Molbap).


## Usage example

### Single image inference

Here's how to load the model and perform inference in half-precision (`torch.float16`):

```python
from transformers import MolmoForConditionalGeneration, AutoProcessor
import torch
from PIL import Image
import requests

model = MolmoForConditionalGeneration.from_pretrained("allenai/Molmo-7B-D-hf", torch_dtype="float16", device_map="auto")
processor = AutoProcessor.from_pretrained("allenai/Molmo-7B-D-hf")


conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://picsum.photos/id/237/536/354"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]
inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        return_dict=True,
        add_generation_prompt=True
        ).to(model.device)

output = model.generate(**inputs, max_new_tokens=100)

print(processor.decode(output[0], skip_special_tokens=True))
```


## MolmoConfig

[[autodoc]] MolmoConfig

## MolmoTextConfig

[[autodoc]] MolmoTextConfig

## MolmoVisionConfig

[[autodoc]] MolmoVisionConfig

## MolmoPoolingConfig

[[autodoc]] MolmoPoolingConfig

## MolmoImageProcessor

[[autodoc]] MolmoImageProcessor

## MolmoImageProcessorFast

[[autodoc]] MolmoImageProcessorFast

## MolmoProcessor

[[autodoc]] MolmoProcessor

## MolmoAdapterModel

[[autodoc]] MolmoAdapterModel
    - forward

## MolmoModel

[[autodoc]] MolmoModel
    - forward

## MolmoTextModel

[[autodoc]] MolmoTextModel
    - forward

## MolmoVisionModel

[[autodoc]] MolmoVisionModel
    - forward

## MolmoForCausalLM

[[autodoc]] MolmoForCausalLM
    - forward
    
## MolmoForConditionalGeneration

[[autodoc]] MolmoForConditionalGeneration
    - forward
