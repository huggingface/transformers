<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
    </div>
</div>

# BLIP

[BLIP](https://huggingface.co/papers/2201.12086) (Bootstrapped Language-Image Pretraining) is a vision-language pretraining (VLP) framework designed for *both* understanding and generation tasks. Most existing pretrained models are only good at one or the other. It uses a captioner to generate captions and a filter to remove the noisy captions. This increases training data quality and more effectively uses the messy web data.


You can find all the original BLIP checkpoints under the [BLIP](https://huggingface.co/collections/Salesforce/blip-models-65242f40f1491fbf6a9e9472) collection.

> [!TIP]
> This model was contributed by [ybelkada](https://huggingface.co/ybelkada).
> 
> Click on the BLIP models in the right sidebar for more examples of how to apply BLIP to different vision language tasks.

The example below demonstrates how to visual question answering with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipeline = pipeline(
    task="visual-question-answering",
    model="Salesforce/blip-vqa-base",
    dtype=torch.float16,
    device=0
)
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
pipeline(question="What is the weather in this image?", image=url)
```

</hfoption>
<hfoption id="AutoModel">

```python
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering

processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = AutoModelForVisualQuestionAnswering.from_pretrained(
    "Salesforce/blip-vqa-base", 
    dtype=torch.float16,
    device_map="auto"
)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

question = "What is the weather in this image?"
inputs = processor(images=image, text=question, return_tensors="pt").to("cuda", torch.float16)

output = model.generate(**inputs)
processor.batch_decode(output, skip_special_tokens=True)[0]
```

</hfoption>
</hfoptions>

## Resources

Refer to this [notebook](https://github.com/huggingface/notebooks/blob/main/examples/image_captioning_blip.ipynb) to learn how to fine-tune BLIP for image captioning on a custom dataset.

## BlipConfig

[[autodoc]] BlipConfig
    - from_text_vision_configs

## BlipTextConfig

[[autodoc]] BlipTextConfig

## BlipVisionConfig

[[autodoc]] BlipVisionConfig

## BlipProcessor

[[autodoc]] BlipProcessor

## BlipImageProcessor

[[autodoc]] BlipImageProcessor
    - preprocess

## BlipImageProcessorFast

[[autodoc]] BlipImageProcessorFast
    - preprocess

<frameworkcontent>
<pt>

## BlipModel

`BlipModel` is going to be deprecated in future versions, please use `BlipForConditionalGeneration`, `BlipForImageTextRetrieval` or `BlipForQuestionAnswering` depending on your usecase.

[[autodoc]] BlipModel
    - forward
    - get_text_features
    - get_image_features

## BlipTextModel

[[autodoc]] BlipTextModel
    - forward

## BlipTextLMHeadModel

[[autodoc]] BlipTextLMHeadModel
- forward

## BlipVisionModel

[[autodoc]] BlipVisionModel
    - forward

## BlipForConditionalGeneration

[[autodoc]] BlipForConditionalGeneration
    - forward

## BlipForImageTextRetrieval

[[autodoc]] BlipForImageTextRetrieval
    - forward

## BlipForQuestionAnswering

[[autodoc]] BlipForQuestionAnswering
    - forward

</pt>
<tf>

## TFBlipModel

[[autodoc]] TFBlipModel
    - call
    - get_text_features
    - get_image_features

## TFBlipTextModel

[[autodoc]] TFBlipTextModel
    - call

## TFBlipTextLMHeadModel

[[autodoc]] TFBlipTextLMHeadModel
- forward

## TFBlipVisionModel

[[autodoc]] TFBlipVisionModel
    - call

## TFBlipForConditionalGeneration

[[autodoc]] TFBlipForConditionalGeneration
    - call

## TFBlipForImageTextRetrieval

[[autodoc]] TFBlipForImageTextRetrieval
    - call

## TFBlipForQuestionAnswering

[[autodoc]] TFBlipForQuestionAnswering
    - call
</tf>
</frameworkcontent>
