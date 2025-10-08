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
*This model was released on 2022-01-28 and added to Hugging Face Transformers on 2022-12-21 and contributed by [ybelkada](https://huggingface.co/ybelkada).*

# BLIP

[BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://huggingface.co/papers/2201.12086) proposes a new VLP framework that excels in both vision-language understanding and generation tasks. BLIP enhances the use of noisy web data through a bootstrapping process involving synthetic caption generation and noise filtering. This approach leads to state-of-the-art results in image-text retrieval, image captioning, and visual question answering, with notable improvements in recall@1, CIDEr, and VQA scores. Additionally, BLIP demonstrates strong generalization to videolanguage tasks in a zero-shot setting.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="visual-question-answering", model="Salesforce/blip-vqa-base", dtype="auto")
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
pipeline(question="What is shown in this image?", image=url)
```

</hfoption>
<hfoption id="AutoModel">

```py
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering

processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base", dtype="auto")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

question = "What is shown in this image?"
inputs = processor(images=image, text=question, return_tensors="pt")

output = model.generate(**inputs)
print(processor.batch_decode(output, skip_special_tokens=True)[0])
```

</hfoption>
</hfoptions>

## BlipConfig

[[autodoc]] BlipConfig

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

## BlipModel

`BlipModel` is going to be deprecated in future versions, please use `BlipForConditionalGeneration`, `BlipForImageTextRetrieval` or `BlipForQuestionAnswering` depending on your usecase.

[[autodoc]] BlipModel
    - forward
    - get_text_features
    - get_image_features

## BlipTextModel

[[autodoc]] BlipTextModel
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

## BlipTextLMHeadModel

[[autodoc]] BlipTextLMHeadModel
    - forward

