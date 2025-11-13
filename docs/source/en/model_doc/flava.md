<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2021-12-08 and added to Hugging Face Transformers on 2022-05-11 and contributed by [aps](https://huggingface.co/aps).*

# FLAVA

[FLAVA: A Foundational Language And Vision Alignment Model](https://huggingface.co/papers/2112.04482) aims to develop a unified foundation model capable of handling vision, language, and vision-and-language multimodal tasks. Unlike existing models that are typically either cross-modal or multi-modal but not both, FLAVA targets all modalities simultaneously. Demonstrating strong results across 35 diverse tasks, FLAVA serves as a comprehensive vision and language foundation model.

<hfoptions id="usage">
<hfoption id="FlavaModel">

```py
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, FlavaModel

model = FlavaModel.from_pretrained("facebook/flava-full", dtype="auto")
processor = AutoProcessor.from_pretrained("facebook/flava-full")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(
  text=["a photo of a cat", "a photo of a dog"], images=[image, image], return_tensors="pt", padding="max_length", max_length=77
)

outputs = model(**inputs)
image_embeddings = outputs.image_embeddings
text_embeddings = outputs.text_embeddings
multimodal_embeddings = outputs.multimodal_embeddings
```

</hfoption>
</hfoptions>

## FlavaConfig

[[autodoc]] FlavaConfig

## FlavaTextConfig

[[autodoc]] FlavaTextConfig

## FlavaImageConfig

[[autodoc]] FlavaImageConfig

## FlavaMultimodalConfig

[[autodoc]] FlavaMultimodalConfig

## FlavaImageCodebookConfig

[[autodoc]] FlavaImageCodebookConfig

## FlavaProcessor

[[autodoc]] FlavaProcessor

## FlavaImageProcessor

[[autodoc]] FlavaImageProcessor
    - preprocess

## FlavaImageProcessorFast

[[autodoc]] FlavaImageProcessorFast
    - preprocess

## FlavaForPreTraining

[[autodoc]] FlavaForPreTraining
    - forward

## FlavaModel

[[autodoc]] FlavaModel
    - forward
    - get_text_features
    - get_image_features

## FlavaImageCodebook

[[autodoc]] FlavaImageCodebook
    - forward
    - get_codebook_indices
    - get_codebook_probs

## FlavaTextModel

[[autodoc]] FlavaTextModel
    - forward

## FlavaImageModel

[[autodoc]] FlavaImageModel
    - forward

## FlavaMultimodalModel

[[autodoc]] FlavaMultimodalModel
    - forward

