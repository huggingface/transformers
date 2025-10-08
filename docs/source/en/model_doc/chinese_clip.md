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
*This model was released on 2022-11-02 and added to Hugging Face Transformers on 2022-12-01 and contributed by [OFA-Sys](https://huggingface.co/OFA-Sys).*

# Chinese-CLIP

[Chinese-CLIP](https://huggingface.co/papers/2211.01335) constructs a large-scale dataset of Chinese image-text pairs and pretrains models of varying sizes, from 77 to 958 million parameters. It employs a two-stage pretraining method, initially freezing the image encoder before optimizing all parameters. Experiments show superior performance on MUGE, Flickr30K-CN, and COCO-CN for zero-shot learning and finetuning, and competitive results in zero-shot image classification on the ELEVATER benchmark.

<hfoptions id="usage">
<hfoption id="ChineseCLIPModel">

```py
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, ChineseCLIPModel

model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16", dtype="auto")
processor = AutoProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

url = "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
# Squirtle, Bulbasaur, Charmander, Pikachu in English
texts = ["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]

inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

print("Text-image similarity probabilities:")
for i, (text, prob) in enumerate(zip(texts, probs[0])):
    print(f"'{text}' -> {prob.item():.4f} ({prob.item()*100:.1f}%)")
```

</hfoption>
</hfoptions>

## ChineseCLIPConfig

[[autodoc]] ChineseCLIPConfig

## ChineseCLIPTextConfig

[[autodoc]] ChineseCLIPTextConfig

## ChineseCLIPVisionConfig

[[autodoc]] ChineseCLIPVisionConfig

## ChineseCLIPImageProcessor

[[autodoc]] ChineseCLIPImageProcessor
    - preprocess

## ChineseCLIPImageProcessorFast

[[autodoc]] ChineseCLIPImageProcessorFast
    - preprocess

## ChineseCLIPFeatureExtractor

[[autodoc]] ChineseCLIPFeatureExtractor

## ChineseCLIPProcessor

[[autodoc]] ChineseCLIPProcessor

## ChineseCLIPModel

[[autodoc]] ChineseCLIPModel
    - forward
    - get_text_features
    - get_image_features

## ChineseCLIPTextModel

[[autodoc]] ChineseCLIPTextModel
    - forward

## ChineseCLIPVisionModel

[[autodoc]] ChineseCLIPVisionModel
    - forward

