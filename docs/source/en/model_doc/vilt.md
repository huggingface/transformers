<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2021-02-05 and added to Hugging Face Transformers on 2022-01-19 and contributed by [nielsr](https://huggingface.co/nielsr).*

# ViLT

[ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://huggingface.co/papers/2102.03334) presents a minimal Vision-and-Language Pre-training (VLP) model that integrates text embeddings into a Vision Transformer (ViT). This approach eliminates the need for convolutional architectures and region supervision, significantly reducing computational requirements while maintaining competitive performance on downstream tasks. ViLT processes visual inputs in a convolution-free manner, similar to text inputs, achieving up to tens of times faster processing speeds compared to previous models.

<hfoptions id="usage">
<hfoption id="ViltForQuestionAnswering">

```py
import torch
import requests
from transformers import AutoProcessor, ViltForQuestionAnswering
from PIL import Image

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
text = "How many cats are there?"

processor = AutoProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa", dtype="auto")

encoding = processor(image, text, return_tensors="pt")
outputs = model(**encoding)
logits = outputs.logits
idx = logits.argmax(-1).item()
print("Predicted answer:", model.config.id2label[idx])
```

</hfoption>
</hfoptions>

## ViltConfig

[[autodoc]] ViltConfig

## ViltImageProcessor

[[autodoc]] ViltImageProcessor
    - preprocess

## ViltProcessor

[[autodoc]] ViltProcessor
    - __call__

## ViltModel

[[autodoc]] ViltModel
    - forward

## ViltForMaskedLM

[[autodoc]] ViltForMaskedLM
    - forward

## ViltForQuestionAnswering

[[autodoc]] ViltForQuestionAnswering
    - forward

## ViltForImagesAndTextClassification

[[autodoc]] ViltForImagesAndTextClassification
    - forward

## ViltForImageAndTextRetrieval

[[autodoc]] ViltForImageAndTextRetrieval
    - forward

## ViltForTokenClassification

[[autodoc]] ViltForTokenClassification
    - forward

## ViltImageProcessorFast

[[autodoc]] ViltImageProcessorFast
    - preprocess

