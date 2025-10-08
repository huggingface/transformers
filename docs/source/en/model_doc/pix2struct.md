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
*This model was released on 2022-10-07 and added to Hugging Face Transformers on 2023-03-22 and contributed by [ybelkada](https://huggingface.co/ybelkada).*

# Pix2Struct

[Pix2Struct](https://huggingface.co/papers/2210.03347) is a pretrained image-to-text model designed for visual language understanding. It learns to parse masked screenshots of web pages into simplified HTML, leveraging the web's diverse visual elements and HTML structure for pretraining. This approach encompasses common pretraining signals like OCR, language modeling, and image captioning. Pix2Struct introduces a variable-resolution input representation and flexible integration of language and vision inputs, rendering language prompts directly onto images. The model achieves state-of-the-art results across six out of nine tasks in four domains: documents, illustrations, user interfaces, and natural images.

<hfoptions id="usage">
<hfoption id="Pix2StructForConditionalGeneration">

```py
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, Pix2StructForConditionalGeneration

processor = AutoProcessor.from_pretrained("google/pix2struct-textcaps-base")
model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-textcaps-base", dtype="auto")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt")

generated_ids = model.generate(**inputs, max_new_tokens=50)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)
```

</hfoption>
</hfoptions>

## Pix2StructConfig

[[autodoc]] Pix2StructConfig

## Pix2StructTextConfig

[[autodoc]] Pix2StructTextConfig

## Pix2StructVisionConfig

[[autodoc]] Pix2StructVisionConfig

## Pix2StructProcessor

[[autodoc]] Pix2StructProcessor

## Pix2StructImageProcessor

[[autodoc]] Pix2StructImageProcessor
    - preprocess

## Pix2StructTextModel

[[autodoc]] Pix2StructTextModel
    - forward

## Pix2StructVisionModel

[[autodoc]] Pix2StructVisionModel
    - forward

## Pix2StructForConditionalGeneration

[[autodoc]] Pix2StructForConditionalGeneration
    - forward

