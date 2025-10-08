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
*This model was released on 2021-09-21 and added to Hugging Face Transformers on 2021-10-13 and contributed by [nielsrogge](https://github.com/nielsrogge).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Vision Encoder Decoder Models

VisionEncoderDecoderModel creates image-to-text models by combining pretrained vision transformers with language models. This approach works well for tasks like optical character recognition, as shown in [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://huggingface.co/papers/2109.10282).

<hfoptions id="usage">
<hfoption id="VisionEncoderDecoderModel">

```py
import torch
import requests
from PIL import Image
from transformers import GPT2TokenizerFast, AutoProcessor, VisionEncoderDecoderModel

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning", dtype="auto")
tokenizer = AutoTokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = AutoProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
pixel_values = image_processor(image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values)
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)
```

</hfoption>
</hfoptions>

## VisionEncoderDecoderConfig

[[autodoc]] VisionEncoderDecoderConfig

## VisionEncoderDecoderModel

[[autodoc]] VisionEncoderDecoderModel
    - forward
    - from_encoder_decoder_pretrained

