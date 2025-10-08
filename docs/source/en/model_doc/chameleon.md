<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2024-05-16 and added to Hugging Face Transformers on 2024-07-17 and contributed by [joaogante](https://huggingface.co/joaogante) and [RaushanTurganbay](https://huggingface.co/RaushanTurganbay).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Chameleon

[Chameleon: Mixed-Modal Early-Fusion Foundation Models](https://huggingface.co/papers/2405.09818v1) is a Vision-Language Model that uses vector quantization to tokenize images, enabling it to generate multimodal output. It handles images and texts in any sequence, including interleaved formats, and produces textual responses. Chameleon demonstrates superior performance in image captioning, outperforms Llama-2 in text-only tasks, and is competitive with Mixtral 8x7B and Gemini-Pro. It also performs non-trivial image generation and matches or exceeds the performance of larger models like Gemini Pro and GPT-4V in long-form mixed-modal generation tasks.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-to-text", model="facebook/chameleon-7b", dtype="auto")
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg", text="What is shown in this image? <image>"
)
```

</hfoption>
<hfoption id="ChameleonForConditionalGeneration">

```py
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, ChameleonForConditionalGeneration

processor = AutoProcessor.from_pretrained("facebook/chameleon-7b")
model = ChameleonForConditionalGeneration.from_pretrained("facebook/chameleon-7b", dtype="auto")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
prompt = "What is shown in this image?<image>"

inputs = processor(images=image, text=prompt, return_tensors="pt").to(torch.bfloat16)
output = model.generate(**inputs, max_new_tokens=50)
print(processor.decode(output[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## ChameleonConfig

[[autodoc]] ChameleonConfig

## ChameleonVQVAEConfig

[[autodoc]] ChameleonVQVAEConfig

## ChameleonProcessor

[[autodoc]] ChameleonProcessor

## ChameleonImageProcessor

[[autodoc]] ChameleonImageProcessor
    - preprocess

## ChameleonImageProcessorFast

[[autodoc]] ChameleonImageProcessorFast
    - preprocess

## ChameleonVQVAE

[[autodoc]] ChameleonVQVAE
    - forward

## ChameleonModel

[[autodoc]] ChameleonModel
    - forward

## ChameleonForConditionalGeneration

[[autodoc]] ChameleonForConditionalGeneration
    - forward

