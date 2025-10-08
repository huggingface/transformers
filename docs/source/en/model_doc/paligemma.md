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
*This model was released on 2024-07-10 and added to Hugging Face Transformers on 2024-05-14 and contributed by [Molbap](https://huggingface.co/Molbap).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# PaliGemma

[PaliGemma](https://huggingface.co/papers/2407.07726) is an open Vision-Language Model that combines the SigLIP-So400m vision encoder with the Gemma-2B language model. It is designed as a versatile base model, optimized for transfer learning across a wide range of tasks. The model demonstrates strong performance on nearly 40 tasks, spanning standard VLM benchmarks as well as specialized areas like remote sensing and segmentation. Its architecture and training enable broad applicability in open-world scenarios.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-text-to-text", model="google/paligemma-3b-mix-224", dtype="auto")
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg", "What is the weather?")
```

</hfoption>
<hfoption id="PaliGemmaForConditionalGeneration">

```py
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma2-3b-mix-224", dtype="auto")
processor = AutoProcessor.from_pretrained("google/paligemma2-3b-mix-224")

prompt = "What is the weather?"
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(image, prompt, return_tensors="pt")

output = model.generate(**inputs, max_new_tokens=50)
print(processor.decode(output[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## PaliGemmaConfig

[[autodoc]] PaliGemmaConfig

## PaliGemmaProcessor

[[autodoc]] PaliGemmaProcessor

## PaliGemmaForConditionalGeneration

[[autodoc]] PaliGemmaForConditionalGeneration
    - forward

## PaliGemmaModel

[[autodoc]] PaliGemmaModel
    - forward
