
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
*This model was released on 2025-04-01 and added to Hugging Face Transformers on 2025-03-20 and contributed by [RyanMullins](https://huggingface.co/RyanMullins).*

# ShieldGemma 2

[ShieldGemma 2](https://huggingface.co/papers/2504.01081) is a 4-billion-parameter image content moderation model built on Gemma 3, designed to detect risks in sexually explicit, violent, and dangerous content for both synthetic and natural images. It demonstrates state-of-the-art performance on internal and external benchmarks, outperforming LlavaGuard, GPT-4o mini, and the base Gemma 3 model. The model leverages a novel adversarial data generation pipeline to produce diverse and robust training examples. ShieldGemma 2 is released as an open tool to support multimodal safety and responsible AI development.

<hfoptions id="usage">
<hfoption id="ShieldGemma2ForImageClassification">

```py
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, ShieldGemma2ForImageClassification

model = ShieldGemma2ForImageClassification.from_pretrained("google/shieldgemma-2-4b-it", dtype="auto")
processor = AutoProcessor.from_pretrained("google/shieldgemma-2-4b-it")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=[image], return_tensors="pt").to(model.device)

output = model(**inputs)
print(output.probabilities)
```

</hfoption>
</hfoptions>

## ShieldGemma2Processor

[[autodoc]] ShieldGemma2Processor

## ShieldGemma2Config

[[autodoc]] ShieldGemma2Config

## ShieldGemma2ForImageClassification

[[autodoc]] ShieldGemma2ForImageClassification
    - forward
