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

*This model was released on 2024-11-21 and added to Hugging Face Transformers on 2025-07-08 and contributed by [yaswanthgali](https://huggingface.co/yaswanthgali).*

# AIMv2

[AIMv2](https://huggingface.co/papers/2411.14402) presents a novel method for pre-training large-scale vision encoders in a multimodal setting, combining images and text. The model, characterized by a straightforward pre-training process and scalability, pairs a vision encoder with a multimodal decoder that autoregressively generates raw image patches and text tokens. AIMV2 excels in both multimodal evaluations and vision benchmarks such as localization, grounding, and classification. Notably, the AIMV2-3B encoder achieves 89.5% accuracy on ImageNet-1k with a frozen trunk and outperforms state-of-the-art contrastive models like CLIP and SigLIP in multimodal image understanding across various settings.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="zero-shot-classification", model="apple/aimv2-large-patch14-native", dtype="auto")
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModel

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
text = ["Picture of a dog.", "Picture of a cat.", "Picture of a horse."]

processor = AutoProcessor.from_pretrained("apple/aimv2-large-patch14-224-lit")
model = AutoModel.from_pretrained("apple/aimv2-large-patch14-224-lit", dtype="auto")

inputs = processor(images=image, text=text, add_special_tokens=True, truncation=True, padding=True, return_tensors="pt",)
outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=-1)
pred_idx = torch.argmax(probs, dim=-1).item()
predicted_label = text[pred_idx]
print(f"Predicted label: {predicted_label}")
```

</hfoption>
</hfoptions>

## Aimv2Config

[[autodoc]] Aimv2Config

## Aimv2TextConfig

[[autodoc]] Aimv2TextConfig

## Aimv2VisionConfig

[[autodoc]] Aimv2VisionConfig

## Aimv2Model

[[autodoc]] Aimv2Model
    - forward

## Aimv2VisionModel

[[autodoc]] Aimv2VisionModel
    - forward

## Aimv2TextModel

[[autodoc]] Aimv2TextModel
    - forward

