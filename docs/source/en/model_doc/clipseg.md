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
*This model was released on 2021-12-18 and added to Hugging Face Transformers on 2022-11-08 and contributed by [nielsr](https://huggingface.co/nielsr).*

# CLIPSeg

[CLIPSeg](https://huggingface.co/papers/2112.10003) extends the CLIP model with a transformer-based decoder to enable zero-shot and one-shot image segmentation using arbitrary text or image prompts. This unified model can handle referring expression segmentation, zero-shot segmentation, and one-shot segmentation tasks. Trained on an extended PhraseCut dataset, CLIPSeg generates binary segmentation maps based on free-text or image queries, demonstrating adaptability to various binary segmentation tasks involving affordances or properties.

<hfoptions id="usage">
<hfoption id="CLIPSegModel">

```py
import torch
from transformers import AutoProcessor, CLIPSegModel
from transformers.image_utils import load_image

processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegModel.from_pretrained("CIDAS/clipseg-rd64-refined", dtype="auto")

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
texts = ["a photo of a cat", "a photo of a dog"]
inputs = processor(
    text=texts, images=image, return_tensors="pt", padding=True
)

with torch.inference_mode():
    outputs = model(**inputs)
logits_per_image = outputs.logits_per_image 
probs = logits_per_image.softmax(dim=1)

print("Text-image similarity probabilities:")
for i, (text, prob) in enumerate(zip(texts, probs[0])):
    print(f"'{text}' -> {prob.item():.4f} ({prob.item()*100:.1f}%)")
```

</hfoption>
</hfoptions>

## Usage tips

- [`CLIPSegForImageSegmentation`] adds a decoder on top of [`CLIPSegModel`]. [`CLIPSegModel`] is identical to [`CLIPModel`].
- [`CLIPSegForImageSegmentation`] generates image segmentations based on arbitrary prompts at test time. Prompts can be text (provided as `input_ids`) or images (provided as `conditional_pixel_values`). Provide custom conditional embeddings as `conditional_embeddings`.

## CLIPSegConfig

[[autodoc]] CLIPSegConfig

## CLIPSegTextConfig

[[autodoc]] CLIPSegTextConfig

## CLIPSegVisionConfig

[[autodoc]] CLIPSegVisionConfig

## CLIPSegProcessor

[[autodoc]] CLIPSegProcessor

## CLIPSegModel

[[autodoc]] CLIPSegModel
    - forward
    - get_text_features
    - get_image_features

## CLIPSegTextModel

[[autodoc]] CLIPSegTextModel
    - forward

## CLIPSegVisionModel

[[autodoc]] CLIPSegVisionModel
    - forward

## CLIPSegForImageSegmentation

[[autodoc]] CLIPSegForImageSegmentation
    - forward

