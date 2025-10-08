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
*This model was released on 2023-06-26 and added to Hugging Face Transformers on 2023-10-30 and contributed by [ydshieh](https://huggingface.co/ydshieh).*

# KOSMOS-2

[Kosmos-2](https://huggingface.co/papers/2306.14824) is a Transformer-based causal language model designed to perceive object descriptions and ground text to the visual world. It represents referential expressions as Markdown-style links, connecting object descriptions to bounding boxes through location tokens. Trained on a large-scale dataset of grounded image-text pairs (GrIT), Kosmos-2 enhances multimodal capabilities by integrating grounding into various applications. The model is evaluated on tasks such as multimodal grounding, referring expression generation, perception-language tasks, and language understanding and generation. This research contributes to the development of Embodiment AI and the convergence of language, multimodal perception, action, and world modeling.

<hfoptions id="usage">
<hfoption id="Kosmos2ForConditionalGeneration">

```py
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, Kosmos2ForConditionalGeneration

model = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224", dtype="auto")
processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

prompt = "<grounding> An image of"

inputs = processor(text=prompt, images=image, return_tensors="pt")

generated_ids = model.generate(
    pixel_values=inputs["pixel_values"],
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    image_embeds=None,
    image_embeds_position_mask=inputs["image_embeds_position_mask"],
    use_cache=True,
    max_new_tokens=64,
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(processor.post_process_generation(generated_text, cleanup_and_extract=False))
```

</hfoption>
</hfoptions>

## Kosmos2Config

[[autodoc]] Kosmos2Config

## Kosmos2ImageProcessor

## Kosmos2Processor

[[autodoc]] Kosmos2Processor
    - __call__

## Kosmos2Model

[[autodoc]] Kosmos2Model
    - forward

## Kosmos2ForConditionalGeneration

[[autodoc]] Kosmos2ForConditionalGeneration
    - forward

