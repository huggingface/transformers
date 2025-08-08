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

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Swin Transformer V2

[Swin Transformer V2](https://huggingface.co/papers/2111.09883) is a 3B parameter model that focuses on how to scale a vision model to billions of parameters. It introduces techniques like residual-post-norm combined with cosine attention for improved training stability, log-spaced continuous position bias to better handle varying image resolutions between pre-training and fine-tuning, and a new pre-training method (SimMIM) to reduce the need for large amounts of labeled data. These improvements enable efficiently training very large models (up to 3 billion parameters) capable of processing high-resolution images.

You can find official Swin Transformer V2 checkpoints under the [Microsoft](https://huggingface.co/microsoft?search_models=swinv2) organization.

> [!TIP]
> Click on the Swin Transformer V2 models in the right sidebar for more examples of how to apply Swin Transformer V2 to vision tasks.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="image-classification",
    model="microsoft/swinv2-tiny-patch4-window8-256",
    dtype=torch.float16,
    device=0
)
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
```

</hfoption>

<hfoption id="AutoModel">

```py
import torch
import requests
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained(
    "microsoft/swinv2-tiny-patch4-window8-256",
)
model = AutoModelForImageClassification.from_pretrained(
    "microsoft/swinv2-tiny-patch4-window8-256",
    device_map="auto"
)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = image_processor(image, return_tensors="pt").to(model.device)

with torch.no_grad():
  logits = model(**inputs).logits

predicted_class_id = logits.argmax(dim=-1).item()
predicted_class_label = model.config.id2label[predicted_class_id]
print(f"The predicted class label is: {predicted_class_label}")
```

</hfoption>
</hfoptions>

## Notes

- Swin Transformer V2 can pad the inputs for any input height and width divisible by `32`. 
- Swin Transformer V2 can be used as a [backbone](../backbones). When `output_hidden_states = True`, it outputs both `hidden_states` and `reshaped_hidden_states`. The `reshaped_hidden_states` have a shape of `(batch, num_channels, height, width)` rather than `(batch_size, sequence_length, num_channels)`.

## Swinv2Config

[[autodoc]] Swinv2Config

## Swinv2Model

[[autodoc]] Swinv2Model
    - forward

## Swinv2ForMaskedImageModeling

[[autodoc]] Swinv2ForMaskedImageModeling
    - forward

## Swinv2ForImageClassification

[[autodoc]] transformers.Swinv2ForImageClassification
    - forward
