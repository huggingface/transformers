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

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# I-JEPA

[I-JEPA](https://huggingface.co/papers/2301.08243) is a self-supervised learning method that learns semantic image representations by predicting parts of an image from other parts of the image. this works on comparing the abstract representations of the image (rather than pixel level comparisons), which avoids the typical pitfalls of data augmentation bias and pixel-level details that don't capture semantic meaning. You can find the original I-JEPA checkpoints under the [facebook](https://huggingface.co/facebook) organization , includingthe variants based on vairants of vit architecture like [facebook/ijepa_vith14_1k](https://huggingface.co/facebook/ijepa_vith14_1k), [facebook/ijepa_vitg16_22k](https://huggingface.co/facebook/facebook/ijepa_vitg16_22k) etc.

> [!TIP]
> This model was contributed by [jmtzt](https://huggingface.co/jmtzt), .
>
> Click on the I-JEPA models in the right sidebar for more examples of how to apply I-JEPA to different image representation and classification tasks.

The example below demonstrates how to extract image features with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

feature_extractor = pipeline(
    task="feature-extraction",
    model="facebook/ijepa_vith14_1k",
    device=0,
    torch_dtype=torch.bfloat16
)

features = feature_extractor("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
print(f"Feature shape: {len(features[0])}")
```

</hfoption>
<hfoption id="AutoModel">

```py
import requests
import torch
from PIL import Image
from torch.nn.functional import cosine_similarity

from transformers import AutoModel, AutoProcessor

url_1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
url_2 = "http://images.cocodataset.org/val2017/000000219578.jpg"
image_1 = Image.open(requests.get(url_1, stream=True).raw)
image_2 = Image.open(requests.get(url_2, stream=True).raw)

model_id = "facebook/ijepa_vith14_1k"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

@torch.no_grad()
def infer(image):
    inputs = processor(image, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

embed_1 = infer(image_1)
embed_2 = infer(image_2)

similarity = cosine_similarity(embed_1, embed_2)
print(f"Similarity: {similarity.item():.4f}")
```

</hfoption>
</hfoptions>

## Notes

- I-JEPA is designed specifically for feature extraction and doesn't generate pixel-level predictions like traditional autoencoders.
- The model uses a masking strategy that samples target blocks with large semantic scale while using spatially distributed context blocks.
- I-JEPA works best with Vision Transformers and scales efficiently to large model sizes.

    ```py
    # Example of using I-JEPA for downstream classification task
    from transformers import IJepaForImageClassification, AutoProcessor

    model = IJepaForImageClassification.from_pretrained("facebook/ijepa_vith14_1k")
    processor = AutoProcessor.from_pretrained("facebook/ijepa_vith14_1k")
    
    inputs = processor(images, return_tensors="pt")
    outputs = model(**inputs)
    ```

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with I-JEPA.

<PipelineTag pipeline="image-classification"/>

- [`IJepaForImageClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
- See also: [Image classification task guide](../tasks/image_classification)

## IJepaConfig

[[autodoc]] IJepaConfig

## IJepaModel

[[autodoc]] IJepaModel
    - forward

## IJepaForImageClassification

[[autodoc]] IJepaForImageClassification
    - forward