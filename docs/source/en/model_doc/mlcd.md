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
*This model was released on 2024-07-24 and added to Hugging Face Transformers on 2025-04-15.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# MLCD

[MLCD](https://huggingface.co/papers/2407.17331) is a new approach to improve contrastive image-text pre-training beyond CLIP’s single-label instance discrimination. Instead of assigning one pseudo-label per image, MLCD clusters the large-scale LAION-400M dataset into one million centers and assigns each image multiple nearest clusters as auxiliary labels to capture multi-object semantics. A novel multi-label classification loss is then applied, separating positive and negative class contributions to reduce boundary ambiguity. Experiments show that MLCD consistently improves representation learning and achieves state-of-the-art results in linear probe, zero-shot classification, and image-text retrieval tasks.

<hfoptions id="usage">
<hfoption id="MLCDVisionModel">

```py
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, MLCDVisionModel

model = MLCDVisionModel.from_pretrained("DeepGlint-AI/mlcd-vit-bigG-patch14-448")
processor = AutoProcessor.from_pretrained("DeepGlint-AI/mlcd-vit-bigG-patch14-448", dtype="auto")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

features = outputs.last_hidden_state
print(f"Extracted features shape: {features.shape}")
```

</hfoption>
</hfoptions>

## MLCDVisionConfig

[[autodoc]] MLCDVisionConfig

## MLCDVisionModel

[[autodoc]] MLCDVisionModel
    - forward
