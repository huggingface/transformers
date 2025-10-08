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

*This model was released on 2024-10-17 and added to Hugging Face Transformers on 2025-04-29 and contributed by [VladOS95-cyber](https://github.com/VladOS95-cyber).*

# D-FINE

[D-FINE](https://huggingface.co/papers/2410.13842) redefines bounding box regression in DETR models through Fine-grained Distribution Refinement (FDR) and Global Optimal Localization Self-Distillation (GO-LSD). FDR iteratively refines probability distributions for enhanced localization accuracy, while GO-LSD optimizes localization knowledge transfer and simplifies residual predictions. D-FINE includes lightweight optimizations for speed and accuracy, achieving 54.0% / 55.8% AP on COCO at 124 / 78 FPS on an NVIDIA T4 GPU. Pretrained on Objects365, D-FINE-L / X reaches 57.1% / 59.3% AP, outperforming existing real-time detectors. The method improves various DETR models by up to 5.3% AP with minimal additional parameters and training costs.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="object-detection", model="ustc-community/dfine-xlarge-coco", dtype="auto")
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("ustc-community/dfine-xlarge-coco")
model = AutoModelForObjectDetection.from_pretrained("ustc-community/dfine-xlarge-coco", dtype="auto")

inputs = image_processor(images=image, return_tensors="pt")
outputs = model(**inputs)
target_sizes = torch.tensor([image.size[::-1]])
results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[
    0
]
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )
```

</hfoption>
</hfoptions>

## DFineConfig

[[autodoc]] DFineConfig

## DFineModel

[[autodoc]] DFineModel
    - forward

## DFineForObjectDetection

[[autodoc]] DFineForObjectDetection
    - forward

