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
*This model was released on 2022-12-12 and added to Hugging Face Transformers on 2023-06-20 and contributed by [nielsr](https://huggingface.co/nielsr).*

> [!WARNING]
> This model is in maintenance mode only, we don’t accept any new PRs changing its code.
>
> If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2. You can do so by running the following command: pip install -U transformers==4.40.2.

# DETA

[DETA](https://huggingface.co/papers/2212.06137) enhances Deformable DETR by substituting the one-to-one bipartite Hungarian matching loss with one-to-many label assignments, a technique commonly used in traditional detectors with non-maximum suppression (NMS). This change results in a significant improvement of up to 2.5 mAP. The model achieves 50.2 COCO mAP within 12 epochs using a ResNet50 backbone, outperforming both traditional and transformer-based detectors in this setting. The study demonstrates that bipartite matching is not essential for effective detection transformers, attributing their success to the expressive transformer architecture.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="object-detection", model="jozhang97/deta-swin-large", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("jozhang97/deta-swin-large")
model = AutoModelForObjectDetection.from_pretrained("jozhang97/deta-swin-large", dtype="auto")

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

## DetaConfig

[[autodoc]] DetaConfig

## DetaImageProcessor

[[autodoc]] DetaImageProcessor
    - preprocess
    - post_process_object_detection

## DetaModel

[[autodoc]] DetaModel
    - forward

## DetaForObjectDetection

[[autodoc]] DetaForObjectDetection
    - forward

