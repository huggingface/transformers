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
*This model was released on 2022-01-28 and added to Hugging Face Transformers on 2025-02-04 and contributed by [davidhajdu](https://huggingface.co/davidhajdu).*

# DAB-DETR

[DAB-DETR](https://huggingface.co/papers/2201.12329) introduces a novel query formulation using dynamic anchor boxes for DETR. This approach directly employs box coordinates as queries in Transformer decoders, updating them iteratively. By leveraging explicit positional priors and box dimensions, it enhances query-to-feature similarity and accelerates training convergence. This method achieves top performance on the MS-COCO benchmark, reaching 45.7% AP with a ResNet-50-DC5 backbone after 50 epochs. Extensive experiments validate the effectiveness of this design.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="object-detection", model="IDEA-Research/dab-detr-resnet-50", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("IDEA-Research/dab-detr-resnet-50")
model = AutoModelForObjectDetection.from_pretrained("IDEA-Research/dab-detr-resnet-50", dtype="auto")

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

## DabDetrConfig

[[autodoc]] DabDetrConfig

## DabDetrModel

[[autodoc]] DabDetrModel
    - forward

## DabDetrForObjectDetection

[[autodoc]] DabDetrForObjectDetection
    - forward

