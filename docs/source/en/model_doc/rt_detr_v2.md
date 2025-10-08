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
*This model was released on 2024-07-24 and added to Hugging Face Transformers on 2025-02-06 and contributed by [jadechoghari](https://huggingface.co/jadechoghari).*

# RT-DETRv2

[RT-DETRv2](https://huggingface.co/papers/2407.17140) refines RT-DETR by implementing selective multi-scale feature extraction through distinct sampling points in deformable attention, enhancing flexibility. It introduces a discrete sampling operator to improve practicality and deployment compatibility. Additionally, RT-DETRv2 optimizes training with dynamic data augmentation and scale-adaptive hyperparameters, maintaining real-time performance.

<hfoptions id="usage">
<hfoption id="Pipeline">

The model is meant to be used on images resized to a size 640x640 with the corresponding ImageProcessor. Reshaping to other sizes will generally degrade performance.

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="object-detection", model=PekingU/rtdetr_v2_r18vd", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
model = AutoModelForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd", dtype="auto")

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

## RTDetrV2Config

[[autodoc]] RTDetrV2Config

## RTDetrV2Model

[[autodoc]] RTDetrV2Model
    - forward
 
## RTDetrV2ForObjectDetection

[[autodoc]] RTDetrV2ForObjectDetection
    - forward

