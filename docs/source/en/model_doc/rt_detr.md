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
*This model was released on 2023-04-17 and added to Hugging Face Transformers on 2024-06-22 and contributed by [rafaelpadilla](https://huggingface.co/rafaelpadilla) and [SangbumChoi](https://github.com/SangbumChoi).*

# RT-DETR

[RT-DETR](https://huggingface.co/papers/2304.08069) is a real-time end-to-end object detection model that leverages the transformer architecture to achieve high accuracy and speed. It addresses the computational cost and inference delay issues associated with traditional DETR models by eliminating the need for non-maximum suppression (NMS). RT-DETR features an efficient hybrid encoder for multi-scale feature processing and an IoU-aware query selection mechanism to enhance object query initialization. The model supports flexible speed adjustment through varying decoder layers without retraining. RT-DETR-L and RT-DETR-X achieve 53.0% and 54.8% AP on COCO val2017 with 114 FPS and 74 FPS on a T4 GPU, respectively, outperforming YOLO models in both speed and accuracy. RT-DETR-R50 also surpasses DINO-Deformable-DETR-R50 in accuracy and FPS.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="object-detection", model="PekingU/rtdetr_r50vd", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
model = AutoModelForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd", dtype="auto")

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

## Usage tips

- RT-DETR processes images using a pretrained ResNet-D variant backbone. This network extracts features from the final three layers of the architecture.
- A hybrid encoder converts the multi-scale features into a sequential array of image features. A decoder with auxiliary prediction heads refines the object queries.
- This process directly generates bounding boxes, eliminating the need for additional post-processing to acquire logits and coordinates.
- Use images resized to 640x640 with the corresponding [`RTDetrImageProcessor`]. Reshaping to other sizes generally degrades performance.

## RTDetrConfig

[[autodoc]] RTDetrConfig

## RTDetrResNetConfig

[[autodoc]] RTDetrResNetConfig

## RTDetrImageProcessor

[[autodoc]] RTDetrImageProcessor
    - preprocess
    - post_process_object_detection

## RTDetrImageProcessorFast

[[autodoc]] RTDetrImageProcessorFast
    - preprocess
    - post_process_object_detection

## RTDetrModel

[[autodoc]] RTDetrModel
    - forward

## RTDetrForObjectDetection

[[autodoc]] RTDetrForObjectDetection
    - forward

## RTDetrResNetBackbone

[[autodoc]] RTDetrResNetBackbone
    - forward

