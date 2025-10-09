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
*This model was released on 2024-03-11 and added to Hugging Face Transformers on 2024-09-25 and contributed by [yonigozlan](https://huggingface.co/yonigozlan).*

# OmDet-Turbo

[OmDet-Turbo](https://huggingface.co/papers/2403.06892) is a transformer-based real-time open-vocabulary object detection model that integrates an Efficient Fusion Head (EFH) module to enhance performance and speed. It achieves up to 100.2 FPS with TensorRT and language cache techniques, and demonstrates high accuracy with an AP of 53.4 on COCO zero-shot. OmDet-Turbo sets new benchmarks on ODinW and OVDEval, showcasing its potential for industrial applications.

<hfoptions id="usage">
<hfoption id="OmDetTurboForObjectDetection">

```py
import torch
import requests
from PIL import Image

from transformers import AutoProcessor, OmDetTurboForObjectDetection

processor = AutoProcessor.from_pretrained("omlab/omdet-turbo-swin-tiny-hf")
model = OmDetTurboForObjectDetection.from_pretrained("omlab/omdet-turbo-swin-tiny-hf", dtype="auto")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
text_labels = ["cat", "remote"]
inputs = processor(image, text=text_labels, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    target_sizes=[(image.height, image.width)],
    text_labels=text_labels,
    threshold=0.3,
    nms_threshold=0.3,
)
result = results[0]
boxes, scores, text_labels = result["boxes"], result["scores"], result["text_labels"]
for box, score, text_label in zip(boxes, scores, text_labels):
    box = [round(i, 2) for i in box.tolist()]
    print(f"Detected {text_label} with confidence {round(score.item(), 3)} at location {box}")
```

</hfoption>
</hfoptions>

## OmDetTurboConfig

[[autodoc]] OmDetTurboConfig

## OmDetTurboProcessor

[[autodoc]] OmDetTurboProcessor
    - post_process_grounded_object_detection

## OmDetTurboForObjectDetection

[[autodoc]] OmDetTurboForObjectDetection
    - forward

