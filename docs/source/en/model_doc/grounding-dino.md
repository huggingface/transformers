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
*This model was released on 2023-03-09 and added to Hugging Face Transformers on 2024-04-11 and contributed by [EduardoPacheco](https://huggingface.co/EduardoPacheco) and [nielsr](https://huggingface.co/nielsr).*

# Grounding DINO

[Grounding DINO](https://huggingface.co/papers/2303.05499) extends DINO with grounded pre-training to enable open-set object detection using human inputs like category names or referring expressions. It divides a closed-set detector into three phases and introduces a feature enhancer, language-guided query selection, and a cross-modality decoder for effective fusion. The model excels in zero-shot detection, achieving 52.5 AP on COCO and setting a new record with 26.1 AP on ODinW. It also performs well on referring expression comprehension tasks.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="zero-shot-object-detection", model="IDEA-Research/grounding-dino-tiny", dtype="auto")
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg", candidate_labels=["cat", "couch"])
```

</hfoption>
<hfoption id="AutoModel">

```py
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, infer_device

processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny", dtype="auto")

image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(image_url, stream=True).raw)
text_labels = [["a cat", "a remote control"]]

inputs = processor(images=image, text=text_labels, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)

result = results[0]
for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
    box = [round(x, 2) for x in box.tolist()]
    print(f"Detected {labels} with confidence {round(score.item(), 3)} at location {box}")
```

</hfoption>
</hfoptions>

## Usage tips

- Use [`GroundingDinoProcessor`] to prepare image-text pairs for the model.
- Separate classes in text with periods. For example: `"a cat. a dog."`
- For multiple classes, use [`GroundingDinoProcessor.post_process_grounded_object_detection`] to post-process outputs. Labels from `post_process_object_detection` represent indices from the model dimension where probability exceeds the threshold.

## GroundingDinoImageProcessor

[[autodoc]] GroundingDinoImageProcessor
    - preprocess
    - post_process_object_detection

## GroundingDinoImageProcessorFast

[[autodoc]] GroundingDinoImageProcessorFast
    - preprocess
    - post_process_object_detection

## GroundingDinoProcessor

[[autodoc]] GroundingDinoProcessor
    - post_process_grounded_object_detection

## GroundingDinoConfig

[[autodoc]] GroundingDinoConfig

## GroundingDinoModel

[[autodoc]] GroundingDinoModel
    - forward

## GroundingDinoForObjectDetection

[[autodoc]] GroundingDinoForObjectDetection
    - forward

