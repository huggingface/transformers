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
*This model was released on 2022-05-12 and added to Hugging Face Transformers on 2022-07-22 and contributed by [adirik](https://huggingface.co/adirik).*

# OWL-ViT

[OWL-ViT](https://huggingface.co/papers/2205.06230) presents a method for adapting large-scale image-text models to open-vocabulary object detection using a standard Vision Transformer with minimal modifications. It combines contrastive image-text pre-training with end-to-end fine-tuning for detection, demonstrating that scaling both pre-training data and model size consistently improves performance. The approach includes specific adaptation strategies and regularizations to achieve strong results in zero-shot text-conditioned and one-shot image-conditioned detection.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="zero-shot-object-detection", model="google/owlvit-base-patch32", dtype="auto")
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
```

</hfoption>
<hfoption id="OwlViTForObjectDetection">

```py
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, OwlViTForObjectDetection

processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32", dtype="auto")

url = https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
text_labels = [["a photo of a cat", "a photo of a dog"]]
inputs = processor(text=text_labels, images=image, return_tensors="pt")
outputs = model(**inputs)

target_sizes = torch.tensor([(image.height, image.width)])
results = processor.post_process_grounded_object_detection(
    outputs=outputs, target_sizes=target_sizes, threshold=0.1, text_labels=text_labels
)
result = results[0]
boxes, scores, text_labels = result["boxes"], result["scores"], result["text_labels"]
for box, score, text_label in zip(boxes, scores, text_labels):
    box = [round(i, 2) for i in box.tolist()]
    print(f"Detected {text_label} with confidence {round(score.item(), 3)} at location {box}")
```

</hfoption>
</hfoptions>

## OwlViTConfig

[[autodoc]] OwlViTConfig

## OwlViTTextConfig

[[autodoc]] OwlViTTextConfig

## OwlViTVisionConfig

[[autodoc]] OwlViTVisionConfig

## OwlViTImageProcessor

[[autodoc]] OwlViTImageProcessor
    - preprocess
    - post_process_object_detection
    - post_process_image_guided_detection

## OwlViTImageProcessorFast

[[autodoc]] OwlViTImageProcessorFast
    - preprocess
    - post_process_object_detection
    - post_process_image_guided_detection

## OwlViTProcessor

[[autodoc]] OwlViTProcessor
    - __call__
    - post_process_grounded_object_detection
    - post_process_image_guided_detection

## OwlViTModel

[[autodoc]] OwlViTModel
    - forward
    - get_text_features
    - get_image_features

## OwlViTTextModel

[[autodoc]] OwlViTTextModel
    - forward

## OwlViTVisionModel

[[autodoc]] OwlViTVisionModel
    - forward

## OwlViTForObjectDetection

[[autodoc]] OwlViTForObjectDetection
    - forward
    - image_guided_detection

