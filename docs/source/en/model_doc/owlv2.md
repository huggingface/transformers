<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2023-06-16 and added to Hugging Face Transformers on 2023-10-13 and contributed by [nielsr](https://huggingface.co/nielsr).*

# OWLv2

[OWLv2](https://huggingface.co/papers/2306.09683) an open-vocabulary object detection model that leverages pretrained vision-language models and self-training on large-scale web image-text pairs. Using the OWL-ST recipe, the model generates pseudo-box annotations from an existing detector, addressing challenges in label space selection, annotation filtering, and training efficiency. OWLv2 already outperforms prior state-of-the-art detectors with ~10M training examples, while OWL-ST scales training to over 1B examples, significantly boosting performance. In particular, it improves AP on LVIS rare classes without human box annotations from 31.2% to 44.6%, demonstrating effective Web-scale open-world localization.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="zero-shot-object-detection", model="google/owlv2-base-patch16-ensemble", dtype="auto")
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg", candidate_labels=["a photo of a cat", "a photo of a dog"])
```

</hfoption>
<hfoption id="Owlv2ForObjectDetection">

```py
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, Owlv2ForObjectDetection

processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble", dtype="auto")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
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

## Owlv2Config

[[autodoc]] Owlv2Config

## Owlv2TextConfig

[[autodoc]] Owlv2TextConfig

## Owlv2VisionConfig

[[autodoc]] Owlv2VisionConfig

## Owlv2ImageProcessor

[[autodoc]] Owlv2ImageProcessor
    - preprocess
    - post_process_object_detection
    - post_process_image_guided_detection

## Owlv2ImageProcessorFast

[[autodoc]] Owlv2ImageProcessorFast
    - preprocess
    - post_process_object_detection
    - post_process_image_guided_detection

## Owlv2Processor

[[autodoc]] Owlv2Processor
    - __call__
    - post_process_grounded_object_detection
    - post_process_image_guided_detection

## Owlv2Model

[[autodoc]] Owlv2Model
    - forward
    - get_text_features
    - get_image_features

## Owlv2TextModel

[[autodoc]] Owlv2TextModel
    - forward

## Owlv2VisionModel

[[autodoc]] Owlv2VisionModel
    - forward

## Owlv2ForObjectDetection

[[autodoc]] Owlv2ForObjectDetection
    - forward
    - image_guided_detection

