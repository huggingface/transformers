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

## Usage tips

- OWLv2 is a zero-shot text-conditioned object detection model like its predecessor OWL-ViT. It uses CLIP as its multi-modal backbone with a ViT-like Transformer for visual features and a causal language model for text features.
- To use CLIP for detection, OWLv2 removes the final token pooling layer of the vision model and attaches lightweight classification and box heads to each transformer output token. Open-vocabulary classification replaces fixed classification layer weights with class-name embeddings from the text model.
- The authors train CLIP from scratch and fine-tune it end-to-end with classification and box heads on standard detection datasets using bipartite matching loss. Use one or multiple text queries per image for zero-shot text-conditioned object detection.
- Use [`Owlv2ImageProcessor`] to resize and normalize images for the model. Use [`CLIPTokenizer`] to encode text. [`Owlv2Processor`] wraps both into a single instance to encode text and prepare images.
- OWLv2's architecture is identical to OWL-ViT, but the object detection head includes an objectness classifier. This predicts the query-agnostic likelihood that a predicted box contains an object (versus background). Use the objectness score to rank or filter predictions independently of text queries.
- OWLv2 usage is identical to OWL-ViT with the new [`Owlv2ImageProcessor`].

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

