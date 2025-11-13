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
*This model was released on 2024-01-04 and added to Hugging Face Transformers on 2025-08-01.*

# MM Grounding DINO

[MM Grounding DINO](https://huggingface.co/papers/2401.02361) is an open-source implementation of the Grounding-DINO model, designed for tasks like Open-Vocabulary Detection, Phrase Grounding, and Referring Expression Comprehension. It is built on the MMDetection framework and uses large-scale vision datasets for pre-training, followed by fine-tuning on specialized detection and grounding datasets. The model provides detailed reproducible settings and analysis, and experiments show that MM-Grounding-DINO-Tiny surpasses the original Grounding-DINO-Tiny in performance. All code and trained models are publicly available for research and downstream applications.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="zero-shot-object-detection", model="openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det", dtype="auto")
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg", candidate_labels=["cat", "couch"])
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from transformers.image_utils import load_image

processor = AutoProcessor.from_pretrained("openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det")
model = AutoModelForZeroShotObjectDetection.from_pretrained("openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det", dtype="auto")

image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = load_image(image_url)
text_labels = [["a cat", "a remote control"]]
inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    threshold=0.4,
    target_sizes=[(image.height, image.width)]
)

result = results[0]
for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
    box = [round(x, 2) for x in box.tolist()]
    print(f"Detected {labels} with confidence {round(score.item(), 3)} at location {box}")
```

</hfoption>
</hfoptions>

## Notes

- Here's a table of models and their object detection performance results on COCO (results from [official repo](https://github.com/open-mmlab/mmdetection/blob/main/configs/mm_grounding_dino/README.md)):

    |                                                              Model                                                             | Backbone |      Pre-Train Data      |   Style   |  COCO mAP  |
    | ------------------------------------------------------------------------------------------------------------------------------ | -------- | ------------------------ | --------- | ---------- |
    |  [mm_grounding_dino_tiny_o365v1_goldg](https://huggingface.co/openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg)                       |  Swin-T  |        O365,GoldG        | Zero-shot | 50.4(+2.3) |
    |  [mm_grounding_dino_tiny_o365v1_goldg_grit](https://huggingface.co/openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_grit)             |  Swin-T  |     O365,GoldG,GRIT      | Zero-shot | 50.5(+2.1) |
    |  [mm_grounding_dino_tiny_o365v1_goldg_v3det](https://huggingface.co/openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det)           |  Swin-T  |     O365,GoldG,V3Det     | Zero-shot | 50.6(+2.2) |
    |  [mm_grounding_dino_tiny_o365v1_goldg_grit_v3det](https://huggingface.co/openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_grit_v3det) |  Swin-T  |  O365,GoldG,GRIT,V3Det   | Zero-shot | 50.4(+2.0) |
    |  [mm_grounding_dino_base_o365v1_goldg_v3det](https://huggingface.co/openmmlab-community/mm_grounding_dino_base_o365v1_goldg_v3det)           |  Swin-B  |     O365,GoldG,V3Det     | Zero-shot |    52.5    |
    |  [mm_grounding_dino_base_all](https://huggingface.co/openmmlab-community/mm_grounding_dino_base_all)                                         |  Swin-B  |         O365,ALL         |     -     |    59.5    |
    |  [mm_grounding_dino_large_o365v2_oiv6_goldg](https://huggingface.co/openmmlab-community/mm_grounding_dino_large_o365v2_oiv6_goldg)           |  Swin-L  | O365V2,OpenImageV6,GoldG | Zero-shot |    53.0    |
    |  [mm_grounding_dino_large_all](https://huggingface.co/openmmlab-community/mm_grounding_dino_large_all)                                       |  Swin-L  |  O365V2,OpenImageV6,ALL  |     -     |    60.3    |

- Here's a table of MM Grounding DINO tiny models and their object detection performance on LVIS (results from [official repo](https://github.com/open-mmlab/mmdetection/blob/main/configs/mm_grounding_dino/README.md)):

    |                                                              Model                                                             |    Pre-Train Data     | MiniVal APr | MiniVal APc | MiniVal APf | MiniVal AP  | Val1.0 APr | Val1.0 APc | Val1.0 APf |  Val1.0 AP  |
    | ------------------------------------------------------------------------------------------------------------------------------ | --------------------- | ----------- | ----------- | ----------- | ----------- | ---------- | ---------- | ---------- | ----------- |
    |  [mm_grounding_dino_tiny_o365v1_goldg](https://huggingface.co/openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg)                       |      O365,GoldG       |    28.1     |    30.2     |    42.0     | 35.7(+6.9)  |    17.1    |    22.4    |    36.5    | 27.0(+6.9)  |
    |  [mm_grounding_dino_tiny_o365v1_goldg_grit](https://huggingface.co/openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_grit)             |    O365,GoldG,GRIT    |    26.6     |    32.4     |    41.8     | 36.5(+7.7)  |    17.3    |    22.6    |    36.4    | 27.1(+7.0)  |
    |  [mm_grounding_dino_tiny_o365v1_goldg_v3det](https://huggingface.co/openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det)           |   O365,GoldG,V3Det    |    33.0     |    36.0     |    45.9     | 40.5(+11.7) |    21.5    |    25.5    |    40.2    | 30.6(+10.5) |
    |  [mm_grounding_dino_tiny_o365v1_goldg_grit_v3det](https://huggingface.co/openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_grit_v3det) | O365,GoldG,GRIT,V3Det |    34.2     |    37.4     |    46.2     | 41.4(+12.6) |    23.6    |    27.6    |    40.5    | 31.9(+11.8) |

- This implementation also supports inference for [LLMDet](https://github.com/iSEE-Laboratory/LLMDet). Here's a table of LLMDet models and their performance on LVIS (results from [official repo](https://github.com/iSEE-Laboratory/LLMDet)):

    |                             Model                         | Pre-Train Data            |  MiniVal APr | MiniVal APc | MiniVal APf | MiniVal AP  | Val1.0 APr | Val1.0 APc | Val1.0 APf |  Val1.0 AP  |
    | --------------------------------------------------------- | -------------------------------------------- | ------------ | ----------- | ----------- | ----------- | ---------- | ---------- | ---------- | ----------- |
    | [llmdet_tiny](https://huggingface.co/iSEE-Laboratory/llmdet_tiny)   | (O365,GoldG,GRIT,V3Det) + GroundingCap-1M    | 44.7         | 37.3        | 39.5        | 50.7        | 34.9       | 26.0       | 30.1       | 44.3        |
    | [llmdet_base](https://huggingface.co/iSEE-Laboratory/llmdet_base)   | (O365,GoldG,V3Det) + GroundingCap-1M         | 48.3         | 40.8        | 43.1        | 54.3        | 38.5       | 28.2       | 34.3       | 47.8        |
    | [llmdet_large](https://huggingface.co/iSEE-Laboratory/llmdet_large) | (O365V2,OpenImageV6,GoldG) + GroundingCap-1M | 51.1         | 45.1        | 46.1        | 56.6        | 42.0       | 31.6       | 38.8       | 50.2        |

## MMGroundingDinoConfig

[[autodoc]] MMGroundingDinoConfig

## MMGroundingDinoModel

[[autodoc]] MMGroundingDinoModel
    - forward

## MMGroundingDinoForObjectDetection

[[autodoc]] MMGroundingDinoForObjectDetection
    - forward

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="zero-shot-object-detection", model="openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det", dtype="auto")
pipeline("path/to/image.png", ["a cat", "a dog"])
```
