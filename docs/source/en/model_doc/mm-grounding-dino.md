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

# MM Grounding DINO

## Overview

The MM Grounding DINO model was proposed in [An Open and Comprehensive Pipeline for Unified Object Grounding and Detection](https://arxiv.org/abs/2401.02361) by Xiangyu Zhao, Yicheng Chen, Shilin Xu, Xiangtai Li, Xinjiang Wang, Yining Li, Haian Huang>.
MM Grounding DINO improves upon the Grounding DINO by improving the contrastive class head and removing the parameter sharing in the decoder, improving zero-shot detection performance on both COCO (50.6(+2.2) AP) and LVIS (31.9(+11.8) val AP and 41.4(+12.6) minival AP).

The abstract from the paper is the following:

*Grounding-DINO is a state-of-the-art open-set detection model that tackles multiple vision tasks including Open-Vocabulary Detection (OVD), Phrase Grounding (PG), and Referring Expression Comprehension (REC). Its effectiveness has led to its widespread adoption as a mainstream architecture for various downstream applications. However, despite its significance, the original Grounding-DINO model lacks comprehensive public technical details due to the unavailability of its training code. To bridge this gap, we present MM-Grounding-DINO, an open-source, comprehensive, and user-friendly baseline, which is built with the MMDetection toolbox. It adopts abundant vision datasets for pre-training and various detection and grounding datasets for fine-tuning. We give a comprehensive analysis of each reported result and detailed settings for reproduction. The extensive experiments on the benchmarks mentioned demonstrate that our MM-Grounding-DINO-Tiny outperforms the Grounding-DINO-Tiny baseline. We release all our models to the research community. Codes and trained models are released at [this https URL](https://github.com/open-mmlab/mmdetection/tree/main/configs/mm_grounding_dino).*

Tips:

Here's a table of models and their object detection performance results on COCO (results from [official repo](https://github.com/open-mmlab/mmdetection/blob/main/configs/mm_grounding_dino/README.md)):

|                                                              Model                                                             | Backbone |      Pre-Train Data      |   Style   |  COCO mAP  |
| ------------------------------------------------------------------------------------------------------------------------------ | -------- | ------------------------ | --------- | ---------- |
|  [mm_grounding_dino_tiny_o365v1_goldg](https://huggingface.co/rziga/mm_grounding_dino_tiny_o365v1_goldg)                       |  Swin-T  |        O365,GoldG        | Zero-shot | 50.4(+2.3) |
|  [mm_grounding_dino_tiny_o365v1_goldg_grit](https://huggingface.co/rziga/mm_grounding_dino_tiny_o365v1_goldg_grit)             |  Swin-T  |     O365,GoldG,GRIT      | Zero-shot | 50.5(+2.1) |
|  [mm_grounding_dino_tiny_o365v1_goldg_v3det](https://huggingface.co/rziga/mm_grounding_dino_tiny_o365v1_goldg_v3det)           |  Swin-T  |     O365,GoldG,V3Det     | Zero-shot | 50.6(+2.2) |
|  [mm_grounding_dino_tiny_o365v1_goldg_grit_v3det](https://huggingface.co/rziga/mm_grounding_dino_tiny_o365v1_goldg_grit_v3det) |  Swin-T  |  O365,GoldG,GRIT,V3Det   | Zero-shot | 50.4(+2.0) |
|  [mm_grounding_dino_base_o365v1_goldg_v3det](https://huggingface.co/rziga/mm_grounding_dino_base_o365v1_goldg_v3det)           |  Swin-B  |     O365,GoldG,V3Det     | Zero-shot |    52.5    |
|  [mm_grounding_dino_base_all](https://huggingface.co/rziga/mm_grounding_dino_base_all)                                         |  Swin-B  |         O365,ALL         |     -     |    59.5    |
|  [mm_grounding_dino_large_o365v2_oiv6_goldg](https://huggingface.co/rziga/mm_grounding_dino_large_o365v2_oiv6_goldg)           |  Swin-L  | O365V2,OpenImageV6,GoldG | Zero-shot |    53.0    |
|  [mm_grounding_dino_large_all](https://huggingface.co/rziga/mm_grounding_dino_large_all)                                       |  Swin-L  |  O365V2,OpenImageV6,ALL  |     -     |    60.3    |


Here's a table of MM Grounding DINO tiny models and their object detection performance on LVIS (results from [official repo](https://github.com/open-mmlab/mmdetection/blob/main/configs/mm_grounding_dino/README.md)):

|                                                              Model                                                             |    Pre-Train Data     | MiniVal APr | MiniVal APc | MiniVal APf | MiniVal AP  | Val1.0 APr | Val1.0 APc | Val1.0 APf |  Val1.0 AP  |
| ------------------------------------------------------------------------------------------------------------------------------ | --------------------- | ----------- | ----------- | ----------- | ----------- | ---------- | ---------- | ---------- | ----------- |
|  [mm_grounding_dino_tiny_o365v1_goldg](https://huggingface.co/rziga/mm_grounding_dino_tiny_o365v1_goldg)                       |      O365,GoldG       |    28.1     |    30.2     |    42.0     | 35.7(+6.9)  |    17.1    |    22.4    |    36.5    | 27.0(+6.9)  |
|  [mm_grounding_dino_tiny_o365v1_goldg_grit](https://huggingface.co/rziga/mm_grounding_dino_tiny_o365v1_goldg_grit)             |    O365,GoldG,GRIT    |    26.6     |    32.4     |    41.8     | 36.5(+7.7)  |    17.3    |    22.6    |    36.4    | 27.1(+7.0)  |
|  [mm_grounding_dino_tiny_o365v1_goldg_v3det](https://huggingface.co/rziga/mm_grounding_dino_tiny_o365v1_goldg_v3det)           |   O365,GoldG,V3Det    |    33.0     |    36.0     |    45.9     | 40.5(+11.7) |    21.5    |    25.5    |    40.2    | 30.6(+10.5) |
|  [mm_grounding_dino_tiny_o365v1_goldg_grit_v3det](https://huggingface.co/rziga/mm_grounding_dino_tiny_o365v1_goldg_grit_v3det) | O365,GoldG,GRIT,V3Det |    34.2     |    37.4     |    46.2     | 41.4(+12.6) |    23.6    |    27.6    |    40.5    | 31.9(+11.8) |


This implementation also supports inference for [LLMDet](https://github.com/iSEE-Laboratory/LLMDet). Here's a table of LLMDet models and their performance on LVIS (results from [official repo](https://github.com/iSEE-Laboratory/LLMDet)):

|                             Model                         | Pre-Train Data            |  MiniVal APr | MiniVal APc | MiniVal APf | MiniVal AP  | Val1.0 APr | Val1.0 APc | Val1.0 APf |  Val1.0 AP  |
| --------------------------------------------------------- | -------------------------------------------- | ------------ | ----------- | ----------- | ----------- | ---------- | ---------- | ---------- | ----------- |
| [llmdet_tiny](https://huggingface.co/rziga/llmdet_tiny)   | (O365,GoldG,GRIT,V3Det) + GroundingCap-1M    | 44.7         | 37.3        | 39.5        | 50.7        | 34.9       | 26.0       | 30.1       | 44.3        |
| [llmdet_base](https://huggingface.co/rziga/llmdet_base)   | (O365,GoldG,V3Det) + GroundingCap-1M         | 48.3         | 40.8        | 43.1        | 54.3        | 38.5       | 28.2       | 34.3       | 47.8        |
| [llmdet_large](https://huggingface.co/rziga/llmdet_large) | (O365V2,OpenImageV6,GoldG) + GroundingCap-1M | 51.1         | 45.1        | 46.1        | 56.6        | 42.0       | 31.6       | 38.8       | 50.2        |


Here's how to use the model for zero-shot object detection:

```python
>>> import requests

>>> import torch
>>> from PIL import Image
>>> from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

>>> model_id = "rziga/mm_grounding_dino_tiny_o365v1_goldg_v3det"
>>> device = "cuda"

>>> processor = AutoProcessor.from_pretrained(model_id)
>>> model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

>>> image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(image_url, stream=True).raw)
>>> # Check for cats and remote controls
>>> text_labels = [["a cat", "a remote control"]]

>>> inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)
>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> results = processor.post_process_grounded_object_detection(
...     outputs,
...     inputs.input_ids,
...     box_threshold=0.4,
...     text_threshold=0.3,
...     target_sizes=[image.size[::-1]]
... )

# Retrieve the first image result
>>> result = results[0]
>>> for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
...     box = [round(x, 2) for x in box.tolist()]
...     print(f"Detected {labels} with confidence {round(score.item(), 3)} at location {box}")
Detected a cat with confidence 0.54 at location [343.29, 23.97, 637.8, 373.7]
Detected a remote control with confidence 0.469 at location [37.98, 70.24, 177.0, 118.54]
Detected a cat with confidence 0.49 at location [9.68, 53.02, 316.85, 474.29]
Detected a remote control with confidence 0.405 at location [331.39, 73.93, 372.08, 187.65]
```

This model was contributed by [rziga](https://huggingface.co/rziga) based on original huggingface implementation by [EduardoPacheco](https://huggingface.co/EduardoPacheco) and [nielsr](https://huggingface.co/nielsr).
The original code can be found [here](https://github.com/open-mmlab/mmdetection).


## MMGroundingDinoConfig

[[autodoc]] MMGroundingDinoConfig

## MMGroundingDinoModel

[[autodoc]] MMGroundingDinoModel
    - forward

## MMGroundingDinoForObjectDetection

[[autodoc]] MMGroundingDinoForObjectDetection
    - forward
