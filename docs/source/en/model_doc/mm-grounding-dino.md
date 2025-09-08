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

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
           <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# MM Grounding DINO

[MM Grounding DINO](https://huggingface.co/papers/2401.02361) model was proposed in [An Open and Comprehensive Pipeline for Unified Object Grounding and Detection](https://huggingface.co/papers/2401.02361) by Xiangyu Zhao, Yicheng Chen, Shilin Xu, Xiangtai Li, Xinjiang Wang, Yining Li, Haian Huang>.

MM Grounding DINO improves upon the [Grounding DINO](https://huggingface.co/docs/transformers/model_doc/grounding-dino) by improving the contrastive class head and removing the parameter sharing in the decoder, improving zero-shot detection performance on both COCO (50.6(+2.2) AP) and LVIS (31.9(+11.8) val AP and 41.4(+12.6) minival AP).

You can find all the original MM Grounding DINO checkpoints under the [MM Grounding DINO](https://huggingface.co/collections/openmmlab-community/mm-grounding-dino-688cbde05b814c4e2832f9df) collection. This model also supports LLMDet inference. You can find LLMDet checkpoints under the [LLMDet](https://huggingface.co/collections/iSEE-Laboratory/llmdet-688475906dc235d5f1dc678e) collection.

> [!TIP]
> Click on the MM Grounding DINO models in the right sidebar for more examples of how to apply MM Grounding DINO to different MM Grounding DINO tasks.

The example below demonstrates how to generate text based on an image with the [`AutoModelForZeroShotObjectDetection`] class.

<hfoptions id="usage">
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, infer_device
from transformers.image_utils import load_image


# Prepare processor and model
model_id = "openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det"
device = infer_device()
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# Prepare inputs
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(image_url)
text_labels = [["a cat", "a remote control"]]
inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)

# Run inference
with torch.no_grad():
    outputs = model(**inputs)

# Postprocess outputs
results = processor.post_process_grounded_object_detection(
    outputs,
    threshold=0.4,
    target_sizes=[(image.height, image.width)]
)

# Retrieve the first image result
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
