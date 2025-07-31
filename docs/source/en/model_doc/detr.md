<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

<div style="float: right;">
	<div class="flex flex-wrap space-x-1">
		<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
	</div>
</div>

# DETR

[DETR](https://huggingface.co/papers/2005.12872) consists of a convolutional backbone followed by an encoder-decoder Transformer which can be trained end-to-end for object detection. It greatly simplifies a lot of the complexity of models like Faster-R-CNN and Mask-R-CNN, which use things like region proposals, non-maximum suppression procedure and anchor generation. Moreover, DETR can also be naturally extended to perform panoptic segmentation, by simply adding a mask head on top of the decoder outputs.

> [!TIP]
> This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found [here](https://github.com/facebookresearch/detr).
>
> Click on the DETR models in the right sidebar for more examples of how to apply DETR to different object detection and segmentation tasks.

The example below demonstrates how to perform Object detection with the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="AutoModel">

```python
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import requests
import torch

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# prepare image for the model
inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=0.3)

for result in results:
    for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
        score, label = score.item(), label_id.item()
        box = [round(i, 2) for i in box.tolist()]
        print(f"{model.config.id2label[label]}: {score:.2f} {box}")
```

</hfoption>
</hfoptions>

There are three other ways to instantiate a DETR model (depending on what you prefer):

Option 1: Instantiate DETR with pre-trained weights for entire model
```python
from transformers import DetrForObjectDetection

model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
```

Option 2: Instantiate DETR with randomly initialized weights for Transformer, but pre-trained weights for backbone
```python
from transformers import DetrConfig, DetrForObjectDetection

config = DetrConfig()
model = DetrForObjectDetection(config)
```

Option 3: Instantiate DETR with randomly initialized weights for backbone + Transformer
```python
config = DetrConfig(use_pretrained_backbone=False)
model = DetrForObjectDetection(config)
```

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview.md) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes.md) to quantize the weights to 4-bits.

```python
from transformers import AutoImageProcessor, AutoModelForObjectDetection, BitsAndBytesConfig
from PIL import Image
import requests
import torch

# Download image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# Enable 4-bit quantization with bnb
bnb_config = BitsAndBytesConfig(load_in_4bit=True)

# Load image processor and quantized model with automatic device placement
image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = AutoModelForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    quantization_config=bnb_config,
    device_map="auto"
)

# Detect the same device where the model's first parameter is
device = next(model.parameters()).device

# Prepare image input and move to same device, convert to fp16 to match quantized model
inputs = image_processor(images=image, return_tensors="pt").to(device)
inputs = {k: v.half() for k, v in inputs.items()}

# Inference
with torch.no_grad():
    outputs = model(**inputs)

# Post-process outputs to get bounding boxes
results = image_processor.post_process_object_detection(
    outputs,
    target_sizes=torch.tensor([image.size[::-1]], device=device),
    threshold=0.3
)

# Print detection results
for result in results:
    for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
        score, label = score.item(), label_id.item()
        box = [round(i, 2) for i in box.tolist()]
        print(f"{model.config.id2label[label]}: {score:.2f} {box}")

```

## Notes

| Task | Object detection | Instance segmentation | Panoptic segmentation |
|------|------------------|-----------------------|-----------------------|
| **Description** | Predicting bounding boxes and class labels around objects in an image | Predicting masks around objects (i.e. instances) in an image | Predicting masks around both objects (i.e. instances) as well as "stuff" (i.e. background things like trees and roads) in an image |
| **Model** | [`~transformers.DetrForObjectDetection`] | [`~transformers.DetrForSegmentation`] | [`~transformers.DetrForSegmentation`] |
| **Example dataset** | COCO detection | COCO detection, COCO panoptic | COCO panoptic  |                                                                        |
| **Format of annotations to provide to**  [`~transformers.DetrImageProcessor`] | {'image_id': `int`, 'annotations': `list[Dict]`} each Dict being a COCO object annotation  | {'image_id': `int`, 'annotations': `list[Dict]`}  (in case of COCO detection) or {'file_name': `str`, 'image_id': `int`, 'segments_info': `list[Dict]`} (in case of COCO panoptic) | {'file_name': `str`, 'image_id': `int`, 'segments_info': `list[Dict]`} and masks_path (path to directory containing PNG files of the masks) |
| **Postprocessing** (i.e. converting the output of the model to Pascal VOC format) | [`~transformers.DetrImageProcessor.post_process`] | [`~transformers.DetrImageProcessor.post_process_segmentation`] | [`~transformers.DetrImageProcessor.post_process_segmentation`], [`~transformers.DetrImageProcessor.post_process_panoptic`] |
| **evaluators** | `CocoEvaluator` with `iou_types="bbox"` | `CocoEvaluator` with `iou_types="bbox"` or `"segm"` | `CocoEvaluator` with `iou_tupes="bbox"` or `"segm"`, `PanopticEvaluator` |

In short, one should prepare the data either in COCO detection or COCO panoptic format, then use
[`~transformers.DetrImageProcessor`] to create `pixel_values`, `pixel_mask` and optional
`labels`, which can then be used to train (or fine-tune) a model. For evaluation, one should first convert the
outputs of the model using one of the postprocessing methods of [`~transformers.DetrImageProcessor`]. These can
be provided to either `CocoEvaluator` or `PanopticEvaluator`, which allow you to calculate metrics like
mean Average Precision (mAP) and Panoptic Quality (PQ). The latter objects are implemented in the [original repository](https://github.com/facebookresearch/detr). See the [example notebooks](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DETR) for more info regarding evaluation.

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with DETR.

<PipelineTag pipeline="object-detection"/>

- All example notebooks illustrating fine-tuning [`DetrForObjectDetection`] and [`DetrForSegmentation`] on a custom dataset can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DETR).
- Scripts for finetuning [`DetrForObjectDetection`] with [`Trainer`] or [Accelerate](https://huggingface.co/docs/accelerate/index) can be found [here](https://github.com/huggingface/transformers/tree/main/examples/pytorch/object-detection).
- See also: [Object detection task guide](../tasks/object_detection).

## DetrConfig

[[autodoc]] DetrConfig

## DetrImageProcessor

[[autodoc]] DetrImageProcessor
    - preprocess
    - post_process_object_detection
    - post_process_semantic_segmentation
    - post_process_instance_segmentation
    - post_process_panoptic_segmentation

## DetrImageProcessorFast

[[autodoc]] DetrImageProcessorFast
    - preprocess
    - post_process_object_detection
    - post_process_semantic_segmentation
    - post_process_instance_segmentation
    - post_process_panoptic_segmentation

## DetrFeatureExtractor

[[autodoc]] DetrFeatureExtractor
    - __call__
    - post_process_object_detection
    - post_process_semantic_segmentation
    - post_process_instance_segmentation
    - post_process_panoptic_segmentation

## DETR specific outputs

[[autodoc]] models.detr.modeling_detr.DetrModelOutput

[[autodoc]] models.detr.modeling_detr.DetrObjectDetectionOutput

[[autodoc]] models.detr.modeling_detr.DetrSegmentationOutput

## DetrModel

[[autodoc]] DetrModel
    - forward

## DetrForObjectDetection

[[autodoc]] DetrForObjectDetection
    - forward

## DetrForSegmentation

[[autodoc]] DetrForSegmentation
    - forward
