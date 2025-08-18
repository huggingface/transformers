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
*This model was released on 2020-05-26 and added to Hugging Face Transformers on 2021-06-09.*

<div style="float: right;">
	<div class="flex flex-wrap space-x-1">
		<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
	</div>
</div>

# DETR

[DETR](https://huggingface.co/papers/2005.12872) consists of a convolutional backbone followed by an encoder-decoder Transformer which can be trained end-to-end for object detection. It greatly simplifies a lot of the complexity of models like Faster-R-CNN and Mask-R-CNN, which use things like region proposals, non-maximum suppression procedure and anchor generation. Moreover, DETR can also be naturally extended to perform panoptic segmentation, by simply adding a mask head on top of the decoder outputs.

You can find all the original DETR checkpoints under the [AI at Meta](https://huggingface.co/facebook/models?search=detr) organization.

> [!TIP]
> This model was contributed by [nielsr](https://huggingface.co/nielsr).
>
> Click on the DETR models in the right sidebar for more examples of how to apply DETR to different object detection and segmentation tasks.

The example below demonstrates how to perform object detection with the [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline
import torch

pipeline = pipeline(
    "object-detection", 
    model="facebook/detr-resnet-50",
    torch_dtype=torch.float16,
    device_map=0
)

pipeline("http://images.cocodataset.org/val2017/000000039769.jpg")
```

</hfoption>
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

<details>
<summary>How DETR works</summary>

Here's a TLDR explaining how [`~transformers.DetrForObjectDetection`] works:

First, an image is sent through a pre-trained convolutional backbone (in the paper, the authors use ResNet-50/ResNet-101). Let's assume we also add a batch dimension. This means that the input to the backbone is a tensor of shape `(batch_size, 3, height, width)`, assuming the image has 3 color channels (RGB). The CNN backbone outputs a new lower-resolution feature map, typically of shape `(batch_size, 2048, height/32, width/32)`. This is then projected to match the hidden dimension of the Transformer of DETR, which is `256` by default, using a `nn.Conv2D` layer. So now, we have a tensor of shape `(batch_size, 256, height/32, width/32).` Next, the feature map is flattened and transposed to obtain a tensor of shape `(batch_size, seq_len, d_model)` = `(batch_size, width/32*height/32, 256)`. So a difference with NLP models is that the sequence length is actually longer than usual, but with a smaller `d_model` (which in NLP is typically 768 or higher).

Next, this is sent through the encoder, outputting `encoder_hidden_states` of the same shape (you can consider these as image features). Next, so-called **object queries** are sent through the decoder. This is a tensor of shape `(batch_size, num_queries, d_model)`, with `num_queries` typically set to 100 and initialized with zeros. These input embeddings are learnt positional encodings that the authors refer to as object queries, and similarly to the encoder, they are added to the input of each attention layer. Each object query will look for a particular object in the image. The decoder updates these embeddings through multiple self-attention and encoder-decoder attention layers to output `decoder_hidden_states` of the same shape: `(batch_size, num_queries, d_model)`. Next, two heads are added on top for object detection: a linear layer for classifying each object query into one of the objects or "no object", and a MLP to predict bounding boxes for each query.

The model is trained using a **bipartite matching loss**: so what we actually do is compare the predicted classes + bounding boxes of each of the N = 100 object queries to the ground truth annotations, padded up to the same length N (so if an image only contains 4 objects, 96 annotations will just have a "no object" as class and "no bounding box" as bounding box). The [Hungarian matching algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm) is used to find an optimal one-to-one mapping of each of the N queries to each of the N annotations. Next, standard cross-entropy (for the classes) and a linear combination of the L1 and [generalized IoU loss](https://giou.stanford.edu/) (for the bounding boxes) are used to optimize the parameters of the model.

DETR can be naturally extended to perform panoptic segmentation (which unifies semantic segmentation and instance segmentation). [`~transformers.DetrForSegmentation`] adds a segmentation mask head on top of [`~transformers.DetrForObjectDetection`]. The mask head can be trained either jointly, or in a two steps process, where one first trains a [`~transformers.DetrForObjectDetection`] model to detect bounding boxes around both "things" (instances) and "stuff" (background things like trees, roads, sky), then freeze all the weights and train only the mask head for 25 epochs. Experimentally, these two approaches give similar results. Note that predicting boxes is required for the training to be possible, since the Hungarian matching is computed using distances between boxes.

</details>

## Notes

- DETR uses so-called **object queries** to detect objects in an image. The number of queries determines the maximum number of objects that can be detected in a single image, and is set to 100 by default (see parameter `num_queries` of [`~transformers.DetrConfig`]). Note that it's good to have some slack (in COCO, the authors used 100, while the maximum number of objects in a COCO image is ~70).
- The decoder of DETR updates the query embeddings in parallel. This is different from language models like GPT-2, which use autoregressive decoding instead of parallel. Hence, no causal attention mask is used.
- DETR adds position embeddings to the hidden states at each self-attention and cross-attention layer before projecting to queries and keys. For the position embeddings of the image, one can choose between fixed sinusoidal or learned absolute position embeddings. By default, the parameter `position_embedding_type` of [`~transformers.DetrConfig`] is set to `"sine"`.
- During training, the authors of DETR did find it helpful to use auxiliary losses in the decoder, especially to help the model output the correct number of objects of each class. If you set the parameter `auxiliary_loss` of [`~transformers.DetrConfig`] to `True`, then prediction feedforward neural networks and Hungarian losses are added after each decoder layer (with the FFNs sharing parameters).
- If you want to train the model in a distributed environment across multiple nodes, then one should update the _num_boxes_ variable in the _DetrLoss_ class of _modeling_detr.py_. When training on multiple nodes, this should be set to the average number of target boxes across all nodes, as can be seen in the original implementation [here](https://github.com/facebookresearch/detr/blob/a54b77800eb8e64e3ad0d8237789fcbf2f8350c5/models/detr.py#L227-L232).
- [`~transformers.DetrForObjectDetection`] and [`~transformers.DetrForSegmentation`] can be initialized with any convolutional backbone available in the [timm library](https://github.com/rwightman/pytorch-image-models). Initializing with a MobileNet backbone for example can be done by setting the `backbone` attribute of [`~transformers.DetrConfig`] to `"tf_mobilenetv3_small_075"`, and then initializing the model with that config.
- DETR resizes the input images such that the shortest side is at least a certain amount of pixels while the longest is at most 1333 pixels. At training time, scale augmentation is used such that the shortest side is randomly set to at least 480 and at most 800 pixels. At inference time, the shortest side is set to 800. One can use [`~transformers.DetrImageProcessor`] to prepare images (and optional annotations in COCO format) for the model. Due to this resizing, images in a batch can have different sizes. DETR solves this by padding images up to the largest size in a batch, and by creating a pixel mask that indicates which pixels are real/which are padding. Alternatively, one can also define a custom `collate_fn` in order to batch images together, using [`~transformers.DetrImageProcessor.pad_and_create_pixel_mask`].
- The size of the images will determine the amount of memory being used, and will thus determine the `batch_size`. It is advised to use a batch size of 2 per GPU. See [this Github thread](https://github.com/facebookresearch/detr/issues/150) for more info.

There are three other ways to instantiate a DETR model (depending on what you prefer):

- Option 1: Instantiate DETR with pre-trained weights for entire model
```python
from transformers import DetrForObjectDetection

model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
```

- Option 2: Instantiate DETR with randomly initialized weights for Transformer, but pre-trained weights for backbone
```python
from transformers import DetrConfig, DetrForObjectDetection

config = DetrConfig()
model = DetrForObjectDetection(config)
```

- Option 3: Instantiate DETR with randomly initialized weights for backbone + Transformer
```python
config = DetrConfig(use_pretrained_backbone=False)
model = DetrForObjectDetection(config)
```

As a summary, consider the following table:

| Task | Object detection | Instance segmentation | Panoptic segmentation |
|------|------------------|-----------------------|-----------------------|
| **Description** | Predicting bounding boxes and class labels around objects in an image | Predicting masks around objects (i.e. instances) in an image | Predicting masks around both objects (i.e. instances) as well as "stuff" (i.e. background things like trees and roads) in an image |
| **Model** | [`~transformers.DetrForObjectDetection`] | [`~transformers.DetrForSegmentation`] | [`~transformers.DetrForSegmentation`] |
| **Example dataset** | COCO detection | COCO detection, COCO panoptic | COCO panoptic  |                                                                        |
| **Format of annotations to provide to**  [`~transformers.DetrImageProcessor`] | {'image_id': `int`, 'annotations': `list[Dict]`} each Dict being a COCO object annotation  | {'image_id': `int`, 'annotations': `list[Dict]`}  (in case of COCO detection) or {'file_name': `str`, 'image_id': `int`, 'segments_info': `list[Dict]`} (in case of COCO panoptic) | {'file_name': `str`, 'image_id': `int`, 'segments_info': `list[Dict]`} and masks_path (path to directory containing PNG files of the masks) |
| **Postprocessing** (i.e. converting the output of the model to Pascal VOC format) | [`~transformers.DetrImageProcessor.post_process`] | [`~transformers.DetrImageProcessor.post_process_segmentation`] | [`~transformers.DetrImageProcessor.post_process_segmentation`], [`~transformers.DetrImageProcessor.post_process_panoptic`] |
| **evaluators** | `CocoEvaluator` with `iou_types="bbox"` | `CocoEvaluator` with `iou_types="bbox"` or `"segm"` | `CocoEvaluator` with `iou_tupes="bbox"` or `"segm"`, `PanopticEvaluator` |

- In short, one should prepare the data either in COCO detection or COCO panoptic format, then use [`~transformers.DetrImageProcessor`] to create `pixel_values`, `pixel_mask` and optional `labels`, which can then be used to train (or fine-tune) a model. 
- For evaluation, one should first convert the outputs of the model using one of the postprocessing methods of [`~transformers.DetrImageProcessor`]. These can be provided to either `CocoEvaluator` or `PanopticEvaluator`, which allow you to calculate metrics like mean Average Precision (mAP) and Panoptic Quality (PQ). The latter objects are implemented in the [original repository](https://github.com/facebookresearch/detr). See the [example notebooks](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DETR) for more info regarding evaluation.

## Resources

- Refer to these [notebooks](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DETR) for examples of fine-tuning [`DetrForObjectDetection`] and [`DetrForSegmentation`] on a custom dataset.

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
