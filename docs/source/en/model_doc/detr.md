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
*This model was released on 2020-05-26 and added to Hugging Face Transformers on 2021-06-09 and contributed by [nielsr](https://huggingface.co/nielsr).*

# DETR

[DETR](https://huggingface.co/papers/2005.12872) presents a novel method for object detection by framing it as a direct set prediction problem. This approach eliminates the need for hand-designed components such as non-maximum suppression and anchor generation. DETR uses a set-based global loss and a transformer encoder-decoder architecture to output predictions in parallel. It achieves accuracy and runtime performance comparable to Faster R-CNN on the COCO dataset and can be extended to panoptic segmentation with superior results.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="object-detection", model="facebook/detr-resnet-50", dtype="auto")
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50", dtype="auto")

inputs = image_processor(images=image, return_tensors="pt")
outputs = model(**inputs)
target_sizes = torch.tensor([image.size[::-1]])
results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[
    0
]
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )
```

</hfoption>
</hfoptions>

## Usage tips

- DETR uses object queries to detect objects in an image. The number of queries determines the maximum number of objects that can be detected in a single image. This is set to 100 by default (see parameter `num_queries` of [`DetrConfig`]). It's good to have some slack. In COCO, the authors used 100 queries while the maximum number of objects in a COCO image is ~70.
- The DETR decoder updates query embeddings in parallel. This differs from language models like GPT-2, which use autoregressive decoding instead of parallel. No causal attention mask is used.
- DETR adds position embeddings to hidden states at each self-attention and cross-attention layer before projecting to queries and keys. For image position embeddings, choose between fixed sinusoidal or learned absolute position embeddings. By default, the `position_embedding_type` parameter of [`DetrConfig`] is set to "sine".
- During training, auxiliary losses in the decoder help the model output the correct number of objects of each class. Set the `auxiliary_loss` parameter of [`DetrConfig`] to `True` to add prediction feedforward neural networks and Hungarian losses after each decoder layer (with FFNs sharing parameters).
- For distributed training across multiple nodes, update the `num_boxes` variable in the `DetrLoss` class of `modeling_detr.py`. When training on multiple nodes, set this to the average number of target boxes across all nodes.
- [`DetrForObjectDetection`] and [`DetrForSegmentation`] initialize with any convolutional backbone available in the timm library. Initialize with a MobileNet backbone by setting the `backbone` attribute of [`DetrConfig`] to "tf_mobilenetv3_small_075", then initialize the model with that config.
- DETR resizes input images so the shortest side is at least a certain amount of pixels while the longest is at most 1333 pixels. At training time, scale augmentation randomly sets the shortest side to at least 480 and at most 800 pixels. At inference time, the shortest side is set to 800.
- Use [`DetrImageProcessor`] to prepare images (and optional annotations in COCO format) for the model. Due to resizing, images in a batch can have different sizes. DETR solves this by padding images up to the largest size in a batch and creating a pixel mask that indicates which pixels are real and which are padding. Alternatively, define a custom `collate_fn` to batch images together using [`~transformers.DetrImageProcessor.pad_and_create_pixel_mask`].
- Image size determines memory usage and batch size. Use a batch size of 2 per GPU.
- Prepare data in COCO detection or COCO panoptic format, then use [`DetrImageProcessor`] to create `pixel_values`, `pixel_mask`, and optional labels for training or fine-tuning.
- For evaluation, convert model outputs using one of the postprocessing methods of [`DetrImageProcessor`]. Provide these to either `CocoEvaluator` or `PanopticEvaluator` to calculate metrics like mean Average Precision (mAP) and Panoptic Quality (PQ). These evaluators are implemented in the original repository.

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

