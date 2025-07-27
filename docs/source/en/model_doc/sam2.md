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

# SAM2

## Overview

SAM2 (Segment Anything Model 2) was proposed in [Segment Anything in Images and Videos](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/) by Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, Christoph Feichtenhofer.

The model can be used to predict segmentation masks of any object of interest given an input image.

![example image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/sam-output.png)

The abstract from the paper is the following:

*We present Segment Anything Model 2 (SAM 2), a foundation model towards solving promptable visual segmentation in images and videos. We build a data engine, which improves model and data via user interaction, to collect the largest video segmentation dataset to date. Our model is a simple transformer architecture with streaming memory for real-time video processing. SAM 2 trained on our data provides strong performance across a wide range of tasks. In video segmentation, we observe better accuracy, using 3x fewer interactions than prior approaches. In image segmentation, our model is more accurate and 6x faster than the Segment Anything Model (SAM). We believe that our data, model, and insights will serve as a significant milestone for video segmentation and related perception tasks. We are releasing a version of our model, the dataset and an interactive demo.*

Tips:

- Batch & Video Support: SAM2 natively supports batch processing and seamless video segmentation, while original SAM is designed for static images and simpler one-image-at-a-time workflows.
- Accuracy & Generalization: SAM2 shows improved segmentation quality, robustness, and zero-shot generalization to new domains compared to the original SAM, especially with mixed prompts.

This model was contributed by [sangbumchoi](https://github.com/SangbumChoi) and [yonigozlan](https://huggingface.co/yonigozlan).
The original code can be found [here](https://github.com/facebookresearch/sam2/tree/main).

Below is an example on how to run mask generation given an image and a 2D point:

```python
import torch
from PIL import Image
import requests
from transformers import Sam2Model, Sam2Processor

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Sam2Model.from_pretrained("danelcsb/sam2.1_hiera_tiny").to(device)
processor = Sam2Processor.from_pretrained("danelcsb/sam2.1_hiera_tiny")

img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
input_points = [[[[450, 600]]]]  # 2D location of a window in the image

inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)
pred_masks = outputs.pred_masks
```

You can also process input ivdeos in the processor to be passed to the model.

```python
from transformers import (
    Sam2Config,
    Sam2ImageProcessorFast,
    Sam2Model,
    Sam2VideoModel,
    Sam2Processor,
    Sam2VideoProcessor,
)

image_processor = Sam2ImageProcessorFast()
video_processor = Sam2VideoProcessor()
processor = Sam2Processor(image_processor=image_processor, video_processor=video_processor)

sam2model = Sam2VideoModel.from_pretrained("danelcsb/sam2.1_hiera_tiny").to("cuda")

# Use your custom video path
video_dir = "./videos/bedroom"

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

videos = []
for frame_name in frame_names:
    videos.append(Image.open(os.path.join(video_dir, frame_name)))
inference_session = processor.init_video_session(video=videos, inference_device="cuda")

ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = [1]  # give a unique id to each object we interact with (it can be any integers)
points = np.array([[[[210, 350]]]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([[[1]]], np.int32)

# Let's add a positive click at (x, y) = (210, 350) to get started
processor.process_new_points_or_boxes_for_video_frame(
    inference_session=inference_session,
    frame_idx=ann_frame_idx,
    obj_ids=ann_obj_id,
    input_points=points,
    input_labels=labels
)

Sam2VideoSegmentationOutput = sam2model(
    inference_session=inference_session,
    frame_idx=ann_frame_idx,
)
```

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with SAM2.

- [Demo notebook](https://github.com/huggingface/notebooks/blob/main/examples/segment_anything_2.ipynb) for using the model.

## Sam2Config

[[autodoc]] Sam2Config

## Sam2HieraDetConfig

[[autodoc]] Sam2HieraDetConfig

## Sam2VisionConfig

[[autodoc]] Sam2VisionConfig

## Sam2MaskDecoderConfig

[[autodoc]] Sam2MaskDecoderConfig

## Sam2PromptEncoderConfig

[[autodoc]] Sam2PromptEncoderConfig

## Sam2Processor

[[autodoc]] Sam2Processor
    - __call__
    - post_process_masks
    - init_video_session
    - add_inputs_to_inference_session

## Sam2ImageProcessorFast

[[autodoc]] Sam2ImageProcessorFast

## Sam2VideoProcessor

[[autodoc]] Sam2VideoProcessor

## Sam2VideoInferenceSession

[[autodoc]] Sam2VideoInferenceSession

## Sam2HieraDetModel

[[autodoc]] Sam2HieraDetModel
    - forward

## Sam2VisionModel

[[autodoc]] Sam2VisionModel
    - forward

## Sam2Model

[[autodoc]] Sam2Model
    - forward

## Sam2VideoModel

[[autodoc]] Sam2VideoModel
    - forward
    - propagate_in_video_iterator
