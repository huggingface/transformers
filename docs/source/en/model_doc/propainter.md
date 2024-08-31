<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the S-Lab License, Version 1.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

https://github.com/sczhou/ProPainter/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# ProPainter

## Overview

The ProPainter model was proposed in [ProPainter: Improving Propagation and Transformer for Video Inpainting](https://arxiv.org/abs/2309.03897) by Shangchen Zhou, Chongyi Li, Kelvin C.K. Chan, Chen Change Loy.

The abstract from the paper is the following:

*Flow-based propagation and spatiotemporal Transformer are two mainstream mechanisms in video inpainting (VI). Despite the effectiveness of these components, they still suffer from some limitations that affect their performance. Previous propagation-based approaches are performed separately either in the image or feature domain. Global image propagation isolated from learning may cause spatial misalignment due to inaccurate optical flow. Moreover, memory or computational constraints limit the temporal range of feature propagation and video Transformer, preventing exploration of correspondence information from distant frames. To address these issues, we propose an improved framework, called ProPainter, which involves enhanced ProPagation and an efficient Transformer. Specifically, we introduce dual-domain propagation that combines the advantages of image and feature warping, exploiting global correspondences reliably. We also propose a mask-guided sparse video Transformer, which achieves high efficiency by discarding unnecessary and redundant tokens. With these components, ProPainter outperforms prior arts by a large margin of 1.46 dB in PSNR while maintaining appealing efficiency.*

## Usage tips:

- The model is used for both video inpainting and video outpainting. To switch between modes, you need to give a value in form of `tuple(h,w)` to `scale_hw` kwarg to the `ProPainterImageProcessor`. In the usage example, we have provided both ways to give video frames and masks whether their data is in form of .mp4 or .jpg or any other image/video extensions.

This model was contributed by [ruffy369](https://huggingface.co/ruffy369). The original code can be found [here](https://github.com/sczhou/ProPainter).

## Usage example

The model can accept videos frames and its corresponding masks frame(s) as input. Here's an example code for inference:

```python
import av
import numpy as np
import cv2
from PIL import Image
from huggingface_hub import hf_hub_download

np.random.seed(0)

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

##########################IF YOU HAVE THE MASK AND FRAMES IN THE FORM OF MP4(comment the below part if you want to use this)###########################
# video clip consists of 80 frames(both masks and original video) (3 seconds at 24 FPS)
video_file_path = hf_hub_download(
    repo_id="ruffy369/propainter-object-removal", filename="object_removal_bmx/bmx.mp4", repo_type="dataset"
)
masks_file_path = hf_hub_download(
    repo_id="ruffy369/propainter-object-removal", filename="object_removal_bmx/bmx_masks.mp4", repo_type="dataset"
)
container_video = av.open(video_file_path)
container_masks = av.open(masks_file_path)


# sample 32 frames
indices = sample_frame_indices(clip_len=32, frame_sample_rate=1, seg_len=container_video.streams.video[0].frames)
video = read_video_pyav(container=container_video, indices=indices)

masks = read_video_pyav(container=container_masks, indices=indices)
video = list(video)
masks = list(masks)


##########################IF YOU HAVE FOLDER WITH JPG IMAGES OF FRAMES AND MASKS(comment the above part if you want to use this########################
from datasets import load_dataset
import numpy as np
ds = load_dataset("ruffy369/propainter-object-removal")
ds_images = ds['train']["image"]
num_frames = 80
video = [np.array(ds_images[i]) for i in range(num_frames)]
#stack to convert H,W mask frame to compatible H,W,C frame
masks = [np.stack([np.array(ds_images[i])] * 3, axis=-1) for i in range(num_frames, 2*num_frames)]

####################################################START OF THE IMAGE PROCESSOR AND MODEL FORWARD PASS################################################
from transformers import ProPainterImageProcessor
image_processor = ProPainterImageProcessor()
inputs = image_processor(images = video, masks = masks)

import transformers as t
model = t.ProPainterModel.from_pretrained("ruffy369/ProPainter")

#For inference and getting that flashy final output(can be commented out)
model.eval()

with torch.no_grad():
    outputs = model(**inputs)
#for training or calculating a simple backward pass

```


## ProPainterConfig

[[autodoc]] ProPainterConfig

## ProPainterModel

[[autodoc]] ProPainterModel
    - forward

</pt>
<tf>
