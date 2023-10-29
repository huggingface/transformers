<!--Copyright 2023 The Intel Team Authors and HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# TVP

## Overview

The text-visual prompting (TVP) framework was proposed in the paper [Text-Visual Prompting for Efficient 2D Temporal Video Grounding](https://arxiv.org/abs/2303.04995) by Yimeng Zhang, Xin Chen, Jinghan Jia, Sijia Liu, Ke Ding.

The abstract from the paper is the following:

*In this paper, we study the problem of temporal video grounding (TVG), which aims to predict the starting/ending time points of moments described by a text sentence within a long untrimmed video. Benefiting from fine-grained 3D visual features, the TVG techniques have achieved remarkable progress in recent years. However, the high complexity of 3D convolutional neural networks (CNNs) makes extracting dense 3D visual features time-consuming, which calls for intensive memory and computing resources. Towards efficient TVG, we propose a novel text-visual prompting (TVP) framework, which incorporates optimized perturbation patterns (that we call ‘prompts’) into both visual inputs and textual features of a TVG model. In sharp contrast to 3D CNNs, we show that TVP allows us to effectively co-train vision encoder and language encoder in a 2D TVG model and improves the performance of cross-modal feature fusion using only low-complexity sparse 2D visual features. Further, we propose a Temporal-Distance IoU (TDIoU) loss for efficient learning of TVG. Experiments on two benchmark datasets, Charades-STA and ActivityNet Captions datasets, empirically show that the proposed TVP significantly boosts the performance of 2D TVG (e.g., 9.79% improvement on Charades-STA and 30.77% improvement on ActivityNet Captions) and achieves 5× inference acceleration over TVG using 3D visual features.*

TVP framework is an effective and efficient framework to train time-efficient 2D TVG models, in which we leverage TVP (text-visual prompting) to improve the utility of sparse 2D visual features without resorting to costly 3D features. To the best of our knowledge, it is the first work to expand the application of prompt learning for resolving TVG problems.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/tvp_architecture.png"
alt="drawing" width="600"/> 

<small> TVP architecture. Taken from the <a href="https://arxiv.org/abs/2209.14156">original paper.</a> </small>

## Usage

TVP consists of a visual encoder and cross-modal encoder, and also textual prompt and visual prompt.
The goal of this model is to incorporate trainable prompts into both visual inputs and textual features to temporal video grounding(TVG) problems.
In principle, one can apply any visual, cross-modal encoder in the proposed architecture.

The [`TvpProcessor`] wraps [`BertTokenizer`] and [`TvpImageProcessor`] into a single instance to both
encode the text and prepare the images respectively.

The following example shows how to run temporal video grounding using [`TvpProcessor`] and [`TvpForVideoGrounding`].
```python
import av
import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, TvpForVideoGrounding


def pyav_decode(container, sampling_rate, num_frames, clip_idx, num_clips, target_fps):
    '''
    Convert the video from its original fps to the target_fps and decode the video with PyAV decoder.
    Returns:
        frames (tensor): decoded frames from the video. Return None if the no
            video stream was found.
        fps (float): the number of frames per second of the video.
    '''
    fps = float(container.streams.video[0].average_rate)
    clip_size = sampling_rate * num_frames / target_fps * fps
    delta = max(container.streams.video[0].frames - clip_size, 0)
    start_idx = delta * clip_idx / num_clips
    end_idx = start_idx + clip_size - 1
    timebase = container.streams.video[0].duration / container.streams.video[0].frames
    video_start_pts = int(start_idx * timebase)
    video_end_pts = int(end_idx * timebase)
    stream_name = {"video": 0}
    seek_offset = max(video_start_pts - 1024, 0)
    container.seek(seek_offset, any_frame=False, backward=True, stream=container.streams.video[0])
    frames = {}
    for frame in container.decode(**stream_name):
        if frame.pts < video_start_pts:
            continue
        if frame.pts <= video_end_pts:
            frames[frame.pts] = frame
        else:
            frames[frame.pts] = frame
            break
    frames = [frames[pts] for pts in sorted(frames)]
    return frames, fps


def decode(container, sampling_rate, num_frames, clip_idx, num_clips, target_fps):
    '''
    Decode the video and perform temporal sampling.
    Args:
        container (container): pyav container.
        sampling_rate (int): frame sampling rate (interval between two sampled frames).
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal sampling.
            If clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the clip_idx-th video clip.
        num_clips (int): overall number of clips to uniformly sample from the given video.
        target_fps (int): the input video may have different fps, convert it to
            the target video fps before frame sampling.
    Returns:
        frames (tensor): decoded frames from the video.
    '''
    assert clip_idx >= -2, "Not a valied clip_idx {}".format(clip_idx)
    frames, fps = pyav_decode(container, sampling_rate, num_frames, clip_idx, num_clips, target_fps)
    clip_size = sampling_rate * num_frames / target_fps * fps
    index = torch.linspace(0, clip_size - 1, num_frames)
    index = torch.clamp(index, 0, len(frames) - 1).long().tolist()
    frames = [frames[idx] for idx in index]
    frames = [frame.to_rgb().to_ndarray() for frame in frames]
    frames = torch.from_numpy(np.stack(frames))
    return frames

def get_resize_size(image, max_size):
    '''
    Args:
        image: np.ndarray
        max_size: The max size of height and width
    Returns:
        (height, width)
    Note the height/width order difference >>> pil_img = Image.open("raw_img_tensor.jpg") >>> pil_img.size (640,
    480) # (width, height) >>> np_img = np.array(pil_img) >>> np_img.shape (480, 640, 3) # (height, width, 3)
    '''
    height, width = image.shape[-2:]
    if height >= width:
        ratio = width * 1.0 / height
        new_height = max_size
        new_width = new_height * ratio
    else:
        ratio = height * 1.0 / width
        new_width = max_size
        new_height = new_width * ratio
    size = {"height": int(new_height), "width": int(new_width)}
    return size

file = hf_hub_download(repo_id="Intel/tvp_demo", filename="3MSZA.mp4", repo_type="dataset")
model = TvpForVideoGrounding.from_pretrained("Intel/tvp-base")

decoder_kwargs = dict(
    container=av.open(file, metadata_errors="ignore"),
    sampling_rate=1,
    num_frames=model.config.num_frames,
    clip_idx=0,
    num_clips=1,
    target_fps=3,
)
raw_sampled_frms = decode(**decoder_kwargs).permute(0, 3, 1, 2)

text = "person turn a light on."
processor = AutoProcessor.from_pretrained("Intel/tvp-base")
size = get_resize_size(raw_sampled_frms, model.config.max_img_size)
model_inputs = processor(
    text=[text], videos=list(raw_sampled_frms.numpy()), return_tensors="pt", max_text_length=100, size=size
)

model_inputs["pixel_values"] = model_inputs["pixel_values"].to(model.dtype)
model_inputs["labels"] = torch.tensor([30.96, 24.3, 30.4])
output = model(**model_inputs)
print(f"The model's output is {output}")

def get_video_duration(filename):
    cap = cv2.VideoCapture(filename)
    if cap.isOpened():
        rate = cap.get(5)
        frame_num = cap.get(7)
        duration = frame_num/rate
        return duration
    return -1

duration = get_video_duration(file)
timestamp = output['logits'].tolist()
start, end = round(timestamp[0][0]*duration, 1), round(timestamp[0][1]*duration, 1)
print(f"The time slot of the video corresponding to the text \"{text}\" is from {start}s to {end}s")
```


This model was contributed by [Jiqing Feng](https://huggingface.co/Jiqing). The original code can be found [here](https://github.com/intel/TVP).


Tips:

- This implementation of TVP uses [`BertTokenizer`] to generate text embeddings and Resnet-50 model to compute visual embeddings.
- Checkpoints for pre-trained [tvp-base](https://huggingface.co/Intel/tvp-base) is released.
- Please refer to [Table 2](https://arxiv.org/pdf/2303.04995.pdf) for TVP's performance on Temporal Video Grounding task.
- The PyTorch version of this model is only available in torch 1.10 and higher.


## TvpConfig

[[autodoc]] TvpConfig

## TvpImageProcessor

[[autodoc]] TvpImageProcessor
    - preprocess

## TvpProcessor

[[autodoc]] TvpProcessor
    - __call__

## TvpModel

[[autodoc]] TvpModel
    - forward

## TvpForVideoGrounding

[[autodoc]] TvpForVideoGrounding
    - forward