<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
*This model was released on 2024-02-20 and added to Hugging Face Transformers on 2026-01-16.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    </div>
</div>

# VideoPrism

The VideoPrism model was proposed in the paper [VideoPrism: A Foundational Visual Encoder for Video Understanding](https://huggingface.co/papers/2402.13217) by Google DeepMind ([blog post](https://research.google/blog/videoprism-a-foundational-visual-encoder-for-video-understanding/)).

VideoPrism is a general-purpose video encoder that tackles diverse video understanding tasks with a single frozen model. The model is pretrained on a large-scale heterogeneous corpus containing 36M high-quality video-caption pairs and 582M video clips with noisy parallel text (e.g., ASR transcripts). The pretraining approach improves upon masked autoencoding through global-local distillation of semantic video embeddings and a token shuffling scheme, enabling the model to focus primarily on the video modality while leveraging text associated with videos. VideoPrism achieves state-of-the-art performance on 31 out of 33 video understanding benchmarks across four broad task groups, from web video question answering to computer vision for science.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/MHRDYN7/videoprism_assets/resolve/main/GuOjZNKWEAAvkyu.jpeg" alt="drawing" width="600"/>
</div>

You can find all original VideoPrism checkpoints under the [VideoPrism](https://huggingface.co/collections/google/videoprism) collection.

Tips:

- VideoPrism uses a factorized spatio-temporal encoder architecture, processing videos through separate spatial and temporal transformers.
- The model supports video-text contrastive learning through `VideoPrismClipModel`, which combines a video encoder and a text encoder. `VideoPrismConfig` must be used with this model.
- For video classification tasks, use `VideoPrismForVideoClassification` which adds a classification head on top of the video encoder. `VideoPrismVisionConfig` must be used with this model.
- The vision encoder can be used standalone via `VideoPrismVisionModel` for extracting video features. `VideoPrismVisionConfig` must be used with this model.
- The default input resolution is 288x288 pixels with 16 frames per video clip for the base models and 8 frames for the large models. Set interpolate_pos_encoding=True to use the models with custom resolution and frames per clip.

This model was contributed by [MHRDYN7](https://github.com/MHRDYN7) and reviewed by [qubvel](https://github.com/qubvel) & [zucchini-nlp](https://github.com/zucchini-nlp).
The original code can be found [here](https://github.com/google-deepmind/videoprism).

## Usage example

The snippet below shows how to load the VideoPrismVisionModel for feature extraction using the `AutoModel` class.

```py
import torch
from torchcodec.decoders import VideoDecoder
import numpy as np

processor = AutoVideoProcessor.from_pretrained("MHRDYN7/videoprism-base-f16r288")
model = AutoModel.from_pretrained(
    "MHRDYN7/videoprism-base-f16r288",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa" # use "eager" to replicate the exact behavior as the original model
)

video_url = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/archery/-Qz25rXdMjE_000014_000024.mp4"

vr = VideoDecoder(video_url)
frame_idx = np.arange(0, 64) # choosing some frames. here, you can define more complex sampling strategy
video = vr.get_frames_at(indices=frame_idx).data  # T x C x H x W

# automatically samples 16 frames by default for the base model
video = processor(video, return_tensors="pt").to(model.device)
outputs = model(**video)

# VideoPrism encoder outputs
encoder_outputs = outputs.last_hidden_state

```

You may also use the original video processing function provided in the VideoPrism repository examples. However, this will be slower than using torchcodec with VideoPrismVideoProcessor for large batches of videos.

```python
import numpy as np

def read_and_preprocess_video(
    filename: str, target_num_frames: int, target_frame_size: tuple[int, int]
):
    """Reads and preprocesses a video."""

    frames = mediapy.read_video(filename)

    # Sample to target number of frames.
    frame_indices = np.linspace(0, len(frames), num=target_num_frames, endpoint=False, dtype=np.int32)
    frames = np.array([frames[i] for i in frame_indices])

    # Resize to target size.
    original_height, original_width = frames.shape[-3:-1]
    target_height, target_width = target_frame_size
    assert original_height * target_width == original_width * target_height, (
        "Currently does not support aspect ratio mismatch."
    )
    frames = mediapy.resize_video(frames, shape=target_frame_size)

    # Normalize pixel values to [0.0, 1.0].
    frames = mediapy.to_float01(frames)

    return frames
```

## VideoPrismVisionConfig

[[autodoc]] VideoPrismVisionConfig

## VideoPrismTextConfig

[[autodoc]] VideoPrismTextConfig

## VideoPrismConfig

[[autodoc]] VideoPrismConfig

## VideoPrismVideoProcessor

[[autodoc]] VideoPrismVideoProcessor

## VideoPrismTokenizer

[[autodoc]] VideoPrismTokenizer

## VideoPrismProcessor

[[autodoc]] VideoPrismProcessor

## VideoPrismVisionModel

[[autodoc]] VideoPrismVisionModel
    - forward

## VideoPrismVideoModel

[[autodoc]] VideoPrismVideoModel
    - forward

## VideoPrismTextModel

[[autodoc]] VideoPrismTextModel
    - forward

## VideoPrismClipModel

[[autodoc]] VideoPrismClipModel
    - forward

## VideoPrismForVideoClassification

[[autodoc]] VideoPrismForVideoClassification
    - forward
