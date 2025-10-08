<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2022-08-04 and added to Hugging Face Transformers on 2023-06-20 and contributed by [nielsr](https://huggingface.co/nielsr).*

# X-CLIP

[X-CLIP](https://huggingface.co/papers/2208.02816) extends CLIP for video recognition by incorporating a cross-frame attention mechanism and a video-specific prompt generator. This approach captures long-range dependencies across frames and leverages video content for generating discriminative textual prompts. Experiments show that X-CLIP achieves high accuracy in fully-supervised, zero-shot, and few-shot video recognition tasks, outperforming existing methods with fewer computational resources.

<hfoptions id="usage">
<hfoption id="AutoModel">

```py
import av
import torch
import numpy as np
from transformers import AutoProcessor, AutoModel
from huggingface_hub import hf_hub_download

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`list[int]`): List of frame indices to decode.
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
        indices (`list[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


file_path = hf_hub_download(
    repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
)
container = av.open(file_path)

indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
video = read_video_pyav(container, indices)
text_labels = ["playing sports", "eating spaghetti", "go shopping"]


processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
model = AutoModel.from_pretrained("microsoft/xclip-base-patch32", dtype="auto")

inputs = processor(
    text=text_labels,
    videos=list(video),
    return_tensors="pt",
    padding=True,
)

with torch.no_grad():
    outputs = model(**inputs)

logits_per_video = outputs.logits_per_video
probs = logits_per_video.softmax(dim=1)
for i, (label, prob) in enumerate(zip(text_labels, probs[0])):
    print(f"{label}: {prob:.4f}")
```

</hfoption>
</hfoptions>

## XCLIPProcessor

[[autodoc]] XCLIPProcessor

## XCLIPConfig

[[autodoc]] XCLIPConfig

## XCLIPTextConfig

[[autodoc]] XCLIPTextConfig

## XCLIPVisionConfig

[[autodoc]] XCLIPVisionConfig

## XCLIPModel

[[autodoc]] XCLIPModel
    - forward
    - get_text_features
    - get_video_features

## XCLIPTextModel

[[autodoc]] XCLIPTextModel
    - forward

## XCLIPVisionModel

[[autodoc]] XCLIPVisionModel
    - forward

