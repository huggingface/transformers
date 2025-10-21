<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
*This model was released on 2021-03-29 and added to Hugging Face Transformers on 2023-07-11 and contributed by [jegormeister](https://huggingface.co/jegormeister).*

# Video Vision Transformer (ViViT)

[Vivit](https://huggingface.co/papers/2103.15691) presents pure-transformer models for video classification, leveraging the success of transformers in image classification. The model extracts spatio-temporal tokens from input videos and encodes them through transformer layers. To manage long sequences of tokens in videos, the model introduces efficient variants that factorize spatial and temporal dimensions. Despite the need for large datasets for transformer models, the paper demonstrates effective regularization techniques and the use of pretrained image models to train on smaller datasets. Comprehensive ablation studies show state-of-the-art results on benchmarks like Kinetics 400 and 600, Epic Kitchens, Something-Something v2, and Moments in Time, surpassing deep 3D convolutional network-based methods.

<hfoptions id="usage">
<hfoption id="AutoModel">

```py
import av
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForVideoClassification
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

indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=container.streams.video[0].frames)
video = read_video_pyav(container=container, indices=indices)

image_processor = AutoImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
model = AutoModelForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")

inputs = image_processor(list(video), return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>
           

## VivitConfig

[[autodoc]] VivitConfig

## VivitImageProcessor

[[autodoc]] VivitImageProcessor
    - preprocess

## VivitModel

[[autodoc]] VivitModel
    - forward

## VivitForVideoClassification

[[autodoc]] transformers.VivitForVideoClassification
    - forward

