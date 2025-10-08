<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2023-11-16 and added to Hugging Face Transformers on 2024-05-15 and contributed by [RaushanTurganbay](https://huggingface.co/RaushanTurganbay).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Video-LLaVA

[Video-LLaVA](https://huggingface.co/papers/2311.10122) unifies visual representations into the language feature space, advancing the foundational Large Language Model (LLM) towards a unified Large Vision-Language Model (LVLM). By learning from a mixed dataset of images and videos, Video-LLaVA enhances both modalities. It achieves superior performance across 9 image benchmarks and outperforms Video-ChatGPT by significant margins on four video datasets. Experiments show that Video-LLaVA benefits both images and videos within a unified visual representation, surpassing models specialized for either modality.

<hfoptions id="usage">
<hfoption id="VideoLlavaForConditionalGeneration">

```py
import av
import torch
import numpy as np
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor

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

model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", dtype="auto")
processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")
container = av.open(video_path)
total_frames = container.streams.video[0].frames
indices = np.arange(0, total_frames, total_frames / 8).astype(int)
video = read_video_pyav(container, indices)

prompt = "USER: <video>\nWhy is this funny? ASSISTANT:"
inputs = processor(text=prompt, videos=video, return_tensors="pt")

out = model.generate(**inputs, max_new_tokens=60)
processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
```

</hfoption>
</hfoptions>

## VideoLlavaConfig

[[autodoc]] VideoLlavaConfig

## VideoLlavaImageProcessor

[[autodoc]] VideoLlavaImageProcessor

## VideoLlavaProcessor

[[autodoc]] VideoLlavaProcessor

## VideoLlavaForConditionalGeneration

[[autodoc]] VideoLlavaForConditionalGeneration
    - forward

## VideoLlavaModel

[[autodoc]] VideoLlavaModel
    - forward

## VideoLlavaVideoProcessor

[[autodoc]] VideoLlavaVideoProcessor

