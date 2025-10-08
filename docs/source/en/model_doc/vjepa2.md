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
*This model was released on 2025-06-11 and added to Hugging Face Transformers on 2025-06-11 and contributed by [koustuvs](https://huggingface.co/koustuvs), [yonigozlan](https://huggingface.co/yonigozlan), and [qubvel-hf](https://huggingface.co/qubvel-hf).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    </div>
</div>

# V-JEPA 2

[V-JEPA 2](https://huggingface.co/papers/2506.09985) is a self-supervised model pre-trained on over 1 million hours of internet video to learn motion, prediction, and planning without action labels. It achieves strong benchmarks in motion understanding and human action anticipation, and when aligned with a large language model, it attains state-of-the-art performance on video question-answering at the 8-billion parameter scale. For robotics, a latent action-conditioned version, V-JEPA 2-AC, is post-trained on under 62 hours of unlabeled robot video and can perform zero-shot object manipulation on Franka arms without task-specific data or rewards. This demonstrates that combining large-scale visual data with minimal robot interaction enables a generalizable world model capable of planning and acting in physical environments.

<hfoptions id="usage">
<hfoption id="AutoModel">

```py
import torch
import numpy as np
from torchcodec.decoders import VideoDecoder
from transformers import AutoVideoProcessor, AutoModelForVideoClassification

processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
model = AutoModelForVideoClassification.from_pretrained("facebook/vjepa2-vitl-fpc64-256", dtype="auto",)

video_url = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/archery/-Qz25rXdMjE_000014_000024.mp4"

vr = VideoDecoder(video_url)
frame_idx = np.arange(0, 64)
video = vr.get_frames_at(indices=frame_idx).data
video = processor(video, return_tensors="pt").to(model.device)
outputs = model(**video)

encoder_outputs = outputs.last_hidden_state
predictor_outputs = outputs.predictor_output.last_hidden_state
```

</hfoption>
</hfoptions>

## VJEPA2Config

[[autodoc]] VJEPA2Config

## VJEPA2Model

[[autodoc]] VJEPA2Model
    - forward

## VJEPA2ForVideoClassification

[[autodoc]] VJEPA2ForVideoClassification
    - forward

## VJEPA2VideoProcessor

[[autodoc]] VJEPA2VideoProcessor
