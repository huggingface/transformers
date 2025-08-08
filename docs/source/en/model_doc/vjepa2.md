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


<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    </div>
</div>

# V-JEPA 2

V-JEPA 2 is a self-supervised approach to training video encoders developed by FAIR, Meta. Using internet-scale video data, V-JEPA 2 attains state-of-the-art performance on motion understanding and human action anticipation tasks. V-JEPA 2-AC is a latent action-conditioned world model post-trained from V-JEPA 2 (using a small amount of robot trajectory interaction data) that solves robot manipulation tasks without environment-specific data collection or task-specific training or calibration.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/vjepa.gif" alt="drawing" width="600"/>
</div>

You can find all original V-JEPA2 checkpoints under the [V-JEPA 2](https://huggingface.co/collections/facebook/v-jepa-2-6841bad8413014e185b497a6) collection.


This model was contributed by [koustuvs](https://huggingface.co/koustuvs), [yonigozlan](https://huggingface.co/yonigozlan) and [qubvel](https://huggingface.co/qubvel-hf). The original code can be found [here](https://github.com/facebookresearch/vjepa2).

## Usage example

The snippet below shows how to load the V-JEPA 2 model for feature extraction using the `AutoModel` class.

```py
import torch
from torchcodec.decoders import VideoDecoder
import numpy as np

processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
model = AutoModel.from_pretrained(
    "facebook/vjepa2-vitl-fpc64-256",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)

video_url = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/archery/-Qz25rXdMjE_000014_000024.mp4"

vr = VideoDecoder(video_url)
frame_idx = np.arange(0, 64) # choosing some frames. here, you can define more complex sampling strategy
video = vr.get_frames_at(indices=frame_idx).data  # T x C x H x W
video = processor(video, return_tensors="pt").to(model.device)
outputs = model(**video)

# V-JEPA 2 encoder outputs, same as calling `model.get_vision_features()`
encoder_outputs = outputs.last_hidden_state

# V-JEPA 2 predictor outputs
predictor_outputs = outputs.predictor_output.last_hidden_state
```

V-JEPA 2 can also be finetuned for video classification. In the following snippet, we show how use finetuned on Something-Something-V2 video classification model.

```python
import torch
import numpy as np

from torchcodec.decoders import VideoDecoder
from transformers import AutoVideoProcessor, AutoModelForVideoClassification

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and video preprocessor
hf_repo = "facebook/vjepa2-vitl-fpc16-256-ssv2"

model = AutoModelForVideoClassification.from_pretrained(hf_repo).to(device)
processor = AutoVideoProcessor.from_pretrained(hf_repo)

# To load a video, sample the number of frames according to the model.
video_url = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/bowling/-WH-lxmGJVY_000005_000015.mp4"
vr = VideoDecoder(video_url)
frame_idx = np.arange(0, model.config.frames_per_clip, 8) # you can define more complex sampling strategy
video = vr.get_frames_at(indices=frame_idx).data  # frames x channels x height x width

# Preprocess and run inference
inputs = processor(video, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits

print("Top 5 predicted class names:")
top5_indices = logits.topk(5).indices[0]
top5_probs = torch.softmax(logits, dim=-1).topk(5).values[0]
for idx, prob in zip(top5_indices, top5_probs):
    text_label = model.config.id2label[idx.item()]
    print(f" - {text_label}: {prob:.2f}")
```

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
