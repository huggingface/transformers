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
*This model was released on {release_date} and added to Hugging Face Transformers on 2025-12-16.*

# PE Video

[PE Video](https://huggingface.co/papers/2504.13181) is the video branch of Meta's Perception Encoder family. It contrastively aligns video clips with text into a shared embedding space, enabling zero-shot video classification and video–text retrieval from a single pretrained backbone.

The encoder's rotary embeddings and patch embedder treat the temporal axis as a first-class dimension, so variable-length clips can be encoded without tiling each frame independently.

You can find all the official PE Audio checkpoints under the [perception-encoder-audio-visual](https://huggingface.co/collections/facebook/perception-encoder-audio-visual) collection.

## Quickstart

```py
import torch
from transformers import AutoProcessor, PeVideoModel
from transformers.video_utils import load_video

processor = AutoProcessor.from_pretrained("facebook/pe-av-large")
model = PeVideoModel.from_pretrained(
    "facebook/pe-av-large",
    device_map="auto",
)

video, _ = load_video("https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4")
labels = ["a person playing tennis", "a person cooking", "a cat sleeping"]

video_inputs = processor.video_processor(video, num_frames=16, return_tensors="pt").to(model.device)
text_inputs = processor.tokenizer(labels, padding=True, return_tensors="pt").to(model.device)
inputs = {**video_inputs, **text_inputs}

with torch.no_grad():
    outputs = model(**inputs)

probs = outputs.logits_video_text.sigmoid()
print({label: p.item() for label, p in zip(labels, probs[0])})
```

## Usage tips and notes

- Variable-length videos use `padding_mask_videos` (not `attention_mask`). The video processor only pads and returns this mask when `return_tensors` is set — without it you get a list of per-clip tensors and no mask.
- Pass `num_frames` to the video processor for fixed-length uniform sampling across `[0, total_frames-1]`. Omit it to fall back to fps-based sampling from the base class. Checkpoints are usually trained at a specific frame count, so match what the checkpoint expects.
- Encoder input is `pixel_values_videos`. The encoder's `main_input_name` is `"pixel_values_videos"` while the full model's is `"input_ids"`, which matters when routing through generic utilities that inspect `main_input_name`.

## PeVideoConfig

[[autodoc]] PeVideoConfig

## PeVideoEncoderConfig

[[autodoc]] PeVideoEncoderConfig

## PeVideoVideoProcessor

[[autodoc]] PeVideoVideoProcessor

## PeVideoProcessor

[[autodoc]] PeVideoProcessor

## PeVideoEncoder

[[autodoc]] PeVideoEncoder
    - forward

## PeVideoModel

[[autodoc]] PeVideoModel
    - forward
