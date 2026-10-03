<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->

# CogVLM2

## Overview

CogVLM2-Video is a video-language model from Zhipu AI / THUDM. The released
CogVLM2-Video Llama 3 checkpoint combines an EVA2-CLIP visual encoder with a
Llama 3-style causal language model.

Each video frame is encoded into 64 spatial visual tokens. A beginning-of-image
and end-of-image embedding are added around those tokens, producing 66 visual
tokens per frame. CogVLM2 uses a compressed position-id scheme for visual spans
so that long visual token runs do not consume one language position per token.

The native Transformers implementation supports the modern cache and generation
APIs and does not depend on the original Triton rotary kernel.

## Inference

Once the checkpoint metadata has been updated for native Transformers support,
CogVLM2-Video can be loaded without remote code:

```python
from transformers import AutoModelForCausalLM, AutoProcessor


model_id = "zai-org/cogvlm2-video-llama3-chat"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

inputs = processor(
    text="What happens in this video?",
    videos="path/to/video.mp4",
    return_tensors="pt",
).to(model.device)

output_ids = model.generate(**inputs, max_new_tokens=128)
print(processor.tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

The original checkpoint was published with remote-code metadata and without a
`model_type` field. Existing Hub repositories need their configuration and
processor metadata updated to use native loading after this model is released.

## CogVLM2Config

[[autodoc]] CogVLM2Config

## CogVLM2VisionConfig

[[autodoc]] CogVLM2VisionConfig

## CogVLM2VideoProcessor

[[autodoc]] CogVLM2VideoProcessor
    - preprocess

## CogVLM2Processor

[[autodoc]] CogVLM2Processor
    - __call__

## CogVLM2VisionModel

[[autodoc]] CogVLM2VisionModel
    - forward

## CogVLM2Model

[[autodoc]] CogVLM2Model
    - forward
    - get_video_features

## CogVLM2ForConditionalGeneration

[[autodoc]] CogVLM2ForConditionalGeneration
    - forward
    - prepare_inputs_for_generation
