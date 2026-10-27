<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-10-23.*


# LFM3-Omni

LFM3-Omni combines the [LFM3](lfm3) language model, the [LFM3-Audio](lfm3_audio) speech encoder and
codec depth decoder, and the [LFM3-VL](lfm3_vl) image encoder. It accepts interleaved text, images,
and audio, and generates text, speech, or interleaved text and speech.

## Usage

```python
import torch
from PIL import Image
from transformers import AutoProcessor, Lfm3OmniForConditionalGeneration

processor = AutoProcessor.from_pretrained("LiquidAI/LFM3-Omni-2B-Preview")
model = Lfm3OmniForConditionalGeneration.from_pretrained("LiquidAI/LFM3-Omni-2B-Preview", dtype=torch.bfloat16, device_map="cuda")
conversation = [{"role": "user", "content": [
    {"type": "image", "image": Image.open("image.png")},
    {"type": "text", "text": "Describe this image."},
]}]
inputs = processor.apply_chat_template(
    conversation, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128, generation_mode="text")
text = processor.decode(outputs.sequences[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
```

Audio inputs are 16 kHz waveforms; output speech uses the bundled 24 kHz codec. Speech generation, streaming,
batching and MaskGIT sampling follow [LFM3-Audio](lfm3_audio).

Videos enter as one timestamped group of placeholders per temporal patch. Images, input clips, videos and their
frame groups are wrapped in `<|vision_start|>` / `<|vision_end|>` like the Omni SFT data. A video's soundtrack is
opt-in: with `use_audio_in_video=True` (and `load_audio_from_video=True` in `apply_chat_template` to read it from the
file) the first clip after each video is interleaved after its frame groups, otherwise every clip is its own media
span.

For training, pass text `labels` and frame-wise `audio_labels`: placeholders take text labels of `-100`, ignored
speech frames have all codebook labels set to `-100`, and the loss averages text tokens and speech frames with their
configured weights as in Omni training. Every component, the vision encoder included, follows the requested `dtype`.

## Lfm3OmniConfig

[[autodoc]] Lfm3OmniConfig

## Lfm3OmniEncoderConfig

[[autodoc]] Lfm3OmniEncoderConfig

## Lfm3OmniProcessor

[[autodoc]] Lfm3OmniProcessor
    - __call__

## Lfm3OmniModel

[[autodoc]] Lfm3OmniModel
    - forward

## Lfm3OmniForConditionalGeneration

[[autodoc]] Lfm3OmniForConditionalGeneration
    - forward
    - generate
    - transcribe_live
