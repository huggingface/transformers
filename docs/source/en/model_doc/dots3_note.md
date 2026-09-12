<!--Copyright 2026 The Dots Studio team and the HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-09-09.*

# Dots 3 Note Preview

Dots 3 Note Preview is a mixture-of-experts causal language model with native text, image, video, and audio inputs. It uses
a shared vision encoder for images and videos and a Whisper-style audio encoder. Both encoders project their outputs
into the language model's hidden space before autoregressive text generation.

Dots 3 Note Preview checkpoints are available with BF16 weights or with fine-grained FP8 language-model weights and BF16
vision, audio, and language-model-head weights. Both formats load through the standard Transformers APIs.

## Usage

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor


model_id = "dots-studio/dots3-note-prev"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

output_ids = model.generate(**inputs, max_new_tokens=128)
print(processor.batch_decode(output_ids, skip_special_tokens=True)[0])
```

Use the same processing and generation calls with any of these user messages:

```python
# Text
conversation = [{"role": "user", "content": [{"type": "text", "text": "Briefly introduce yourself."}]}]

# Audio: replace the path with a local recording.
conversation = [{"role": "user", "content": [
    {"type": "audio", "path": "speech.wav"},
    {"type": "text", "text": "Transcribe this recording."},
]}]

# Native video, including its audio track when present.
conversation = [{"role": "user", "content": [
    {"type": "video", "path": "concert.mp4"},
    {"type": "text", "text": "Describe what you see and hear."},
]}]
```

`Dots3NoteModel` combines the text, vision and audio encoders. `Dots3NoteForConditionalGeneration` adds the
language-model head and generation interface. `Dots3NoteTextForCausalLM` provides the text-only variant;
`Dots3NoteForCausalLM` remains a compatibility name for the original multimodal checkpoints.

## Dots3NoteConfig

[[autodoc]] Dots3NoteConfig

## Dots3NoteVisionConfig

[[autodoc]] Dots3NoteVisionConfig

## Dots3NoteAudioConfig

[[autodoc]] Dots3NoteAudioConfig

## Dots3NoteForCausalLM

[[autodoc]] Dots3NoteForCausalLM
    - forward

## Dots3NoteModel

[[autodoc]] Dots3NoteModel
    - forward

## Dots3NoteForConditionalGeneration

[[autodoc]] Dots3NoteForConditionalGeneration

## Dots3NoteTextModel

[[autodoc]] Dots3NoteTextModel
    - forward

## Dots3NoteTextForCausalLM

[[autodoc]] Dots3NoteTextForCausalLM
    - forward

## Dots3NoteVisionModel

[[autodoc]] Dots3NoteVisionModel
    - forward

## Dots3NoteAudioModel

[[autodoc]] Dots3NoteAudioModel
    - forward

## Dots3NoteProcessor

[[autodoc]] Dots3NoteProcessor
    - __call__

## Dots3NoteImageProcessor

[[autodoc]] Dots3NoteImageProcessor

## Dots3NoteVideoProcessor

[[autodoc]] Dots3NoteVideoProcessor

## Dots3NoteFeatureExtractor

[[autodoc]] Dots3NoteFeatureExtractor
