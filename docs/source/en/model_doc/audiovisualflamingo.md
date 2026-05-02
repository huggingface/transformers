<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2026-04-22.*

# Audio-Visual Flamingo

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

Audio-Visual Flamingo (AVF) is a fully open audio-visual large language model for joint understanding and reasoning over
audio, images, and videos. In Transformers, AVF pairs a SigLIP vision tower with an AF-Whisper audio encoder and a
Qwen2.5-7B causal language model, with separate projectors for visual and audio features.

For video plus audio inputs, AVF does not simply concatenate visual and sound features. Instead, it aligns synchronized
visual and audio chunks, interleaves them along the time axis, applies Constrained Rotary Time Embeddings (CRTE), and
then feeds the fused sequence to the language model. In the Transformers interface, the processor prepares the required
media placeholder spans and the model replaces those token positions with projected multimodal embeddings during the
forward pass.

The model checkpoint is available at: [nvidia/audio-visual-flamingo-hf](https://huggingface.co/nvidia/audio-visual-flamingo-hf)

Highlights:

- Unified prompting across image, video, and audio inputs.
- Joint video plus audio understanding from a single container when `load_audio_in_video=True`.
- Dynamic-S2 visual preprocessing for high-resolution images and sampled video frames.
- Temporal audio-visual interleaving with CRTE before the Qwen2.5-7B backbone.
- Replace-in-place multimodal fusion through processor-prepared media spans and projected media embeddings.

This model was contributed by [Lasha Koroshinadze](https://huggingface.co/lashahub) and [Eric Bezzam](https://huggingface.co/bezzam).

### Paper

Audio-Visual Flamingo: Open Audio-Visual Intelligence for Long and Complex Videos  
S. Ghosh, A. Goel, K. Jayakumar, L. Koroshinadze, N. Anand, Z. Kong, S. Gururani, S. Lee, J. Kim, A. Aljafari, C.-H. H. Yang, S. Kim, R. Duraiswami, D. Manocha, M. Shoeybi, B. Catanzaro, M.-Y. Liu, W. Ping  
NVIDIA and University of Maryland

The paper presents AVF as a fully open audio-visual model trained for long and complex real-world videos. It introduces
AVF-Skills, a three-stage training curriculum, and Temporal Audio-Visual Interleaved Chain-of-Thought (TAVIT) for
temporally grounded reasoning. The paper also discusses a streaming TTS component; this page focuses on the public
conditional-generation checkpoint for multimodal understanding and text generation.

## Usage

### Audio-Visual Instruct Mode

The model supports chat-template conversations mixing text, images, videos, and audio. When
`load_audio_in_video=True`, a `video` content item can contribute both sampled frames and audio from the same
container.

➡️ video + audio from a single container

```python
from transformers import AudioVisualFlamingoForConditionalGeneration, AutoProcessor

model_id = "nvidia/audio-visual-flamingo-hf"

model = AudioVisualFlamingoForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    load_audio_in_video=True,
).eval()
processor = AutoProcessor.from_pretrained(
    model_id,
    padding_side="left",
    use_fast=False,
    load_audio_in_video=True,
    num_video_frames=128,
    audio_chunk_length="max_3600",
)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "/path/to/video.mp4"},
            {
                "type": "text",
                "text": "Describe both the visual scene and the spoken or environmental audio content.",
            },
        ],
    }
]

inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
).to(model.device)

generated_ids = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=False,
)

new_tokens = generated_ids[:, inputs.input_ids.shape[1] :]
print(processor.batch_decode(new_tokens, skip_special_tokens=True)[0])
```

### Prompt format

AVF uses chat-template content items with media placeholders:

- `{"type": "image", "image": "/path/to/image.jpg"}`
- `{"type": "video", "video": "/path/to/video.mp4"}`
- `{"type": "audio", "path": "/path/to/audio.wav"}`
- `{"type": "text", "text": "Describe the media."}`

You can mix these items within the same turn. When `load_audio_in_video=True`, a `video` content item can contribute
both visual frames and audio features from the same container.

## How the model works

### Architecture

* **Vision tower**
  SigLIP encodes images and sampled video frames. AVF uses Dynamic-S2 preprocessing to preserve fine-grained detail in
  high-resolution visual inputs while keeping the visual token sequence compact.

* **Audio tower**
  AVF uses AF-Whisper, the Audio Flamingo series' Whisper-based audio encoder. Audio is resampled to 16 kHz mono,
  converted to 128-bin log-mel spectrograms, and encoded in non-overlapping 30-second windows for long-form audio.

* **Multimodal projectors**
  Two 2-layer MLP projectors map visual and audio encoder features into the shared language-model hidden size.

* **Temporal interleaving + CRTE**
  After projection, synchronized visual and audio chunks are interleaved along the time axis rather than naively
  concatenated. AVF then applies Constrained Rotary Time Embeddings (CRTE) to the interleaved sequence so the language
  model can preserve absolute temporal structure while attending across co-occurring visual and auditory events.

* **Language model**
  A decoder-only multimodal language model built on a Qwen2.5-7B text backbone. In the Transformers interface, the
  processor expands the required media spans and the model replaces those token positions with projected multimodal
  embeddings during the forward pass; subsequent decode steps reuse the language-model cache.

### Processor-level alignment

1. The processor loads images, sampled video frames, and audio waveforms from the chat-template content items.
2. For `video` inputs, it can also decode the container audio stream when `load_audio_in_video=True`, so a single
   video item yields synchronized visual and audio features.
3. Visual inputs go through the Dynamic-S2 preprocessing path, while audio inputs are converted into AF-Whisper
   features with temporal chunk metadata for later alignment.
4. During the forward pass, the model projects the visual and audio features, interleaves synchronized chunks along the
   time axis, applies CRTE, and replaces the prepared media spans with the fused multimodal embeddings.

## AudioVisualFlamingoConfig

[[autodoc]] AudioVisualFlamingoConfig

## AudioVisualFlamingoProcessor

[[autodoc]] AudioVisualFlamingoProcessor
    - __call__

## AudioVisualFlamingoForConditionalGeneration

[[autodoc]] AudioVisualFlamingoForConditionalGeneration
    - forward
