<!--Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2025-03-26 and added to Hugging Face Transformers on 2025-09-21.*

# Qwen3-Omni-MOE

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The Qwen3-Omni-MOE model is a unified multiple modalities model proposed in [Qwen3-Omni Technical Report](https://huggingface.co/papers/2509.17765) from Qwen team, Alibaba Group.

The abstract from the technical report is the following:

*We present Qwen3-Omni, a single multimodal model that, for the first time, maintains state-of-the-art performance across text, image, audio, and video without any degradation relative to single-modal counterparts. Qwen3-Omni matches the performance of same-sized single-modal models within the Qwen series and excels particularly on audio tasks. Across 36 audio and audio-visual benchmarks, Qwen3-Omni achieves open-source SOTA on 32 benchmarks and overall SOTA on 22, outperforming strong closed-source models such as Gemini-2.5-Pro, Seed-ASR, and GPT-4o-Transcribe. Qwen3-Omni adopts a Thinker-Talker MoE architecture that unifies perception and generation across text, images, audio, and video, yielding fluent text and natural real-time speech. It supports text interaction in 119 languages, speech understanding in 19 languages, and speech generation in 10 languages. To reduce first-packet latency in streaming synthesis, Talker autoregressively predicts discrete speech codecs using a multi-codebook scheme. Leveraging the representational capacity of these codebooks, we replace computationally intensive block-wise diffusion with a lightweight causal ConvNet, enabling streaming from the first codec frame. In cold-start settings, Qwen3-Omni achieves a theoretical end-to-end first-packet latency of 234 ms. To further strengthen multimodal reasoning, we introduce a Thinking model that explicitly reasons over inputs from any modality. Since the research community currently lacks a general-purpose audio captioning model, we fine-tuned Qwen3-Omni-30B-A3B to obtain Qwen3-Omni-30B-A3B-Captioner, which produces detailed, low-hallucination captions for arbitrary audio inputs. Qwen3-Omni-30B-A3B, Qwen3-Omni-30B-A3B-Thinking, and Qwen3-Omni-30B-A3B-Captioner are publicly released under the Apache 2.0 license.

## Notes

- Use [`Qwen3OmniMoeForConditionalGeneration`] to generate audio and text output. To generate only one output type, use [`Qwen3OmniMoeThinkerForConditionalGeneration`] for text-only and [`Qwen3OmniMoeTalkerForConditionalGeneration`] for audio-only outputs.
- Audio generation with [`Qwen3OmniMoeForConditionalGeneration`] supports only single batch size at the moment.
- In case out out-of-memory errors hwen working with video input, decrease `processor.max_pixels`. By default the maximum is set to a very arge value and high resolution visuals will not be resized, unless resolution exceeds `processor.max_pixels`.
- The processor has its own [`~ProcessorMixin.apply_chat_template`] method to convert chat messages to model inputs.

## Usage example

`Qwen3-Omni` can be found on the [Huggingface Hub](https://huggingface.co/Qwen).

### Single Media inference

The model can accept text, images, audio and videos as input. Here's an example code for inference.

```python
import soundfile as sf
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    dtype="auto",
    device_map="auto"
)
processor = Qwen3OmniMoeProcessor.from_pretrained("Qwen/Qwen3-Omni-30B-A3B-Instruct")

conversations = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "/path/to/video.mp4"},
            {"type": "text", "text": "What cant you hear and see in this video?"},
        ],
    },
]

inputs = processor.apply_chat_template(
    conversations,
    load_audio_from_video=True,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    video_fps=1,

    # kwargs to be passed to `Qwen3OmniMoeProcessor`
    padding=True,
    use_audio_in_video=True,
).to(model.device)

# Generation params for audio or text can be different and have to be prefixed with `thinker_` or `talker_`
text_ids, audio = model.generate(**inputs, use_audio_in_video=True, thinker_do_sample=False, talker_do_sample=True)
text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

sf.write(
    "output.wav",
    audio.reshape(-1).detach().cpu().numpy(),
    samplerate=24000,
)
print(text)
```

### Text-only generation

To generate only text output and save compute by not loading the audio generation model, we can use `Qwen3OmniMoeThinkerForConditionalGeneration` model.

```python
from transformers import Qwen3OmniMoeThinkerForConditionalGeneration, Qwen3OmniMoeProcessor

model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    dtype="auto",
    device_map="auto",
)
processor = Qwen3OmniMoeProcessor.from_pretrained("Qwen/Qwen3-Omni-30B-A3B-Instruct")

conversations = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "/path/to/video.mp4"},
            {"type": "text", "text": "What cant you hear and see in this video?"},
        ],
    },
]

inputs = processor.apply_chat_template(
    conversations,
    load_audio_from_video=True,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    video_fps=1,

    # kwargs to be passed to `Qwen3OmniMoeProcessor`
    padding=True,
    use_audio_in_video=True,
).to(model.device)


text_ids = model.generate(**inputs, use_audio_in_video=True)
text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

sf.write(
    "output.wav",
    audio.reshape(-1).detach().cpu().numpy(),
    samplerate=24000,
)
print(text)
```

### Batch Mixed Media Inference

The model can batch inputs composed of mixed samples of various types such as text, images, audio and videos as input when using `Qwen3OmniMoeThinkerForConditionalGeneration` model. Here is an example.

```python
import soundfile as sf
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    dtype="auto",
    device_map="auto"
)
processor = Qwen3OmniMoeProcessor.from_pretrained("Qwen/Qwen3-Omni-30B-A3B-Instruct")

# Conversation with video only
conversation1 = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "path": "/path/to/video.mp4"},
        ]
    }
]

# Conversation with audio only
conversation2 = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "audio", "path": "/path/to/audio.wav"},
        ]
    }
]

# Conversation with pure text
conversation3 = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": "who are you?"}],
    }
]


# Conversation with mixed media
conversation4 = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "path": "/path/to/image.jpg"},
            {"type": "video", "path": "/path/to/video.mp4"},
            {"type": "audio", "path": "/path/to/audio.wav"},
            {"type": "text", "text": "What are the elements can you see and hear in these medias?"},
        ],
    }
]

conversations = [conversation1, conversation2, conversation3, conversation4]

inputs = processor.apply_chat_template(
    conversations,
    load_audio_from_video=True,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    video_fps=1,

    # kwargs to be passed to `Qwen3OmniMoeProcessor`
    padding=True,
    use_audio_in_video=True,
).to(model.thinker.device)

text_ids = model.generate(**inputs, use_audio_in_video=True)
text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

print(text)
```

### Usage Tips

#### Image Resolution trade-off

The model supports a wide range of resolution inputs. By default, it uses the native resolution for input, but higher resolutions can enhance performance at the cost of more computation. Users can set the minimum and maximum number of pixels to achieve an optimal configuration for their needs.

```python
min_pixels = 128*28*28
max_pixels = 768*28*28
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-Omni-30B-A3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
```

#### Prompt for audio output

If users need audio output, the system prompt must be set as "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.", otherwise the audio output may not work as expected.

```json
{
    "role": "system",
    "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
}
```

#### Use audio output or not

The model supports both text and audio outputs, if users do not need audio outputs, they can set `enable_audio_output` in the `from_pretrained` function. This option will save about `~2GB` of GPU memory but the `return_audio` option for `generate` function will only allow to be set at `False`.

```python
model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    dtype="auto",
    device_map="auto",
    enable_audio_output=False,
)
```

In order to obtain a flexible experience, we recommend that users set `enable_audio_output` at `True` when initializing the model through `from_pretrained` function, and then decide whether to return audio when `generate` function is called. When `return_audio` is set to `False`, the model will only return text outputs to get text responses faster.

```python
model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    dtype="auto",
    device_map="auto",
    enable_audio_output=True,
)
...
text_ids = model.generate(**inputs, return_audio=False)
```

#### Change voice type of output audio

Qwen3-Omni-MOE supports the ability to change the voice of the output audio. Users can use the `spk` parameter of `generate` function to specify the voice type. The `"Qwen/Qwen3-Omni-30B-A3B-Instruct"` checkpoint support two voice types: `Chelsie` and `Ethan`, while `Chelsie` is a female voice and `Ethan` is a male voice. By default, if `spk` is not specified, the default voice type is `Chelsie`.

```python
text_ids, audio = model.generate(**inputs, spk="Chelsie")
```

```python
text_ids, audio = model.generate(**inputs, spk="Ethan")
```

#### Flash-Attention 2 to speed up generation

First, make sure to install the latest version of Flash Attention 2:

```bash
pip install -U flash-attn --no-build-isolation
```

Also, you should have hardware that is compatible with FlashAttention 2. Read more about it in the official documentation of the [flash attention repository](https://github.com/Dao-AILab/flash-attention). FlashAttention-2 can only be used when a model is loaded in `torch.float16` or `torch.bfloat16`.

To load and run a model using FlashAttention-2, add `attn_implementation="flash_attention_2"` when loading the model:

```python
from transformers import Qwen3OmniMoeForConditionalGeneration

model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    device_map="auto",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
```

## Qwen3OmniMoeConfig

[[autodoc]] Qwen3OmniMoeConfig

## Qwen3OmniMoeThinkerConfig

[[autodoc]] Qwen3OmniMoeThinkerConfig

## Qwen3OmniMoeTalkerConfig

[[autodoc]] Qwen3OmniMoeTalkerConfig

## Qwen3OmniMoeForConditionalGeneration

[[autodoc]] Qwen3OmniMoeForConditionalGeneration

## Qwen3OmniMoeThinkerTextModel

[[autodoc]] Qwen3OmniMoeThinkerTextModel

## Qwen3OmniMoeThinkerForConditionalGeneration

[[autodoc]] Qwen3OmniMoeThinkerForConditionalGeneration

## Qwen3OmniMoeTalkerForConditionalGeneration

[[autodoc]] Qwen3OmniMoeTalkerForConditionalGeneration

## Qwen3OmniMoePreTrainedModel

[[autodoc]] Qwen3OmniMoePreTrainedModel

## Qwen3OmniMoePreTrainedModelForConditionalGeneration

[[autodoc]] Qwen3OmniMoePreTrainedModelForConditionalGeneration

## Qwen3OmniMoeTalkerModel

[[autodoc]] Qwen3OmniMoeTalkerModel

## Qwen3OmniMoeThinkerTextPreTrainedModel

[[autodoc]] Qwen3OmniMoeThinkerTextPreTrainedModel

## Qwen3OmniMoeProcessor

[[autodoc]] Qwen3OmniMoeProcessor

## Qwen3OmniMoeCode2Wav

[[autodoc]] Qwen3OmniMoeCode2Wav

## Qwen3OmniMoeCode2WavDecoderBlock

[[autodoc]] Qwen3OmniMoeCode2WavDecoderBlock

## Qwen3OmniMoeCode2WavTransformerModel

[[autodoc]] Qwen3OmniMoeCode2WavTransformerModel

## Qwen3OmniMoeTalkerCodePredictorModel

[[autodoc]] Qwen3OmniMoeTalkerCodePredictorModel

## Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration

[[autodoc]] Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration
