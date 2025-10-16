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

*This model was released on 2025-03-26 and added to Hugging Face Transformers on 2025-04-14.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Qwen2.5-Omni

[Qwen2.5-Omni](https://huggingface.co/papers/2503.20215) is an end-to-end multimodal model capable of processing text, images, audio, and video, and generating text and natural speech responses in real-time. It uses block-wise processing for audio and visual encoders to manage long sequences, enhancing modality fusion through shared attention mechanisms. A novel position embedding approach, TMRoPE, aligns audio and video timestamps. The Thinker-Talker architecture separates text generation (Thinker) from speech generation (Talker), both trained and inferred end-to-end. A sliding-window DiT is used for efficient audio token decoding. Qwen2.5-Omni surpasses Qwen2-VL and Qwen2-Audio in image and audio tasks and achieves top performance on multimodal benchmarks like Omni-Bench. It is the first open-source model to match text input capabilities in end-to-end speech instruction following, as shown in MMLU and GSM8K benchmarks. Its streaming speech generation excels in robustness and naturalness compared to existing alternatives.

<hfoptions id="usage">
<hfoption id="Qwen2_5OmniThinkerForConditionGeneration">

```py
import torch
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, AutoProcessor

model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B", dtype="auto")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

conversations = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are a meterologist."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg},
            {"type": "text", "text": "Describe the weather in this image."},
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
    padding=True,
    use_audio_in_video=True,
)

text_ids = model.generate(**inputs, use_audio_in_video=True)
text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

sf.write(
    "output.wav",
    audio.reshape(-1).detach().cpu().numpy(),
    samplerate=24000,
)
print(text)
```

</hfoption>
</hfoptions>

## Usage tips

- Use [`Qwen2_5OmniForConditionalGeneration`] to generate both audio and text output. For single output types, use [`Qwen2_5OmniThinkerForConditionalGeneration`] for text-only or [`Qwen2_5OmniTalkersForConditionalGeneration`] for audio-only outputs.
- Audio generation with [`Qwen2_5OmniForConditionalGeneration`] supports only single batch size currently.
- Decrease `processor.max_pixels` if you encounter out-of-memory errors when working with video input. By default, the maximum is set to a very large value and high-resolution visuals won't resize unless resolution exceeds `processor.max_pixels`.
- The processor includes [`apply_chat_template`] to convert chat messages to model inputs.
- The model supports a wide range of resolution inputs. It uses native resolution by default, but higher resolutions enhance performance at the cost of more computation. Set minimum and maximum pixel counts to achieve optimal configuration for your needs.
- For audio output, set the system prompt to: "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech." Otherwise, audio output won't work as expected.
- The model supports both text and audio outputs. Set `enable_audio_output` in the `from_pretrained` function if you don't need audio outputs. This saves about 2GB of GPU memory but limits `return_audio` to `False` in the generate function. For flexibility, set `enable_audio_output=True` when initializing the model, then decide whether to return audio when calling generate. When `return_audio=False`, the model returns only text outputs for faster responses.
- Qwen2.5-Omni supports voice changes for output audio. Use the `spk` parameter in the generate function to specify voice type. The "Qwen/Qwen2.5-Omni-7B" checkpoint supports two voice types: Chelsie (female) and Ethan (male). Chelsie is the default voice if `spk` isn't specified.

## Qwen2_5OmniConfig

[[autodoc]] Qwen2_5OmniConfig

## Qwen2_5OmniProcessor

[[autodoc]] Qwen2_5OmniProcessor

## Qwen2_5OmniForConditionalGeneration

[[autodoc]] Qwen2_5OmniForConditionalGeneration
    - forward

## Qwen2_5OmniPreTrainedModelForConditionalGeneration

[[autodoc]] Qwen2_5OmniPreTrainedModelForConditionalGeneration

## Qwen2_5OmniThinkerConfig

[[autodoc]] Qwen2_5OmniThinkerConfig

## Qwen2_5OmniThinkerForConditionalGeneration

[[autodoc]] Qwen2_5OmniThinkerForConditionalGeneration

## Qwen2_5OmniThinkerTextModel

[[autodoc]] Qwen2_5OmniThinkerTextModel

## Qwen2_5OmniTalkerConfig

[[autodoc]] Qwen2_5OmniTalkerConfig

## Qwen2_5OmniTalkerForConditionalGeneration

[[autodoc]] Qwen2_5OmniTalkerForConditionalGeneration

## Qwen2_5OmniTalkerModel

[[autodoc]] Qwen2_5OmniTalkerModel

## Qwen2_5OmniToken2WavConfig

[[autodoc]] Qwen2_5OmniToken2WavConfig

## Qwen2_5OmniToken2WavModel

[[autodoc]] Qwen2_5OmniToken2WavModel

## Qwen2_5OmniToken2WavDiTModel

[[autodoc]] Qwen2_5OmniToken2WavDiTModel

## Qwen2_5OmniToken2WavBigVGANModel

[[autodoc]] Qwen2_5OmniToken2WavBigVGANModel

