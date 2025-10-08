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

*This model was released on 2025-03-26 and added to Hugging Face Transformers on 2025-10-07.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Qwen3-Omni-MOE

[Qwen3-Omni-MOE](https://huggingface.co/papers/2509.17765) is a unified multimodal model that achieves state-of-the-art performance across text, image, audio, and video without compromising on single-modal tasks. It excels in audio tasks, outperforming models like Gemini-2.5-Pro and GPT-4o-Transcribe on 36 audio and audio-visual benchmarks. The model uses a Thinker-Talker MoE architecture to unify perception and generation, supporting text in 119 languages, speech understanding in 19 languages, and speech generation in 10 languages. It reduces first-packet latency in streaming synthesis through a multi-codebook scheme and a lightweight causal ConvNet. Additionally, a Thinking model enhances multimodal reasoning, and a fine-tuned version, Qwen3-Omni-30B-A3B-Captioner, provides detailed audio captions. The models are released under the Apache 2.0 license.

<hfoptions id="usage">
<hfoption id="Qwen3OmniMoeForConditionalGeneration">

```py
import soundfile as sf
from transformers import Qwen3OmniMoeForConditionalGeneration, AutoProcessor

model = Qwen3OmniMoeForConditionalGeneration.from_pretrained("Qwen/Qwen3-Omni-30B-A3B-Instruct", dtype="auto")
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-Omni-30B-A3B-Instruct")

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

</hfoption>
</hfoptions>

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

