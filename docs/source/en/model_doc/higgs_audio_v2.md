<!--Copyright 2025 Boson AI and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2025-07-22 and added to Hugging Face Transformers on 2026-02-19.*

# Higgs Audio V2

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

## Overview

Higgs Audio V2 is a powerful audio foundation model developed by [Boson AI](https://www.boson.ai/). 
The model was pretrained on over 10 million hours of audio data and a diverse set of text data. 
Despite having no post-training or fine-tuning, Higgs Audio v2 excels in expressive audio generation, thanks to its deep language and acoustic understanding.

**Model Architecture:**
<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/eustlb/documentation-images/resolve/main/higgs_audio_v2_architecture_combined.png"/>
</div>
Higgs Audio v2 adopts the "generation variant" depicted in the architecture figure above. Its strong performance is driven by three key technical innovations:

- Developed an automated annotation pipeline that leverages multiple ASR models, sound event classification models, and our in-house audio understanding model. Using this pipeline, we cleaned and annotated 10 million hours audio data, which we refer to as AudioVerse. The in-house understanding model is finetuned on top of Higgs Audio v1 Understanding, which adopts the "understanding variant" shown in the architecture figure.
- Trained a unified audio tokenizer from scratch that captures both semantic and acoustic features.
- Proposed DualFFN architecture, which enhances the LLM’s ability to model acoustics tokens with minimal computational overhead.

## Usage

All of the snippets below mirror the integration tests in `test_higgs_audio.py`, ensuring the doc stays in sync with the officially supported workflows.

### Single-speaker smart voice

```python
from transformers import AutoProcessor, HiggsAudioV2ForConditionalGeneration

model_id = "eustlb/higgs-audio-v2-generation-3B-base"
processor = AutoProcessor.from_pretrained(model_id, device_map="auto")
model = HiggsAudioV2ForConditionalGeneration.from_pretrained(model_id, device_map="auto")

conversation = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "Generate audio following instruction."
            }
        ],
    },
    {
        "role": "scene",
        "content": [
            {
                "type": "text",
                "text": "Audio is recorded from a quiet room."
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years."
            }
        ]
    }
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    sampling_rate=24000,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=1000, do_sample=False)
decoded = processor.batch_decode(outputs)
processor.save_audio(decoded, "output_single_speaker_smart_voice.wav")
```

### Multi-speaker smart voice

```python
from transformers import AutoProcessor, HiggsAudioV2ForConditionalGeneration

model_id = "eustlb/higgs-audio-v2-generation-3B-base"
processor = AutoProcessor.from_pretrained(model_id, device_map="auto")
model = HiggsAudioV2ForConditionalGeneration.from_pretrained(model_id, device_map="auto")

system_message = """You are an AI assistant designed to convert text into speech.
If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.
If no speaker tag is present, select a suitable voice on your own."""

user_message = """[SPEAKER0] I can't believe you did that without even asking me first!
[SPEAKER1] Oh, come on! It wasn't a big deal, and I knew you would overreact like this.
[SPEAKER0] Overreact? You made a decision that affects both of us without even considering my opinion!
[SPEAKER1] Because I didn't have time to sit around waiting for you to make up your mind! Someone had to act."""

conversation = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": system_message
            }
        ]
    },
    {
        "role": "scene",
        "content": [
            {
                "type": "text",
                "text": "Audio is recorded from a quiet room."
            },
            {
                "type": "text",
                "text": "SPEAKER0: feminine"
            },
            {
                "type": "text",
                "text": "SPEAKER1: masculine"
            },
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": user_message
            }
        ]
    }
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    sampling_rate=24000,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
decoded = processor.batch_decode(outputs)
processor.save_audio(decoded, "output_multi_speaker_smart_voice.wav")
```

### Zero-shot voice cloning

```python
from transformers import AutoProcessor, HiggsAudioV2ForConditionalGeneration

model_id = "eustlb/higgs-audio-v2-generation-3B-base"
processor = AutoProcessor.from_pretrained(model_id, device_map="auto")
model = HiggsAudioV2ForConditionalGeneration.from_pretrained(model_id, device_map="auto")

conversation = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "Generate audio following instruction."
            }
        ]
    },
    {
        "role": "scene",
        "content": [
            {
                "type": "text",
                "text": "Audio is recorded from a quiet room."
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "It was the night before my birthday. Hooray! It's almost here! It may not be a holiday, but it's the best day of the year."
            }
        ]
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "audio",
                "url": "https://huggingface.co/datasets/eustlb/dummy-audio-samples-higgs/resolve/main/belinda.wav"
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years."
            }
        ]
    }
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    sampling_rate=24000,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=1000, do_sample=False)
decoded = processor.batch_decode(outputs)
processor.save_audio(decoded, "output_zero_shot_voice_cloning.wav")
```

### Multi-speaker voice cloning

```python
from transformers import AutoProcessor, HiggsAudioV2ForConditionalGeneration

model_id = "eustlb/higgs-audio-v2-generation-3B-base"
processor = AutoProcessor.from_pretrained(model_id, device_map="auto")
model = HiggsAudioV2ForConditionalGeneration.from_pretrained(model_id, device_map="auto")

user_message = """[SPEAKER0] I can't believe you did that without even asking me first!
[SPEAKER1] Oh, come on! It wasn't a big deal, and I knew you would overreact like this.
[SPEAKER0] Overreact? You made a decision that affects both of us without even considering my opinion!
[SPEAKER1] Because I didn't have time to sit around waiting for you to make up your mind! Someone had to act."""

conversation = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "Generate audio following instruction."
            }
        ]
    },
    {
        "role": "scene",
        "content": [
            {
                "type": "text",
                "text": "Audio is recorded from a quiet room."
            },
            {
                "type": "text",
                "text": "SPEAKER0:"
            },
            {
                "type": "audio",
                "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav"
            },
            {
                "type": "text",
                "text": "SPEAKER1:"
            },
            {
                "type": "audio",
                "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/1272-128104-0000.flac"
            },
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": user_message
            }
        ]
    },
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    sampling_rate=24000,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=1000, do_sample=False)
decoded = processor.batch_decode(outputs)
processor.save_audio(decoded, "output_multi_speaker_voice_cloning.wav")
```

### Batched inference

```python
from transformers import AutoProcessor, HiggsAudioV2ForConditionalGeneration

model_id = "eustlb/higgs-audio-v2-generation-3B-base"
processor = AutoProcessor.from_pretrained(model_id, device_map="auto")
model = HiggsAudioV2ForConditionalGeneration.from_pretrained(model_id, device_map="auto")

conversation1 = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "Generate audio following instruction."
            }
        ]
    },
    {
        "role": "scene",
        "content": [
            {
                "type": "text",
                "text": "Audio is recorded from a quiet room."
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "It was the night before my birthday. Hooray! It's almost here! It may not be a holiday, but it's the best day of the year."
            }
        ]
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "audio",
                "url": "https://huggingface.co/datasets/eustlb/dummy-audio-samples-higgs/resolve/main/belinda.wav"
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years."
            }
        ]
    }
]

conversation2 = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "Generate audio following instruction."
            }
        ]
    },
    {
        "role": "scene",
        "content": [
            {
                "type": "text",
                "text": "Audio is recorded from a quiet room."
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": " It's super important to assess fairly the fact that our former model is over. And this is not a question of adjustment. This is not the same world, 2024, 2025. And on top of that, we are making the same mistakes, on top of the key elements I mentioned. We are over-regulating and under-investing. So just if, in the two to three years to come, if we follow our classical agenda, we will be out of the market. I have no doubts."
            }
        ]
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "audio",
                "url": "https://huggingface.co/datasets/eustlb/dummy-audio-samples-higgs/resolve/main/macron.wav"
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Hey, here is a clone from the given voice."
            }
        ]
    }
]

inputs = processor.apply_chat_template(
    [conversation1, conversation2],
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    sampling_rate=24000,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=1000, do_sample=False)
decoded = processor.batch_decode(outputs)
processor.save_audio(decoded, ["output_batched_1.wav", "output_batched_2.wav"])
```

### Training

> [!TIP]
> By default, the model does not load the text language modeling head to save memory (~1.5GiB reduction), as it's not required for generation.
> However, when training the model, you need the text head to compute loss on text tokens. To enable it, set `use_text_head=True` when instantiating the model (see example below).

```python
from transformers import AutoProcessor, HiggsAudioV2ForConditionalGeneration

model_id = "eustlb/higgs-audio-v2-generation-3B-base"
processor = AutoProcessor.from_pretrained(model_id, device_map="auto")
model = HiggsAudioV2ForConditionalGeneration.from_pretrained(model_id, device_map="auto", use_text_head=True)

conversation1 = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "Generate audio following instruction."
            }
        ]
    },
    {
        "role": "scene",
        "content": [
            {
                "type": "text",
                "text": "Audio is recorded from a quiet room."
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "It was the night before my birthday. Hooray! It's almost here! It may not be a holiday, but it's the best day of the year."
            }
        ]
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "audio",
                "url": "https://huggingface.co/datasets/eustlb/dummy-audio-samples-higgs/resolve/main/belinda.wav"
            }
        ]
    }
]

conversation2 = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "Generate audio following instruction."
            }
        ]
    },
    {
        "role": "scene",
        "content": [
            {
                "type": "text",
                "text": "Audio is recorded from a quiet room."
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": " I would imagine so. A wand with a dragon heartstring core is capable of dazzling magic, and the bond between you and your wand should only grow stronger. Do not be surprised at your new wand's ability to perceive your intentions, particularly in a moment of need"
            }
        ]
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "audio",
                "url": "https://huggingface.co/datasets/eustlb/dummy-audio-samples-higgs/resolve/main/broom_salesman.wav"
            }
        ]
    }
]

inputs = processor.apply_chat_template(
    [conversation1, conversation2],
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    sampling_rate=24000,
    return_tensors="pt",
    output_labels=True,
).to(model.device)

outputs = model(**inputs)
outputs.loss.backward()
```

This model was contributed by [Shuai Zheng](https://huggingface.co/szhengac) and [Eustache Le Bihan](https://huggingface.co/eustlb). The original code can be found [here](https://github.com/boson-ai/higgs-audio).


## HiggsAudioV2Config

[[autodoc]] HiggsAudioV2Config

## HiggsAudioV2Processor

[[autodoc]] HiggsAudioV2Processor
    - __call__
    - decode

## HiggsAudioV2Model

[[autodoc]] HiggsAudioV2Model
    - forward

## HiggsAudioV2ForConditionalGeneration

[[autodoc]] HiggsAudioV2ForConditionalGeneration
    - forward
    - generate
