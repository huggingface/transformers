<!--Copyright 2025 OpenMOSS and The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
http://www.apache.org/licenses/LICENSE-2.0
-->


# MOSS-TTSD

<div style="float: right;">
<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
</div>
</div>

## Overview

MOSS-TTSD (text to spoken dialogue) is an open-source bilingual spoken dialogue synthesis model that supports both Chinese and English.

It can transform dialogue scripts between two speakers into natural, expressive conversational speech.

MOSS-TTSD supports voice cloning and long single-session speech generation, making it ideal for AI podcast production, interviews, and chats.

For detailed information about the model and demos, please refer to our [Blog-en](https://www.open-moss.com/en/moss-ttsd/) and [中文博客](https://www.open-moss.com/cn/moss-ttsd/). You can also find the model on [Hugging Face](https://huggingface.co/fnlp/MOSS-TTSD-v0.5) and try it out in the [Spaces demo](https://huggingface.co/spaces/fnlp/MOSS-TTSD).

## Highlights

- **Highly Expressive Dialogue Speech**: Built on unified semantic-acoustic neural audio codec, a pre-trained large language model, millions of hours of TTS data, and 400k hours synthetic and real conversational speech, MOSS-TTSD generates highly expressive, human-like dialogue speech with natural conversational prosody.
- **Two-Speaker Voice Cloning**: MOSS-TTSD supports zero-shot two speakers voice cloning and can generate conversational speech with accurate speaker swithcing based on dialogue scripts. Only 10 to 20 seconds of reference audio is needed.
- **Chinese-English Bilingual Support**: MOSS-TTSD enables highly expressive speech generation in both Chinese and English.
- **Long-Form Speech Generation**: Thanks to low-bitrate codec and training framework optimization, MOSS-TTSD has been trained for long speech generation (Training maximum length is 960s).
- **Fully Open Source & Commercial-Ready**: MOSS-TTSD and its future updates will be fully open-source and support free commercial use.


## Usage

```python
import os
import torch
import requests
import hashlib
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.moss_ttsd.processing_moss_ttsd import MossTTSDProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

def download_audio_file(url):
    os.makedirs("example_audios", exist_ok=True)
    local_path = os.path.join("example_audios", f"audio_{hashlib.md5(url.encode()).hexdigest()[:8]}.wav")
    if os.path.exists(local_path):
        return local_path
    response = requests.get(url)
    response.raise_for_status()
    with open(local_path, 'wb') as f:
        f.write(response.content)
    return local_path

processor = MossTTSDProcessor.from_pretrained(
    "fnlp/MOSS-TTSD-v0.5",
    audio_tokenizer_path="fnlp/XY_Tokenizer_TTSD_V0_hf"
)
tokenizer = AutoTokenizer.from_pretrained("fnlp/MOSS-TTSD-v0.5")
model = AutoModelForCausalLM.from_pretrained("fnlp/MOSS-TTSD-v0.5").to(device).eval()

data = [
    { # Case 1: Text Generation (No voice cloning)
        "text": "人工智能浪潮正在席卷全球，给我们带来深刻变化",
        "system_prompt": "You are a speech synthesizer that generates natural, realistic, and human-like conversational audio from dialogue text.",
    },
    { # Case 2: Single Voice Cloning - clone one speaker's voice from reference audio
        "text": "人工智能浪潮正在席卷全球，给我们带来深刻变化",
        "system_prompt": "You are a speech synthesizer that generates natural, realistic, and human-like conversational audio from dialogue text.",
        "prompt_text": "周一到周五，每天早晨七点半到九点半的直播片段，言下之意呢就是废话有点儿多，大家也别嫌弃，因为这都是直播间最真实的状态",
        "prompt_audio": download_audio_file("https://raw.githubusercontent.com/OpenMOSS/MOSS-TTSD/main/examples/zh_spk1_moon.wav"),
    },
    { # Case 3: Dual Dialogue Voice Cloning - generate dialogue with two different speaker voices
        "text": "[S1]你听说了吗，人工智能现在变得非常厉害！[S2]是啊，我听说现在TTS模型生成的声音已经非常逼真了",
        "system_prompt": "You are a speech synthesizer that generates natural, realistic, and human-like conversational audio from dialogue text.",
        "prompt_audio_speaker1": download_audio_file("https://raw.githubusercontent.com/OpenMOSS/MOSS-TTSD/main/examples/zh_spk1_moon.wav"),
        "prompt_text_speaker1": "周一到周五，每天早晨七点半到九点半的直播片段，言下之意呢就是废话有点儿多，大家也别嫌弃，因为这都是直播间最真实的状态",
        "prompt_audio_speaker2": download_audio_file("https://raw.githubusercontent.com/OpenMOSS/MOSS-TTSD/main/examples/zh_spk2_moon.wav"),
        "prompt_text_speaker2": "如果大家想听到更丰富、更及时的直播内容，记得在周一到周五准时进入直播间，和大家一起，畅聊新消费、新科技、新趋势"
    }
]

inputs = processor(data)

token_ids = model.generate(
    input_ids=inputs["input_ids"].to(device), 
    attention_mask=inputs["attention_mask"].to(device), 
    tokenizer=tokenizer,
    do_sample=True,
    temperature=0.7,
    top_p=0.8
)

text, audios = processor.batch_decode(token_ids)

processor.save_audio(audios, output_dir="output", prefix="audio")
```


## MossTTSDConfig

[[autodoc]] MossTTSDConfig

## MossTTSDModel

[[autodoc]] MossTTSDModel
    - forward

## MossTTSDForCausalLM

[[autodoc]] MossTTSDForCausalLM
    - forward

## MossTTSDProcessor

[[autodoc]] MossTTSDProcessor
    - __call__
    - from_pretrained
    - save_pretrained
