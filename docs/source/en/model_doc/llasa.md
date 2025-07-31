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

# Llasa TTS

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

## Overview

Llasa comprises a set of open-source text-to-speech (TTS) model developed by researchers at [Prof. Wei Xue's lab](https://huggingface.co/HKUSTAudio) at the Hong Kong University of Science and Technology (HKUST).
It was proposed in their paper [Llasa: Scaling Train-Time and Inference-Time Compute for Llama-based Speech Synthesis](https://huggingface.co/papers/2502.04128).
Three models are available in different sizes.

Model cards (TODO ask them to do new checkpoints or different endpoint?)
- [1B](https://huggingface.co/HKUSTAudio/Llasa-1B)
- [3B](https://huggingface.co/HKUSTAudio/Llasa-3B)
- [8B](https://huggingface.co/HKUSTAudio/Llasa-8B)

This model was contributed by [Eric Bezzam](https://huggingface.co/bezzam).

**Model Architecture:**
Llasa is designed with the standard text LLM paradigm in mind, consisting of two main components: (1) a tokenizer and (2) a single Transformer-based LLM.

1. The tokenizer combines a standard text LLM tokenizer (for the input text) and a speech tokenizer for representing waveforms as speech tokens. To this end, the authors introduced [X-Codec2](./xcodec2), which employs a single codebook (unlike [DAC](./dac) and [EnCodec](./encodec)) for convenient and efficient conversion between speech tokens and audio waveforms.
2. The Transformer-based LLM is trained to handle both conventional text and speech tokens. It outputs speech tokens, which can be decoded by the speech tokenizer. The Llasa LLM is initialized with parameters from:
   - [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) for Llasa-1B  
   - [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) for Llasa-3B  
   - [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) for Llasa-8B

## Usage Tips

### Generation with Text

TODO: switch to using AutoProcessor? and use XCodec2 from transformers

```python
"""
pip install torchao xcodec2==0.1.3
"""

import torch
from transformers import LlasaTokenizer, LlasaForCausalLM, LlasaProcessor
import soundfile as sf
from xcodec2.modeling_xcodec2 import XCodec2Model

model_repo = "bezzam/Llasa-1B"
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# load tokenizer
processor = LlasaProcessor(
    LlasaTokenizer.from_pretrained(model_repo),
    XCodec2Model.from_pretrained("HKUSTAudio/xcodec2").eval().to(torch_device)
)

# load model
model = LlasaForCausalLM.from_pretrained(model_repo)
model.eval().to(torch_device)

# TTS
input_text = "How much wood would a woodchuck chuck if a woodchuck could chuck speech tokens?"
with torch.no_grad():

    # Tokenize the text
    encoded_text = processor(input_text).to(torch_device)

    # Generate the speech autoregressively
    outputs = model.generate(
        encoded_text["input_ids"],
        do_sample=False,
        max_length=600,    # up to ~10 seconds. Max allowed length is 2048, as Llasa was trained with max length 2048
        top_p=1,           # Adjusts the diversity of generated content
        temperature=0.8,   # Controls randomness in output
    )

# decode to audio
gen_wav = processor.decode(outputs, input_offset=encoded_text["input_offset"])
sf.write("llasa_out.wav", gen_wav.cpu().numpy(), 16000)
print("Generated speech saved to llasa_out.wav")
```

### Training

The original training code can be found [here](https://github.com/zhenye234/LLaSA_training).

An extendable training code, e.g., to train the codec or pick an LLM model other than Llama can be found [here](https://github.com/inworld-ai/tts).


## LlasaConfig

[[autodoc]] LlasaConfig

## LlasaTokenizer

[[autodoc]] LlasaTokenizer
    - __call__

## LlasaProcessor

[[autodoc]] LlasaProcessor
    - __call__
    - batch_decode
    - decode

## LlasaForCausalLM

[[autodoc]] LlasaForCausalLM
    - forward
    - generate
