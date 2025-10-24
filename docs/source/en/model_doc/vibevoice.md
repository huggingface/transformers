<!--Copyright 2025 The Microsoft Team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2025-08-25 and added to Hugging Face Transformers on 2025-08-25.*

# VibeVoice

## Overview

VibeVoice is a novel framework for synthesizing high-fidelity, long-form speech with multiple speakers by employing a next-token diffusion approach within a Large Language Model (LLM) structure. It's designed to capture the authentic conversational "vibe" and is particularly suited for generating audio content like podcasts and multi-participant audiobooks.

**Core Architecture**

The VibeVoice framework integrates three key components:
- **Speech Tokenizers:** Utilize specialized acoustic and semantic tokenizers, where the acoustic tokenizer uses a $\sigma$-VAE to achieve ultra-low compression (7.5 tokens/sec, 3200x) for scalability and fidelity, and the semantic tokenizer uses an ASR proxy task for content-centric feature extraction.
- **Large Language Model (LLM):** Use Qwen2.5 (in 1.5B and 7B versions) as its core sequence model.
- **Token-Level Diffusion Head:** condition on the LLM's hidden state and be responsible for predicting the continuous VAE features in a streaming way.


## Key Features

- **Long-Form Synthesis**: Can synthesize multi-speaker conversational speech for up to 90 minutes.
- **Multi-Speaker Dialogue**: Capable of synthesizing audio with a maximum of 4 speakers.
- **State-of-the-Art Quality**: Outperforms baselines on both subjective and objective metrics.
- **High Compression**: Achieved by a novel acoustic tokenizer operating at an ultra-low 7.5 Hz frame rate.
- **Scalable LLM**: Scaling the core LLM from 1.5B to 7B significantly improves perceptual quality.

## Usage
```python
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.processor.input_processor import input_process
import torch

model_path = "microsoft/VibeVoice-1.5b"
cfg_scale = "1.3"
generated_audio_name = "./outputs/generated.wav"
# Please follow the format below to add new scripts and voices
scripts = """Speaker 0: Hello, how are you?
Speaker 1: I'm fine, thank you! And you?
Speaker 0: I'm doing well, thanks for asking.
Speaker 1: That's great to hear. What have you been up to lately?
Speaker 0: Just working and spending time with family."""

# Only five voices for use
# 'en-Alice_woman', 'en-Ben_man', 'en-Carter_man', 'en-Maya_woman', 'in-Samuel_man'
# TODO: need to check voices later
voices = ["en-Alice_woman", "en-Ben_man"]

processor = VibeVoiceProcessor.from_pretrained(model_path)
model = VibeVoiceForConditionalGenerationInference.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map='cuda',
    attn_implementation="flash_attention_2"
)

model.eval()
model.set_ddpm_inference_steps(num_steps=10)

full_script, voice_samples = input_process(scripts, voices)

inputs = processor(
    text=[full_script], 
    voice_samples=[voice_samples], 
    padding=True,
    return_tensors="pt",
    return_attention_mask=True,
)

outputs = model.generate(
    **inputs,
    max_new_tokens=None,
    cfg_scale=cfg_scale,
    tokenizer=processor.tokenizer,
    generation_config={'do_sample': False},
    verbose=True,
)

processor.save_audio(
    outputs.speech_outputs[0],  # First (and only) batch item
    output_path=generated_audio_name,
)

```