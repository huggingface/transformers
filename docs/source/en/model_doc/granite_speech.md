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
*This model was released on 2025-04-16 and added to Hugging Face Transformers on 2025-04-11.*

# Granite Speech

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The [Granite Speech](https://huggingface.co/papers/2505.08699) model ([blog post](https://www.ibm.com/new/announcements/ibm-granite-3-3-speech-recognition-refined-reasoning-rag-loras)) is a multimodal language model, consisting of a speech encoder, speech projector, large language model, and LoRA adapter(s). More details regarding each component for the current (Granite 3.2 Speech) model architecture may be found below.

1. Speech Encoder: A [Conformer](https://huggingface.co/papers/2005.08100) encoder trained with Connectionist Temporal Classification (CTC) on character-level targets on ASR corpora. The encoder uses block-attention and self-conditioned CTC from the middle layer.

2. Speech Projector: A query transformer (q-former) operating on the outputs of the last encoder block. The encoder and projector temporally downsample the audio features to be merged into the multimodal embeddings to be processed by the llm.

3. Large Language Model: The Granite Speech model leverages Granite LLMs, which were originally proposed in [this paper](https://huggingface.co/papers/2408.13359).

4. LoRA adapter(s): The Granite Speech model contains a modality specific LoRA, which will be enabled when audio features are provided, and disabled otherwise.

Note that most of the aforementioned components are implemented generically to enable compatibility and potential integration with other model architectures in transformers.

This model was contributed by [Alexander Brooks](https://huggingface.co/abrooks9944), [Avihu Dekel](https://huggingface.co/Avihu), and [George Saon](https://huggingface.co/gsaon).

## Usage tips

- This model bundles its own LoRA adapter, which will be automatically loaded and enabled/disabled as needed during inference calls. Be sure to install [PEFT](https://github.com/huggingface/peft) to ensure the LoRA is correctly applied!

## Usage Example

Granite Speech is a multimodal model that can process both text and audio inputs for speech-to-text transcription and audio understanding tasks. Here's how to use it:

```python
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from datasets import load_dataset


# Load model and processor
model_id = "ibm-granite/granite-speech-3.3-8b"
processor = AutoProcessor.from_pretrained(model_id)
tokenizer = processor.tokenizer
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, device_map="auto", dtype="auto"
)

# Load audio from dummy dataset
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
wav = dataset[0]["audio"].get_all_samples().data.unsqueeze(0)  # add batch dimension

# Create chat conversation with audio
system_prompt = "Knowledge Cutoff Date: April 2024.\nToday's Date: April 9, 2025.\nYou are Granite, developed by IBM. You are a helpful AI assistant"
user_prompt = "<|audio|>can you transcribe the speech into a written format?"
chat = [
    dict(role="system", content=system_prompt),
    dict(role="user", content=user_prompt),
]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

# Process audio and text together
model_inputs = processor(prompt, wav, device=device, return_tensors="pt").to(device)

# Generate response
model_outputs = model.generate(**model_inputs, max_new_tokens=200, do_sample=False, num_beams=1)

# Extract only the new tokens (response)
num_input_tokens = model_inputs["input_ids"].shape[-1]
new_tokens = torch.unsqueeze(model_outputs[0, num_input_tokens:], dim=0)
output_text = tokenizer.batch_decode(
    new_tokens, add_special_tokens=False, skip_special_tokens=True
)
print(f"STT output = {output_text[0].upper()}")
```

## GraniteSpeechConfig

[[autodoc]] GraniteSpeechConfig

## GraniteSpeechEncoderConfig

[[autodoc]] GraniteSpeechEncoderConfig

## GraniteSpeechProcessor

[[autodoc]] GraniteSpeechProcessor

## GraniteSpeechFeatureExtractor

[[autodoc]] GraniteSpeechFeatureExtractor

## GraniteSpeechForConditionalGeneration

[[autodoc]] GraniteSpeechForConditionalGeneration
    - forward
