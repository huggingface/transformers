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
- The model expects 16kHz sampling rate audio. The processor will automatically resample if needed.
- The LoRA adapter is automatically enabled when audio features are present and disabled for text-only inputs, so you don't need to manage it manually.

## Usage example

Granite Speech is a multimodal speech-to-text model that can transcribe audio and respond to text prompts. Here's how to use it:

### Basic Speech Transcription

```python
from transformers import GraniteSpeechForConditionalGeneration, GraniteSpeechProcessor
from datasets import load_dataset, Audio
import torch

# Load model and processor
model = GraniteSpeechForConditionalGeneration.from_pretrained(
    "ibm-granite/granite-3.2-8b-speech",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = GraniteSpeechProcessor.from_pretrained("ibm-granite/granite-3.2-8b-speech")

# Load audio from dataset (16kHz sampling rate required)
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
audio = ds['audio'][0]['array']

# Process audio
inputs = processor(audio=audio, return_tensors="pt").to(model.device)

# Generate transcription
generated_ids = model.generate(**inputs, max_new_tokens=256)
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(transcription)
```

### Speech-to-Text with Chat Template

For instruction-following with audio, use the chat template with audio directly in the conversation format:

```python
from transformers import GraniteSpeechForConditionalGeneration, GraniteSpeechProcessor
from datasets import load_dataset, Audio
import torch

model = GraniteSpeechForConditionalGeneration.from_pretrained(
    "ibm-granite/granite-3.2-8b-speech",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = GraniteSpeechProcessor.from_pretrained("ibm-granite/granite-3.2-8b-speech")

# Load audio from dataset
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
audio = ds['audio'][0]

# Prepare conversation with audio and text
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio},
            {"type": "text", "text": "Transcribe the following audio:"},
        ],
    }
]

# Apply chat template with audio - processor handles both tokenization and audio processing
inputs = processor.apply_chat_template(conversation, tokenize=True, return_tensors="pt").to(model.device)

# Generate transcription
generated_ids = model.generate(**inputs, max_new_tokens=512)
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(output_text)
```

### Batch Processing

Process multiple audio files efficiently:

```python
from transformers import GraniteSpeechForConditionalGeneration, GraniteSpeechProcessor
from datasets import load_dataset, Audio
import torch

model = GraniteSpeechForConditionalGeneration.from_pretrained(
    "ibm-granite/granite-3.2-8b-speech",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = GraniteSpeechProcessor.from_pretrained("ibm-granite/granite-3.2-8b-speech")

# Load multiple audio samples from dataset
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
audio_samples = [ds['audio'][i]['array'] for i in range(3)]

# Process batch
inputs = processor(audio=audio_samples, return_tensors="pt", padding=True).to(model.device)

# Generate for all inputs
generated_ids = model.generate(**inputs, max_new_tokens=256)
transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)

for i, transcription in enumerate(transcriptions):
    print(f"Audio {i+1}: {transcription}")
```

## GraniteSpeechConfig

[[autodoc]] GraniteSpeechConfig

## GraniteSpeechEncoderConfig

[[autodoc]] GraniteSpeechEncoderConfig

## GraniteSpeechProcessor

[[autodoc]] GraniteSpeechProcessor
    - __call__

## GraniteSpeechFeatureExtractor

[[autodoc]] GraniteSpeechFeatureExtractor

## GraniteSpeechForConditionalGeneration

[[autodoc]] GraniteSpeechForConditionalGeneration
    - forward
