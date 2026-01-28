<!--Copyright 2026 Microsoft and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2025-08-26 and added to Hugging Face Transformers on 2025-12-09.*

# VibeVoice

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

[VibeVoice](https://huggingface.co/papers/2508.19205) is a novel framework for synthesizing high-fidelity, long-form speech with multiple speakers by employing a next-token diffusion approach within a Large Language Model (LLM) structure. It's designed to capture the authentic conversational "vibe" and is particularly suited for generating audio content like podcasts and multi-participant audiobooks.

Two model checkpoint are available at:
- [bezzam/VibeVoice-1.5B](https://huggingface.co/bezzam/VibeVoice-1.5B)
- [bezzam/VibeVoice-7B](https://huggingface.co/bezzam/VibeVoice-7B)

This model was contributed by [Eric Bezzam](https://huggingface.co/bezzam).

## Architecture

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/bezzam/documentation-images/resolve/main/vibevoice_arch.png"/>
</div>

The VibeVoice framework integrates three key components:
1. **Continuous Speech Tokenizers:** Specialized [acoustic](./vibevoice_acoustic_tokenizer) and [semantic](./vibevoice_semantic_tokenizer) tokenizers, where the acoustic tokenizer uses a $\sigma$-VAE to achieve ultra-low compression (7.5 tokens/sec, 3200x) for scalability and fidelity, and the semantic tokenizer uses an ASR proxy task for content-centric feature extraction.
2. **Large Language Model (LLM):** Use Qwen2.5 (in 1.5B and 7B versions) as its core sequence model.
3. **Token-Level Diffusion Head:** conditioned on the LLM's hidden state and responsible for predicting the continuous VAE features in a streaming fashion.

The original VibeVoice-1.5B checkpoint is available under the [Microsoft](https://huggingface.co/microsoft/VibeVoice-1.5B) organization on Hugging Face.


## Key Features

- **Long-Form Synthesis**: Can synthesize up to 90 minutes multi-speaker conversational speech.
- **Multi-Speaker Dialogue**: Capable of synthesizing audio with a maximum of 4 speakers.
- **State-of-the-Art Quality**: Outperforms baselines on both subjective and objective metrics.
- **High Compression**: Achieved by a novel acoustic tokenizer operating at an ultra-low 7.5 Hz frame rate.
- **Scalable LLM**: Scaling the core LLM from 1.5B to 7B significantly improves perceptual quality.


## Usage

### Setup 

The `diffusers` library is needed as a diffusion process is used to generate chunks of audio.
```
pip install diffusers
pip install soundfile   # for saving audio
```

### Loading the model

```python
from transformers import AutoProcessor, VibeVoiceForConditionalGeneration

model_id = "bezzam/VibeVoice-1.5B"
# model_id = "bezzam/VibeVoice-7B"
processor = AutoProcessor.from_pretrained(model_id)
model = VibeVoiceForConditionalGeneration.from_pretrained(model_id)
```

### Text-to-speech (TTS) example

```python
import os
from transformers import AutoProcessor, VibeVoiceForConditionalGeneration, set_seed


model_id = "bezzam/VibeVoice-1.5B"
# model_id = "bezzam/VibeVoice-7B"
text = "Hello, nice to meet you. How are you?"
set_seed(42)  # for deterministic results

# Load model
processor = AutoProcessor.from_pretrained(model_id)
model = VibeVoiceForConditionalGeneration.from_pretrained(model_id, device_map="auto")
sampling_rate = processor.feature_extractor.sampling_rate

# Prepare input
chat_template = [{"role": "0", "content": [{"type": "text", "text": text}]}]
inputs = processor.apply_chat_template(
    chat_template,
    tokenize=True,
    return_dict=True,
).to(model.device, model.dtype)

# Generate!
audio = model.generate(**inputs)

# Save to file
fn = f"{os.path.basename(model_id)}_tts.wav"
processor.save_audio(audio, fn)
print(f"Saved output to {fn}")
```

### TTS voice cloning example

A voice can be cloned by providing a reference audio alongside the text within the chat template dictionary.

A url (`url`), local path (`path`), or loaded audio array (`audio`) can be provided as a reference audio.

```python
import os
from transformers import AutoProcessor, VibeVoiceForConditionalGeneration, set_seed


model_id = "bezzam/VibeVoice-1.5B"
# model_id = "bezzam/VibeVoice-7B"
text = "Hello, nice to meet you. How are you?"
set_seed(42)  # for deterministic results

# Load model
processor = AutoProcessor.from_pretrained(model_id)
model = VibeVoiceForConditionalGeneration.from_pretrained(model_id, device_map="auto")
sampling_rate = processor.feature_extractor.sampling_rate

# Prepare input
chat_template = [
    {
        "role": "0",
        "content": [
            {"type": "text", "text": text},
            {
                "type": "audio",
                "url": "https://hf.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Alice_woman.wav",
            },
        ],
    }
]
inputs = processor.apply_chat_template(
    chat_template,
    tokenize=True,
    return_dict=True,
    sampling_rate=sampling_rate,
).to(model.device, model.dtype)

# Generate!
audio = model.generate(**inputs)

# Save to file
fn = f"{os.path.basename(model_id)}_tts_clone.wav"
processor.save_audio(audio, fn)
print(f"Saved output to {fn}")
```

### Generating a podcast from a script

Below is an example to generate a conversation between two speakers, whose voices are cloned by providing a refence audio for each unique role ID in the chat template.

The example below also used the `monitor_progress` option to track the generation progress.

```python
import os
import time
import torch
from tqdm import tqdm
from transformers import AutoProcessor, VibeVoiceForConditionalGeneration, set_seed


model_id = "bezzam/VibeVoice-1.5B"
# model_id = "bezzam/VibeVoice-7B"
max_new_tokens = 400  # `None` to ensure full generation
set_seed(42)  # for deterministic results

# create conversation with an audio for the first time a speaker appears to clone that particular voice
chat_template = [
    {
        "role": "0",
        "content": [
            {
                "type": "text",
                "text": "Hello everyone, and welcome to the VibeVoice podcast. I'm your host, Linda, and today we're getting into one of the biggest debates in all of sports: who's the greatest basketball player of all time? I'm so excited to have Thomas here to talk about it with me.",
            },
            {
                "type": "audio",
                "url": "https://hf.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Alice_woman.wav",
            },
        ],
    },
    {
        "role": "1",
        "content": [
            {
                "type": "text",
                "text": "Thanks so much for having me, Linda. You're absolutely right—this question always brings out some seriously strong feelings.",
            },
            {
                "type": "audio",
                "url": "https://hf.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Frank_man.wav",
            },
        ],
    },
    {
        "role": "0",
        "content": [
            {
                "type": "text",
                "text": "Okay, so let's get right into it. For me, it has to be Michael Jordan. Six trips to the Finals, six championships. That kind of perfection is just incredible.",
            },
        ],
    },
    {
        "role": "1",
        "content": [
            {
                "type": "text",
                "text": "Oh man, the first thing that always pops into my head is that shot against the Cleveland Cavaliers back in '89. Jordan just rises, hangs in the air forever, and just sinks it",
            },
        ],
    },
]

# Load model
processor = AutoProcessor.from_pretrained(model_id)
model = VibeVoiceForConditionalGeneration.from_pretrained(model_id, device_map="auto")
sampling_rate = processor.feature_extractor.sampling_rate

# prepare inputs
inputs = processor.apply_chat_template(
    chat_template, 
    tokenize=True, 
    return_dict=True, 
    sampling_rate=sampling_rate
).to(model.device, model.dtype)

# Generate audio with a callback to track progress
start_time = time.time()
completed_samples = set()
with tqdm(desc="Generating") as pbar:

    def monitor_progress(p_batch):
        # p_batch format: [current_step, max_step] for each sample
        active_samples = p_batch[:, 0] < p_batch[:, 1]
        if active_samples.any():
            active_progress = p_batch[active_samples]
            max_active_idx = torch.argmax(active_progress[:, 0])
            p = active_progress[max_active_idx].detach().cpu()
        else:
            p = p_batch[0].detach().cpu()

        pbar.total = int(p[1])
        pbar.n = int(p[0])
        pbar.update()

    audio = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        monitor_progress=monitor_progress,
    )
generation_time = time.time() - start_time
print(f"Generation time: {generation_time:.2f} seconds")

# Save audio
fn = f"{os.path.basename(model_id)}_script.wav"
processor.save_audio(audio, fn)
print(f"Saved output to {fn}")
```

### Batched inference

For batch processing, a list of conversations can be passed to `processor.apply_chat_template`: 

```python
import os
import time
import torch
from tqdm import tqdm
from transformers import AutoProcessor, VibeVoiceForConditionalGeneration, set_seed


model_id = "bezzam/VibeVoice-1.5B"
# model_id = "bezzam/VibeVoice-7B"
max_new_tokens = 400  # `None` to ensure full generation
set_seed(42)  # for deterministic results

chat_template = [
    [
        {
            "role": "0",
            "content": [
                {
                    "type": "text",
                    "text": "Hello everyone, and welcome to the VibeVoice podcast. I'm your host, Linda, and today we're getting into one of the biggest debates in all of sports: who's the greatest basketball player of all time? I'm so excited to have Thomas here to talk about it with me.",
                },
                {
                    "type": "audio",
                    "url": "https://hf.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Alice_woman.wav",
                },
            ],
        },
        {
            "role": "1",
            "content": [
                {
                    "type": "text",
                    "text": "Thanks so much for having me, Linda. You're absolutely right—this question always brings out some seriously strong feelings.",
                },
                {
                    "type": "audio",
                    "url": "https://hf.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Frank_man.wav",
                },
            ],
        },
        {
            "role": "0",
            "content": [
                {
                    "type": "text",
                    "text": "Okay, so let's get right into it. For me, it has to be Michael Jordan. Six trips to the Finals, six championships. That kind of perfection is just incredible.",
                },
            ],
        },
        {
            "role": "1",
            "content": [
                {
                    "type": "text",
                    "text": "Oh man, the first thing that always pops into my head is that shot against the Cleveland Cavaliers back in '89. Jordan just rises, hangs in the air forever, and just sinks it",
                },
            ],
        },
    ],
    [
        {
            "role": "0",
            "content": [
                {
                    "type": "text",
                    "text": "Hello and welcome to Planet in Peril. I'm your host, Alice. We're here today to discuss a really sobering new report that looks back at the last ten years of climate change, from 2015 to 2025. It paints a picture not just of steady warming, but of a dangerous acceleration. And to help us unpack this, I'm joined by our expert panel. Welcome Carter, Frank, and Maya.",
                },
                {
                    "type": "audio",
                    "url": "https://hf.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Alice_woman.wav",
                },
            ],
        },
        {
            "role": "1",
            "content": [
                {"type": "text", "text": "Hi Alice, it's great to be here. I'm Carter."},
                {
                    "type": "audio",
                    "url": "https://hf.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Carter_man.wav",
                },
            ],
        },
        {
            "role": "2",
            "content": [
                {"type": "text", "text": "Hello, uh, I'm Frank. Good to be on."},
                {
                    "type": "audio",
                    "url": "https://hf.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Frank_man.wav",
                },
            ],
        },
        {
            "role": "3",
            "content": [
                {"type": "text", "text": "And I'm Maya. Thanks for having me."},
                {
                    "type": "audio",
                    "url": "https://hf.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Maya_woman.wav",
                },
            ],
        },
    ],
]

# Load model
processor = AutoProcessor.from_pretrained(model_id)
model = VibeVoiceForConditionalGeneration.from_pretrained(model_id, device_map="auto")
sampling_rate = processor.feature_extractor.sampling_rate

# prepare inputs
inputs = processor.apply_chat_template(
    chat_template,
    return_dict=True,
    tokenize=True,
    sampling_rate=sampling_rate,
).to(model.device, model.dtype)

# Generate audio with a callback to track progress
start_time = time.time()
completed_samples = set()
with tqdm(desc="Generating") as pbar:

    def monitor_progress(p_batch):
        # p_batch format: [current_step, max_step] for each sample
        active_samples = p_batch[:, 0] < p_batch[:, 1]
        if active_samples.any():
            active_progress = p_batch[active_samples]
            max_active_idx = torch.argmax(active_progress[:, 0])
            p = active_progress[max_active_idx].detach().cpu()
        else:
            p = p_batch[0].detach().cpu()

        pbar.total = int(p[1])
        pbar.n = int(p[0])
        pbar.update()

    audio = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        monitor_progress=monitor_progress,
    )
generation_time = time.time() - start_time
print(f"Generation time: {generation_time:.2f} seconds")

# Save audio
output_dir = f"{os.path.basename(model_id)}_batch"
processor.save_audio(audio, output_dir)
print(f"Saved output to {output_dir}")
```

### Pipeline usage

VibeVoice can also be loaded as a pipeline:

```python
import os
import soundfile as sf
from transformers import pipeline, set_seed


model_id = "bezzam/VibeVoice-1.5B"
# model_id = "bezzam/VibeVoice-7B"
text = "Hello, nice to meet you. How are you?"
set_seed(42)  # for deterministic results

# Load pipeline
pipe = pipeline("text-to-speech", model=model_id)

# Generate!
chat_template = [
    {
        "role": "0",
        "content": [
            {"type": "text", "text": text},
            {
                "type": "audio",
                "url": "https://hf.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Alice_woman.wav",
            },
        ],
    }
]
generate_kwargs = {"cfg_scale": 1.5}
output = pipe(chat_template, generate_kwargs=generate_kwargs)

# Save to file
fn = f"{os.path.basename(model_id)}_pipeline.wav"
sf.write(fn, output["audio"], output["sampling_rate"])
print(f"Saved output to {fn}")
```

## VibeVoiceConfig

[[autodoc]] VibeVoiceConfig

## VibeVoiceProcessor

[[autodoc]] VibeVoiceProcessor
    - __call__

## VibeVoiceForConditionalGeneration

[[autodoc]] VibeVoiceForConditionalGeneration
    - forward
    - generate

## VibeVoiceModel

[[autodoc]] VibeVoiceModel

## VibeVoiceSemanticTokenizerConfig

[[autodoc]] VibeVoiceSemanticTokenizerConfig

## VibeVoiceSemanticTokenizerModel

[[autodoc]] VibeVoiceSemanticTokenizerModel

