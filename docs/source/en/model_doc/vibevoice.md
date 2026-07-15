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
*This model was published in HF papers on 2025-08-26 and contributed to Hugging Face Transformers on 2026-07-15.*


# VibeVoice

<div class="flex flex-wrap space-x-1">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

[VibeVoice](https://huggingface.co/papers/2508.19205) is a novel framework for synthesizing high-fidelity, long-form speech with multiple speakers by employing a next-token diffusion approach within a Large Language Model (LLM) structure. It's designed to capture the authentic conversational "vibe" and is particularly suited for generating audio content like podcasts and multi-participant audiobooks.

Two model checkpoint are available at:
- [bezzam/VibeVoice-1.5B-hf](https://huggingface.co/bezzam/VibeVoice-1.5B-hf)
- [bezzam/VibeVoice-7B-hf](https://huggingface.co/bezzam/VibeVoice-7B-hf)

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

A noise scheduler is needed as audio generation relies on a diffusion process. The easiest approach (and as done by the model developers) is to use a noise scheduler from the `diffusers` library. By default, the model will create a noise scheduler with `diffusers` internally.
```
pip install diffusers
pip install soundfile   # for saving audio
```

### Loading the model

```python
from transformers import AutoProcessor, AutoModelForTextToWaveform

model_id = "bezzam/VibeVoice-1.5B-hf"  # "bezzam/VibeVoice-7B-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForTextToWaveform.from_pretrained(model_id)
```

### Text-to-speech (TTS)

```python
import os
from transformers import AutoProcessor, AutoModelForTextToWaveform

model_id = "bezzam/VibeVoice-1.5B-hf"   # "bezzam/VibeVoice-7B-hf"
text = "Hello, nice to meet you. How are you?"

# Load model
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForTextToWaveform.from_pretrained(model_id, device_map="auto")

# Prepare input
chat_template = [{"role": "0", "content": [{"type": "text", "text": text}]}]
inputs = processor.apply_chat_template(
    chat_template, return_dict=True, tokenize=True,
).to(model.device, model.dtype)

# Generate!
audio = model.generate(**inputs)

# Save to file
file_name = f"{os.path.basename(model_id)}_tts.wav"
processor.save_audio(audio, file_name)
print(f"Saved output to {file_name}")
```

### TTS voice cloning

A voice can be cloned by providing a reference audio alongside the text within the chat template dictionary.

```python
import os
from transformers import AutoProcessor, AutoModelForTextToWaveform, set_seed

model_id = "bezzam/VibeVoice-1.5B-hf"   # "bezzam/VibeVoice-7B-hf"
text = "Hello, nice to meet you. How are you?"
set_seed(42)  # for deterministic results

# Load model
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForTextToWaveform.from_pretrained(model_id, device_map="auto")
sampling_rate = processor.feature_extractor.sampling_rate

# Prepare input
chat_template = [
    {
        "role": "0",
        "content": [
            {"type": "text", "text": text},
            {
                "type": "audio",
                "url": "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Alice_woman.wav",
            },
        ],
    }
]
inputs = processor.apply_chat_template(
    chat_template, return_dict=True, tokenize=True,
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
from transformers import AutoProcessor, AutoModelForTextToWaveform

model_id = "bezzam/VibeVoice-1.5B-hf"   # "bezzam/VibeVoice-7B-hf"
max_new_tokens = 400  # `None` to ensure full generation

# create conversation with an audio for the first time a speaker appears to clone that particular voice
chat_template = [
    {
        "role": "0",
        "content": [
            {
                "type": "text", "text": "Hello everyone, and welcome to the VibeVoice podcast. I'm your host, Linda, and today we're getting into one of the biggest debates in all of sports: who's the greatest basketball player of all time? I'm so excited to have Thomas here to talk about it with me.",
            },
            {
                "type": "audio", "url": "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Alice_woman.wav",
            },
        ],
    },
    {
        "role": "1",
        "content": [
            {
                "type": "text", "text": "Thanks so much for having me, Linda. You're absolutely right—this question always brings out some seriously strong feelings.",
            },
            {
                "type": "audio", "url": "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Frank_man.wav",
            },
        ],
    },
    {
        "role": "0",
        "content": [
            {
                "type": "text", "text": "Okay, so let's get right into it. For me, it has to be Michael Jordan. Six trips to the Finals, six championships. That kind of perfection is just incredible.",
            },
        ],
    },
    {
        "role": "1",
        "content": [
            {
                "type": "text", "text": "Oh man, the first thing that always pops into my head is that shot against the Cleveland Cavaliers back in '89. Jordan just rises, hangs in the air forever, and just sinks it",
            },
        ],
    },
]

# Load model
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForTextToWaveform.from_pretrained(model_id, device_map="auto")

# prepare inputs
inputs = processor.apply_chat_template(
    chat_template, return_dict=True, tokenize=True,
).to(model.device, model.dtype)

# Generate audio with a progress bar to track generation
model.generation_config.max_new_tokens = max_new_tokens
start_time = time.time()
audio = model.generate(**inputs, monitor_progress=True)
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
from transformers import AutoProcessor, AutoModelForTextToWaveform

model_id = "bezzam/VibeVoice-1.5B-hf"   # "bezzam/VibeVoice-7B-hf"
max_new_tokens = 400  # `None` to ensure full generation

chat_template = [
    [
        {
            "role": "0",
            "content": [
                {
                    "type": "text", "text": "Hello everyone, and welcome to the VibeVoice podcast. I'm your host, Linda, and today we're getting into one of the biggest debates in all of sports: who's the greatest basketball player of all time? I'm so excited to have Thomas here to talk about it with me.",
                },
                {
                    "type": "audio",
                    "url": "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Alice_woman.wav",
                },
            ],
        },
        {
            "role": "1",
            "content": [
                {
                    "type": "text", "text": "Thanks so much for having me, Linda.",
                },
                {
                    "type": "audio", "url": "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Frank_man.wav",
                },
            ],
        },
    ],
    [
        {
            "role": "0",
            "content": [
                {
                    "type": "text", "text": "Hello and welcome to Planet in Peril. I'm your host, Alice. We're here today to discuss a really sobering new report that looks back at the last ten years of climate change. I'm joined by our expert panel. Welcome Carter, Frank, and Maya.",
                },
                {
                    "type": "audio", "url": "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Alice_woman.wav",
                },
            ],
        },
        {
            "role": "1",
            "content": [
                {"type": "text", "text": "Hi Alice, it's great to be here. I'm Carter."},
                {
                    "type": "audio", "url": "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Carter_man.wav",
                },
            ],
        },
        {
            "role": "2",
            "content": [
                {"type": "text", "text": "Hello, uh, I'm Frank. Good to be on."},
                {
                    "type": "audio", "url": "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Frank_man.wav",
                },
            ],
        },
        {
            "role": "3",
            "content": [
                {"type": "text", "text": "And I'm Maya. Thanks for having me."},
                {
                    "type": "audio", "url": "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Maya_woman.wav",
                },
            ],
        },
    ],
]

# Load model
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForTextToWaveform.from_pretrained(model_id, device_map="auto")

# prepare inputs
inputs = processor.apply_chat_template(
    chat_template, return_dict=True, tokenize=True,
).to(model.device, model.dtype)

# Generate audio with a progress bar to track generation
model.generation_config.max_new_tokens = max_new_tokens
start_time = time.time()
audio = model.generate(**inputs, monitor_progress=True)
generation_time = time.time() - start_time
print(f"Generation time: {generation_time:.2f} seconds")

# Save audio
output_dir = f"{os.path.basename(model_id)}_batch"
processor.save_audio(audio, output_dir)
print(f"Saved output to {output_dir}")
```

### Pipeline usage

VibeVoice can also be loaded as a pipeline. We also show below how the diffusion parameters can be adjusted.

```python
import os
import soundfile as sf
from transformers import pipeline

model_id = "bezzam/VibeVoice-1.5B-hf"   # "bezzam/VibeVoice-7B-hf"
text = "Hello, nice to meet you. How are you?"
pipe = pipeline("text-to-speech", model=model_id)

# Generate!
chat_template = [
    {
        "role": "0",
        "content": [
            {"type": "text", "text": text},
            {
                "type": "audio", "url": "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Alice_woman.wav",
            },
        ],
    }
]
# optional kwargs for generation
generate_kwargs = {"guidance_scale": 1.3, "num_diffusion_steps": 10}
output = pipe(chat_template, generate_kwargs=generate_kwargs)

# Save to file
fn = f"{os.path.basename(model_id)}_pipeline.wav"
sf.write(fn, output["audio"], output["sampling_rate"])
print(f"Saved output to {fn}")
```

### Training

VibeVoice can be trained with the loss outputted by the model.

```python
from transformers import AutoProcessor, AutoModelForTextToWaveform


model_id = "bezzam/VibeVoice-1.5B-hf"   # "bezzam/VibeVoice-7B-hf"

# Load model and processor
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForTextToWaveform.from_pretrained(model_id, device_map="auto")
model.train()

# Prepare batch of 2
chat_template = [
    [
        {
            "role": "0",
            "content": [
                {
                    "type": "text", "text": "VibeVoice is this novel framework designed for generating expressive, long-form, multi-speaker, conversational audio.",
                },
                {
                    "type": "audio", "url": "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/realtime_model/vibevoice_tts_german.wav",
                },
            ],
        }
    ],
    # NOTE: multiple speakers not supported yet
    [
        {
            "role": "0",
            "content": [
                {
                    "type": "text", "text": "Hello everyone and welcome to the VibeVoice podcast. I'm your host, Alex, and today we're getting into one of the biggest debates in all of sports: who's the greatest basketball player of all time? I'm so excited to have Sam here to talk about it with me. Thanks so much for having me, Alex. And you're absolutely right. This question always brings out some seriously strong feelings. Okay, so let's get right into it. For me, it has to be Michael Jordan. Six trips to the finals, six championships. That kind of perfection is just incredible. Oh man, the first thing that always pops into my head is that shot against the Cleveland Cavaliers back in '89. Jordan just rises, hangs in the air forever, and just sinks it.",
                },
                {
                    "type": "audio",
                    "url": "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/example_output/VibeVoice-1.5B_output.wav",
                },
            ],
        }
    ],
]

# Process with apply_chat_template and output_labels=True for training
inputs = processor.apply_chat_template(
    chat_template,
    tokenize=True,
    return_dict=True,
    processor_kwargs={"output_labels": True},
).to(model.device, model.dtype)

# Forward pass
outputs = model(**inputs, ddpm_batch_multiplier=2, num_diffusion_steps=2)

# Compute loss as simple sum, but they can be weighted differently
lm_loss = outputs.loss
diffusion_loss = outputs.diffusion_loss
total_loss = lm_loss + diffusion_loss

print(f"LM loss: {lm_loss.item():.4f}")
print(f"Diffusion loss: {diffusion_loss.item():.4f}")
print(f"Total loss: {total_loss.item():.4f}")

# Backward pass
total_loss.backward()
```

### Torch compile

The model can be compiled with `torch.compile` for faster inference. A few warmup runs are needed before the compiled model reaches full speed.

On an A100 with batch size 4, we observed a ~1.5x speed-up between compiled vs. non-compiled inference, see [this script](https://gist.github.com/ebezzam/c45b9fdee65f3029e17d566e30c59399).

```python
import os
import time
import torch
from transformers import AutoModelForTextToWaveform, AutoProcessor, CompileConfig


model_id = "bezzam/VibeVoice-1.5B-hf"   # "bezzam/VibeVoice-7B-hf"
num_warmup = 5
max_new_tokens = 128

torch.set_float32_matmul_precision("high")

# Load processor + model
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForTextToWaveform.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto").eval()

# Prepare inputs
chat_template = [
    [
        {
            "role": "0",
            "content": [
                {"type": "text", "text": "VibeVoice is a novel framework for generating expressive audio."},
                {
                    "type": "audio", "url": "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/realtime_model/vibevoice_tts_german.wav",
                },
            ],
        }
    ],
] * 4  # batch size 4
inputs = processor.apply_chat_template(
    chat_template, tokenize=True, return_dict=True,
).to(model.device, model.dtype)

compile_config = CompileConfig(mode="default" dynamic=False)

generate_kwargs = dict(
    **inputs,
    max_new_tokens=max_new_tokens,
    cache_implementation="static",
    compile_config=compile_config,
)

# Warmup
print("Warming up...")
warmup_start = time.time()
with torch.inference_mode():
    for _ in range(num_warmup):
        torch.compiler.cudagraph_mark_step_begin()
        _ = model.generate(**generate_kwargs)
torch.cuda.synchronize()
print(f"Warmup complete in {time.time() - warmup_start:.2f}s. Ready!")

# Apply model
with torch.inference_mode():
    torch.compiler.cudagraph_mark_step_begin()
    audio = model.generate(**generate_kwargs)
fn = f"{os.path.basename(model_id)}_compiled_output.wav"
processor.save_audio(audio, fn)
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
