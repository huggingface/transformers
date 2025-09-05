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
*This model was released on 2025-02-27 and added to Hugging Face Transformers on 2025-05-07.*

# Csm

## Overview

The Conversational Speech Model (CSM) is the first open-source contextual text-to-speech model [released by Sesame](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice). It is designed to generate natural-sounding speech with or without conversational context. This context typically consists of multi-turn dialogue between speakers, represented as sequences of text and corresponding spoken audio.

**Model Architecture:**
CSM is composed of two LLaMA-style auto-regressive transformer decoders: a backbone decoder that predicts the first codebook token and a depth decoder that generates the remaining tokens. It uses the pretrained codec model [Mimi](./mimi), introduced by Kyutai, to encode speech into discrete codebook tokens and decode them back into audio.

The original csm-1b checkpoint is available under the [Sesame](https://huggingface.co/sesame/csm-1b) organization on Hugging Face.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/eustlb/documentation-images/resolve/main/csm_architecture.png"/>
</div>

## Usage Tips

### Without Conversational Context

CSM can be used to simply generate speech from a text prompt:

```python
import torch
from transformers import CsmForConditionalGeneration, AutoProcessor, infer_device

model_id = "sesame/csm-1b"
device = infer_device()

# load the model and the processor
processor = AutoProcessor.from_pretrained(model_id)
model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=device)

# prepare the inputs
text = "[0]The past is just a story we tell ourselves." # `[0]` for speaker id 0
inputs = processor(text, add_special_tokens=True).to(device)

# another equivalent way to prepare the inputs
conversation = [
    {"role": "0", "content": [{"type": "text", "text": "The past is just a story we tell ourselves."}]},
]
inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
).to(model.device)

# infer the model
audio = model.generate(**inputs, output_audio=True)
processor.save_audio(audio, "example_without_context.wav")
```

### With Conversational Context

CSM can be used to generate speech given a conversation, allowing consistency in the voices and content-aware generation:

```python
import torch
from transformers import CsmForConditionalGeneration, AutoProcessor, infer_device
from datasets import load_dataset, Audio

model_id = "sesame/csm-1b"
device = infer_device()

# load the model and the processor
processor = AutoProcessor.from_pretrained(model_id)
model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=device)

# prepare the inputs
ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
# ensure the audio is 24kHz
ds = ds.cast_column("audio", Audio(sampling_rate=24000))
conversation = []

# 1. context
for text, audio, speaker_id in zip(ds[:4]["text"], ds[:4]["audio"], ds[:4]["speaker_id"]):
    conversation.append(
        {
            "role": f"{speaker_id}",
            "content": [{"type": "text", "text": text}, {"type": "audio", "path": audio["array"]}],
        }
    )

# 2. text prompt
conversation.append({"role": f"{ds[4]['speaker_id']}", "content": [{"type": "text", "text": ds[4]["text"]}]})

inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
).to(model.device)

# infer the model
audio = model.generate(**inputs, output_audio=True)
processor.save_audio(audio, "example_with_context.wav")
```

### Batched Inference

CSM supports batched inference!

```python
import torch
from transformers import CsmForConditionalGeneration, AutoProcessor, infer_device
from datasets import load_dataset, Audio

model_id = "sesame/csm-1b"
device = infer_device()

# load the model and the processor
processor = AutoProcessor.from_pretrained(model_id)
model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=device)

# prepare the inputs 
ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
# ensure the audio is 24kHz
ds = ds.cast_column("audio", Audio(sampling_rate=24000))
# here a batch with two prompts
conversation = [
    [
        {
            "role": f"{ds[0]['speaker_id']}",
            "content": [
                {"type": "text", "text": ds[0]["text"]},
                {"type": "audio", "path": ds[0]["audio"]["array"]},
            ],
        },
        {
            "role": f"{ds[1]['speaker_id']}",
            "content": [
                {"type": "text", "text": ds[1]["text"]},
            ],
        },
    ],
    [
        {
            "role": f"{ds[0]['speaker_id']}",
            "content": [
                {"type": "text", "text": ds[0]["text"]},
            ],
        }
    ],
]
inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
).to(model.device)

audio = model.generate(**inputs, output_audio=True)
processor.save_audio(audio, [f"speech_batch_idx_{i}.wav" for i in range(len(audio))])
```

### Making The Model Go Brrr

CSM supports full-graph compilation with CUDA graphs!

```python
import torch
import copy
from transformers import CsmForConditionalGeneration, AutoProcessor
from datasets import load_dataset

model_id = "sesame/csm-1b"
device = "cuda"

# set logs to ensure no recompilation and graph breaks
torch._logging.set_logs(graph_breaks=True, recompiles=True, cudagraphs=True)

# load the model and the processor
processor = AutoProcessor.from_pretrained(model_id)
model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=device)

# use static cache, enabling automatically torch compile with fullgraph and reduce-overhead
model.generation_config.max_length = 250 # big enough to avoid recompilation
model.generation_config.max_new_tokens = None # would take precedence over max_length
model.generation_config.cache_implementation = "static"
model.depth_decoder.generation_config.cache_implementation = "static"

# generation kwargs
gen_kwargs = {
    "do_sample": False,
    "depth_decoder_do_sample": False,
    "temperature": 1.0,
    "depth_decoder_temperature": 1.0,
}

# Define a timing decorator
class TimerContext:
    def __init__(self, name="Execution"):
        self.name = name
        self.start_event = None
        self.end_event = None
        
    def __enter__(self):
        # Use CUDA events for more accurate GPU timing
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()
        return self

    def __exit__(self, *args):
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed_time = self.start_event.elapsed_time(self.end_event) / 1000.0
        print(f"{self.name} time: {elapsed_time:.4f} seconds")

# prepare the inputs 
ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")

conversation = [
    {
        "role": f"{ds[0]['speaker_id']}",
        "content": [
            {"type": "text", "text": ds[0]["text"]},
            {"type": "audio", "path": ds[0]["audio"]["array"]},
        ],
    },
    {
        "role": f"{ds[1]['speaker_id']}",
        "content": [
            {"type": "text", "text": ds[1]["text"]},
            {"type": "audio", "path": ds[1]["audio"]["array"]},
        ],
    },
    {
        "role": f"{ds[2]['speaker_id']}",
        "content": [
            {"type": "text", "text": ds[2]["text"]},
        ],
    },
]

padded_inputs_1 = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
).to(model.device)

print("\n" + "="*50)
print("First generation - compiling and recording CUDA graphs...")
with TimerContext("First generation"):
    _ = model.generate(**padded_inputs_1, **gen_kwargs)
print("="*50)

print("\n" + "="*50)
print("Second generation - fast !!!")
with TimerContext("Second generation"):
    _ = model.generate(**padded_inputs_1, **gen_kwargs)
print("="*50)

# now with different inputs
conversation = [
    {
        "role": f"{ds[0]['speaker_id']}",
        "content": [
            {"type": "text", "text": ds[2]["text"]},
            {"type": "audio", "path": ds[2]["audio"]["array"]},
        ],
    },
    {
        "role": f"{ds[1]['speaker_id']}",
        "content": [
            {"type": "text", "text": ds[3]["text"]},
            {"type": "audio", "path": ds[3]["audio"]["array"]},
        ],
    },
    {
        "role": f"{ds[2]['speaker_id']}",
        "content": [
            {"type": "text", "text": ds[4]["text"]},
        ],
    },
]
padded_inputs_2 = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
).to(model.device)

print("\n" + "="*50)
print("Generation with other inputs!")
with TimerContext("Generation with different inputs"):
    _ = model.generate(**padded_inputs_2, **gen_kwargs)
print("="*50)
```

### Training

CSM Transformers integration supports training!

```python
from transformers import CsmForConditionalGeneration, AutoProcessor, infer_device
from datasets import load_dataset, Audio

model_id = "sesame/csm-1b"
device = infer_device()

# load the model and the processor
processor = AutoProcessor.from_pretrained(model_id)
model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=device)
model.train()
model.codec_model.eval()

ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
# ensure the audio is 24kHz
ds = ds.cast_column("audio", Audio(sampling_rate=24000))
conversation = []

# context
for text, audio, speaker_id in zip(ds[:4]["text"], ds[:4]["audio"], ds[:4]["speaker_id"]):
    conversation.append(
        {
            "role": f"{speaker_id}",
            "content": [{"type": "text", "text": text}, {"type": "audio", "path": audio["array"]}],
        }
    )

inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
    output_labels=True,
).to(model.device)

out = model(**inputs)
out.loss.backward()
```

This model was contributed by [Eustache Le Bihan](https://huggingface.co/eustlb).
The original code can be found [here](https://github.com/SesameAILabs/csm).


## CsmConfig

[[autodoc]] CsmConfig

## CsmDepthDecoderConfig

[[autodoc]] CsmDepthDecoderConfig

## CsmProcessor

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/eustlb/documentation-images/resolve/main/fig1.jpg"/>
</div>

[[autodoc]] CsmProcessor
    - __call__

## CsmForConditionalGeneration

[[autodoc]] CsmForConditionalGeneration
    - forward
    - generate

## CsmDepthDecoderForCausalLM

[[autodoc]] CsmDepthDecoderForCausalLM

## CsmDepthDecoderModel

[[autodoc]] CsmDepthDecoderModel

## CsmBackboneModel

[[autodoc]] CsmBackboneModel
