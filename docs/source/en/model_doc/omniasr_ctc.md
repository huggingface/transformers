<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was published in HF papers on 2025-11-12 and contributed to Hugging Face Transformers on 2026-10-21.*

# OmniASR CTC

<div class="flex flex-wrap space-x-1">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

Check the [OmniASR](./omniasr) page for more information about this model family and checkpoints.

## Usage

The CTC variant transcribes in lowercase without punctuation, and does not take a language hint.

### Simple transcription

```python
from datasets import Audio, load_dataset
from transformers import AutoModelForCTC, AutoProcessor

model_id = "bezzam/omniasr-ctc-300m-v2"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCTC.from_pretrained(model_id, device_map="auto")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

inputs = processor(ds[0]["audio"]["array"], sampling_rate=processor.feature_extractor.sampling_rate)
inputs.to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs)
print(processor.decode(outputs, skip_special_tokens=True)[0])
```

### Batch inference

Batch inference is possible by passing a list of audios, which the processor pads to the longest one.

```python
from datasets import Audio, load_dataset
from transformers import AutoModelForCTC, AutoProcessor

model_id = "bezzam/omniasr-ctc-300m-v2"
NUM_SAMPLES = 5

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCTC.from_pretrained(model_id, device_map="auto")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
speech_samples = [el["array"] for el in ds["audio"][:NUM_SAMPLES]]

inputs = processor(speech_samples, sampling_rate=processor.feature_extractor.sampling_rate)
inputs.to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs)
for i, text in enumerate(processor.decode(outputs, skip_special_tokens=True)):
    print(f"Audio {i + 1}: {text}")
```

### Training

The model can be trained with the loss it outputs. Passing `text` to the processor prepares the CTC `labels`, with
padding positions masked out. Note that OmniASR transcribes in lowercase, so the (uppercase) LibriSpeech transcripts
are lowercased below.

```python
from datasets import Audio, load_dataset
from transformers import AutoModelForCTC, AutoProcessor

model_id = "bezzam/omniasr-ctc-300m-v2"
NUM_SAMPLES = 5

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCTC.from_pretrained(model_id, device_map="auto")
model.train()

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
speech_samples = [el["array"] for el in ds["audio"][:NUM_SAMPLES]]
text_samples = [text.lower() for text in ds["text"][:NUM_SAMPLES]]

# passing `text` to the processor will prepare inputs' `labels` key
inputs = processor(audio=speech_samples, text=text_samples, sampling_rate=processor.feature_extractor.sampling_rate)
inputs.to(model.device, dtype=model.dtype)

outputs = model(**inputs)
print("Loss:", outputs.loss.item())
outputs.loss.backward()
```

### Torch compile

The CTC variant runs a single forward pass, which is compiled by passing a `CompileConfig` to `generate`. Padding all
inputs to the same length (here 30 seconds) keeps the input shapes static, so that new inputs don't trigger a
recompilation.

On an A100, we observed a speed-up of ~1.3 for a batch size of 4 ([script](https://gist.github.com/ebezzam/26af2bd40fa207af322de39701179650#file-example_compile_ctc-py)).

```python
import torch
from datasets import Audio, load_dataset
from transformers import AutoModelForCTC, AutoProcessor, CompileConfig

model_id = "bezzam/omniasr-ctc-300m-v2"
NUM_SAMPLES = 4
num_warmup = 3

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCTC.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda").eval()

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
speech_samples = [el["array"] for el in ds["audio"][:NUM_SAMPLES]]

inputs = processor(
    speech_samples,
    sampling_rate=processor.feature_extractor.sampling_rate,
    padding="max_length",
    max_length=30 * processor.feature_extractor.sampling_rate,
).to(model.device, dtype=model.dtype)

compile_config = CompileConfig(fullgraph=True, mode="reduce-overhead")

# Warmup
for _ in range(num_warmup):
    _ = model.generate(**inputs, compile_config=compile_config)
torch.cuda.synchronize()

# Apply model
outputs = model.generate(**inputs, compile_config=compile_config)
print(processor.decode(outputs, skip_special_tokens=True)[0])
```

## OmniASRCTCConfig

[[autodoc]] OmniASRCTCConfig

## OmniASRCTCProcessor

[[autodoc]] OmniASRCTCProcessor

## OmniASRCTCModel

[[autodoc]] OmniASRCTCModel
    - forward

## OmniASRForCTC

[[autodoc]] OmniASRForCTC
    - forward
