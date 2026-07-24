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

*This model was released on 2025-11-12 and added to Hugging Face Transformers on 2026-07-10.*

# OmniASR

<div class="flex flex-wrap space-x-1">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

OmniASR (Omnilingual ASR) is a suite of multilingual speech-recognition models from Meta FAIR that covers more than
1,600 languages — including hundreds never previously supported by any ASR system. Each model pairs a Wav2Vec2-style
audio encoder (a convolutional feature extractor that downsamples the 16 kHz waveform ~320× to a 50 Hz frame rate,
followed by a pre-norm Transformer encoder) with one of two heads:

- **CTC variant** ([`OmniASRForCTC`]): a single linear projection to the vocabulary, decoded non-autoregressively with
  greedy CTC. Fast and simple.
- **LLM variant** ([`OmniASRForConditionalGeneration`]): the audio embeddings are linearly projected into a Llama
  decoder which autoregressively generates the transcription. This variant additionally supports optional **language
  conditioning**: passing a language code such as `"eng_Latn"` selects a learned language embedding that is inserted
  into the decoder context, which generally improves transcription quality.

The converted development checkpoints used in the examples below are
[bezzam/omniasr-ctc-300m-v2](https://huggingface.co/bezzam/omniasr-ctc-300m-v2) and
[bezzam/omniasr-llm-300m-v2](https://huggingface.co/bezzam/omniasr-llm-300m-v2). Original weights are released by Meta
under [facebook/omniASR-*](https://huggingface.co/facebook).

> [!TIP]
> Audio must be mono and sampled at **16 kHz**. The CTC and LLM model suites are validated for clips **shorter than 40
> seconds**.

This model was contributed by [Eric Bezzam](https://huggingface.co/bezzam) and Stephen Fernandes.

### Paper

[Omnilingual ASR: Open-Source Multilingual Speech Recognition for 1600+ Languages](https://huggingface.co/papers/2511.09690),
Omnilingual ASR team, FAIR at Meta.

## Usage

### CTC variant

```python
from datasets import Audio, load_dataset
from transformers import AutoModelForCTC, AutoProcessor

model_id = "bezzam/omniasr-ctc-300m-v2"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCTC.from_pretrained(model_id, device_map="auto")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
speech_samples = [el["array"] for el in ds["audio"][:5]]

inputs = processor(
    speech_samples,
    sampling_rate=processor.feature_extractor.sampling_rate,
)
inputs.to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs)
print(processor.decode(outputs, skip_special_tokens=True))
```

#### Training

TODO: something is off with the labels, probably need a new tokenizer object rather than using LasrTokenizer in the conversion script
```
Labels 0: 'MISTERQUILTERISTHEAPOSTLEOFTHEMIDLECLASESANDWEAREGLADTOWELCOMEHISGOSPEL'
Labels 1: "NORISMISTERQUILTER'SMANERLESINTERESTINGTHANHISMATER"
Labels 2: 'HETELSUSTHATATTHISFESTIVESEASONOFTHEYEARWITHCHRISTMASANDROASTBEFLOMINGBEFOREUSSIMILESDRAWNFROMEATINGANDITSRESULTSOCURMOSTREADILYTOTHEMIND'
Labels 3: "HEHASGRAVEDOUBTSWHETHERSIRFREDERICKLEIGHTON'SWORKISREALYGREKAFTERALANDCANDISCOVERINITBUTLITLEOFROCKYITHACA"
Labels 4: "LINEL'SPICTURESAREASORTOFUPGUARDSANDATEMPAINTINGSANDMASON'SEXQUISITEIDYLSAREASNATIONALASAJINGOPOEMMISTERBIRKETFOSTER'SLANDSCAPESSMILEATONEMUCHINTHESAMEWAYTHATMISTERCARKERUSEDTOFLASHHISTETHANDMISTERJOHNCOLIERGIVESHISSITERACHERFULSLAPONTHEBACKBEFOREHESAYSLIKEASHAMPOERINATURKISHBATHNEXTMAN"
```

```python
from datasets import Audio, load_dataset
from transformers import AutoProcessor, AutoModelForCTC

model_id = "bezzam/omniasr-ctc-300m-v2"
NUM_SAMPLES = 5

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCTC.from_pretrained(model_id, device_map="auto")
model.train()

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
speech_samples = [el["array"] for el in ds["audio"][:NUM_SAMPLES]]
text_samples = ds["text"][:NUM_SAMPLES]

# passing `text` to the processor will prepare inputs' `labels` key
inputs = processor(
    audio=speech_samples,
    text=text_samples,
    sampling_rate=processor.feature_extractor.sampling_rate,
    padding=True,
    return_tensors="pt",
).to(model.device)

# What the model is actually trained to predict (-100 = padding, filtered by CTC loss)
for i, labels in enumerate(inputs["labels"]):
    print(f"Labels {i}:", repr(processor.decode(labels[labels != -100])))

loss = model(**inputs).loss
loss.backward()
print("Loss:", loss.item())
```

### LLM variant (with language conditioning)

```python
from datasets import load_dataset, Audio
from transformers import AutoProcessor, OmniASRForConditionalGeneration

model_id = "bezzam/omniasr-llm-300m-v2"
processor = AutoProcessor.from_pretrained(model_id)
model = OmniASRForConditionalGeneration.from_pretrained(model_id)

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
audio = ds[0]["audio"]["array"]

# `language` is optional; pass a `{lang}_{script}` code (e.g. "eng_Latn") for better quality.
inputs = processor(audio, sampling_rate=16000, language=["eng_Latn"], return_tensors="pt")
generated_ids = model.generate(**inputs, max_new_tokens=256)
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(transcription)
```

## OmniASRConfig

[[autodoc]] OmniASRConfig

## OmniASRCTCConfig

[[autodoc]] OmniASRCTCConfig

## OmniASRLLMConfig

[[autodoc]] OmniASRLLMConfig

## OmniASRFeatureExtractor

[[autodoc]] OmniASRFeatureExtractor

## OmniASRProcessor

[[autodoc]] OmniASRProcessor

## OmniASRModel

[[autodoc]] OmniASRModel
    - forward

## OmniASRForCTC

[[autodoc]] OmniASRForCTC
    - forward

## OmniASRForConditionalGeneration

[[autodoc]] OmniASRForConditionalGeneration
    - forward