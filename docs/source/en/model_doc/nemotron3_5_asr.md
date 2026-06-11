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
*This model was contributed to Hugging Face Transformers on 2026-06-10.*

# Nemotron3_5Asr

## Overview

Nemotron3_5Asr is the **multilingual** extension of [NemotronAsr](./nemotron_asr) (`nvidia/nemotron-3.5-asr-streaming-0.6b`).
It reuses the entire cache-aware streaming [Fast Conformer](https://huggingface.co/papers/2305.05084) encoder, RNN-T
(Recurrent Neural Network Transducer) head, feature extraction, and streaming generation of [`NemotronAsr`], and adds
**language-ID prompt conditioning** so a single model transcribes 40 language-locales.

The target language is turned into a one-hot vector, broadcast across the encoder time axis, concatenated with the
encoder output, and fused back to the encoder hidden size by a small MLP (`prompt_kernel`) before the joint network.
Pass the language through the processor's `target_lang` argument (a locale such as `"en-US"`/`"de-DE"`, a bare code such
as `"de"`, or `"auto"` for automatic language detection). In `auto` mode the model appends an `<xx-XX>` language tag
after the transcript's terminal punctuation; the processor's `decode`/`batch_decode` strip it by default
(`strip_lang_tags=True`).

## Usage

```python
from transformers import AutoProcessor, Nemotron3_5AsrForRNNT
from datasets import load_dataset, Audio

model_id = "nvidia/nemotron-3.5-asr-streaming-0.6b"
processor = AutoProcessor.from_pretrained(model_id)
model = Nemotron3_5AsrForRNNT.from_pretrained(model_id).eval()

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

# Condition on a known language ...
inputs = processor(ds[0]["audio"]["array"], sampling_rate=16000, target_lang="en-US")
generated = model.generate(**inputs)
print(processor.batch_decode(generated.sequences, skip_special_tokens=True))

# ... or let the model detect it and keep the emitted language tag.
inputs = processor(ds[0]["audio"]["array"], sampling_rate=16000, target_lang="auto")
generated = model.generate(**inputs)
print(processor.batch_decode(generated.sequences, skip_special_tokens=True, strip_lang_tags=False))
```

## Nemotron3_5AsrConfig

[[autodoc]] Nemotron3_5AsrConfig

## Nemotron3_5AsrEncoderConfig

[[autodoc]] Nemotron3_5AsrEncoderConfig

## Nemotron3_5AsrFeatureExtractor

[[autodoc]] Nemotron3_5AsrFeatureExtractor

## Nemotron3_5AsrProcessor

[[autodoc]] Nemotron3_5AsrProcessor

## Nemotron3_5AsrRNNTOutput

[[autodoc]] Nemotron3_5AsrRNNTOutput

## Nemotron3_5AsrForRNNT

[[autodoc]] Nemotron3_5AsrForRNNT
    - forward
    - generate
