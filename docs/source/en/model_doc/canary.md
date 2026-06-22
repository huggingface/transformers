<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was published in HF papers on 2025-09-17 and contributed to Hugging Face Transformers on 2026-06-21.*

<div class="flex flex-wrap space-x-1">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

# Canary

## Overview

Canary-1B-v2 was proposed in [Canary-1B-v2 & Parakeet-TDT-0.6B-v3: Efficient and High-Performance Models for Multilingual ASR and AST](https://huggingface.co/papers/2509.14128) by Monica Sekoyan, Nithin Rao Koluguri, Nune Tadevosyan, Piotr Zelasko, Travis Bartley, Nikolay Karpov, Jagadeesh Balam, and Boris Ginsburg.

The abstract from the paper is the following:

*This report introduces Canary-1B-v2, a fast, robust multilingual model for Automatic Speech Recognition (ASR) and Speech-to-Text Translation (AST). Built with a FastConformer encoder and Transformer decoder, it supports 25 European languages. The model was trained on 1.7M hours of total data samples, including Granary and NeMo ASR Set 3.0, with non-speech audio added to reduce hallucinations for ASR and AST. We describe its two-stage pre-training and fine-tuning process with dynamic data balancing, as well as experiments with an nGPT encoder. Results show nGPT scales well with massive data, while FastConformer excels after fine-tuning. For timestamps, Canary-1B-v2 uses the NeMo Forced Aligner (NFA) with an auxiliary CTC model, providing reliable segment-level timestamps for ASR and AST. Evaluations show Canary-1B-v2 outperforms Whisper-large-v3 on English ASR while being 10× faster, and delivers competitive multilingual ASR and AST performance against larger models like Seamless-M4T-v2-large and LLM-based systems. We also release Parakeet-TDT-0.6B-v3, a successor to v2, offering multilingual ASR across the same 25 languages with just 600M parameters.*

Canary reuses the [Fast Conformer](https://huggingface.co/papers/2305.05084) encoder from [Parakeet](./parakeet.md) (loaded through [`ParakeetEncoder`] / [`ParakeetEncoderConfig`]) and pairs it with a Transformer decoder that uses fixed sinusoidal positional embeddings, cross-attention to the encoder outputs and tied input/output embeddings. The task is selected through a decoder prompt prefix built by [`CanaryProcessor`] of the form `<|startofcontext|> <|startoftranscript|> <source_lang> <target_lang> <pnc|nopnc> <timestamp|notimestamp> ...`, where `source_lang == target_lang` selects transcription and otherwise selects translation.

The original implementation can be found in [NVIDIA NeMo](https://github.com/NVIDIA/NeMo). A Transformers-compatible checkpoint is available at [harshaljanjani/canary-1b-v2-hf](https://huggingface.co/harshaljanjani/canary-1b-v2-hf).

## Usage

```python
from datasets import load_dataset, Audio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

processor = AutoProcessor.from_pretrained("harshaljanjani/canary-1b-v2-hf")
model = AutoModelForSpeechSeq2Seq.from_pretrained("harshaljanjani/canary-1b-v2-hf", device_map="auto")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

# transcription (English -> English)
inputs = processor(ds[0]["audio"]["array"], source_lang="en", target_lang="en", return_tensors="pt").to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=128)
print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])

# translation (English speech -> German text)
inputs = processor(ds[0]["audio"]["array"], source_lang="en", target_lang="de", return_tensors="pt").to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=128)
print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
```

## CanaryConfig

[[autodoc]] CanaryConfig

## CanaryProcessor

[[autodoc]] CanaryProcessor

## CanaryModel

[[autodoc]] CanaryModel
    - forward

## CanaryForConditionalGeneration

[[autodoc]] CanaryForConditionalGeneration
    - forward
