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
*This model was released on 2025-04-16 and added to Hugging Face Transformers on 2025-04-11 and contributed by [abrooks9944](https://huggingface.co/abrooks9944), [Avihu](https://huggingface.co/Avihu), and [gsaon](https://huggingface.co/gsaon).*

# Granite Speech

[GraniteSpeech](https://huggingface.co/papers/2505.08699) is a family of compact speech language models (2B and 8B parameters) derived from Granite-3.3-Instruct and optimized for English automatic speech recognition (ASR) and speech translation (AST). The models integrate a Conformer acoustic encoder with block attention and self-conditioning trained using connectionist temporal classification (CTC), a windowed query-transformer adapter for temporal downsampling and alignment to the LLM’s text space, and LoRA adapters for fine-tuning. Granite-speech operates in two modes: a speech mode (ASR/AST with encoder and adapters) and a text mode (standard Granite-3.3-Instruct). Despite being trained only on open data, it surpasses larger proprietary ASR models and achieves competitive AST performance across several languages.

<hfoptions id="usage">
<hfoption id="GraniteSpeechForConditionalGeneration">

```py
import torch
from datasets import load_dataset
from transformers import AutoProcessor, GraniteSpeechForConditionalGeneration

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation").sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

processor = AutoProcessor.from_pretrained("ibm-granite/granite-speech-3.3-2b")
model = GraniteSpeechForConditionalGeneration.from_pretrained("ibm-granite/granite-speech-3.3-2b", dtype="auto")

inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)
print(f"Transcription: {processor.batch_decode(predicted_ids)[0]}")
```

</hfoption>
</hfoptions>

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
