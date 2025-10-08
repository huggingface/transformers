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
*This model was released on 2025-06-17 and added to Hugging Face Transformers on 2025-06-25 and contributed by [eustlb](https://huggingface.co/eustlb).*

# Kyutai Speech-To-Text

[Kyutai STT](https://huggingface.co/papers/2509.08753) is a decoder-only framework for streaming, multimodal sequence-to-sequence learning that aligns input and output streams in advance rather than learning alignment during training. By introducing controlled delays between already time-aligned streams, DSM enables real-time generation for tasks like automatic speech recognition (ASR) and text-to-speech (TTS) using the same underlying model. This approach allows efficient streaming inference with arbitrary sequence lengths and multimodal inputs while maintaining low latency. Experiments show DSM achieves state-of-the-art performance on both ASR and TTS, rivaling even non-streaming (offline) models.

<hfoptions id="usage">
<hfoption id="KyutaiSpeechToTextForConditionalGeneration">

```py
import torch
from datasets import load_dataset, Audio
from transformers import KyutaiSpeechToTextProcessor, KyutaiSpeechToTextForConditionalGeneration
from accelerate import Accelerator

processor = KyutaiSpeechToTextProcessor.from_pretrained("kyutai/stt-2.6b-en-trfs")
model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained("kyutai/stt-2.6b-en-trfs", dtype="auto")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=24000))

inputs = processor(ds[0]["audio"]["array"],)
output_tokens = model.generate(**inputs)
print(f"Transcription: {processor.batch_decode(output_tokens, skip_special_tokens=True)}")
```

</hfoption>
</hfoptions>

## KyutaiSpeechToTextConfig

[[autodoc]] KyutaiSpeechToTextConfig

## KyutaiSpeechToTextProcessor

[[autodoc]] KyutaiSpeechToTextProcessor
    - __call__

## KyutaiSpeechToTextFeatureExtractor

[[autodoc]] KyutaiSpeechToTextFeatureExtractor

## KyutaiSpeechToTextForConditionalGeneration

[[autodoc]] KyutaiSpeechToTextForConditionalGeneration
    - forward
    - generate

## KyutaiSpeechToTextModel

[[autodoc]] KyutaiSpeechToTextModel
