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
*This model was released on 2025-07-22 and added to Hugging Face Transformers on 2026-02-19.*

# Higgs Audio V2 Tokenizer

## Overview

- Low Frame Rate: At 25 fps, our tokenizer halves the frame rate of many baselines when still maintaining high audio quality.
- Unified 24 kHz Training: We mix speech, music, and sound-event clips in one model, capturing both semantic and acoustic details, hugely facilitating the training of audio language models.
- Fast Inference: By avoiding diffusion steps, our encoder/decoder processes batches quickly, making it practical for real-time or large-scale tasks.

**Model Architecture:**
<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/eustlb/documentation-images/resolve/main/higgs_audio_tokenizer_architecture.png"/>
</div>

## Usage

```python
from transformers import HiggsAudioV2TokenizerModel, AutoFeatureExtractor
from datasets import load_dataset, Audio

# load model and feature extractor
model_id = "eustlb/higgs-audio-v2-tokenizer"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
model = HiggsAudioV2TokenizerModel.from_pretrained(model_id, device_map="auto")

# load audio sample
dummy_dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dummy_dataset = dummy_dataset.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
audio_sample = dummy_dataset[-1]["audio"]["array"]
inputs = feature_extractor(raw_audio=audio_sample, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")

# encode and decode
encoder_outputs = model.encode(inputs["input_values"])
decoder_outputs = model.decode(encoder_outputs.audio_codes)
audio_values = decoder_outputs.audio_values

# or the equivalent with a forward pass
audio_values = model(inputs["input_values"]).audio_values
```

## HiggsAudioV2TokenizerConfig

[[autodoc]] HiggsAudioV2TokenizerConfig

## HiggsAudioV2TokenizerModel

[[autodoc]] HiggsAudioV2TokenizerModel
    - decode
    - encode
    - forward