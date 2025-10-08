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

*This model was released on 2024-08-30 and added to Hugging Face Transformers on 2025-08-15.*

# X-Codec

[X-Codec](https://huggingface.co/papers/2408.17175) integrates semantic information from self-supervised models into a neural audio codec, enhancing traditional acoustic information. This integration improves music continuation by better modeling musical semantics, facilitates text-to-sound synthesis by capturing semantic alignment between text and audio, and supports semantic-aware audio tokenization. X-Codec reduces word error rates in speech synthesis by incorporating semantic features before the Residual Vector Quantization stage and using a semantic reconstruction loss afterward. Experiments show significant performance improvements in text-to-speech, music continuation, and text-to-sound tasks.

<hfoptions id="usage">
<hfoption id="XcodecModel">

```py
import torch
import soundfile as sf
from datasets import load_dataset, Audio
from transformers import XcodecModel, AutoFeatureExtractor

model = XcodecModel.from_pretrained("hf-audio/xcodec-hubert-librispeech", dtype="auto")
feature_extractor = AutoFeatureExtractor.from_pretrained("hf-audio/xcodec-hubert-librispeech")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
audio_sample = ds[-1]["audio"]["array"]
inputs = feature_extractor(raw_audio=audio_sample, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")

encoder_outputs = model.encode(inputs["input_values"])
decoder_outputs = model.decode(encoder_outputs.audio_codes)
audio_values = decoder_outputs.audio_values
audio_values = model(inputs["input_values"]).audio_values

original = audio_sample
reconstruction = audio_values[0].cpu().detach().numpy()
sampling_rate = feature_extractor.sampling_rate

sf.write("original.wav", original, sampling_rate)
sf.write("reconstruction.wav", reconstruction.T, sampling_rate)
```

</hfoption>
</hfoptions>

## XcodecConfig

[[autodoc]] XcodecConfig

## XcodecModel

[[autodoc]] XcodecModel
    - decode
    - encode
    - forward

