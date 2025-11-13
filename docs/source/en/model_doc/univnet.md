<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2021-06-15 and added to Hugging Face Transformers on 2023-11-22 and contributed by [dg845](https://huggingface.co/dg845).*

# UnivNet

[UnivNet](https://huggingface.co/papers/2106.07889) is a neural vocoder designed to synthesize high-fidelity speech waveforms in real time. It uses full-band mel-spectrograms as input and incorporates a multi-resolution spectrogram discriminator to mitigate over-smoothing issues. The discriminator employs multiple linear spectrogram magnitudes with varying parameters. Evaluated on a dataset with hundreds of speakers, UnivNet achieved the best objective and subjective results, including top scores for text-to-speech, demonstrating its capability for fast adaptation to new speakers without retraining.

<hfoptions id="usage">
<hfoption id="UnivNetModel">

```py
import torch
from scipy.io.wavfile import write
from datasets import Audio, load_dataset
from transformers import UnivNetFeatureExtractor, UnivNetModel

model = UnivNetModel.from_pretrained("dg845/univnet-dev", dtype="auto")
feature_extractor = UnivNetFeatureExtractor.from_pretrained("dg845/univnet-dev")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
inputs = feature_extractor(
    ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], pad_end=True, return_tensors="pt"
)

with torch.no_grad():
    audio = model(**inputs)

audio = feature_extractor.batch_decode(**audio)[0]
write("sample_audio.wav", feature_extractor.sampling_rate, audio)
```

</hfoption>
</hfoptions>

## Usage tips

- The `noise_sequence` argument for [`UnivNetModel.forward`] requires standard Gaussian noise (like from `torch.randn`) with shape `([batch_size], noise_length, model.config.model_in_channels)`. The `noise_length` must match the length dimension (dimension 1) of the `input_features` argument. If not supplied, noise generates randomly. Supply a `torch.Generator` to the `generator` argument to reproduce the forward pass. [`UnivNetFeatureExtractor`] returns generated noise by default, so manual noise generation isn't necessary.
- Remove padding added by [`UnivNetFeatureExtractor`] from [`UnivNetModel`] output using [`UnivNetFeatureExtractor.batch_decode`], as shown in the usage example.
- Pad the end of each waveform with silence to reduce artifacts at the end of generated audio samples. Set `pad_end=True` in [`UnivNetFeatureExtractor.call`]. See this [issue](https://github.com/mindslab-ai/univnet/issues/8) for more details.

## UnivNetConfig

[[autodoc]] UnivNetConfig

## UnivNetFeatureExtractor

[[autodoc]] UnivNetFeatureExtractor
    - __call__

## UnivNetModel

[[autodoc]] UnivNetModel
    - forward

