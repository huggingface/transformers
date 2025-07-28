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

<div style="float: right;">
  <div class="flex flex-wrap space-x-1">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
  </div>
</div>

# CLAP

[CLAP (Contrastive Language-Audio Pretraining)](https://huggingface.co/papers/2211.06687) is a model trained on a large number of (audio, text) pairs to learn a shared space between audio and language.

It uses a SWINTransformer to get features from audio (converted into log-Mel spectrograms) and a RoBERTa model for text. Both are projected into the same space so that similar audio and text end up close together. This lets CLAP work well for things like zero-shot audio classification and text-to-audio retrieval.

You can find all the original CLAP checkpoints under the [laion](https://huggingface.co/laion?search=clap) organization.

> [!TIP]
> This model was contributed by [@ybelkada](https://huggingface.co/ybelkada) and [@ArthurZ](https://huggingface.co/ArthurZ).
>
> Click on the CLAP models in the right sidebar for more examples of how to apply CLAP to different audio retrieval and classification tasks.

The example below demonstrates how to extract embeddings with [`Pipeline`], [`AutoModel`] and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipeline = pipeline(
    task="feature-extraction",
    model="laion/clap-htsat-unfused",
    device=0
)
embedding = pipeline("A sound of waves crashing on the beach.")
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
import torchaudio
from transformers import AutoProcessor, ClapModel

processor = AutoProcessor.from_pretrained(
    "laion/clap-htsat-unfused"
)
model = ClapModel.from_pretrained(
    "laion/clap-htsat-unfused",
    torch_dtype=torch.float16,
    device_map="auto"
)

waveform, sample_rate = torchaudio.load("path/to/audio.wav")
inputs = processor(audios=waveform, sampling_rate=sample_rate, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}
audio_embed = model.get_audio_features(**inputs)

text_inputs = processor(text=["A dog is barking"], return_tensors="pt")
text_inputs = {k: v.to("cuda") for k, v in text_inputs.items()}
text_embed = model.get_text_features(**text_inputs)
```

</hfoption>
<hfoption id="transformers-cli">

This model is not supported via `transformers-cli` at this time.
</hfoption>
</hfoptions>

## Quantization

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

Quantization is not currently supported for this model.

## Notes

- CLAP works with both raw audio (as waveform tensors) and text input.
- For best results, resample your audio to 48 kHz mono before inference:

    ```python
    import torchaudio

    waveform, sample_rate = torchaudio.load("my_audio.wav")
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=48000)
    mono_waveform = resampler(waveform.mean(dim=0, keepdim=True))
    ```
  
## Resources

- [Original paper](https://huggingface.co/papers/2211.06687)
- [Official GitHub repo](https://github.com/LAION-AI/CLAP)

## ClapConfig

[[autodoc]] ClapConfig
    - from_text_audio_configs

## ClapTextConfig

[[autodoc]] ClapTextConfig

## ClapAudioConfig

[[autodoc]] ClapAudioConfig

## ClapFeatureExtractor

[[autodoc]] ClapFeatureExtractor

## ClapProcessor

[[autodoc]] ClapProcessor

## ClapModel

[[autodoc]] ClapModel
    - forward
    - get_text_features
    - get_audio_features

## ClapTextModel

[[autodoc]] ClapTextModel
    - forward

## ClapTextModelWithProjection

[[autodoc]] ClapTextModelWithProjection
    - forward

## ClapAudioModel

[[autodoc]] ClapAudioModel
    - forward

## ClapAudioModelWithProjection

[[autodoc]] ClapAudioModelWithProjection
    - forward
