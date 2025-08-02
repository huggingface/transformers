<!--Copyright 2022 The HuggingFace Team. All rights reserved.

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

# Data2Vec

[Data2Vec](https://huggingface.co/papers/2202.03555) is a self-supervised model developed by Meta AI that works across multiple modalities including speech, vision, and text. This model card focuses on the audio variant: `data2vec-audio-base-960h`, which learns contextualized representations directly from raw audio using a masked prediction task.

You can find all the original Data2Vec checkpoints under the [Data2Vec](https://huggingface.co/facebook/data2vec-audio-base-960h) collection.

> [!TIP]
> This model was contributed by [facebook](https://huggingface.co/facebook).
>
> Click on the Data2Vec models in the right sidebar for more examples of how to apply Data2Vec to speech tasks.

The example below demonstrates how to transcribe audio using [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline

pipe = pipeline(model="facebook/data2vec-audio-base-960h", task="automatic-speech-recognition")
pipe("path/to/audio.wav")
```

</hfoption>
<hfoption id="AutoModel">

```python
from transformers import AutoProcessor, AutoModelForAudioClassification
import torch
import torchaudio

processor = AutoProcessor.from_pretrained("facebook/data2vec-audio-base-960h")
model = AutoModelForAudioClassification.from_pretrained("facebook/data2vec-audio-base-960h")

waveform, sample_rate = torchaudio.load("path/to/audio.wav")
inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = torch.argmax(logits).item()
```

</hfoption>
<hfoption id="transformers CLI">

<!-- CLI not supported for this model -->

</hfoption>
</hfoptions>

## Notes

- This model supports audio feature extraction and is pretrained on LibriSpeech.
- Input audio must be sampled at 16kHz.
- It's recommended to use the corresponding AutoProcessor to ensure correct preprocessing.

## Data2VecAudioConfig

[[autodoc]] Data2VecAudioConfig
  - all

## Data2VecAudioModel

[[autodoc]] Data2VecAudioModel
  - forward

## Data2VecAudioPreTrainedModel

[[autodoc]] Data2VecAudioPreTrainedModel

## Data2Vec specific outputs

[[autodoc]] models.data2vec.modeling_data2vec_audio.Data2VecAudioBaseModelOutput