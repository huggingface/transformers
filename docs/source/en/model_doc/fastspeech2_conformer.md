<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
*This model was released on 2020-10-26 and added to Hugging Face Transformers on 2024-01-03 and contributed by [connor-henderson](https://huggingface.co/connor-henderson).*

# FastSpeech2Conformer

[FastSpeech2Conformer](https://huggingface.co/papers/2010.13956) enhances text-to-speech (TTS) by addressing limitations in FastSpeech. It eliminates the need for a teacher-student distillation pipeline and instead uses ground-truth target data for training. FastSpeech2 incorporates additional speech variation information, such as pitch and energy, as conditional inputs. This model achieves faster training and inference speeds while improving voice quality, surpassing even autoregressive models.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-to-audio", model="espnet/fastspeech2_conformer_with_hifigan", dtype="auto")
output = pipeline("Plants create energy through a process known as photosynthesis.")
audio = output["audio"]
```

</hfoption>
<hfoption id="FastSpeech2ConformerWithHifiGan">

```py
import torch
import soundfile as sf
from transformers import FastSpeech2ConformerTokenizer, FastSpeech2ConformerWithHifiGan

tokenizer = FastSpeech2ConformerTokenizer.from_pretrained("espnet/fastspeech2_conformer_with_hifigan"")
model = FastSpeech2ConformerWithHifiGan.from_pretrained("espnet/fastspeech2_conformer_with_hifigan", dtype="auto")

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
input_ids = inputs["input_ids"]

output_dict = model(input_ids, return_dict=True)
waveform = output_dict["waveform"]
sf.write("speech.wav", waveform.squeeze().detach().numpy(), samplerate=22050)
```

</hfoption>
</hfoptions>

## FastSpeech2ConformerConfig

[[autodoc]] FastSpeech2ConformerConfig

## FastSpeech2ConformerHifiGanConfig

[[autodoc]] FastSpeech2ConformerHifiGanConfig

## FastSpeech2ConformerWithHifiGanConfig

[[autodoc]] FastSpeech2ConformerWithHifiGanConfig

## FastSpeech2ConformerTokenizer

[[autodoc]] FastSpeech2ConformerTokenizer
    - __call__
    - save_vocabulary
    - decode
    - batch_decode

## FastSpeech2ConformerModel

[[autodoc]] FastSpeech2ConformerModel
    - forward

## FastSpeech2ConformerHifiGan

[[autodoc]] FastSpeech2ConformerHifiGan
    - forward

## FastSpeech2ConformerWithHifiGan

[[autodoc]] FastSpeech2ConformerWithHifiGan
    - forward
