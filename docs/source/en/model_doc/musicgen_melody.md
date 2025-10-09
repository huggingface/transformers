<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

:warning: Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2023-06-08 and added to Hugging Face Transformers on 2024-03-18 and contributed by [ylacombe](https://huggingface.co/ylacombe).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# MusicGen Melody

[MusicGen Melody](https://huggingface.co/papers/2306.05284) is a single-stage auto-regressive Transformer model designed for generating high-quality music samples based on text descriptions or audio prompts. It uses a frozen text encoder to convert text into hidden-state representations, which are then used to predict discrete audio tokens. These tokens are decoded into audio waveforms using an audio compression model like EnCodec. The model employs an efficient token interleaving pattern, eliminating the need for multiple cascaded models, and can generate all codebooks in a single forward pass. This approach allows for better control over the generated output and has been shown to outperform existing baselines in text-to-music generation tasks.

<hfoptions id="usage">
<hfoption id="MusicgenForConditionalGeneration">

```py
import torch
import scipy.io.wavfile
from transformers import AutoProcessor, MusicgenForConditionalGeneration

processor = AutoProcessor.from_pretrained("facebook/musicgen-melody")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-melody", dtype="auto")

inputs = processor(
    text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
    padding=True,
    return_tensors="pt",
)
audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)
scipy.io.wavfile.write("generated_music.wav", rate=processor.sampling_rate, data=audio_values[0, 0].cpu().numpy())
```

</hfoption>
</hfoptions>

## MusicgenMelodyDecoderConfig

[[autodoc]] MusicgenMelodyDecoderConfig

## MusicgenMelodyProcessor

[[autodoc]] MusicgenMelodyProcessor
    - get_unconditional_inputs

## MusicgenMelodyFeatureExtractor

[[autodoc]] MusicgenMelodyFeatureExtractor

## MusicgenMelodyConfig

[[autodoc]] MusicgenMelodyConfig

## MusicgenMelodyModel

[[autodoc]] MusicgenMelodyModel
    - forward

## MusicgenMelodyForCausalLM

[[autodoc]] MusicgenMelodyForCausalLM
    - forward

## MusicgenMelodyForConditionalGeneration

[[autodoc]] MusicgenMelodyForConditionalGeneration
    - forward

