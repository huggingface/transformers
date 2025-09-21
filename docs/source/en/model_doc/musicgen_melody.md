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
*This model was released on 2023-06-08 and added to Hugging Face Transformers on 2024-03-18.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# MusicGen Melody

[MusicGen Melody](https://huggingface.co/papers/2306.05284) is a single-stage, auto-regressive Transformer model designed for high-quality music generation, conditioned on both text and audio prompts. Unlike its predecessor, MusicGen Melody uses the audio prompt as a direct melodic guide, allowing for more precise control over the generated music.

You can find all the original [MusicGen Melody](https://huggingface.co/models?sort=downloads&search=facebook%2Fmusicgen) checkpoints on the Hugging Face Hub.

> [!TIP]
> Click on the MusicGen Melody models in the right sidebar for more examples of how to apply the model to various music generation tasks.

The example below demonstrates how to generate music conditioned on an audio melody and a text description using the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline

# This pipeline is for text-to-audio, melody conditioning is best done with the AutoModel approach
pipe = pipeline("text-to-audio", model="facebook/musicgen-melody")
audio = pipe("80s pop track with bassy drums and synth")
```

</hfoption>
<hfoption id="AutoModel">

```python
from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration
from datasets import load_dataset
import torch

processor = AutoProcessor.from_pretrained("facebook/musicgen-melody")
model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody")

# Load audio prompt
dataset = load_dataset("sanchit-gandhi/gtzan", split="train", streaming=True)
sample = next(iter(dataset))["audio"]
# For better results, it's recommended to use a melody-only audio prompt (e.g., using Demucs)
# Here, we use the raw audio for simplicity
inputs = processor(
    audio=sample["array"],
    sampling_rate=sample["sampling_rate"],
    text=["80s blues track with groovy saxophone"],
    padding=True,
    return_tensors="pt",
)
audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes quantization](https://huggingface.co/docs/transformers/main/en/main_classes/quantization) to quantize the weights to 8-bit:

```python
from transformers import MusicgenMelodyForConditionalGeneration
import torch

model = MusicgenMelodyForConditionalGeneration.from_pretrained(
    "facebook/musicgen-melody",
    load_in_8bit=True,
    device_map="auto"
)
```

## Notes

### Differences with MusicGen
- The audio prompt serves as a conditional signal for the melody, whereas in the original MusicGen, it's used for audio continuation.
- Conditional text and audio signals are concatenated to the decoder's hidden states, not used as a cross-attention signal.

### Audio Prompting with Demucs
For best results, the audio prompt should contain only the melody. Use a source separation model like [Demucs](https://github.com/adefossez/demucs/tree/main) to isolate vocals or other melodic instruments from drums and bass.

```python
# Example of using Demucs for source separation
from demucs import pretrained
from demucs.apply import apply_model
from demucs.audio import convert_audio
import torch

# Assuming 'wav' is a loaded audio tensor and 'sample_rate' is its sample rate
wav = torch.tensor(sample["array"]).to(torch.float32)
demucs_model = pretrained.get_model('htdemucs')
wav = convert_audio(wav[None], sample_rate, demucs_model.samplerate, demucs_model.audio_channels)
separated_audio = apply_model(demucs_model, wav[None])
melody_prompt = separated_audio[0, demucs_model.sources.index('vocals')] # or other source
```

### Other Generation Examples

**Text-only Conditional Generation**
```python
from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration

processor = AutoProcessor.from_pretrained("facebook/musicgen-melody")
model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody")

inputs = processor(
    text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
    padding=True,
    return_tensors="pt",
)
audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)
```

**Unconditional Generation**
```python
from transformers import MusicgenMelodyForConditionalGeneration, MusicgenMelodyProcessor

model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody")
processor = MusicgenMelodyProcessor.from_pretrained("facebook/musicgen-melody")
unconditional_inputs = processor.get_unconditional_inputs(num_samples=1)
audio_values = model.generate(**unconditional_inputs, do_sample=True, max_new_tokens=256)
```

**Generation Configuration**
You can inspect and update the model's generation configuration.
```python
from transformers import MusicgenMelodyForConditionalGeneration

model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody")
# Inspect config
print(model.generation_config)
# Update config
model.generation_config.guidance_scale = 4.0
model.generation_config.max_length = 256
```

### Other Information
- **Checkpoint Conversion**: Convert original checkpoints using the script at `src/transformers/models/musicgen_melody/convert_musicgen_melody_transformers.py`.
- **`head_mask`**: The `head_mask` argument is only effective with `attn_implementation="eager"`.
- **Sampling**: For best results, use sampling (`do_sample=True`).

## Model Structure

The model consists of three parts:
1.  **Text encoder**: A frozen T5 or Flan-T5 model that creates hidden-state representations from text.
2.  **MusicGen Melody decoder**: An auto-regressive language model that generates audio tokens based on the text and audio prompts.
3.  **Audio decoder**: An EnCodec model that converts audio tokens back into a waveform.

You can work with the standalone decoder [`MusicgenMelodyForCausalLM`] or the full [`MusicgenMelodyForConditionalGeneration`] model.

```python
from transformers import AutoConfig, MusicgenMelodyForCausalLM, MusicgenMelodyForConditionalGeneration

# Option 1: Load the decoder directly
decoder_config = AutoConfig.from_pretrained("facebook/musicgen-melody").decoder
decoder = MusicgenMelodyForCausalLM.from_pretrained("facebook/musicgen-melody", **decoder_config.to_dict())

# Option 2: Access the decoder from the composite model
model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody")
decoder = model.decoder
```

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