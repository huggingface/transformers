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
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# MusicGen

[MusicGen](https://huggingface.co/papers/2306.05284) is a single-stage auto-regressive Transformer model for generating high-quality music samples from text descriptions or audio prompts. What makes MusicGen unique is its ability to generate all codebooks in a single forward pass, without needing to cascade multiple models. It's like having a music studio in your pocket—just describe the vibe, and MusicGen brings it to life!

You can find all the original [MusicGen](https://huggingface.co/models?sort=downloads&search=facebook%2Fmusicgen-) checkpoints under the [MusicGen collection](https://huggingface.co/models?sort=downloads&search=facebook%2Fmusicgen-).

> [!TIP]
> Click on the MusicGen models in the right sidebar for more examples of how to apply MusicGen to different music generation tasks.

The example below demonstrates how to generate music from text or audio prompts using the [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline
pipe = pipeline("text-to-audio", model="facebook/musicgen-small")
audio = pipe("80s pop track with bassy drums and synth")
```

</hfoption>
<hfoption id="AutoModel">

```python
from transformers import AutoProcessor, MusicgenForConditionalGeneration

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

inputs = processor(
    text=["80s pop track with bassy drums and synth"],
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
from transformers import MusicgenForConditionalGeneration
import torch

model = MusicgenForConditionalGeneration.from_pretrained(
    "facebook/musicgen-small",
    load_in_8bit=True,
    device_map="auto"
)
```

## Notes

- MusicGen supports both mono (1-channel) and stereo (2-channel) variants. The stereo version generates two sets of codebooks (left/right channels) and combines them for the final output.
- Generation is limited to 30 seconds of audio (1503 tokens) due to positional embedding constraints. Input audio for audio-prompted generation counts toward this limit.
- For best results, use sampling mode (`do_sample=True`) and a guidance scale of 3 for classifier-free guidance.
- MusicGen is trained on the 32kHz checkpoint of EnCodec. Ensure you use a compatible EnCodec model.
- The `head_mask` argument is only effective with `attn_implementation="eager"`.
- You can convert original checkpoints using the script at `src/transformers/models/musicgen/convert_musicgen_transformers.py`.

```python
# Convert original checkpoints
python src/transformers/models/musicgen/convert_musicgen_transformers.py \
    --checkpoint small --pytorch_dump_folder /output/path --safe_serialization
```

### Unconditional Generation

```python
from transformers import MusicgenForConditionalGeneration
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
unconditional_inputs = model.get_unconditional_inputs(num_samples=1)
audio_values = model.generate(**unconditional_inputs, do_sample=True, max_new_tokens=256)
```

### Listen to Generated Audio

```python
from IPython.display import Audio
sampling_rate = model.config.audio_encoder.sampling_rate
Audio(audio_values[0].numpy(), rate=sampling_rate)
```

### Save as WAV

```python
import scipy
sampling_rate = model.config.audio_encoder.sampling_rate
scipy.io.wavfile.write("musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())
```

### Audio-Prompted Generation

```python
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from datasets import load_dataset

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
dataset = load_dataset("sanchit-gandhi/gtzan", split="train", streaming=True)
sample = next(iter(dataset))["audio"]
sample["array"] = sample["array"][: len(sample["array"]) // 2]
inputs = processor(
    audio=sample["array"],
    sampling_rate=sample["sampling_rate"],
    text=["80s blues track with groovy saxophone"],
    padding=True,
    return_tensors="pt",
)
audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)
```

### Batched Audio-Prompted Generation

```python
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from datasets import load_dataset

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
dataset = load_dataset("sanchit-gandhi/gtzan", split="train", streaming=True)
sample = next(iter(dataset))["audio"]
sample_1 = sample["array"][: len(sample["array"]) // 4]
sample_2 = sample["array"][: len(sample["array"]) // 2]
inputs = processor(
    audio=[sample_1, sample_2],
    sampling_rate=sample["sampling_rate"],
    text=["80s blues track with groovy saxophone", "90s rock song with loud guitars and heavy drums"],
    padding=True,
    return_tensors="pt",
)
audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)
# Remove padding from batched audio
audio_values = processor.batch_decode(audio_values, padding_mask=inputs.padding_mask)
```

### Generation Configuration

```python
from transformers import MusicgenForConditionalGeneration
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
# Inspect the default generation config
print(model.generation_config)
# Update guidance scale and max length
model.generation_config.guidance_scale = 4.0
model.generation_config.max_length = 256
```

### Model Structure

- **Text encoder**: Maps text to hidden-state representations (frozen T5 or Flan-T5).
- **MusicGen decoder**: Auto-regressively generates audio tokens (codes) from encoder hidden-states.
- **Audio encoder/decoder**: Encodes audio prompts and decodes audio tokens to waveforms.

You can use MusicGen as a standalone decoder ([`MusicgenForCausalLM`]) or as a composite model ([`MusicgenForConditionalGeneration`]).

```python
from transformers import AutoConfig, MusicgenForCausalLM, MusicgenForConditionalGeneration
# Option 1: get decoder config and pass to `.from_pretrained`
decoder_config = AutoConfig.from_pretrained("facebook/musicgen-small").decoder
decoder = MusicgenForCausalLM.from_pretrained("facebook/musicgen-small", **decoder_config)
# Option 2: load the entire composite model, but only return the decoder
decoder = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").decoder
```

---

## API Reference

### MusicgenDecoderConfig

[[autodoc]] MusicgenDecoderConfig

### MusicgenConfig

[[autodoc]] MusicgenConfig

### MusicgenProcessor

[[autodoc]] MusicgenProcessor

### MusicgenModel

[[autodoc]] MusicgenModel
    - forward

### MusicgenForCausalLM

[[autodoc]] MusicgenForCausalLM
    - forward

### MusicgenForConditionalGeneration

[[autodoc]] MusicgenForConditionalGeneration
    - forward
