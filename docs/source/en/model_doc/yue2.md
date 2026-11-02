<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was published in HF papers on 2026-09-27 and contributed to Hugging Face Transformers on 2026-10-29.*

# YuE2

## Overview

[YuE2](https://huggingface.co/m-a-p/YuE2-3B) is a music generation model by [M-A-P](https://huggingface.co/m-a-p): from lyrics and a style prompt, it generates a complete song with vocals and accompaniment, as 48 kHz stereo audio. It can first write the song as an editable symbolic score (ABC notation, melody and chords), which it then follows.

One 3B-parameter AR–NAR mixture-of-transformers backbone does everything. Each decoder layer has two sets of weights:

- the autoregressive (AR) weights write the symbolic score and then the song's semantic tokens, one per 40 ms frame;
- the non-autoregressive (NAR) weights generate the acoustic latents of these frames by flow matching, each latent attending to the prompt, the score and the semantic tokens, and to the other latents in both directions.

The [YuE2 acoustic tokenizer](./yue2_acoustic_tokenizer) (a variational autoencoder) decodes the latents into audio.

The weights are released under the [CC-BY-NC-4.0](https://huggingface.co/m-a-p/YuE2-3B/blob/main/LICENSE) license.

This model was contributed by [Eric Bezzam](https://huggingface.co/bezzam). The original code can be found on [GitHub](https://github.com/multimodal-art-projection/YuE/blob/main/src/yue2/modeling_yue2.py). See the [technical report](https://huggingface.co/papers/2609.33757) for more details.

## Usage

[`YuE2Processor`] builds the prompt from the lyrics and the style, [`~YuE2ForConditionalGeneration.generate`] writes the score and the semantic tokens and generates the acoustic latents, and the processor decodes them into audio.

```python
import soundfile as sf
import torch
from transformers import AutoProcessor, YuE2ForConditionalGeneration

model_id = "bezzam/yue2-3b"
processor = AutoProcessor.from_pretrained(model_id, device_map="auto")
model = YuE2ForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")

lyrics = """[verse]
Golden light on the morning street
Coffee steam and a steady beat
[chorus]
Sing it loud, let the city hear
"""
inputs = processor(lyrics=lyrics, style="Upbeat pop, female vocal, acoustic guitar, handclaps").to(model.device)

torch.manual_seed(831001)  # the flow matching draws its noise from the global CPU generator
# `max_new_tokens` caps the semantic tokens (25 per second of audio); the generation config allows 6 minutes
outputs = model.generate(**inputs, max_new_tokens=750)

audio = processor.decode(outputs.audio_latents, outputs.audio_latents_mask)[0]  # (2, num_samples) at 48 kHz
sf.write("song.wav", audio.T.numpy(), samplerate=48000)
```

The processor holds the acoustic tokenizer, which runs in float32 as in the original.

### Symbolic score

`cot` sets how the model plans the song, as in the original:

- `cot="full"` (default) writes a chord-annotated score, then the music;
- `cot="melody"` writes a melody-only score;
- `cot="off"` writes the music directly.

The score is in the generated `sequences`, between the `<abc>` and `</abc>` tokens:

```python
sequence = outputs.sequences[0].tolist()
abc_start, abc_end = processor.tokenizer.convert_tokens_to_ids(["<abc>", "</abc>"])
score = processor.tokenizer.decode(sequence[sequence.index(abc_start) + 1 : sequence.index(abc_end)])
print(score)
```

Pass a score with `abc` to follow it instead of writing one, for instance an edited version of a generated score, or the melody of an existing song transcribed with [SheetSage2](https://huggingface.co/m-a-p/SheetSage2) to make a cover (with `cot="melody"`, recommended for covers):

```python
inputs = processor(lyrics=lyrics, style="Jazz-funk, warm lead vocal, Rhodes piano", cot="melody", abc=edited_score)
```

### Generation parameters

The semantic tokens are sampled with the generation config (`temperature`, `top_p`, `top_k`, `repetition_penalty`, `min_new_tokens`, `max_new_tokens`), and the score with its `abc_generation_config` overrides. `repetition_penalty` is YuE2's: it compounds with the number of occurrences of a token among the last `repetition_penalty_window` generated tokens. `num_flow_steps` sets the number of midpoint steps of the flow-matching ODE (32 by default).

`guidance_scale` sets the classifier-free guidance of the semantic tokens, with the instruction (and the score) as negative prompt. When it is not set, the semantic tokens are guided by 1.01 with `cot="off"` and not guided otherwise, as in the original; the original suggests trying `guidance_scale=1.2` for stronger text guidance.

## YuE2Config

[[autodoc]] YuE2Config

## YuE2Processor

[[autodoc]] YuE2Processor
    - __call__
    - decode

## YuE2Model

[[autodoc]] YuE2Model
    - forward

## YuE2ForConditionalGeneration

[[autodoc]] YuE2ForConditionalGeneration
    - forward
    - generate
    - get_audio_latent_velocity
