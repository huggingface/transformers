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
*This model was released on 2023-05-12 and added to Hugging Face Transformers on 2023-11-10 and contributed by [susnato](https://huggingface.co/susnato).*

# CLVP

[CLVP](https://huggingface.co/papers/2305.07243) applies advancements from image generation, specifically autoregressive transformers and DDPMs, to speech synthesis. The result is TorToise, an expressive, multi-voice text-to-speech system.

<hfoptions id="usage">
<hfoption id="ClvpModelForConditionalGeneration">

```py
import datasets
import torch
from transformers import AutoProcessor, ClvpModelForConditionalGeneration

text = "Plants create energy through a process known as photosynthesis."

ds = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", datasets.Audio(sampling_rate=22050))
sample = ds[0]["audio"]

processor = AutoProcessor.from_pretrained("susnato/clvp_dev")
model = ClvpModelForConditionalGeneration.from_pretrained("susnato/clvp_dev", dtype="auto")

processor_output = processor(raw_speech=sample["array"], sampling_rate=sample["sampling_rate"], text=text, return_tensors="pt")
outputs = model(**processor_output)
```

</hfoption>
</hfoptions>

## Usage tips

- CLVP is an integral part of the Tortoise TTS model.
- CLVP compares different generated speech candidates with provided text. The best speech tokens forward to the diffusion model.
- Use [`ClvpModelForConditionalGeneration.generate`] for Tortoise usage.
- CLVP expects audio sampled at 22.05 kHz, unlike other audio models that expect 16 kHz.

## ClvpConfig

[[autodoc]] ClvpConfig

## ClvpEncoderConfig

[[autodoc]] ClvpEncoderConfig

## ClvpDecoderConfig

[[autodoc]] ClvpDecoderConfig

## ClvpTokenizer

[[autodoc]] ClvpTokenizer
    - save_vocabulary

## ClvpFeatureExtractor

[[autodoc]] ClvpFeatureExtractor
    - __call__

## ClvpProcessor

[[autodoc]] ClvpProcessor
    - __call__
    - decode
    - batch_decode

## ClvpModelForConditionalGeneration

[[autodoc]] ClvpModelForConditionalGeneration
    - forward
    - generate
    - get_text_features
    - get_speech_features

## ClvpForCausalLM

[[autodoc]] ClvpForCausalLM

## ClvpModel

[[autodoc]] ClvpModel

## ClvpEncoder

[[autodoc]] ClvpEncoder

## ClvpDecoder

[[autodoc]] ClvpDecoder

