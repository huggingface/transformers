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

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
          <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
          <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
          <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Moonshine

[Moonshine](https://huggingface.co/papers/2410.15608) is a speech recognition model that is optimized for real-time transcription and voice command. Instead of using traditional absolute position embeddings, Moonshine uses Rotary Position Embedding (RoPE).


You can find all the original Moonshine checkpoints under the [Useful Sensors](https://huggingface.co/UsefulSensors) organization.

> [!TIP]
> Click on the Moonshine models in the right sidebar for more examples of how to apply Moonshine to different speech recognition tasks.

The example below demonstrates how to generate a transcription based on an audio file with [`Pipeline`] or the [`AutoModel`] class.



<hfoptions id="usage">
<hfoption id="Pipeline">

```py
# uncomment to install ffmpeg which is needed to decode the audio file
# !brew install ffmpeg

from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="UsefulSensors/moonshine-base")

result = asr("path_to_audio_file")

#Prints the transcription from the audio file
print(result["text"])
```

</hfoption>
<hfoption id="AutoModel">

```py
# uncomment to install rjieba which is needed for the tokenizer
# !pip install rjieba
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained(
    "junnyu/roformer_chinese_base", torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("junnyu/roformer_chinese_base")

input_ids = tokenizer("水在零度时会[MASK]", return_tensors="pt").to(model.device)
outputs = model(**input_ids)
decoded = tokenizer.batch_decode(outputs.logits.argmax(-1), skip_special_tokens=True)
print(decoded)
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "水在零度时会[MASK]" | transformers-cli run --task fill-mask --model junnyu/roformer_chinese_base --device 0
```

</hfoption>
</hfoptions>






The Moonshine model was proposed in [Moonshine: Speech Recognition for Live Transcription and Voice Commands
](https://arxiv.org/abs/2410.15608) by Nat Jeffries, Evan King, Manjunath Kudlur, Guy Nicholson, James Wang, Pete Warden.

The abstract from the paper is the following:

*This paper introduces Moonshine, a family of speech recognition models optimized for live transcription and voice command processing. Moonshine is based on an encoder-decoder transformer architecture and employs Rotary Position Embedding (RoPE) instead of traditional absolute position embeddings. The model is trained on speech segments of various lengths, but without using zero-padding, leading to greater efficiency for the encoder during inference time. When benchmarked against OpenAI's Whisper tiny-en, Moonshine Tiny demonstrates a 5x reduction in compute requirements for transcribing a 10-second speech segment while incurring no increase in word error rates across standard evaluation datasets. These results highlight Moonshine's potential for real-time and resource-constrained applications.*

Tips:

- Moonshine improves upon Whisper's architecture:
  1. It uses SwiGLU activation instead of GELU in the decoder layers
  2. Most importantly, it replaces absolute position embeddings with Rotary Position Embeddings (RoPE). This allows Moonshine to handle audio inputs of any length, unlike Whisper which is restricted to fixed 30-second windows.

- A guide for automatic speech recognition can be found [here](../tasks/asr)

## MoonshineConfig

[[autodoc]] MoonshineConfig

## MoonshineModel

[[autodoc]] MoonshineModel
    - forward
    - _mask_input_features

## MoonshineForConditionalGeneration

[[autodoc]] MoonshineForConditionalGeneration
    - forward
    - generate

