<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2024-10-16 and contributed by [ylacombe](https://huggingface.co/ylacombe).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Moshi

[Moshi](https://huggingface.co/papers/2310.15199) is a speech-text foundation model that treats spoken dialogue as speech-to-speech generation. Built on a text language model backbone, Moshi generates speech from the residual quantizer of a neural audio codec, modeling its own speech and the user's speech in parallel streams. This approach eliminates the need for explicit speaker turns and allows for the modeling of complex conversational dynamics. Moshi introduces an "Inner Monologue" method, which predicts time-aligned text tokens before audio tokens, enhancing linguistic quality and enabling streaming speech recognition and text-to-speech. The model achieves real-time full-duplex spoken dialogue with a theoretical latency of 160ms and a practical latency of 200ms. Moshi consists of three main components: the main decoder (similar to a text LLM), the depth decoder (which generates over the codebook dimension), and the audio encoder (used to tokenize audio). 

<hfoptions id="usage">
<hfoption id="MoshiForConditionalGeneration">

```py
import torch, math
from datasets import load_dataset, Audio
from transformers import MoshiForConditionalGeneration, AutoFeatureExtractor, AutoTokenizer

feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/moshiko-pytorch-bf16")
tokenizer = AutoTokenizer.from_pretrained("kyutai/moshiko-pytorch-bf16")
model = MoshiForConditionalGeneration.from_pretrained("kyutai/moshiko-pytorch-bf16", dtype="auto")

librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
audio_sample = librispeech_dummy[-1]["audio"]["array"]
user_input_values = feature_extractor(raw_audio=audio_sample, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt").to(device=device, dtype=dtype)

moshi_input_values = torch.zeros_like(user_input_values.input_values)

num_tokens = math.ceil(moshi_input_values.shape[-1] * waveform_to_token_ratio)
input_ids = torch.ones((1, num_tokens), device=device, dtype=torch.int64) * tokenizer.encode("<pad>")[0]

output = model.generate(input_ids=input_ids, user_input_values=user_input_values.input_values, moshi_input_values=moshi_input_values, max_new_tokens=25)

text_tokens = output.sequences
audio_waveforms = output.audio_sequences
```

</hfoption>
</hfoptions>

## MoshiConfig

[[autodoc]] MoshiConfig

## MoshiDepthConfig

[[autodoc]] MoshiDepthConfig

## MoshiModel

[[autodoc]] MoshiModel
    - forward

## MoshiForCausalLM

[[autodoc]] MoshiForCausalLM
    - forward

## MoshiForConditionalGeneration

[[autodoc]] MoshiForConditionalGeneration
    - forward
    - generate
    - get_unconditional_inputs

