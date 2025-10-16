<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2021-04-14 and added to Hugging Face Transformers on 2021-09-01.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Speech Encoder Decoder Models

SpeechEncoderDecoderModel creates speech-to-text models by combining pretrained speech encoders with autoregressive decoders. This approach works well for speech recognition and translation tasks, as shown in [Large-Scale Self- and Semi-Supervised Learning for Speech Translation](https://huggingface.co/papers/2104.06678).

<hfoptions id="usage">
<hfoption id="SpeechEncoderDecoderModel">

```py
import torch
from transformers import Wav2Vec2Processor, SpeechEncoderDecoderModel
from datasets import load_dataset

model = SpeechEncoderDecoderModel.from_pretrained("facebook/wav2vec2-xls-r-300m-en-to-15", dtype="auto")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xls-r-300m-en-to-15")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
input_values = processor(ds[0]["audio"]["array"], return_tensors="pt").input_values

generated_ids = model.generate(input_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)
```

</hfoption>
</hfoptions>

## Usage tips

- The model requires only 2 inputs to compute loss: `input_values` (speech inputs) and `labels` (input_ids of the encoded target sequence).

## SpeechEncoderDecoderConfig

[[autodoc]] SpeechEncoderDecoderConfig

## SpeechEncoderDecoderModel

[[autodoc]] SpeechEncoderDecoderModel
    - forward
    - from_encoder_decoder_pretrained