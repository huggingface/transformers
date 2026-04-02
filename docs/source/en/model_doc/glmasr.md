<!--Copyright 2025 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was released on 2025-12-08 and added to Hugging Face Transformers on 2025-12-24.*

# GlmAsr

## Overview

**GLM-ASR-Nano-2512** is a robust, open-source speech recognition model with **1.5B parameters**. Designed for
real-world complexity, it outperforms OpenAI Whisper V3 on multiple benchmarks while maintaining a compact size.

Key capabilities include:

* **Exceptional Dialect Support**
  Beyond standard Mandarin and English, the model is highly optimized for **Cantonese (粤语)** and other dialects,
  effectively bridging the gap in dialectal speech recognition.

* **Low-Volume Speech Robustness**
  Specifically trained for **"Whisper/Quiet Speech"** scenarios. It captures and accurately transcribes extremely
  low-volume audio that traditional models often miss.

* **SOTA Performance**
  Achieves the **lowest average error rate (4.10)** among comparable open-source models, showing significant advantages
  in Chinese benchmarks (Wenet Meeting, Aishell-1, etc..).

This model was contributed by [Eustache Le Bihan](https://huggingface.co/eustlb) and [Yuxuan Zhang](https://huggingface.co/ZHANGYUXUAN-zR).
you can check the [model card](https://huggingface.co/zai-org/GLM-ASR-Nano-2512) for more details and our
[github repo](https://github.com/zai-org/GLM-ASR).

## Usage

### Basic usage

<options id="usage">
<hfoption id="AutoModel">

```py runnable:test_basic
# pytest-decorator: transformers.testing_utils.slow, transformers.testing_utils.require_torch
from transformers import AutoModelForSeq2SeqLM, AutoProcessor

processor = AutoProcessor.from_pretrained("zai-org/GLM-ASR-Nano-2512")
model = AutoModelForSeq2SeqLM.from_pretrained("zai-org/GLM-ASR-Nano-2512", dtype="auto", device_map="auto")

inputs = processor.apply_transcription_request("https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3")

inputs = inputs.to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)

decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)
assert len(decoded_outputs) == 1  # nodoc
print(decoded_outputs)
```

</hfoption>
</hfoptions>

### Advanced usage

The processor's `apply_transcription_request` is equivalent to using the chat template in the following manner:

```py runnable:test_advanced
# pytest-decorator: transformers.testing_utils.slow, transformers.testing_utils.require_torch
from transformers import GlmAsrForConditionalGeneration, AutoProcessor

processor = AutoProcessor.from_pretrained("zai-org/GLM-ASR-Nano-2512")
model = GlmAsrForConditionalGeneration.from_pretrained("zai-org/GLM-ASR-Nano-2512", dtype="auto", device_map="auto")

inputs = processor.apply_transcription_request("https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3")

# which is equivalent to
conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "url": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3",
            },
            {"type": "text", "text": "Please transcribe this audio into text"},
        ],
    },
]

inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
)

inputs = inputs.to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)

decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)
print(decoded_outputs)
```

One can also use audio arrays directly:

```py runnable:test_audio_array
# pytest-decorator: transformers.testing_utils.slow, transformers.testing_utils.require_torch
from transformers import GlmAsrForConditionalGeneration, AutoProcessor
from datasets import load_dataset, Audio

processor = AutoProcessor.from_pretrained("zai-org/GLM-ASR-Nano-2512")
model = GlmAsrForConditionalGeneration.from_pretrained("zai-org/GLM-ASR-Nano-2512", dtype="auto", device_map="auto")

# loading audio directly from dataset
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
audio_array = ds[0]["audio"]["array"]

inputs = processor.apply_transcription_request(audio_array)

inputs = inputs.to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)

decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)
print(decoded_outputs)
```

### Batched inference

You can process multiple audio files at once:

```py runnable:test_batched
# pytest-decorator: transformers.testing_utils.slow, transformers.testing_utils.require_torch
import torch
from transformers import AutoProcessor, GlmAsrForConditionalGeneration

checkpoint_name = "zai-org/GLM-ASR-Nano-2512"
processor = AutoProcessor.from_pretrained(checkpoint_name)

conversation = [
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "url": "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/bcn_weather.mp3",
                },
                {"type": "text", "text": "Please transcribe this audio into text"},
            ],
        },
    ],
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "url": "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/obama2.mp3",
                },
                {"type": "text", "text": "Please transcribe this audio into text"},
            ],
        },
    ],
]

model = GlmAsrForConditionalGeneration.from_pretrained(checkpoint_name, device_map="auto", dtype="auto")

inputs = processor.apply_chat_template(
    conversation, tokenize=True, add_generation_prompt=True, return_dict=True
).to(model.device, dtype=model.dtype)

inputs_transcription = processor.apply_transcription_request(
    [
        "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/bcn_weather.mp3",
        "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/obama2.mp3",
    ],
).to(model.device, dtype=model.dtype)

for key in inputs:  # doc-builder: ignore-bare-assert
    assert torch.equal(inputs[key], inputs_transcription[key])

outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)

decoded_outputs = processor.batch_decode(
    outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
)

EXPECTED_OUTPUT = [
    "Yesterday it was thirty five degrees in Barcelona, but today the temperature will go down to minus twenty degrees.",
    "This week, I traveled to Chicago to deliver my final farewell address to the nation, following in the tradition of presidents before me. It was an opportunity to say thank you. Whether we've seen eye to eye or rarely agreed at all, my conversations with you, the American people, in living rooms and schools, at farms and on factory floors, at diners and on distant military outposts, all these conversations are what have kept me honest, kept me inspired, and kept me going. Every day, I learned from you. You made me a better president, and you made me a better man. Over the",
]
assert decoded_outputs == EXPECTED_OUTPUT
```

## GlmAsrEncoderConfig

[[autodoc]] GlmAsrEncoderConfig

## GlmAsrConfig

[[autodoc]] GlmAsrConfig

## GlmAsrPreTrainedModel

[[autodoc]] GlmAsrPreTrainedModel
    - forward

## GlmAsrProcessor

[[autodoc]] GlmAsrProcessor
    - __call__

## GlmAsrEncoder

[[autodoc]] GlmAsrEncoder
    - forward

## GlmAsrForConditionalGeneration

[[autodoc]] GlmAsrForConditionalGeneration
    - forward
