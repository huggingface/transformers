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
*This model was released on {release_date} and added to Hugging Face Transformers on 2025-12-24.*


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

```py
from transformers import AutoModelForSeq2SeqLM, AutoProcessor

processor = AutoProcessor.from_pretrained("zai-org/GLM-ASR-Nano-2512")
model = AutoModelForSeq2SeqLM.from_pretrained("zai-org/GLM-ASR-Nano-2512", dtype="auto", device_map="auto")

inputs = processor.apply_transcription_request("https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3")

inputs = inputs.to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)

decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)
print(decoded_outputs)
```
</hfoption>
</hfoptions>

### Advanced usage

The processor's `apply_transcription_request` is equivalent to using the chat template in the following manner:

```py
from transformers import GlmAsrForConditionalGeneration, AutoProcessor

processor = GlmAsrForConditionalGeneration.from_pretrained("zai-org/GLM-ASR-Nano-2512")

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
```

One can also use audio arrays directly:

```py
from transformers import GlmAsrForConditionalGeneration, AutoProcessor
from datasets import load_dataset

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

```py
from transformers import GlmAsrForConditionalGeneration, AutoProcessor

processor = AutoProcessor.from_pretrained("zai-org/GLM-ASR-Nano-2512")
model = GlmAsrForConditionalGeneration.from_pretrained("zai-org/GLM-ASR-Nano-2512", dtype="auto", device_map="auto")

inputs = processor.apply_transcription_request([
    "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3",
    "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3",
])

inputs = inputs.to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)

decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)
print(decoded_outputs)
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