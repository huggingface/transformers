<!--Copyright 2026 OpenMOSS and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was published in HF papers on None and contributed to Hugging Face Transformers on 2026-06-14.*

# MOSS-TTS Delay

[MOSS-TTS-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5) is a multilingual text-to-speech model from
OpenMOSS. The model uses a Qwen3 language backbone and predicts delayed audio codebook tokens for speech generation.

## Usage

```python
from scipy.io.wavfile import write

from transformers import AutoProcessor, AutoModelForTextToWaveform


model_id = "OpenMOSS-Team/MOSS-TTS-v1.5"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForTextToWaveform.from_pretrained(model_id, dtype="auto", device_map="auto")

message = processor.build_user_message(text="Hello from MOSS-TTS.", language="English")
inputs = processor([message]).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=1024)

messages = processor.decode(outputs)
audio_values = messages[0].audio_codes_list[0]
write("moss_tts.wav", processor.model_config.sampling_rate, audio_values.cpu().numpy())
```

## MossTTSDelayConfig

[[autodoc]] MossTTSDelayConfig

## MossTTSDelayProcessor

[[autodoc]] MossTTSDelayProcessor

## Message

[[autodoc]] Message

## UserMessage

[[autodoc]] UserMessage

## AssistantMessage

[[autodoc]] AssistantMessage

## MossTTSDelayModel

[[autodoc]] MossTTSDelayModel
    - forward

## MossTTSDelayOutputWithPast

[[autodoc]] MossTTSDelayOutputWithPast
