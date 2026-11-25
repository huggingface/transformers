<!--Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was published in HF papers on 2024-08-03 and contributed to Hugging Face Transformers on 2026-11-25.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# MiniCPM-o 2.6

[MiniCPM-o 2.6](https://huggingface.co/openbmb/MiniCPM-o-2_6) is an omni-modal model from [OpenBMB](https://github.com/OpenBMB) ([paper](https://huggingface.co/papers/2408.01800)) that adds speech to [MiniCPM-V 2.6](minicpmv2_6). Audio is encoded by a [Whisper](whisper)-medium encoder whose frames attend to their own 1 s chunk and to all earlier ones, then projected to the [Qwen2](qwen2) 7B language model and average-pooled to 25 tokens per second. Images are handled as in MiniCPM-V 2.6. This implementation covers image and audio understanding with text output; the checkpoint's speech decoder (`tts.*`) is not loaded.

The original code can be found [here](https://github.com/OpenBMB/MiniCPM-o).

## Usage example

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "openbmb/MiniCPM-o-2_6"
model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "url": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3"},
            {"type": "text", "text": "Please listen to the audio snippet carefully and transcribe the content."},
        ],
    }
]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device)
generated = model.generate(**inputs, max_new_tokens=64)
print(processor.post_process_image_text_to_text(generated[:, inputs["input_ids"].shape[1] :])[0])
```

## Notes

- In a prompt, each audio is marked with `<|audio|>`. The processor splits audio longer than 30 s into 30 s windows and expands the marker to 25 `<|audio|>` tokens per second of audio, each second wrapped in `<|audio_start|>` and `<|audio_end|>`.
- Images are marked with `<unk>` and expanded as in [MiniCPM-V 2.6](minicpmv2_6); the chat template adds `<image_id>N</image_id>` before each image.
- When the conversation contains audio, the chat template drops the default system prompt and opens the assistant turn with `<|spk_bos|><|spk|><|spk_eos|><|tts_bos|>`, as the original `chat()` does. The model then ends its answer with `<|tts_eos|>`, a special token.
- Every audio is encoded on its own. The original's omni mode, which merges the audios of one message (one per second of video) into a single stream, isn't supported.

## MiniCPMO2_6Config

[[autodoc]] MiniCPMO2_6Config

## MiniCPMO2_6VisionConfig

[[autodoc]] MiniCPMO2_6VisionConfig

## MiniCPMO2_6AudioConfig

[[autodoc]] MiniCPMO2_6AudioConfig

## MiniCPMO2_6VisionTransformer

[[autodoc]] MiniCPMO2_6VisionTransformer
    - forward

## MiniCPMO2_6AudioEncoder

[[autodoc]] MiniCPMO2_6AudioEncoder
    - forward

## MiniCPMO2_6Model

[[autodoc]] MiniCPMO2_6Model
    - forward
    - get_image_features
    - get_audio_features

## MiniCPMO2_6ForConditionalGeneration

[[autodoc]] MiniCPMO2_6ForConditionalGeneration
    - forward
    - get_image_features
    - get_audio_features

## MiniCPMO2_6Processor

[[autodoc]] MiniCPMO2_6Processor
    - __call__
