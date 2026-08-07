<!--Copyright 2026 The rednote-hilab team and the HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-08-07.*

# Dots3-Note Omni

Dots3-Note Omni is a mixture-of-experts causal language model with native text, image, video, and audio inputs. It uses
a shared vision encoder for images and videos and a Whisper-style audio encoder. Both encoders project their outputs
into the language model's hidden space before autoregressive text generation.

Dots3-Note Omni checkpoints are available with BF16 weights or with fine-grained FP8 language-model weights and BF16
vision, audio, and language-model-head weights. Both formats load through the standard Transformers APIs.

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor


model_id = "rednote-hilab/dots3.note.omni"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

output_ids = model.generate(**inputs, max_new_tokens=128)
print(processor.batch_decode(output_ids, skip_special_tokens=True)[0])
```

## Dots3NoteOmniConfig

[[autodoc]] Dots3NoteOmniConfig

## Dots3NoteOmniVisionConfig

[[autodoc]] Dots3NoteOmniVisionConfig

## Dots3NoteOmniAudioConfig

[[autodoc]] Dots3NoteOmniAudioConfig

## Dots3NoteOmniForCausalLM

[[autodoc]] Dots3NoteOmniForCausalLM
    - forward

## Dots3NoteOmniForConditionalGeneration

[[autodoc]] Dots3NoteOmniForConditionalGeneration

## Dots3NoteOmniTextModel

[[autodoc]] Dots3NoteOmniTextModel
    - forward

## Dots3NoteOmniTextForCausalLM

[[autodoc]] Dots3NoteOmniTextForCausalLM
    - forward

## Dots3NoteOmniVisionModel

[[autodoc]] Dots3NoteOmniVisionModel
    - forward

## Dots3NoteOmniAudioModel

[[autodoc]] Dots3NoteOmniAudioModel
    - forward

## Dots3NoteOmniProcessor

[[autodoc]] Dots3NoteOmniProcessor
    - __call__

## Dots3NoteOmniFeatureExtractor

[[autodoc]] Dots3NoteOmniFeatureExtractor
