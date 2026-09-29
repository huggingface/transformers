<!--Copyright 2026 the HuggingFace Team. All rights reserved.

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
*This model was contributed to Hugging Face Transformers on 2026-09-29.*


# Apertus 1.5

## Overview

Apertus 1.5 is a multimodal model from the [Swiss AI Initiative](https://huggingface.co/swiss-ai) that accepts
text, images, and audio and generates text. It extends the [Apertus](./apertus) language model with the
[EMU3.5 vision tokenizer](https://huggingface.co/BAAI/Emu3.5-VisionTokenizer) and [WavTokenizer](./wavtokenizer)
for audio.

The media tokenizers convert images and audio into discrete tokens, which the language backbone processes
alongside text in a shared input vocabulary. The output vocabulary contains only text tokens. The integration
includes the media encoders and quantizers; it does not support image or audio generation, or training the
vision tokenizer.

## Usage example

Use the processor's chat template to prepare a conversation with images and audio. It loads the referenced
media, resamples audio, and constructs the model inputs automatically.

```python
from transformers import Apertus1p5ForConditionalGeneration, AutoProcessor

model = Apertus1p5ForConditionalGeneration.from_pretrained(
    "swiss-ai/Apertus-v1.5-8B", device_map="auto"
)
processor = AutoProcessor.from_pretrained("swiss-ai/Apertus-v1.5-8B")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
            {"type": "audio", "url": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/belinda.wav"},
            {"type": "text", "text": "Describe the image and transcribe the audio."},
        ],
    }
]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device)

generated = model.generate(**inputs, max_new_tokens=128)
print(processor.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
```

## Usage notes

- **Tokenizer precision:** Both media tokenizers require `float32` for stable code assignments.
  `from_pretrained` preserves this precision when loading the language backbone in half precision. Avoid
  casting the entire model to a lower precision or running the tokenizers under mixed-precision autocast.
  If quantizing the model, ensure the backend excludes both media tokenizers.
- **Output vocabulary:** Returned logits cover only `text_config.output_vocab_size` when set, otherwise
  `text_config.vocab_size`. Input-only multimodal tokens cannot be generated. Mask these tokens with `-100`
  in training labels, and restrict generation constraints to tokens within the output vocabulary.

## Apertus1p5Config

[[autodoc]] Apertus1p5Config

## Apertus1p5TextConfig

[[autodoc]] Apertus1p5TextConfig

## Apertus1p5VisionTokenizerConfig

[[autodoc]] Apertus1p5VisionTokenizerConfig

## Apertus1p5Processor

[[autodoc]] Apertus1p5Processor
    - __call__

## Apertus1p5ImageProcessor

[[autodoc]] Apertus1p5ImageProcessor
    - preprocess

## Apertus1p5VisionTokenizerModel

[[autodoc]] Apertus1p5VisionTokenizerModel
    - encode

## Apertus1p5TextModel

[[autodoc]] Apertus1p5TextModel
    - forward

## Apertus1p5TextForCausalLM

[[autodoc]] Apertus1p5TextForCausalLM
    - forward

## Apertus1p5Model

[[autodoc]] Apertus1p5Model
    - forward

## Apertus1p5ForConditionalGeneration

[[autodoc]] Apertus1p5ForConditionalGeneration
    - forward
