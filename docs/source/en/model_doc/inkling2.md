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


⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-11-03.*

# Inkling2

Inkling2 is a multimodal model from [Thinking Machines Lab](https://huggingface.co/thinkingmachines) that takes text and images and generates text. It keeps the decoder of [Inkling](./inkling), with learned relative attention, short causal convolutions and mixture-of-experts layers, and adds block attention residuals, gated attention with a learned sink and dynamic key/value convolutions. Every decoder layer is a mixture-of-experts layer. Images are encoded by a vision transformer based on the Perception Encoder and projected into the language model embeddings, one embedding per 40x40 pixel patch.

The example below generates text from an image and a prompt.
<TODO> update with autoprocessor only!
```py
import torch

from transformers import AutoModelForImageTextToText, AutoTokenizer, Inkling2ImageProcessor, Inkling2Processor
from transformers.image_utils import load_image

model_id = "thinkingmachines/model-share"
processor = Inkling2Processor(image_processor=Inkling2ImageProcessor(), tokenizer=AutoTokenizer.from_pretrained(model_id))
model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
prompt = (
    "<|message_user|><|content_image|><|unused_200054|><|end_message|>"
    "<|message_user|><|content_text|>What animal is in this image, and where is it?<|end_message|>"
)
inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(processor.decode(output[0, inputs["input_ids"].shape[1] :]))
```

## Notes

- The checkpoint stores routed experts in NVFP4 and MXFP8. They are dequantized to the model dtype when loading.
- Attention runs in eager mode only, because every layer adds a relative position bias and a sink.

## Inkling2Config

[[autodoc]] Inkling2Config

## Inkling2TextConfig

[[autodoc]] Inkling2TextConfig

## Inkling2VisionConfig

[[autodoc]] Inkling2VisionConfig

## Inkling2PreTrainedModel

[[autodoc]] Inkling2PreTrainedModel
    - forward

## Inkling2TextModel

[[autodoc]] Inkling2TextModel
    - forward

## Inkling2ForCausalLM

[[autodoc]] Inkling2ForCausalLM
    - forward

## Inkling2VisionModel

[[autodoc]] Inkling2VisionModel
    - forward

## Inkling2Model

[[autodoc]] Inkling2Model
    - forward

## Inkling2ForConditionalGeneration

[[autodoc]] Inkling2ForConditionalGeneration
    - forward

## Inkling2ImageProcessor

[[autodoc]] Inkling2ImageProcessor

## Inkling2Processor

[[autodoc]] Inkling2Processor
