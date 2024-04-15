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

# Idefics2

## Overview

The Idefics2 model was created by the [Hugging Face M4](https://huggingface.co/HuggingFaceM4) team and authored by Léo Tronchon, Hugo Laurencon, Victor Sanh.
The accompanying blog post can be found [here](https://huggingface.co/blog/idefics2).

Idefics2 is an open multimodal model that accepts arbitrary sequences of image and text inputs and produces text
outputs. The model can answer questions about images, describe visual content, create stories grounded on multiple
images, or simply behave as a pure language model without visual inputs. It improves upon IDEFICS-1, notably on
document understanding, OCR, or visual reasoning. Idefics2 is lightweight (8 billion parameters) and treats
images in their native aspect ratio and resolution, which allows for varying inference efficiency.

Tips:
- Each sample can contain multiple images, and the number of images can vary between samples. The processor will pad the inputs to the maximum number of images in a batch for input to the model.
- The processor has a `do_image_splitting` option. If `True`, each input image will be split into 4 sub-images, and concatenated with the original to form 5 images. This is useful for increasing model performance. Make sure `processor.image_processor.do_image_splitting` is set to `False` if the model was not trained with this option.
- `text` passed to the processor should have the `<image>` tokens where the images should be inserted. And `<end_of_utterance>` at the end of each utterance if the text is a chat message.
- The processor has its own `apply_chat_template` method to convert chat messages to text that can then be passed as `text` to the processor.

Example of how to use the processor on chat messages:
```python
import requests
from PIL import Image
from transformers import Idefics2Processor, Idefics2ForConditionalGeneration

url_1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
url_2 = "http://images.cocodataset.org/val2017/000000219578.jpg"

image_1 = Image.open(requests.get(url_1, stream=True).raw)
image_2 = Image.open(requests.get(url_2, stream=True).raw)
images = [image_1, image_2]

messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "What’s the difference between these two images?"},
        {"type": "image"},
        {"type": "image"},
    ],
}]

processor = Idefics2Processor.from_pretrained("HuggingFaceM4/idefics2-8b")
model = Idefics2ForConditionalGeneration.from_pretrained("HuggingFaceM4/idefics2-8b")

text = processor.apply_chat_template(messages)
# "User: What’s the difference between these two images?<image><image><end_of_utterance>\n"
print(text)

inputs = processor(images=images, text=text)

generated_text = model.generate(**inputs)
```

This model was contributed by [amyeroberts](https://huggingface.co/amyeroberts).
The original code can be found [here](https://huggingface.co/HuggingFaceM4/idefics2).


## Idefics2Config

[[autodoc]] Idefics2Config


## Idefics2Model

[[autodoc]] Idefics2Model
    - forward


## Idefics2ForConditionalGeneration

[[autodoc]] Idefics2ForConditionalGeneration
    - forward


## Idefics2ImageProcessor
[[autodoc]] Idefics2ImageProcessor
    - preprocess


## Idefics2Processor
[[autodoc]] Idefics2Processor
    - __call__
