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
*This model was released on 2024-05-03 and added to Hugging Face Transformers on 2024-04-15 and contributed by [amyeroberts](https://huggingface.co/amyeroberts).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Idefics2

[Idefics2](https://huggingface.co/papers/2405.02246) is an open multimodal model designed to handle arbitrary sequences of image and text inputs, producing text outputs. It excels in answering questions about images, describing visual content, and creating stories based on multiple images, while also functioning as a pure language model without visual inputs. Idefics2 improves upon its predecessor, IDEFICS-1, in areas such as document understanding, OCR, and visual reasoning. The model is lightweight, featuring 8 billion parameters, and processes images in their native aspect ratio and resolution, enhancing inference efficiency. Idefics2 achieves state-of-the-art performance within its size category across various multimodal benchmarks, often matching the performance of models four times its size.

<hfoptions id="usage">
<hfoption id="AutoModelForVision2Seq">

```py
import requests
import torch
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
image2 = load_image("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg")
image3 = load_image("https://cdn.britannica.com/68/170868-050-8DDE8263/Golden-Gate-Bridge-San-Francisco.jpg")

processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b-base")
model = AutoModelForVision2Seq.from_pretrained("HuggingFaceM4/idefics2-8b-base", dtype="auto")

BAD_WORDS_IDS = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
EOS_WORDS_IDS = [processor.tokenizer.eos_token_id]

prompts = [
  "<image>In this image, we can see the city of New York, and more specifically the Statue of Liberty.<image>In this image,",
  "In which city is that bridge located?<image>",
]
images = [[image1, image2], [image3]]
inputs = processor(images=images, text=prompts, padding=True, return_tensors="pt").to("cuda")

generated_ids = model.generate(**inputs, bad_words_ids=BAD_WORDS_IDS, max_new_tokens=20)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_texts)
```

</hfoption>
</hfoptions>

## Usage tips

- Each sample contains multiple images with varying counts. The processor pads inputs to the maximum number of images in a batch.
- Enable `do_image_splitting=True` to split each input image into 4 sub-images and concatenate with the original to form 5 images. This improves model performance. Set `processor.image_processor.do_image_splitting=False` if the model wasn't trained with this option.
- Add `<image>` tokens where images should be inserted in text. Add `<end_of_utterance>` at the end of each utterance for chat messages.
- The processor includes [`apply_chat_template`] to convert chat messages to text for processing.
- During training, determine which tokens the model shouldn't learn. For Idefics2, this typically includes image and padding tokens.
- For multi-turn conversations between user and assistant, set all tokens corresponding to user messages to -100.

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

## Idefics2ImageProcessorFast

[[autodoc]] Idefics2ImageProcessorFast
    - preprocess

## Idefics2Processor

[[autodoc]] Idefics2Processor
    - __call__

