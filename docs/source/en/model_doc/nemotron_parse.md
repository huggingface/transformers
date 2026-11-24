<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-11-24.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Nemotron-Parse

[Nemotron-Parse](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-2.0) is NVIDIA's document parsing model, released as [v1.2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.2) and [2.0](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-2.0). It reads a page image and writes its text in reading order, with the bounding box and semantic class (title, table, caption, ...) of every element. A [C-RADIOv2-H](https://huggingface.co/nvidia/C-RADIOv2-H) vision encoder is followed by a small convolutional neck and an mBART-style decoder without position embeddings.

The task prompt selects the output format. The default prompt below asks for boxes, classes and markdown text.

```py
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.image_utils import load_image

model_id = "nvidia/NVIDIA-Nemotron-Parse-2.0"
model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

image = load_image("https://huggingface.co/datasets/hf-internal-testing/transformers-synthetic-assets/resolve/main/images/paddle_doc_test.jpg")
prompt = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=4096)
print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
```

The image processor follows the 2.0 preprocessing for both checkpoints: pages are shrunk with antialiasing and padded with white. v1.2's original processor pads with black and resizes without antialiasing, so v1.2 outputs can differ from it in single box coordinates.

Box coordinates are relative to the padded 1664x2048 page; the `postprocessing.py` script in the model repositories maps them back to the original image. For training, pass `labels` together with `decoder_input_ids` (the labels shifted right by one); the model does not derive `decoder_input_ids` from `labels`.

## NemotronParseConfig

[[autodoc]] NemotronParseConfig

## NemotronParseTextConfig

[[autodoc]] NemotronParseTextConfig

## NemotronParseImageProcessor

[[autodoc]] NemotronParseImageProcessor
    - preprocess

## NemotronParseProcessor

[[autodoc]] NemotronParseProcessor
    - __call__

## NemotronParseModel

[[autodoc]] NemotronParseModel
    - forward

## NemotronParseTextModel

[[autodoc]] NemotronParseTextModel
    - forward

## NemotronParseForConditionalGeneration

[[autodoc]] NemotronParseForConditionalGeneration
    - forward
