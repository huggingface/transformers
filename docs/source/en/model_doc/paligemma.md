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

# PaliGemma

## Overview

The PaliGemma model was proposed in [PaliGemma – Google's Cutting-Edge Open Vision Language Model](https://huggingface.co/blog/paligemma) by Google. It is a 3B vision-language model composed by a [SigLIP](siglip) vision encoder and a [Gemma](gemma) language decoder linked by a multimodal linear projection. It cuts an image into a fixed number of VIT tokens and prepends it to an optional prompt. One particularity is that the model uses full block attention on all the image tokens plus the input text tokens. It comes in 3 resolutions, 224x224, 448x448 and 896x896 with 3 base models, with 55 fine-tuned versions for different tasks, and 2 mix models.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/paligemma/paligemma_arch.png"
alt="drawing" width="600"/>

<small> PaliGemma architecture. Taken from the <a href="https://huggingface.co/blog/paligemma">blog post.</a> </small>

This model was contributed by [Molbap](https://huggingface.co/Molbap).

## Usage tips

Inference with PaliGemma can be performed as follows:

```python
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

model_id = "google/paligemma-3b-mix-224"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

prompt = "What is on the flower?"
image_file = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg?download=true"
raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(prompt, raw_image, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=20)

print(processor.decode(output[0], skip_special_tokens=True)[len(prompt):])
```

- PaliGemma is not meant for conversational use, and it works best when fine-tuning to a specific use case. Some downstream tasks on which PaliGemma can be fine-tuned include image captioning, visual question answering (VQA), object detection, referring expression segmentation and document understanding.
- One can use `PaliGemmaProcessor` to prepare images, text and optional labels for the model. When fine-tuning a PaliGemma model, the `suffix` argument can be passed to the processor which creates the `labels` for the model:

```python
prompt = "What is on the flower?"
answer = "a bee"
inputs = processor(text=prompt, images=raw_image, suffix=answer, return_tensors="pt")
```

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with PaliGemma. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

- A blog post introducing all the features of PaliGemma can be found [here](https://huggingface.co/blog/paligemma).
- Demo notebooks on how to fine-tune PaliGemma for VQA with the Trainer API along with inference can be found [here](https://github.com/huggingface/notebooks/tree/main/examples/paligemma).
- Demo notebooks on how to fine-tune PaliGemma on a custom dataset (receipt image -> JSON) along with inference can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/PaliGemma). 🌎

## PaliGemmaConfig

[[autodoc]] PaliGemmaConfig

## PaliGemmaProcessor

[[autodoc]] PaliGemmaProcessor

## PaliGemmaForConditionalGeneration

[[autodoc]] PaliGemmaForConditionalGeneration
    - forward
