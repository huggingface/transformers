
<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Gemma3

## Overview

The Gemma 3 model was proposed in the [Gemma 3 Techncial Report](https://goo.gle/Gemma3Report) by Google. It is a vision-language model composed by a [SigLIP](siglip) vision encoder and a [Gemma 2](gemma_2) language decoder, linked by a multimodal linear projection. It cuts an image into a fixed number of tokens, in the same way as SigLIP, as long as the image does not exceed certain aspect ratio. For images that exceed the given aspect ratio, it crops the image into multiple smaller patches and concatenates them with the base image embedding. One particularity is that the model uses bidirectional attention on all the image tokens. In addition, the model interleaves sliding window local attention with full causal attention in the language backbone, where each sixth layer is a full causal attention layer.

This model was contributed by [Ryan Mullins](https://huggingface.co/RyanMullins), [Raushan Turganbay](https://huggingface.co/RaushanTurganbay) [Arthur Zucker](https://huggingface.co/ArthurZ), and [Pedro Cuenca](https://huggingface.co/pcuenq).


## Usage tips


- For image+text and image-only inputs use `Gemma3ForConditionalGeneration`.
- For text-only inputs use `Gemma3ForCausalLM` for generation to avoid loading the vision tower.
- Each sample can contain multiple images, and the number of images can vary between samples. However, make sure to pass correctly batched images to the processor, where each batch is a list of one or more images.
- The text passed to the processor should have a `<start_of_image>` token wherever an image should be inserted.
- The processor has its own `apply_chat_template` method to convert chat messages to model inputs. See the examples below for more details on how to use it.


### Image cropping for high resolution images

The model supports cropping images into smaller patches when the image aspect ratio exceeds a certain value. By default the images are not cropped and only the base image is forwarded to the model. Users can set `do_pan_and_scan=True` to obtain several crops per image along with the base image to improve the quality in DocVQA or similar tasks requiring higher resolution images.

Pan and scan is an inference time optimization to handle images with skewed aspect ratios. When enabled, it improves performance on tasks related to document understanding, infographics, OCR, etc.

```python

processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it", padding_side="left")

url = "https://media.istockphoto.com/id/1192867753/photo/cow-in-berchida-beach-siniscola.jpg?s=612x612&w=0&k=20&c=v0hjjniwsMNfJSuKWZuIn8pssmD5h5bSN1peBd1CmH4="
messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are a helpful assistant."}
        ]
    },
    {
        "role": "user", "content": [
            {"type": "image", "url": url},
            {"type": "text", "text": "What is shown in this image?"},
        ]
    },
]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
    do_pan_and_scan=True,
).to(model.device)

```


## Usage Example

### Single-image Inference

```python
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

model_id = "google/gemma-3-4b-it"
model = Gemma3ForConditionalGeneration.from_pretrained(model_id, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id, padding_side="left")

url = "https://media.istockphoto.com/id/1192867753/photo/cow-in-berchida-beach-siniscola.jpg?s=612x612&w=0&k=20&c=v0hjjniwsMNfJSuKWZuIn8pssmD5h5bSN1peBd1CmH4="
messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are a helpful assistant."}
        ]
    },
    {
        "role": "user", "content": [
            {"type": "image", "url": url},
            {"type": "text", "text": "What is shown in this image?"},
        ]
    },
]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)

output = model.generate(**inputs, max_new_tokens=50)
print(processor.decode(output[0], skip_special_tokens=True)[inputs.input_ids.shape[1]: ])
```

### Multi-image Inference

```python
model_id = "google/gemma-3-4b-it"
model = Gemma3ForConditionalGeneration.from_pretrained(model_id, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id, padding_side="left")

url_cow = "https://media.istockphoto.com/id/1192867753/photo/cow-in-berchida-beach-siniscola.jpg?s=612x612&w=0&k=20&c=v0hjjniwsMNfJSuKWZuIn8pssmD5h5bSN1peBd1CmH4="
url_stop = "https://www.ilankelman.org/stopsigns/australia.jpg"
messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are a helpful assistant."}
        ]
    },
    {
        "role": "user", "content": [
            {"type": "image", "url": url_cow},
            {"type": "image", "url": url_stop},
            {"type": "text", "text": "Are these two images identical?"},
        ]
    },
]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)

output = model.generate(**inputs, max_new_tokens=50)
print(processor.decode(output[0], skip_special_tokens=True)[inputs.input_ids.shape[1]: ])

```

### Text-only inference

You can use the VLMs for text-only generation by omitting images in your input. However, you can also load the models in text-only mode as shown below. This will skip loading the vision tower and will save resources when you just need the LLM capabilities.
```python
from transformers import AutoTokenizer, Gemma3ForCausalLM

model_id = "google/gemma-3-1b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = Gemma3ForCausalLM.from_pretrained(model_id, device_map="auto")

input_ids = tokenizer("Write me a poem about Machine Learning.", return_tensors="pt").to(model.device)

outputs = model.generate(**input_ids, max_new_tokens=100)
text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

print(text)

```


## Gemma3ImageProcessor

[[autodoc]] Gemma3ImageProcessor

## Gemma3ImageProcessorFast

[[autodoc]] Gemma3ImageProcessorFast

## Gemma3Processor

[[autodoc]] Gemma3Processor

## Gemma3TextConfig

[[autodoc]] Gemma3TextConfig

## Gemma3Config

[[autodoc]] Gemma3Config

## Gemma3TextModel

[[autodoc]] Gemma3TextModel
    - forward

## Gemma3ForCausalLM

[[autodoc]] Gemma3ForCausalLM
    - forward

## Gemma3ForConditionalGeneration

[[autodoc]] Gemma3ForConditionalGeneration
    - forward
