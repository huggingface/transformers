
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

The gemma3 model was proposed in [<INSERT PAPER NAME HERE>](<INSERT PAPER LINK HERE>) by Google. It is a 3B vision-language model composed by a [SigLIP](siglip) vision encoder and a [Gemma-2](gemma_2) language decoder linked by a multimodal linear projection. It cuts an image into a fixed number of tokens same way as Siglip if the image does not exceed certain aspect ratio. For images that exceed the givenn aspect ratio, it crops the image into multiple smaller pacthes and concatenates them with the base image embedding. One particularity is that the model uses bidiredctional attention on all the image tokens. Also the model interleaves sliding window local attention with full causal attention in the language backbone, where each sixth layer is a full causal attention.

This model was contributed by [INSERT](INSERT).


## Usage tips

- For image+text and image-only inputs use `Gemma3ForConditionalGeneration`.
- For text-only inputs use `Gemma3ForCausalLM` for generation to avoid loading the vision tower.
- Each sample can contain multiple images, and the number of images can vary between samples. However make sure to pass correctly batched images to the processor, where each batch is a list of one or more images.
- The text passed to the processor should have the `"<start_of_image_>"` token where the images should be inserted.
- The processor has its own `apply_chat_template` method to convert chat messages to text that can then be passed as text to the processor. You can also get a vectorized output from `apply_chat_template`. See the examples below for more details on how to use it.


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

### Text-onlty inference

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

## Gemma3Config

[[autodoc]] Gemma3Config

## Gemma3Model

[[autodoc]] Gemma3Model
    - forward

## Gemma3ForCausalLM

[[autodoc]] Gemma3ForCausalLM
    - forward

## Gemma3ForSequenceClassification

[[autodoc]] Gemma3ForSequenceClassification
    - forward

## Gemma3ForTokenClassification

[[autodoc]] Gemma3ForTokenClassification
    - forward
