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

# Qwen2_VL


## Overview

The Qwen2_VL is the new model series of large vision-language models from the Qwen team. 

#### What’s New in Qwen2-VL?

1. Enhanced Image Comprehension: We've significantly improved the model's ability to understand and interpret visual information, setting new benchmarks across key performance metrics.
2. Advanced Video Understanding: Qwen2-VL now features superior online streaming capabilities, enabling real-time analysis of dynamic video content with remarkable accuracy.
3. Integrated Visual Agent Functionality: Our model now seamlessly incorporates sophisticated system integration, transforming Qwen2-VL into a powerful visual agent capable of complex reasoning and decision-making.
4. Expanded Multilingual Support: We've broadened our language capabilities to better serve a diverse global user base, making Qwen2-VL more accessible and effective across different linguistic contexts.



## Usage example

### Single image inference

```python
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

# Load the model in half-precision on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# You can directly insert a local file path, a URL, or a base64-encoded image into the position where you want in the text.
## Local file path
messages = [{"role": "user", "content": [{"type": "image", "image": "file:///path/to/your/image.jpg"}, {"type": "text", "text": "Describe this image."}]}]
## Image URL
messages = [{"role": "user", "content": [{"type": "image", "image": "http://path/to/your/image.jpg"}, {"type": "text", "text": "Describe this image."}]}]
## Base64 encoded image
messages = [{"role": "user", "content": [{"type": "image", "image": "data:image;base64,/9j/..."}, {"type": "text", "text": "Describe this image."}]}]

# Model dynamically adjusts image size, specify dimensions if required.
messages = [{"role": "user", "content": [{"type": "image", "image": "file:///path/to/your/image.jpg", "resized_height": 280, "resized_width": 420}, {"type": "text", "text": "Describe this image."}]}]

# Preparation for inference
text, vision_infos = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=[text], vision_infos=[vision_infos], padding=True, return_tensors="pt")
inputs = inputs.to('cuda')

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
output_text = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(output_text)
```

### Multi image inference


```python

# Messages containing multiple images and a text query
messages = [{"role": "user", "content": [{"type": "image", "image": "file:///path/to/image1.jpg"}, {"type": "image", "image": "file:///path/to/image2.jpg"}, {"type": "text", "text": "Identify the similarities between these images."}]}]

# Preparation for inference
text, vision_infos = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=[text], vision_infos=[vision_infos], padding=True, return_tensors="pt")
inputs = inputs.to('cuda')

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
output_text = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(output_text)
```

### Video inference


```python


# Messages containing a images list as a video and a text query
messages = [{"role": "user", "content": [{"type": "video", "video": ["file:///path/to/frame1.jpg", "file:///path/to/frame2.jpg", "file:///path/to/frame3.jpg", "file:///path/to/frame4.jpg"], 'fps': 1.0}, {"type": "text", "text": "Describe this video."}]}]
# Messages containing a video and a text query
messages = [{"role": "user", "content": [{"type": "video", "video": "file:///path/to/video1.mp4", 'max_pixels': 360*420, 'fps': 1.0}, {"type": "text", "text": "Describe this video."}]}]

# Preparation for inference
text, vision_infos = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=[text], vision_infos=[vision_infos], padding=True, return_tensors="pt")
inputs = inputs.to('cuda')

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
output_text = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(output_text)
```


### Batch inference

```python

# Sample messages for batch inference
messages1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "file:///path/to/image1.jpg"},
            {"type": "image", "image": "file:///path/to/image2.jpg"},
            {"type": "text", "text": "What are the common elements in these pictures?"},
        ],
    }
]
messages2 = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Who are you?"}]
# Combine messages for batch processing
messages = [messages1, messages1]

# Preparation for batch inference
texts, vision_infos = zip(*[processor.apply_chat_template(msg, add_generation_prompt=True) for msg in messages])
inputs = processor(text=texts, vision_infos=vision_infos, padding=True, return_tensors="pt")
inputs = inputs.to('cuda')

# Batch Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
output_texts = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(output_texts)
```



### We strongly recommend using Flash-Attention 2 to speed up generation

First make sure to install flash-attn. Refer to the [original repository of Flash Attention](https://github.com/Dao-AILab/flash-attention) regarding that package installation. Simply set the `use_flash_attention=True` in the `config.json` file.


## Qwen2VLConfig

[[autodoc]] Qwen2VLConfig

## Qwen2VLImageProcessor

[[autodoc]] Qwen2VLImageProcessor
    - preprocess

## Qwen2VLProcessor

[[autodoc]] Qwen2VLProcessor

## Qwen2VLModel

[[autodoc]] Qwen2VLModel
    - forward

## Qwen2VLForConditionalGeneration

[[autodoc]] Qwen2VLForConditionalGeneration
    - forward
