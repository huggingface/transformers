<!--Copyright 2025 The Vita Team and The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Long_Vita

## Overview
The long_vita model was proposed in [<Long-VITA: Scaling Large Multi-modal Models to 1 Million Tokens with Leading Short-Context Accuracy>](<https://arxiv.org/pdf/2502.05177>) by Yunhang Shen, Chaoyou Fu, Shaoqi Dong, Xiong Wang, Yi-Fan Zhang, Peixian Chen, Mengdan Zhang, Haoyu Cao, Ke Li, Xiawu Zheng, Yan Zhang, Yiyi Zhou, Ran He, Caifeng Shan, Rongrong Ji, Xing Sun.

Long-VITA is a strong long-context visual language model and supports more than 1 million tokens.


The abstract from the paper is the following:

*< We introduce Long-VITA, a simple yet effective large multi-modal model for long-context visual-language understanding tasks. It is adept at concurrently processing and analyzing modalities of image, video, and text over 4K frames or 1M tokens while delivering advanced performances on short-context multi-modal tasks. We propose an effective multi-modal training schema that starts with large language models and proceeds through vision-language alignment, general knowledge learning, and two sequential stages of long-sequence fine-tuning. We further implement context-parallelism distributed inference and logits-masked language modeling head to scale Long-VITA to infinitely long inputs of images and texts during model inference. Regarding training data, Long-VITA is built on a mix of 17M samples from public datasets only and demonstrates the state-of-the-art performance on various multi-modal benchmarks, compared against recent cutting-edge models with internal data. Long-VITA is fully reproducible and supports both NPU and GPU platforms for training and testing. By leveraging our inference designs, Long-VITA models achieve a remarkable 2× prefill speedup and 4× context length extension in single node with 8 GPUs. We hope Long-VITA can serve as a competitive baseline and offer valuable insights for the open-source community in advancing long-context multi-modal understanding.>*

This model was contributed by [VITA Team](https://huggingface.co/<VITA-MLLM/Long-VITA-16K_HF>).
The original code can be found [here](<https://github.com/VITA-MLLM/Long-VITA>).

## Usage example

### Inference with a single image
This example demonstrates how to perform inference on a single image with the Long_vita model using chat templates.

```python
from transformers import AutoTokenizer
from transformers import pipeline

chat_template = (
        "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n'}}"
            "{% if message['content'] is string %}"
                "{{ message['content'] }}"
            "{% else %}"
                "{% for content in message['content'] %}"
                    "{% if content['type'] == 'image' %}"
                        "{{ '<image>\n' }}"
                    "{% elif content['type'] == 'video' %}"
                        "{{ '<video>' + '\n'}}"
                        "{{'<video_path>' + content['video'] + '<video_path>' }}"
                    "{% elif content['type'] == 'text' %}"
                        "{{ content['text'] }}"
                    "{% endif %}"
                "{% endfor %}"
            "{% endif %}"
            "{{'<|im_end|>\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
            "{{'<|im_start|>assistant\n' }}"
        "{% endif %}"
    )

model_name = 'VITA-MLLM/Long-VITA-16K_HF'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, chat_template=chat_template)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "http://images.cocodataset.org/val2017/000000039769.jpg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    },
]
pipe = pipeline("image-text-to-text", model="VITA-MLLM/Long-VITA-16K_HF", device_map="sequential", chat_template=chat_template, torch_dtype=torch.bfloat16)
outputs = pipe(text=messages, max_new_tokens=50, return_full_text=False)
outputs[0]["generated_text"]
print(outputs[0]["generated_text"])

'The image depicts two cats lying on a pink sofa. The sofa is plush and has a smooth, velvety texture. Both cats are tabby with distinct black and brown stripes, which are characteristic of the tabby pattern in feline fur.'
```

### Inference with multiple images or videos
This example demonstrates how to perform inference on multiple images or videos with the Long_vita model using chat templates.

```python
from transformers import AutoTokenizer
from transformers import pipeline

chat_template = (
        "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n'}}"
            "{% if message['content'] is string %}"
                "{{ message['content'] }}"
            "{% else %}"
                "{% for content in message['content'] %}"
                    "{% if content['type'] == 'image' %}"
                        "{{ '<image>\n' }}"
                    "{% elif content['type'] == 'video' %}"
                        "{{ '<video>' + '\n'}}"
                        "{{'<video_path>' + content['video'] + '<video_path>' }}"
                    "{% elif content['type'] == 'text' %}"
                        "{{ content['text'] }}"
                    "{% endif %}"
                "{% endfor %}"
            "{% endif %}"
            "{{'<|im_end|>\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
            "{{'<|im_start|>assistant\n' }}"
        "{% endif %}"
    )

model_name = 'VITA-MLLM/Long-VITA-16K_HF'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, chat_template=chat_template)

messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "http://images.cocodataset.org/val2017/000000039769.jpg",
                },
                {
                    "type": "image",
                    "image": "http://images.cocodataset.org/val2017/000000039769.jpg",
                },
                {
                    "type": "video",
                    "video": "local_path/test_video.mp4"
                },
                {
                    "type": "video",
                    "video": "local_path/test_video.mp4"
                },
                {
                    "type": "text",
                    "text": "Describe this video."
                },
            ],
        },
    ]
pipe = pipeline("image-text-to-text", model="VITA-MLLM/Long-VITA-16K_HF", device_map="sequential", chat_template=chat_template, torch_dtype=torch.bfloat16)
outputs = pipe(text=messages, max_new_tokens=50, return_full_text=False)
outputs[0]["generated_text"]
```

## Long_vitaConfig

[[autodoc]] Long_vitaConfig

## Long_vitaProcessor

[[autodoc]] Long_vitaProcessor

## Long_vitaModel

[[autodoc]] Long_vitaModel
    - forward

## Long_vitaForConditionalGeneration

[[autodoc]] Long_vitaForConditionalGeneration
    - forward
