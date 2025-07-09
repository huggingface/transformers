<!--Copyright 2025 The Keye Team and The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">    </div>
</div>

# Keye-VL
[Keye-VL](https://huggingface.co/papers/2507.01949) is an 8-billion-parameter multimodal foundation model, excels in short-video understanding while maintaining robust general-purpose vision-language abilities through a comprehensive pre- and post-training process, including reinforcement learning and alignment.


You can find the original Keye-VL checkpoint under the [Keye-VL](https://huggingface.co/Kwai-Keye) collection.

The example below demonstrates how to generate text based on an image with the [`AutoModel`] class.

<hfoptions id="usage">

<hfoption id="AutoModel">

```py
import torch
from transformers import KeyeForConditionalGeneration, AutoProcessor
from PIL import Image
import requests

model = KeyeForConditionalGeneration.from_pretrained(
    "Kwai-Keye/Keye-VL-8B-Preview",
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained("Kwai-Keye/Keye-VL-8B-Preview", trust_remote_code=True)
url = "https://s1-11508.kwimgs.com/kos/nlav11508/mllm_all/ziran_jiafeimao_11.jpg"
messages = [
    {
        "role":"user",
        "content":[
            {
                "type":"image",
                "image": url,
            },
            {
                "type":"text",
                "text":"Describe this image."
            }
        ]
    }

]

image_inputs = [Image.open(requests.get(url, stream=True).raw)]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=None,
    padding=True,
    return_tensors="pt",
).to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(output_text)
```
</hfoption>

### Notes

- Use Keye-VL for video inputs by setting `"type": "video"` as shown below.
    ```python
    # pip3 install keye_vl_utils
    from keye_vl_utils import process_vision_info
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "/path/to/video.mp4"},
                {"type": "text", "text": "What happened in the video?"},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(model.device)
    
    # Inference: Generation of the output
    output_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(output_text)
    ```
- Use Keye-VL for a mixed batch of inputs (images, videos, text).
    ```python
    import torch
    from transformers import KeyeForConditionalGeneration, AutoProcessor
    
    model = KeyeForConditionalGeneration.from_pretrained(
        "Kwai-Keye/Keye-VL-8B-Preview",
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    processor = AutoProcessor.from_pretrained("Kwai-Keye/Keye-VL-8B-Preview")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"}, 
                {"type": "text", "text": "Hello, how are you?"}
            ]
        },
        {
            "role": "assistant",
            "content": "I'm doing well, thank you for asking. How can I assist you today?"
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Can you describe these images and video?"}, 
                {"type": "image"}, 
                {"type": "image"}, 
                {"type": "video"}, 
                {"type": "text", "text": "These are from my vacation."}
            ]
        },
        {
            "role": "assistant",
            "content": "I'd be happy to describe the images and video for you. Could you please provide more context about your vacation?"
        },
        {
            "role": "user",
            "content": "It was a trip to the mountains. Can you see the details in the images and video?"
        }
    ]
    
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Hello, how are you?<|im_end|>\n<|im_start|>assistant\nI'm doing well, thank you for asking. How can I assist you today?<|im_end|>\n<|im_start|>user\nCan you describe these images and video?<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|><|vision_start|><|video_pad|><|vision_end|>These are from my vacation.<|im_end|>\n<|im_start|>assistant\nI'd be happy to describe the images and video for you. Could you please provide more context about your vacation?<|im_end|>\n<|im_start|>user\nIt was a trip to the mountains. Can you see the details in the images and video?<|im_end|>\n<|im_start|>assistant\n'
    
    ```

## KeyeConfig

[[autodoc]] KeyeConfig

## KeyeProcessor

[[autodoc]] KeyeProcessor

## KeyeForConditionalGeneration

[[autodoc]] KeyeForConditionalGeneration
    - forward
