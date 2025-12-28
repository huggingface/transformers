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

# Video-text-to-text

[[open-in-colab]]

Video-text-to-text, also known as video language models are models that can process video and output text. These models can tackle various tasks, from video question answering to video captioning.

These models have nearly the same architecture as [image-text-to-text](../image_text_to_text) models except for some changes to accept video data, since video data is essentially image frames with temporal dependencies. Some image-text-to-text models take in multiple images, but this alone is inadequate for a model to accept videos.

Moreover, video-text-to-text models are often trained with all vision modalities. Each example might have videos, multiple videos, images and multiple images. Some of these models can also take interleaved inputs. For example, you can refer to a specific video inside a string of text by adding a video token in text like "What is happening in this video? `<video>`".

Note that these models process videos with no audio. [Any-to-any](../any-to-any) models on the other hand can process videos with audio in them.

In this guide, we provide a brief overview of video LMs and show how to use them with Transformers for inference.

To begin with, there are multiple types of video LMs:

- base models used for fine-tuning
- chat fine-tuned models for conversation
- instruction fine-tuned models

This guide focuses on inference with an instruction-tuned model, [llava-hf/llava-onevision-qwen2-0.5b-ov-hf](https://huggingface.co/llava-hf/llava-interleave-qwen-7b-hf) which can take in interleaved data. Alternatively, you can try [llava-interleave-qwen-0.5b-hf](https://huggingface.co/llava-hf/llava-interleave-qwen-0.5b-hf) if your hardware doesn't allow running a 7B model.

Let's begin installing the dependencies.

```bash
pip install -q transformers accelerate flash_attn torchcodec
```

Let's initialize the model and the processor.

```python
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"

processor = AutoProcessor.from_pretrained(model_id, device="cuda")

model = LlavaForConditionalGeneration.from_pretrained(model_id, device_map="auto", dtype=torch.float16)
```

We will infer with two videos, both have cats.

<div class="container">
  <div class="video-container">
    <video width="400" controls>
      <source src="https://huggingface.co/spaces/merve/llava-interleave/resolve/main/cats_1.mp4" type="video/mp4">
    </video>
  </div>

  <div class="video-container">
    <video width="400" controls>
      <source src="https://huggingface.co/spaces/merve/llava-interleave/resolve/main/cats_2.mp4" type="video/mp4">
    </video>
  </div>
</div>

Videos are series of image frames. Depending on the hardware limitations, downsampling is required. If the number of downsampled frames are too little, predictions will be low quality.

Video-text-to-text models have processors with video processor abstracted in them. You can pass video inference related arguments to [`~ProcessorMixin.apply_chat_template`] function.

> [!WARNING]
> You can learn more about video processors [here](../main_classes/video_processor).

We can define our chat history, passing in video with a URL like below.

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "https://huggingface.co/spaces/merve/llava-interleave/resolve/main/cats_1.mp4"},
            {"type": "text", "text": "Describe what is happening in this video."},
        ],
    }
]
```

You can preprocess the videos by passing in messages, setting `do_sample_frames` to True and passing in `num_frames`. Here we sample 10 frames.

```python
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
    num_frames=10,
    do_sample_frames=True
)
inputs.to(model.device)
```

The inputs contain `input_ids` for tokenized text, `pixel_values_videos` for 10 frames and `attention_mask` for which tokens .

We can now infer with our preprocessed inputs and decode them.

```python
generated_ids = model.generate(**inputs, max_new_tokens=128)
input_length = len(inputs["input_ids"][0])
output_text = processor.batch_decode(
    generated_ids[:, input_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False
)
output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text[0])

#"The video features a fluffy, long-haired cat with a mix of brown and white fur, lying on a beige carpeted floor. The cat's eyes are wide open, and its whiskers are prominently visible. The cat appears to be in a relaxed state, with its head slightly"
```

You can also interleave multiple videos with text directly in chat template like below.

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Here's a video."},
            {"type": "video", "video": "https://huggingface.co/spaces/merve/llava-interleave/resolve/main/cats_1.mp4"},
            {"type": "text", "text": "Here's another video."},
            {"type": "video", "video": "https://huggingface.co/spaces/merve/llava-interleave/resolve/main/cats_2.mp4"},
            {"type": "text", "text": "Describe similarities in these videos."},
        ],
    }
]
```

The inference remains the same as the previous example.

```python
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
    num_frames=100,
    do_sample_frames=True
)
inputs.to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=50)
input_length = len(inputs["input_ids"][0])
output_text = processor.batch_decode(
    generated_ids[:, input_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
#['Both videos feature a cat with a similar appearance, characterized by a fluffy white coat with black markings, a pink nose, and a pink tongue. The cat\'s eyes are wide open, and it appears to be in a state of alertness or excitement. ']
```
