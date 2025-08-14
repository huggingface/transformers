<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Multimodal templates

Multimodal model chat templates expect a similar [template](./chat_templating) as text-only models. It needs `messages` that includes a dictionary of the `role` and `content`.

Multimodal templates are included in the [Processor](./processors) class and require an additional `type` key for specifying whether the included content is an image, video, or text.

This guide will show you how to format chat templates for multimodal models as well as some best practices for configuring the template

## ImageTextToTextPipeline

[`ImageTextToTextPipeline`] is a high-level image and text generation class with a “chat mode”. Chat mode is enabled when a conversational model is detected and the chat prompt is [properly formatted](./llm_tutorial#wrong-prompt-format).

Start by building a chat history with the following two roles.

- `system` describes how the model should behave and respond when you’re chatting with it. This role isn’t supported by all chat models.
- `user` is where you enter your first message to the model.

```py
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a friendly chatbot who always responds in the style of a pirate"}],
    },
    {
      "role": "user",
      "content": [
            {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
            {"type": "text", "text": "What are these?"},
        ],
    },
]
```

Create a [`ImageTextToTextPipeline`] and pass the chat to it. For large models, setting [device_map=“auto”](./models#big-model-inference) helps load the model quicker and automatically places it on the fastest device available. Changing the data type to [torch.bfloat16](./models#model-data-type) also helps save memory.

> [!TIP]
> The [`ImageTextToTextPipeline`] accepts chats in the OpenAI format to make inference easier and more accessible. 

```python
import torch
from transformers import pipeline

pipeline = pipeline("image-text-to-text", model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf", device_map="auto", dtype=torch.float16)
pipeline(text=messages, max_new_tokens=50, return_full_text=False)
[{'input_text': [{'role': 'system',
    'content': [{'type': 'text',
      'text': 'You are a friendly chatbot who always responds in the style of a pirate'}]},
   {'role': 'user',
    'content': [{'type': 'image',
      'url': 'http://images.cocodataset.org/val2017/000000039769.jpg'},
     {'type': 'text', 'text': 'What are these?'}]}],
  'generated_text': 'The image shows two cats lying on a pink surface, which appears to be a cushion or a soft blanket. The cat on the left has a striped coat, typical of tabby cats, and is lying on its side with its head resting on the'}]
```

## Image inputs

For multimodal models that accept images like [LLaVA](./model_doc/llava), include the following in `content` as shown below.

- The content `"type"` can be an `"image"` or `"text"`.
- For images, it can be a link to the image (`"url"`), a file path (`"path"`), or `"base64"`. Images are automatically loaded, processed, and prepared into pixel values as inputs to the model.

```python
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")

messages = [
    {
      "role": "system",
      "content": [{"type": "text", "text": "You are a friendly chatbot who always responds in the style of a pirate"}],
    },
    {
      "role": "user",
      "content": [
            {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
            {"type": "text", "text": "What are these?"},
        ],
    },
]
```

Pass `messages` to [`~ProcessorMixin.apply_chat_template`] to tokenize the input content and return the `input_ids` and `pixel_values`.

```py
processed_chat = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
print(processed_chat.keys())
```

These inputs are now ready to be used in [`~GenerationMixin.generate`].

## Video inputs

Some vision models also support video inputs. The message format is very similar to the format for [image inputs](#image-inputs).

- The content `"type"` should be `"video"` to indicate the content is a video.
- For videos, it can be a link to the video (`"url"`) or it could be a file path (`"path"`). Videos loaded from a URL can only be decoded with [PyAV](https://pyav.basswood-io.com/docs/stable/) or [Decord](https://github.com/dmlc/decord).
- In addition to loading videos from a URL or file path, you can also pass decoded video data directly. This is useful if you’ve already preprocessed or decoded video frames elsewhere in memory (e.g., using OpenCV, decord, or torchvision). You don't need to save to files or store it in an URL.

> [!WARNING]
> Loading a video from `"url"` is only supported by the PyAV or Decord backends.

```python
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

messages = [
    {
      "role": "system",
      "content": [{"type": "text", "text": "You are a friendly chatbot who always responds in the style of a pirate"}],
    },
    {
      "role": "user",
      "content": [
            {"type": "video", "url": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_10MB.mp4"},
            {"type": "text", "text": "What do you see in this video?"},
        ],
    },
]
```

### Example: Passing decoded video objects
```python
import numpy as np

video_object1 = np.random.randint(0, 255, size=(16, 224, 224, 3), dtype=np.uint8),

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a friendly chatbot who always responds in the style of a pirate"}],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": video_object1},
            {"type": "text", "text": "What do you see in this video?"}
        ],
    },
]
```
You can also use existing (`"load_video()"`) function to load a video, edit the video in memory and pass it in the messages.
```python

# Make sure a video backend library (pyav, decord, or torchvision) is available.
from transformers.video_utils import load_video

# load a video file in memory for testing
video_object2, _ = load_video(
    "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_10MB.mp4"
)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a friendly chatbot who always responds in the style of a pirate"}],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": video_object2},
            {"type": "text", "text": "What do you see in this video?"}
        ],
    },
]
```

Pass `messages` to [`~ProcessorMixin.apply_chat_template`] to tokenize the input content. There are a few extra parameters to include in [`~ProcessorMixin.apply_chat_template`] that controls the sampling process.

The `video_load_backend` parameter refers to a specific framework to load a video. It supports [PyAV](https://pyav.basswood-io.com/docs/stable/), [Decord](https://github.com/dmlc/decord), [OpenCV](https://github.com/opencv/opencv), and [torchvision](https://pytorch.org/vision/stable/index.html).

The examples below use Decord as the backend because it is a bit faster than PyAV.

<hfoptions id="sampling">
<hfoption id="fixed number of frames">

The `num_frames` parameter controls how many frames to uniformly sample from the video. Each checkpoint has a maximum frame count it was pretrained with and exceeding this count can significantly lower generation quality. It's important to choose a frame count that fits both the model capacity and your hardware resources. If `num_frames` isn't specified, the entire video is loaded without any frame sampling.


```python
processed_chat = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    num_frames=32,
    video_load_backend="decord",
)
print(processed_chat.keys())
```

These inputs are now ready to be used in [`~GenerationMixin.generate`].

</hfoption>
<hfoption id="fps">

For longer videos, it may be better to sample more frames for better representation with the `video_fps` parameter. This determines how many frames per second to extract. As an example, if a video is 10 seconds long and `video_fps=2`, then the model samples 20 frames. In other words, 2 frames are uniformly sampled every 10 seconds.

```py
processed_chat = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    video_fps=16,
    video_load_backend="decord",
)
print(processed_chat.keys())
```

</hfoption>
<hfoption id="list of image frames">

Videos may also exist as a set of sampled frames stored as images rather than the full video file.

In this case, pass a list of image file paths and the processor automatically concatenates them into a video. Make sure all images are the same size since they are assumed to be from the same video.

```py
frames_paths = ["/path/to/frame0.png", "/path/to/frame5.png", "/path/to/frame10.png"]
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a friendly chatbot who always responds in the style of a pirate"}],
    },
    {
      "role": "user",
      "content": [
            {"type": "video", "path": frames_paths},
            {"type": "text", "text": "What do you see in this video?"},
        ],
    },
]

processed_chat = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
)
print(processed_chat.keys())
```

</hfoption>
</hfoptions>

## Template configuration

You can create a custom chat template with [Jinja](https://jinja.palletsprojects.com/en/3.1.x/templates/) and set it with [`~ProcessorMixin.apply_chat_template`]. Refer to the [Template writing](./chat_templating_writing) guide for more details.

For example, to enable a template to handle a *list of content* from multiple modalities while still supporting plain strings for text-only inference, specify how to handle the `content['type']` if it is an image or text as shown below in the Llama 3.2 Vision Instruct [template](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct/blob/main/chat_template.json).

```jinja
{% for message in messages %}
{% if loop.index0 == 0 %}{{ bos_token }}{% endif %}
{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}
{% if message['content'] is string %}
{{ message['content'] }}
{% else %}
{% for content in message['content'] %}
{% if content['type'] == 'image' %}
{{ '<|image|>' }}
{% elif content['type'] == 'text' %}
{{ content['text'] }}
{% endif %}
{% endfor %}
{% endif %}
{{ '<|eot_id|>' }}
{% endfor %}
{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}
```
