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

# Multimodal chat templates

Multimodal chat models accept inputs like images, audio or video, in addition to text. The `content` key in a multimodal chat history is a list containing multiple items of different types. This is unlike text-only chat models whose `content` key is a single string.

In the same way the [Tokenizer](./fast_tokenizer) class handles chat templates and tokenization for text-only models, 
the [Processor](./processors) class handles preprocessing, tokenization and chat templates for multimodal models. Their [`~ProcessorMixin.apply_chat_template`] methods are almost identical.

This guide covers chatting with image and video models at a lower level using the [`~ProcessorMixin.apply_chat_template`] and [`~GenerationMixin.generate`] methods,
and is intended for more advanced users. If you just want to quickly get started chatting with a VLM, check out the "Including images in chats" section
in the [Chat basics](./conversations) guide.


## Using `apply_chat_template`

Like [text-only models](./chat_templating), use the [`~ProcessorMixin.apply_chat_template`] method to prepare the chat messages for multimodal models. 
This method handles the tokenization and formatting of the chat messages, including images and other media types. The resulting inputs are passed to the model for generation.


```python
from transformers import AutoProcessor, AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", device_map="auto", torch_dtype="auto")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

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

Pass `messages` to [`~ProcessorMixin.apply_chat_template`] to tokenize the input content. Unlike text models, the output of `apply_chat_template`
contains a `pixel_values` key with the preprocessed image data, in addition to the tokenized text.

```py
processed_chat = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
print(list(processed_chat.keys()))
```


```
['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw']
```

Pass these inputs to [`~GenerationMixin.generate`].

```python
out = model.generate(**processed_chat.to(model.device), max_new_tokens=128)
print(processor.decode(out[0]))
```

The decoded output contains the full conversation so far, including the user message and the placeholder tokens that contain the image information. You may need to trim the previous conversation from the output before displaying it to the user.

## Response parsing

TODO section on response parsing with a processor here

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
)
print(processed_chat.keys())
```

These inputs are now ready to be used in [`~GenerationMixin.generate`].

</hfoption>
<hfoption id="fps">

For longer videos, it may be better to sample more frames for better representation with the `fps` parameter. This determines how many frames per second to extract. As an example, if a video is 10 seconds long and `fps=2`, then the model samples 20 frames. In other words, 2 frames are uniformly sampled every 10 seconds.

```py
processed_chat = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    fps=16,
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

