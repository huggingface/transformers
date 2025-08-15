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

Multimodal chat models are similar to normal chat models, but they can also accept non-text inputs like images, audio or video.

Chatting with multimodal models is very similar to chatting with text-only models, with the key difference being the `content` key of the messages. Instead of a single string,
as it is for text-only models, `content` can be a list containing multiple items of different types.

In the same way that the [Tokenizer](./tokenizer_summary.md) class handles chat templates and tokenization for text-only models, 
the [Processor](./processors) class handles preprocessing, tokenization and chat templates for multimodal models. Methods like [`ProcessorMixin.apply_chat_template`] are almost identical.

This guide will show you how to chat with multimodal models, first at a high level using the [`ImageTextToTextPipeline`] and then at a lower level using the [`ProcessorMixin.apply_chat_template`] and [`GenerationMixin.generate`] methods.

## ImageTextToTextPipeline

[`ImageTextToTextPipeline`] is a high-level image and text generation class with a “chat mode”. Chat mode is enabled when a conversational model is detected and the chat prompt is [properly formatted](./llm_tutorial#wrong-prompt-format). You can think of this pipeline
as the equivalent of the [`TextGenerationPipeline`] for multimodal vision-language models (VLMs).

We can see this pipeline in action by building a sample chat. Note how `content` is a list here!

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

Next, we create an [`ImageTextToTextPipeline`] and pass the chat to it. For large models, setting [device_map=“auto”](./models#big-model-inference) helps load the model quicker and automatically places it on the fastest device available. Setting the data type to [auto](./models#model-data-type) also helps save memory and improve speed.

```python
import torch
from transformers import pipeline

pipe = pipeline("image-text-to-text", model="Qwen/Qwen2.5-VL-3B-Instruct", device_map="auto", torch_dtype="auto")
out = pipe(text=messages, max_new_tokens=128)
print(out[0]['generated_text'][-1]['content'])
```

And we get:

```
Ahoy, me hearty! These be two feline friends, likely some tabby cats, taking a siesta on a cozy pink blanket. They're resting near remote controls, perhaps after watching some TV or just enjoying some quiet time together. Cats sure know how to find comfort and relaxation, don't they?
```

Aside from the gradual descent from pirate-speak into modern American English (it **is** only a 3B model, after all), this is correct!

## Using `apply_chat_template` directly

Similarly to [text-only models](./chat_templating.md), you can use the [`ProcessorMixin.apply_chat_template`] method to prepare the chat messages for multimodal models. 
This method handles the tokenization and formatting of the chat messages, including images and other media types. You can then pass the resulting inputs to the model for generation.

Let's see the example above, but using the low-level methods directly instead of a `pipeline`:

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

Pass `messages` to [`~ProcessorMixin.apply_chat_template`] to tokenize the input content. Note that, unlike text models, the output of `apply_chat_template` will
contain a `pixel_values` key with the preprocessed image data, in addition to the tokenized text.

```py
processed_chat = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
print(list(processed_chat.keys()))
```

and you should see:

```
['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw']
```

These inputs are now ready to be used in [`~GenerationMixin.generate`]:

```python
out = model.generate(**processed_chat.to(model.device), max_new_tokens=128)
print(processor.decode(out[0]))
```

If you try this, note that because we used lower-level methods the decoded output is the full conversation so far, including
the user message and the placeholder tokens that contain the image information. As a result, I won't paste it all here,
as it might blow up the document a bit! Just be aware that if you want to use the lower-level methods in practice,
you may need to trim the previous conversation from the output before displaying it to the user.


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

