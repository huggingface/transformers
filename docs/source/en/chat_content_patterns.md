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

# Chat message patterns

Chat models expect conversations as a list of dictionaries. Each dictionary uses `role` and `content` keys. The `content` key holds the user message passed to the model. Large language models accept text and tools and multimodal models combine text with images, videos, and audio.

Transformers uses a unified format where each modality type is specified explicitly, making it straightforward to mix and match inputs in a single message.

This guide covers message formatting patterns for each modality, tools, batch inference, and multi-turn conversations.

## Text

Text is the most basic content type. It's the foundation for all other patterns. Pass your message to `"content"` as a string.

```py
message = [
    {
        "role": "user",
        "content": "Explain the French Bread Law."
    }
]
```

You could also use the explicit `"type": "text"` format to keep your code consistent when you add images, videos, or audio later.

```py
message = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "Explain the French Bread Law."}]
    }
]
```

## Tools

[Tools](./chat_extras) are functions a chat model can call, like getting real-time weather data, instead of generating it on its own.

The `assistant` role handles the tool request. Set `"type": "function"` in the `"tool_calls"` key and provide your tool to the `"function"` key. Append the assistant's tool request to your message.

```py
weather = {"name": "get_current_temperature", "arguments": {"location": "Paris, France", "unit": "celsius"}}
message.append(
    {
        "role": "assistant", 
        "tool_calls": [{"type": "function", "function": weather}]
    }
)
```

The `tool` role handles the result. Append it in `"content"`. This value should always be a string.

```py
message.append({"role": "tool", "content": "22"})
```

## Multimodal

Multimodal models extend this format to handle images, videos, and audio. Each input specifies its `"type"` and provides the media with `"url"` or `"path"`.

### Image

Set `"type": "image"` and use `"url"` for links or `"path"` for local files.

```py
message = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://assets.bonappetit.com/photos/57ad4ebc53e63daf11a4ddc7/master/w_1280,c_limit/kouign-amann.jpg"},
            {"type": "text", "text": "What pastry is shown in the image?"}
        ]
    }
]
```

### Video

Set `"type": "video"` and use `"url"` for links or `"path"` for local files.

```py
message = [
    {
        "role": "user",
        "content": [
            {"type": "video", "url": "https://static01.nyt.com/images/2019/10/01/dining/01Sourdough-GIF-1/01Sourdough-GIF-1-superJumbo.gif"},
            {"type": "text", "text": "What is shown in this video?"}
        ]
    }
]
```

### Audio

Set `"type": "audio"` and use `"url"` for links or `"path"` for local files.

```py
message = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "url": "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"},
            {"type": "text", "text": "Transcribe the speech."}
        ]
    }
]
```

### Mixed multiple

The `content` list accepts any combination of types. The model processes all inputs together, enabling comparisons or cross-modal reasoning.

```py
message = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://assets.bonappetit.com/photos/57ad4ebc53e63daf11a4ddc7/master/w_1280,c_limit/kouign-amann.jpg"},
            {"type": "video", "url": "https://static01.nyt.com/images/2019/10/01/dining/01Sourdough-GIF-1/01Sourdough-GIF-1-superJumbo.gif"},
            {"type": "text", "text": "What does the image and video share in common?"},
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://assets.bonappetit.com/photos/57ad4ebc53e63daf11a4ddc7/master/w_1280,c_limit/kouign-amann.jpg"},
            {"type": "image", "url": "https://assets.bonappetit.com/photos/57e191f49f19b4610e6b7693/master/w_1600%2Cc_limit/undefined"},
            {"type": "text", "text": "What type of pastries are these?"},
        ],
    }
]
```

## Batched

Batched inference processes multiple conversations in a single forward pass to improve throughput and efficiency. Wrap each conversation in its own list, then pass them together as a list of lists. 

```py
messages = [
    [
        {"role": "user",
        "content": [
                {"type": "image", "url": "https://assets.bonappetit.com/photos/57ad4ebc53e63daf11a4ddc7/master/w_1280,c_limit/kouign-amann.jpg"},
                {"type": "text", "text": "What type of pastry is this?"}
            ]
        },
    ],
    [
        {"role": "user",
        "content": [
                {"type": "image", "url": "https://assets.bonappetit.com/photos/57e191f49f19b4610e6b7693/master/w_1600%2Cc_limit/undefined"},
                {"type": "text", "text": "What type of pastry is this?"}
            ]
        },
    ],
]
```

## Multi-turn

Conversations span multiple exchanges, alternating between `"user"` and `"assistant"` roles. Each turn adds a new message to the list, giving the model access to the full conversation history. This context helps the model generate more appropriate responses.

```py
message = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://assets.bonappetit.com/photos/57ad4ebc53e63daf11a4ddc7/master/w_1280,c_limit/kouign-amann.jpg"},
            {"type": "text", "text": "What pastry is shown in the image?"}
        ]
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "This is kouign amann, a laminated dough pastry (i.e., dough folded with layers of butter) that also incorporates sugar between layers so that during baking the sugar caramelizes."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://static01.nyt.com/images/2023/07/21/multimedia/21baguettesrex-hbkc/21baguettesrex-hbkc-videoSixteenByNineJumbo1600.jpg"},
            {"type": "text", "text": "Compare it to this image now."}
        ]
    }
]
```