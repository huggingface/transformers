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

# Serve CLI

The `transformers serve` CLI is a lightweight option for local or self-hosted servers. It avoids the extra runtime and operational overhead of dedicated inference engines like vLLM. Use it for evaluation, experimentation, and moderate load deployments. Features like [continuous batching](../continuous_batching) increase throughput and lower latency.

> [!TIP]
> For large scale production deployments, use vLLM or SGLang with a Transformer model as the backend. Learn more in the [Inference backends](../community_integrations/transformers_as_backend) guide.

The `transformers serve` command spawns a local server compatible with the [OpenAI SDK](https://platform.openai.com/docs/overview). The server works with many third-party applications and supports the REST APIs below.

- `/v1/chat/completions` for text, image, audio, and video requests
- `/v1/responses` supports the [Responses API](https://platform.openai.com/docs/api-reference/responses)
- `/v1/audio/transcriptions` for audio transcriptions
- `/v1/models` lists available models for third-party integrations
- `/load_model` streams model loading progress via SSE

Install the serving dependencies.

```bash
pip install transformers[serving]
```

Run `transformers serve` to launch a server. The default server address is http://localhost:8000.

```shell
transformers serve
```

## v1/chat/completions

The `v1/chat/completions` API is based on the [Chat Completions API](https://platform.openai.com/docs/api-reference/chat). It supports text, image, audio, and video requests for LLMs, VLMs, and multimodal models. Use it with `curl`, the [`~huggingface_hub.InferenceClient`], or the [OpenAI](https://platform.openai.com/docs/quickstart) client.

### Text-based completions

<hfoptions id="chat-completion-http">
<hfoption id="huggingface_hub">

```python
from huggingface_hub import InferenceClient

messages = [{"role": "user", "content": "What is the Transformers library known for?"}]
client = InferenceClient("http://localhost:8000")

result = client.chat_completion(messages, model="Qwen/Qwen2.5-0.5B-Instruct", max_tokens=256)
print(result.choices[0].message.content)
```

The [`~huggingface_hub.InferenceClient`] returns a printed string.

```shell
The Transformers library is primarily known for its ability to create and manipulate large-scale language models [...]
```

</hfoption>
<hfoption id="huggingface_hub (stream)">

```python
from huggingface_hub import InferenceClient

messages = [{"role": "user", "content": "What is the Transformers library known for?"}]
client = InferenceClient("http://localhost:8000")

stream = client.chat_completion(messages, model="Qwen/Qwen2.5-0.5B-Instruct", max_tokens=256, stream=True)
for chunk in stream:
    token = chunk.choices[0].delta.content
    if token:
        print(token, end="")
```

The [`~huggingface_hub.InferenceClient`] returns a printed string.

```shell
The Transformers library is primarily known for its ability to create and manipulate large-scale language models [...]
```

</hfoption>
<hfoption id="openai">

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="<random_string>")

completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "What is the Transformers library known for?"
        }
    ],
)
print(completion.choices[0].message.content)
```

The [OpenAI](https://platform.openai.com/docs/quickstart) client returns a printed string.

```shell
The Transformers library is primarily known for its ability to create and manipulate large-scale language models [...]
```

</hfoption>
<hfoption id="openai (stream)">

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="<random_string>")

completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "What is the Transformers library known for?"
        }
    ],
    stream=True,
)

for chunk in completion:
    token = chunk.choices[0].delta.content
    if token:
        print(token, end="")
```

The [OpenAI](https://platform.openai.com/docs/quickstart) client returns a printed string.

```shell
The Transformers library is primarily known for its ability to create and manipulate large-scale language models [...]
```

</hfoption>
<hfoption id="curl">

```shell
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "What is the Transformers library known for?"}],
    "temperature": 0.9,
    "max_tokens": 256
  }'
```

The command returns a JSON string.

```json
{
  "id": "ade3578d-e2c3-47ed-b651-c014991a92f6",
  "choices": [
    {
      "finish_reason": "length",
      "index": 0,
      "message": {
        "content": "The Transformers library is primarily known for its ability to create and manipulate large-scale language models [...]",
        "role": "assistant"
      }
    }
  ],
  "created": 1776184004,
  "model": "Qwen/Qwen2.5-0.5B-Instruct@main",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 256,
    "prompt_tokens": 37,
    "total_tokens": 293
  }
}
```

</hfoption>
<hfoption id="curl (stream)">

```shell
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "What is the Transformers library known for?"}],
    "temperature": 0.9,
    "max_tokens": 256,
    "stream": true
  }'
```

The command returns a stream of chunks.

```shell
data: {"id":"596d3941-754d-43b4-8773-9bf9940049f5","choices":[{"delta":{"role":"assistant"},"index":0}],"created":1776183469,"model":"Qwen/Qwen2.5-0.5B-Instruct@main","object":"chat.completion.chunk","system_fingerprint":""}

data: {"id":"596d3941-754d-43b4-8773-9bf9940049f5","choices":[{"delta":{"content":"The"},"index":0}],"created":1776183469,"model":"Qwen/Qwen2.5-0.5B-Instruct@main","object":"chat.completion.chunk","system_fingerprint":""}

(...)
```

</hfoption>
</hfoptions>

### Image-based completions

<hfoptions id="chat-completion-http-images">
<hfoption id="huggingface_hub">

```python
from huggingface_hub import InferenceClient

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg",
                }
            },
        ],
    }
]
client = InferenceClient("http://localhost:8000")

result = client.chat_completion(messages, model="Qwen/Qwen2.5-VL-7B-Instruct", max_tokens=256)
print(result.choices[0].message.content)
```

The [`~huggingface_hub.InferenceClient`] returns a printed string.

```xmp
The image depicts an astronaut in a space suit standing on what appears to be the surface of the moon, given the barren, rocky landscape and the dark sky in the background. The astronaut is holding a large egg that has cracked open, revealing a small creature inside. The scene is imaginative and playful, combining elements of space exploration with a whimsical twist involving the egg and the creature.
```

</hfoption>
<hfoption id="huggingface_hub (stream)">

```python
from huggingface_hub import InferenceClient

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg",
                }
            },
        ],
    }
]
client = InferenceClient("http://localhost:8000")

stream = client.chat_completion(messages, model="Qwen/Qwen2.5-VL-7B-Instruct", max_tokens=256, stream=True)
for chunk in stream:
    token = chunk.choices[0].delta.content
    if token:
        print(token, end="")
```

The [`~huggingface_hub.InferenceClient`] returns a printed string.

```xmp
The image depicts an astronaut in a space suit standing on what appears to be the surface of the moon, given the barren, rocky landscape and the dark sky in the background. The astronaut is holding a large egg that has cracked open, revealing a small creature inside. The scene is imaginative and playful, combining elements of space exploration with a whimsical twist involving the egg and the creature.
```

</hfoption>
<hfoption id="openai">

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="<random_string>")

completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg",
                    }
                },
            ],
        }
    ],
)
print(completion.choices[0].message.content)
```

The [OpenAI](https://platform.openai.com/docs/quickstart) client returns a printed string.

```xmp
The image depicts an astronaut in a space suit standing on what appears to be the surface of the moon, given the barren, rocky landscape and the dark sky in the background. The astronaut is holding a large egg that has cracked open, revealing a small creature inside. The scene is imaginative and playful, combining elements of space exploration with a whimsical twist involving the egg and the creature.
```

</hfoption>
<hfoption id="openai (stream)">

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="<random_string>")

completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg",
                    }
                },
            ],
        }
    ],
    stream=True,
)

for chunk in completion:
    token = chunk.choices[0].delta.content
    if token:
        print(token, end="")
```

The [OpenAI](https://platform.openai.com/docs/quickstart) client returns a printed string.

```xmp
The image depicts an astronaut in a space suit standing on what appears to be the surface of the moon, given the barren, rocky landscape and the dark sky in the background. The astronaut is holding a large egg that has cracked open, revealing a small creature inside. The scene is imaginative and playful, combining elements of space exploration with a whimsical twist involving the egg and the creature.
```

</hfoption>
<hfoption id="curl">

```shell
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What's in this image?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
            }
          }
        ]
      }
    ],
    "max_tokens": 256
  }'
```

The command returns a JSON string.

```json
{
  "id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "The image depicts an astronaut in a space suit standing on what appears to be the surface of the moon [...]",
        "role": "assistant"
      }
    }
  ],
  "created": 1753366665,
  "model": "Qwen/Qwen2.5-VL-7B-Instruct@main",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 50,
    "prompt_tokens": 120,
    "total_tokens": 170
  }
}
```

</hfoption>
<hfoption id="curl (stream)">

```shell
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "stream": true,
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What's in this image?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
            }
          }
        ]
      }
    ],
    "max_tokens": 256
  }'
```

The command returns a stream of chunks.

```shell
data: {"id":"f47ac10b-58cc-4372-a567-0e02b2c3d479","choices":[{"delta":{"role":"assistant"},"index":0}],"created":1753366665,"model":"Qwen/Qwen2.5-VL-7B-Instruct@main","object":"chat.completion.chunk","system_fingerprint":""}

data: {"id":"f47ac10b-58cc-4372-a567-0e02b2c3d479","choices":[{"delta":{"content":"The "},"index":0}],"created":1753366701,"model":"Qwen/Qwen2.5-VL-7B-Instruct@main","object":"chat.completion.chunk","system_fingerprint":""}

(...)
```

</hfoption>
</hfoptions>

### Audio completions

Multimodal models like [Gemma 4](https://huggingface.co/google/gemma-4-E2B-it) and [Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) accept audio input using the OpenAI `input_audio` content type. The audio must be base64-encoded and the format (`mp3` or `wav`) must be specified.

<hfoptions id="audio-completions">
<hfoption id="huggingface_hub">

```python
import base64
import httpx
from huggingface_hub import InferenceClient

audio_url = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama_first_45_secs.mp3"
audio_b64 = base64.b64encode(httpx.get(audio_url, follow_redirects=True).content).decode()

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Transcribe this audio."},
            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "mp3"}},
        ],
    }
]
client = InferenceClient("http://localhost:8000")

result = client.chat_completion(messages, model="google/gemma-4-E2B-it", max_tokens=256)
print(result.choices[0].message.content)
```

The [`~huggingface_hub.InferenceClient`] returns a printed string.

```
This week, I traveled to Chicago to deliver my final farewell address to the nation, following in the tradition of presidents before me. It was an opportunity to say thank you. Whether we've seen eye to eye or rarely agreed at all, my conversations with you, the American people, in living rooms and schools, at farms and on factory floors, at diners, and on distant military outposts, all these conversations are what have kept me honest.
```

</hfoption>
<hfoption id="huggingface_hub (stream)">

```python
import base64
import httpx
from huggingface_hub import InferenceClient

audio_url = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama_first_45_secs.mp3"
audio_b64 = base64.b64encode(httpx.get(audio_url, follow_redirects=True).content).decode()

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Transcribe this audio."},
            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "mp3"}},
        ],
    }
]
client = InferenceClient("http://localhost:8000")

stream = client.chat_completion(messages, model="google/gemma-4-E2B-it", max_tokens=256, stream=True)
for chunk in stream:
    token = chunk.choices[0].delta.content
    if token:
        print(token, end="")
```

The [`~huggingface_hub.InferenceClient`] returns a printed string.

```
This week, I traveled to Chicago to deliver my final farewell address to the nation, following in the tradition of presidents before me. It was an opportunity to say thank you. Whether we've seen eye to eye or rarely agreed at all, my conversations with you, the American people, in living rooms and schools, at farms and on factory floors, at diners, and on distant military outposts, all these conversations are what have kept me honest.
```

</hfoption>
<hfoption id="openai">

```python
import base64
import httpx
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="<random_string>")

audio_url = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama_first_45_secs.mp3"
audio_b64 = base64.b64encode(httpx.get(audio_url, follow_redirects=True).content).decode()

completion = client.chat.completions.create(
    model="google/gemma-4-E2B-it",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe this audio."},
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "mp3"}},
            ],
        }
    ],
)
print(completion.choices[0].message.content)
```

The [OpenAI](https://platform.openai.com/docs/quickstart) client returns a printed string.

```
This week, I traveled to Chicago to deliver my final farewell address to the nation, following in the tradition of presidents before me. It was an opportunity to say thank you. Whether we've seen eye to eye or rarely agreed at all, my conversations with you, the American people, in living rooms and schools, at farms and on factory floors, at diners, and on distant military outposts, all these conversations are what have kept me honest.
```

</hfoption>
<hfoption id="openai (stream)">

```python
import base64
import httpx
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="<random_string>")

audio_url = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama_first_45_secs.mp3"
audio_b64 = base64.b64encode(httpx.get(audio_url, follow_redirects=True).content).decode()

completion = client.chat.completions.create(
    model="google/gemma-4-E2B-it",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe this audio."},
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "mp3"}},
            ],
        }
    ],
    stream=True,
)

for chunk in completion:
    token = chunk.choices[0].delta.content
    if token:
        print(token, end="")
```

The [OpenAI](https://platform.openai.com/docs/quickstart) client returns a printed string.

```
This week, I traveled to Chicago to deliver my final farewell address to the nation, following in the tradition of presidents before me. It was an opportunity to say thank you. Whether we've seen eye to eye or rarely agreed at all, my conversations with you, the American people, in living rooms and schools, at farms and on factory floors, at diners, and on distant military outposts, all these conversations are what have kept me honest.
```

</hfoption>
<hfoption id="curl">

```shell
# First, base64-encode the audio file and build the JSON payload
AUDIO_B64=$(curl -sL https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama_first_45_secs.mp3 | base64 -w 0)

cat <<EOF > /tmp/audio_request.json
{
  "model": "google/gemma-4-E2B-it",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Transcribe this audio."},
        {"type": "input_audio", "input_audio": {"data": "$AUDIO_B64", "format": "mp3"}}
      ]
    }
  ]
}
EOF

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @/tmp/audio_request.json
```

The command returns a JSON string.

```json
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "This week, I traveled to Chicago to deliver my final farewell address to the nation [...]",
        "role": "assistant"
      }
    }
  ],
  "created": 1753366665,
  "model": "google/gemma-4-E2B-it@main",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 85,
    "prompt_tokens": 200,
    "total_tokens": 285
  }
}
```

</hfoption>
<hfoption id="curl (stream)">

```shell
# First, base64-encode the audio file and build the JSON payload
AUDIO_B64=$(curl -sL https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama_first_45_secs.mp3 | base64 -w 0)

cat <<EOF > /tmp/audio_request.json
{
  "model": "google/gemma-4-E2B-it",
  "stream": true,
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Transcribe this audio."},
        {"type": "input_audio", "input_audio": {"data": "$AUDIO_B64", "format": "mp3"}}
      ]
    }
  ]
}
EOF

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @/tmp/audio_request.json
```

The command returns a stream of chunks.

```shell
data: {"id":"cb997e1d-98b9-414a-be89-1880288610ef","choices":[{"delta":{"role":"assistant"},"index":0}],"created":1776254856,"model":"google/gemma-4-E2B-it@main","object":"chat.completion.chunk","system_fingerprint":""}

data: {"id":"cb997e1d-98b9-414a-be89-1880288610ef","choices":[{"delta":{"content":"This"},"index":0}],"created":1776254856,"model":"google/gemma-4-E2B-it@main","object":"chat.completion.chunk","system_fingerprint":""}

(...)
```

</hfoption>
</hfoptions>

> [!WARNING]
> The `audio_url` content type is an extension not part of the OpenAI standard and may change in future versions.

As a convenience, audio can also be passed by URL using the `audio_url` content type, avoiding the need for base64 encoding.

```python
completion = client.chat.completions.create(
    model="google/gemma-4-E2B-it",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe this audio."},
                {"type": "audio_url", "audio_url": {"url": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama_first_45_secs.mp3"}},
            ],
        }
    ],
)
```

### Video completions

> [!WARNING]
> The `video_url` content type is an extension not part of the OpenAI standard and may change in future versions.

Video input is supported using the `video_url` content type. If the model supports audio (e.g. Gemma 4, Qwen2.5-Omni), the audio track is automatically extracted from the video and processed alongside the visual frames.

> [!TIP]
> Video processing requires [torchcodec](https://github.com/pytorch/torchcodec). Install it with `pip install torchcodec`.

<hfoptions id="video-completions">
<hfoption id="huggingface_hub">

```python
from huggingface_hub import InferenceClient

messages = [
    {
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": {"url": "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/concert.mp4"}},
            {"type": "text", "text": "What is happening in the video and what is the song about?"},
        ],
    }
]
client = InferenceClient("http://localhost:8000")

result = client.chat_completion(messages, model="google/gemma-4-E2B-it", max_tokens=256)
print(result.choices[0].message.content)
```

The [`~huggingface_hub.InferenceClient`] returns a printed string.

```
The video captures a live music performance at a music festival or a large concert. There are several musicians on stage, including a central figure playing an acoustic guitar and singing. The foreground is filled with the backs of the audience, indicating a large crowd watching the show. The stage is dramatically lit with bright spotlights and blue and white stage lighting, with haze and smoke creating an immersive atmosphere.

The lyrics of the song are: "I don't care 'bout street, from that fresh street, 'cause there's no problem, another one I want to be, in the storm..."
```

</hfoption>
<hfoption id="huggingface_hub (stream)">

```python
from huggingface_hub import InferenceClient

messages = [
    {
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": {"url": "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/concert.mp4"}},
            {"type": "text", "text": "What is happening in the video and what is the song about?"},
        ],
    }
]
client = InferenceClient("http://localhost:8000")

stream = client.chat_completion(messages, model="google/gemma-4-E2B-it", max_tokens=256, stream=True)
for chunk in stream:
    token = chunk.choices[0].delta.content
    if token:
        print(token, end="")
```

The [`~huggingface_hub.InferenceClient`] returns a printed string.

```
The video captures a live music performance at a music festival or a large concert. There are several musicians on stage, including a central figure playing an acoustic guitar and singing. The foreground is filled with the backs of the audience, indicating a large crowd watching the show. The stage is dramatically lit with bright spotlights and blue and white stage lighting, with haze and smoke creating an immersive atmosphere.

The lyrics of the song are: "I don't care 'bout street, from that fresh street, 'cause there's no problem, another one I want to be, in the storm..."
```

</hfoption>
<hfoption id="openai">

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="<random_string>")

completion = client.chat.completions.create(
    model="google/gemma-4-E2B-it",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/concert.mp4"}},
                {"type": "text", "text": "What is happening in the video and what is the song about?"},
            ],
        }
    ],
    max_tokens=256,
)
print(completion.choices[0].message.content)
```

The [OpenAI](https://platform.openai.com/docs/quickstart) client returns a printed string.

```
The video captures a live music performance at a music festival or a large concert. There are several musicians on stage, including a central figure playing an acoustic guitar and singing. The foreground is filled with the backs of the audience, indicating a large crowd watching the show. The stage is dramatically lit with bright spotlights and blue and white stage lighting, with haze and smoke creating an immersive atmosphere.

The lyrics of the song are: "I don't care 'bout street, from that fresh street, 'cause there's no problem, another one I want to be, in the storm..."
```

</hfoption>
<hfoption id="openai (stream)">

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="<random_string>")

completion = client.chat.completions.create(
    model="google/gemma-4-E2B-it",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/concert.mp4"}},
                {"type": "text", "text": "What is happening in the video and what is the song about?"},
            ],
        }
    ],
    max_tokens=256,
    stream=True,
)

for chunk in completion:
    token = chunk.choices[0].delta.content
    if token:
        print(token, end="")
```

The [OpenAI](https://platform.openai.com/docs/quickstart) client returns a printed string.

```
The video captures a live music performance at a music festival or a large concert. There are several musicians on stage, including a central figure playing an acoustic guitar and singing. The foreground is filled with the backs of the audience, indicating a large crowd watching the show. The stage is dramatically lit with bright spotlights and blue and white stage lighting, with haze and smoke creating an immersive atmosphere.

The lyrics of the song are: "I don't care 'bout street, from that fresh street, 'cause there's no problem, another one I want to be, in the storm..."
```

</hfoption>
<hfoption id="curl">

```shell
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-E2B-it",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "video_url", "video_url": {"url": "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/concert.mp4"}},
          {"type": "text", "text": "What is happening in the video and what is the song about?"}
        ]
      }
    ],
    "max_tokens": 256
  }'
```

The command returns a JSON string.

```json
{
  "id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
  "choices": [
    {
      "finish_reason": "length",
      "index": 0,
      "message": {
        "content": "The video captures a live music performance at a music festival or a large concert [...]",
        "role": "assistant"
      }
    }
  ],
  "created": 1753366665,
  "model": "google/gemma-4-E2B-it@main",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 256,
    "prompt_tokens": 350,
    "total_tokens": 606
  }
}
```

</hfoption>
<hfoption id="curl (stream)">

```shell
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-E2B-it",
    "stream": true,
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "video_url", "video_url": {"url": "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/concert.mp4"}},
          {"type": "text", "text": "What is happening in the video and what is the song about?"}
        ]
      }
    ],
    "max_tokens": 256
  }'
```

The command returns a stream of chunks.

```shell
data: {"id":"cb997e1d-98b9-414a-be89-1880288610ef","choices":[{"delta":{"role":"assistant"},"index":0}],"created":1776254856,"model":"google/gemma-4-E2B-it@main","object":"chat.completion.chunk","system_fingerprint":""}

data: {"id":"cb997e1d-98b9-414a-be89-1880288610ef","choices":[{"delta":{"content":"Based"},"index":0}],"created":1776254856,"model":"google/gemma-4-E2B-it@main","object":"chat.completion.chunk","system_fingerprint":""}

(...)
```

</hfoption>
</hfoptions>

## v1/responses

The [Responses API](https://platform.openai.com/docs/api-reference/responses) is OpenAI's latest API endpoint for generation. It supports stateful interactions and integrates built-in tools to extend a model's capabilities. OpenAI [recommends](https://platform.openai.com/docs/guides/migrate-to-responses) using the Responses API over the Chat Completions API for new projects.

The `v1/responses` API supports text, image, audio, and video requests through the `curl` command and [OpenAI](https://platform.openai.com/docs/quickstart) client.

### Text-based responses

<hfoptions id="responses">
<hfoption id="openai">

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="<random_string>")

response = client.responses.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    input="Tell me a three sentence bedtime story about a unicorn.",
    max_output_tokens=256,
    stream=False,
)
print(response.output[0].content[0].text)
```

The [OpenAI](https://platform.openai.com/docs/quickstart) client returns a printed string.

```shell
Once upon a time, in a faraway land, there lived a beautiful unicorn named Luna [...]
```

</hfoption>
<hfoption id="openai (stream)">

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="<random_string>")

response = client.responses.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    input="Tell me a three sentence bedtime story about a unicorn.",
    max_output_tokens=256,
    stream=True,
)

for event in response:
    if hasattr(event, "delta") and event.delta:
        print(event.delta, end="")
```

The [OpenAI](https://platform.openai.com/docs/quickstart) client returns a printed string.

```shell
Once upon a time, in a faraway land, there lived a beautiful unicorn named Luna [...]
```

</hfoption>
<hfoption id="curl">

```shell
curl http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "input": "Tell me a three sentence bedtime story about a unicorn.",
    "max_output_tokens": 256,
    "stream": false
  }'
```

The command returns a JSON string.

```json
{
  "id": "resp_8693af9b-b561-4267-bd1a-22d1b43284cb",
  "created_at": 1776258580.7114341,
  "model": "Qwen/Qwen2.5-0.5B-Instruct@main",
  "object": "response",
  "output": [
    {
      "id": "msg_8693af9b-b561-4267-bd1a-22d1b43284cb",
      "content": [
        {
          "annotations": [],
          "text": "Once upon a time, in a faraway land, there lived a beautiful unicorn named Luna [...]",
          "type": "output_text"
        }
      ],
      "role": "assistant",
      "status": "completed",
      "type": "message"
    }
  ],
  "status": "completed"
}
```

</hfoption>
<hfoption id="curl (stream)">

```shell
curl http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "stream": true,
    "input": "Tell me a three sentence bedtime story about a unicorn.",
    "max_output_tokens": 256
  }'
```

The command returns a stream of chunks.

```shell
data: {"response":{"id":"resp_req_0","created_at":1754059817.783648,"model":"Qwen/Qwen2.5-0.5B-Instruct@main","object":"response","output":[],"status":"queued"},"sequence_number":0,"type":"response.created"}

data: {"content_index":0,"delta":"Once ","item_id":"msg_req_0","output_index":0,"sequence_number":5,"type":"response.output_text.delta"}

data: {"content_index":0,"delta":"upon ","item_id":"msg_req_0","output_index":0,"sequence_number":6,"type":"response.output_text.delta"}

(...)
```

</hfoption>
</hfoptions>

### Image-based responses

The Responses API also supports image, audio, and video inputs. Pass them as a list of messages using the same content types as [v1/chat/completions](#text-and-image-based-completions).

<hfoptions id="responses-images">
<hfoption id="openai">

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="<random_string>")

response = client.responses.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg",
                    }
                },
            ],
        }
    ],
    max_output_tokens=256,
    stream=False,
)
print(response.output[0].content[0].text)
```

The [OpenAI](https://platform.openai.com/docs/quickstart) client returns a printed string.

```xmp
The image depicts an astronaut in a space suit standing on what appears to be the surface of the moon, given the barren, rocky landscape and the dark sky in the background. The astronaut is holding a large egg that has cracked open, revealing a small creature inside. The scene is imaginative and playful, combining elements of space exploration with a whimsical twist involving the egg and the creature.
```

</hfoption>
<hfoption id="openai (stream)">

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="<random_string>")

response = client.responses.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg",
                    }
                },
            ],
        }
    ],
    max_output_tokens=256,
    stream=True,
)

for event in response:
    if hasattr(event, "delta") and event.delta:
        print(event.delta, end="")
```

The [OpenAI](https://platform.openai.com/docs/quickstart) client returns a printed string.

```xmp
The image depicts an astronaut in a space suit standing on what appears to be the surface of the moon, given the barren, rocky landscape and the dark sky in the background. The astronaut is holding a large egg that has cracked open, revealing a small creature inside. The scene is imaginative and playful, combining elements of space exploration with a whimsical twist involving the egg and the creature.
```

</hfoption>
<hfoption id="curl">

```shell
curl http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "input": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What'\''s in this image?"},
          {"type": "image_url", "image_url": {"url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"}}
        ]
      }
    ],
    "max_output_tokens": 256,
    "stream": false
  }'
```

The command returns a JSON string.

```json
{
  "id": "resp_f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "created_at": 1776258580.7114341,
  "model": "Qwen/Qwen2.5-VL-7B-Instruct@main",
  "object": "response",
  "output": [
    {
      "id": "msg_f47ac10b-58cc-4372-a567-0e02b2c3d479",
      "content": [
        {
          "annotations": [],
          "text": "The image depicts an astronaut in a space suit standing on what appears to be the surface of the moon [...]",
          "type": "output_text"
        }
      ],
      "role": "assistant",
      "status": "completed",
      "type": "message"
    }
  ],
  "status": "completed"
}
```

</hfoption>
<hfoption id="curl (stream)">

```shell
curl http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "stream": true,
    "input": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What'\''s in this image?"},
          {"type": "image_url", "image_url": {"url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"}}
        ]
      }
    ],
    "max_output_tokens": 256
  }'
```

The command returns a stream of chunks.

```shell
data: {"response":{"id":"resp_f47ac10b-58cc-4372-a567-0e02b2c3d479","created_at":1776258580.7114341,"model":"Qwen/Qwen2.5-VL-7B-Instruct@main","object":"response","output":[],"status":"queued"},"sequence_number":0,"type":"response.created"}

data: {"content_index":0,"delta":"The ","item_id":"msg_f47ac10b","output_index":0,"sequence_number":5,"type":"response.output_text.delta"}

(...)
```

</hfoption>
</hfoptions>

### Audio-based responses

<hfoptions id="responses-audio">
<hfoption id="openai">

```python
import base64
import httpx
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="<random_string>")

audio_url = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama_first_45_secs.mp3"
audio_b64 = base64.b64encode(httpx.get(audio_url, follow_redirects=True).content).decode()

response = client.responses.create(
    model="google/gemma-4-E2B-it",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe this audio."},
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "mp3"}},
            ],
        }
    ],
    max_output_tokens=256,
    stream=False,
)
print(response.output[0].content[0].text)
```

The [OpenAI](https://platform.openai.com/docs/quickstart) client returns a printed string.

```
This week, I traveled to Chicago to deliver my final farewell address to the nation, following in the tradition of presidents before me. It was an opportunity to say thank you. Whether we've seen eye to eye or rarely agreed at all, my conversations with you, the American people, in living rooms and schools, at farms and on factory floors, at diners, and on distant military outposts, all these conversations are what have kept me honest.
```

</hfoption>
<hfoption id="openai (stream)">

```python
import base64
import httpx
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="<random_string>")

audio_url = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama_first_45_secs.mp3"
audio_b64 = base64.b64encode(httpx.get(audio_url, follow_redirects=True).content).decode()

response = client.responses.create(
    model="google/gemma-4-E2B-it",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe this audio."},
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "mp3"}},
            ],
        }
    ],
    max_output_tokens=256,
    stream=True,
)

for event in response:
    if hasattr(event, "delta") and event.delta:
        print(event.delta, end="")
```

The [OpenAI](https://platform.openai.com/docs/quickstart) client returns a printed string.

```
This week, I traveled to Chicago to deliver my final farewell address to the nation, following in the tradition of presidents before me. It was an opportunity to say thank you. Whether we've seen eye to eye or rarely agreed at all, my conversations with you, the American people, in living rooms and schools, at farms and on factory floors, at diners, and on distant military outposts, all these conversations are what have kept me honest.
```

</hfoption>
<hfoption id="curl">

```shell
# First, base64-encode the audio file and build the JSON payload
AUDIO_B64=$(curl -sL https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama_first_45_secs.mp3 | base64 -w 0)

cat <<EOF > /tmp/audio_request.json
{
  "model": "google/gemma-4-E2B-it",
  "input": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Transcribe this audio."},
        {"type": "input_audio", "input_audio": {"data": "$AUDIO_B64", "format": "mp3"}}
      ]
    }
  ],
  "max_output_tokens": 256,
  "stream": false
}
EOF

curl http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d @/tmp/audio_request.json
```

The command returns a JSON string.

```json
{
  "id": "resp_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "created_at": 1776258580.7114341,
  "model": "google/gemma-4-E2B-it@main",
  "object": "response",
  "output": [
    {
      "id": "msg_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "content": [
        {
          "annotations": [],
          "text": "This week, I traveled to Chicago to deliver my final farewell address to the nation [...]",
          "type": "output_text"
        }
      ],
      "role": "assistant",
      "status": "completed",
      "type": "message"
    }
  ],
  "status": "completed"
}
```

</hfoption>
<hfoption id="curl (stream)">

```shell
# First, base64-encode the audio file and build the JSON payload
AUDIO_B64=$(curl -sL https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama_first_45_secs.mp3 | base64 -w 0)

cat <<EOF > /tmp/audio_request.json
{
  "model": "google/gemma-4-E2B-it",
  "stream": true,
  "input": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Transcribe this audio."},
        {"type": "input_audio", "input_audio": {"data": "$AUDIO_B64", "format": "mp3"}}
      ]
    }
  ],
  "max_output_tokens": 256
}
EOF

curl http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d @/tmp/audio_request.json
```

The command returns a stream of chunks.

```shell
data: {"response":{"id":"resp_a1b2c3d4-e5f6-7890-abcd-ef1234567890","created_at":1776258580.7114341,"model":"google/gemma-4-E2B-it@main","object":"response","output":[],"status":"queued"},"sequence_number":0,"type":"response.created"}

data: {"content_index":0,"delta":"This ","item_id":"msg_a1b2c3d4","output_index":0,"sequence_number":5,"type":"response.output_text.delta"}

(...)
```

</hfoption>
</hfoptions>

### Video-based responses

> [!TIP]
> Video processing requires [torchcodec](https://github.com/pytorch/torchcodec). Install it with `pip install torchcodec`.

<hfoptions id="responses-video">
<hfoption id="openai">

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="<random_string>")

response = client.responses.create(
    model="google/gemma-4-E2B-it",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/concert.mp4"}},
                {"type": "text", "text": "What is happening in the video and what is the song about?"},
            ],
        }
    ],
    max_output_tokens=256,
    stream=False,
)
print(response.output[0].content[0].text)
```

The [OpenAI](https://platform.openai.com/docs/quickstart) client returns a printed string.

```
The video captures a live music performance at a music festival or a large concert. There are several musicians on stage, including a central figure playing an acoustic guitar and singing. The foreground is filled with the backs of the audience, indicating a large crowd watching the show. The stage is dramatically lit with bright spotlights and blue and white stage lighting, with haze and smoke creating an immersive atmosphere.

The lyrics of the song are: "I don't care 'bout street, from that fresh street, 'cause there's no problem, another one I want to be, in the storm..."
```

</hfoption>
<hfoption id="openai (stream)">

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="<random_string>")

response = client.responses.create(
    model="google/gemma-4-E2B-it",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/concert.mp4"}},
                {"type": "text", "text": "What is happening in the video and what is the song about?"},
            ],
        }
    ],
    max_output_tokens=256,
    stream=True,
)

for event in response:
    if hasattr(event, "delta") and event.delta:
        print(event.delta, end="")
```

The [OpenAI](https://platform.openai.com/docs/quickstart) client returns a printed string.

```
The video captures a live music performance at a music festival or a large concert. There are several musicians on stage, including a central figure playing an acoustic guitar and singing. The foreground is filled with the backs of the audience, indicating a large crowd watching the show. The stage is dramatically lit with bright spotlights and blue and white stage lighting, with haze and smoke creating an immersive atmosphere.

The lyrics of the song are: "I don't care 'bout street, from that fresh street, 'cause there's no problem, another one I want to be, in the storm..."
```

</hfoption>
<hfoption id="curl">

```shell
curl http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-E2B-it",
    "input": [
      {
        "role": "user",
        "content": [
          {"type": "video_url", "video_url": {"url": "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/concert.mp4"}},
          {"type": "text", "text": "What is happening in the video and what is the song about?"}
        ]
      }
    ],
    "max_output_tokens": 256,
    "stream": false
  }'
```

The command returns a JSON string.

```json
{
  "id": "resp_b2c3d4e5-f6a7-8901-bcde-f12345678901",
  "created_at": 1776258580.7114341,
  "model": "google/gemma-4-E2B-it@main",
  "object": "response",
  "output": [
    {
      "id": "msg_b2c3d4e5-f6a7-8901-bcde-f12345678901",
      "content": [
        {
          "annotations": [],
          "text": "The video captures a live music performance at a music festival or a large concert [...]",
          "type": "output_text"
        }
      ],
      "role": "assistant",
      "status": "completed",
      "type": "message"
    }
  ],
  "status": "completed"
}
```

</hfoption>
<hfoption id="curl (stream)">

```shell
curl http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-E2B-it",
    "stream": true,
    "input": [
      {
        "role": "user",
        "content": [
          {"type": "video_url", "video_url": {"url": "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/concert.mp4"}},
          {"type": "text", "text": "What is happening in the video and what is the song about?"}
        ]
      }
    ],
    "max_output_tokens": 256
  }'
```

The command returns a stream of chunks.

```shell
data: {"response":{"id":"resp_b2c3d4e5-f6a7-8901-bcde-f12345678901","created_at":1776258580.7114341,"model":"google/gemma-4-E2B-it@main","object":"response","output":[],"status":"queued"},"sequence_number":0,"type":"response.created"}

data: {"content_index":0,"delta":"Based ","item_id":"msg_b2c3d4e5","output_index":0,"sequence_number":5,"type":"response.output_text.delta"}

(...)
```

</hfoption>
</hfoptions>

## v1/audio/transcriptions

The `v1/audio/transcriptions` endpoint transcribes audio using speech-to-text models. It follows the [Audio transcription API](https://platform.openai.com/docs/api-reference/audio/createTranscription) format.

```shell
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/audio.wav" \
  -F "model=openai/whisper-large-v3"
```

The command returns the following response.

```shell
{
  "text": "Transcribed text from the audio file",
}
```

## v1/models

The `v1/models` endpoint scans your local Hugging Face cache and returns a list of downloaded models in the OpenAI-compatible format. Third-party tools use this endpoint to discover available models.

Download a model before running `transformers serve`.

```bash
transformers download Qwen/Qwen2.5-0.5B-Instruct
```

Once downloaded, the model appears in `/v1/models` responses.

```shell
curl http://localhost:8000/v1/models
```

The endpoint returns a JSON object with available models.

## Loading models

The `/load_model` endpoint pre-loads a model and streams progress via [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) (SSE). The `transformers chat` CLI uses it automatically so users see download and loading progress instead of a hanging prompt. Use it to warm up a model before sending inference requests.

### Request

```shell
curl -N -X POST http://localhost:8000/load_model \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-0.5B-Instruct"}'
```

The `model` field is a Hugging Face model identifier, optionally with an `@revision` suffix (`Qwen/Qwen2.5-0.5B-Instruct@main`). Omitting the revision defaults to `main`.

### Response

The response is an SSE stream (`Content-Type: text/event-stream`). Each frame is a JSON object on a `data:` line.

```
data: {"status": "loading", "model": "Qwen/Qwen2.5-0.5B-Instruct@main", "stage": "processor"}
```

Every event contains at minimum a `status` and `model` field. Additional fields depend on the status.

| Field | Present when | Description |
|-------|-------------|-------------|
| `status` | Always | `loading`, `ready`, or `error` |
| `model` | Always | Canonical `model_id@revision` |
| `stage` | `status == "loading"` | One of `processor`, `config`, `download`, `weights` (see stages below) |
| `progress` | `download` and `weights` stages | Object with `current` and `total` (integer or null) |
| `cached` | `status == "ready"` | `true` if the model was already in memory |
| `message` | `status == "error"` | Error description |

### Stages

Loading progresses through these stages in order. Some may be skipped (`download` is skipped when files are already cached locally).

| Stage | Has progress? | Description |
|-------|---------------|-------------|
| `processor` | No | Loading the tokenizer/processor |
| `config` | No | Loading model configuration |
| `download` | Yes (bytes) | Downloading model files |
| `weights` | Yes (items) | Loading weight tensors into memory |

The stream ends with exactly one terminal event, `ready` (success) or `error` (failure).

## Timeout

`transformers serve` supports different requests by different models. Each model loads on demand and stays in GPU memory. Models unload automatically after 300 seconds of inactivity to free up GPU memory. Set `--model-timeout` to a different value in seconds, or `-1` to disable unloading entirely.

```shell
transformers serve --model-timeout 400
```

### Loading examples

See the example responses below for a freshly downloaded model, a model loaded from your local cache (skips the download stage), and a model that already exists in memory.

<hfoptions id="load-model-examples">
<hfoption id="fresh load">

```
data: {"status": "loading", "model": "org/model@main", "stage": "processor"}
data: {"status": "loading", "model": "org/model@main", "stage": "config"}
data: {"status": "loading", "model": "org/model@main", "stage": "download", "progress": {"current": 0, "total": 269100000}}
data: {"status": "loading", "model": "org/model@main", "stage": "download", "progress": {"current": 134600000, "total": 269100000}}
data: {"status": "loading", "model": "org/model@main", "stage": "download", "progress": {"current": 269100000, "total": 269100000}}
data: {"status": "loading", "model": "org/model@main", "stage": "weights", "progress": {"current": 1, "total": 272}}
data: {"status": "loading", "model": "org/model@main", "stage": "weights", "progress": {"current": 272, "total": 272}}
data: {"status": "ready", "model": "org/model@main", "cached": false}
```

</hfoption>
<hfoption id="cached files">

```
data: {"status": "loading", "model": "org/model@main", "stage": "processor"}
data: {"status": "loading", "model": "org/model@main", "stage": "config"}
data: {"status": "loading", "model": "org/model@main", "stage": "weights", "progress": {"current": 1, "total": 272}}
data: {"status": "loading", "model": "org/model@main", "stage": "weights", "progress": {"current": 272, "total": 272}}
data: {"status": "ready", "model": "org/model@main", "cached": false}
```

</hfoption>
<hfoption id="in memory">

```
data: {"status": "ready", "model": "org/model@main", "cached": true}
```

</hfoption>
</hfoptions>

## Tool calling

The `transformers serve` server supports OpenAI-style function calling. Models trained for tool-use generate structured function calls that your application executes.

> [!NOTE]
> Tool calling is currently limited to the Qwen model family.

Define tools as a list of function specifications following the OpenAI format.

```py
import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="<KEY>")

tools = [
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get the current weather in a location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city name, e.g. San Francisco"
          },
          "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "temperature unit"
          }
        },
        "required": ["location"]
      }
    }
  }
]
```

Customize generation by passing [`GenerationConfig`] parameters to the `extra_body` argument in [create](https://platform.openai.com/docs/api-reference/responses/create).

```py
generation_config = {
  "max_new_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "do_sample": True,
  "repetition_penalty": 1.1,
  "no_repeat_ngram_size": 3,
}

response = client.responses.create(
  model="Qwen/Qwen2.5-7B-Instruct",
  instructions="You are a helpful weather assistant. Use the get_weather tool to answer questions.",
  input="What's the weather like in San Francisco?",
  tools=tools,
  stream=True,
  extra_body={"generation_config": json.dumps(generation_config)}
)

for event in response:
  print(event)
```

## Port forwarding

Port forwarding lets you serve models from a remote server. Make sure you have SSH access to the server, then run this command on your local machine.

```bash
ssh -N -f -L 8000:localhost:8000 your_server_account@your_server_IP -p port_to_ssh_into_your_server
```

## Reproducibility

Use `--force-model <repo_id>` to avoid per-request model hints and produce stable, repeatable runs.

```sh
transformers serve \
  --force-model Qwen/Qwen2.5-0.5B-Instruct \
  --continuous-batching \
  --dtype "bfloat16"
```