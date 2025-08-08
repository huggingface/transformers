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

# Serving

Transformer models can be efficiently deployed using libraries such as vLLM, Text Generation Inference (TGI), and others. These libraries are designed for production-grade user-facing services, and can scale to multiple servers and millions of concurrent users. Refer to [Transformers as Backend for Inference Servers](./transformers_as_backends) for usage examples.

> [!TIP]
> Responses API is now supported as an experimental API! Read more about it [here](#responses-api).

Apart from that you can also serve transformer models easily using the `transformers serve` CLI. This is ideal for experimentation purposes, or to run models locally for personal and private use.

In this document, we dive into the different supported endpoints and modalities; we also cover the setup of several user interfaces that can be used on top of `transformers serve` in the following guides:
- [Jan (text and MCP user interface)](./jan.md)
- [Cursor (IDE)](./cursor.md)
- [Open WebUI (text, image, speech user interface)](./open_webui.md)
- [Tiny-Agents (text and MCP CLI tool)](./tiny_agents.md)

## Serve CLI

> [!WARNING]
> This section is experimental and subject to change in future versions

You can serve models of diverse modalities supported by `transformers` with the `transformers serve` CLI. It spawns a local server that offers compatibility with the OpenAI SDK, which is the _de facto_ standard for LLM conversations and other related tasks. This way, you can use the server from many third party applications, or test it using the `transformers chat` CLI ([docs](conversations.md#chat-cli)).

The server supports the following REST APIs:
- `/v1/chat/completions`
- `/v1/responses`
- `/v1/audio/transcriptions`
- `/v1/models`

To launch a server, simply use the `transformers serve` CLI command:

```shell
transformers serve
```

The simplest way to interact with the server is through our `transformers chat` CLI

```shell
transformers chat localhost:8000 --model-name-or-path Qwen/Qwen3-4B
```

or by sending an HTTP request, like we'll see below.

## Chat Completions - text-based

See below for examples for text-based requests. Both LLMs and VLMs should handle 

<hfoptions id="chat-completion-http">
<hfoption id="curl">

```shell
curl -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"messages": [{"role": "system", "content": "hello"}], "temperature": 0.9, "max_tokens": 1000, "stream": true, "model": "Qwen/Qwen2.5-0.5B-Instruct"}'
```

from which you'll receive multiple chunks in the Completions API format

```shell
data: {"object": "chat.completion.chunk", "id": "req_0", "created": 1751377863, "model": "Qwen/Qwen2.5-0.5B-Instruct", "system_fingerprint": "", "choices": [{"delta": {"role": "assistant", "content": "", "tool_call_id": null, "tool_calls": null}, "index": 0, "finish_reason": null, "logprobs": null}]}

data: {"object": "chat.completion.chunk", "id": "req_0", "created": 1751377863, "model": "Qwen/Qwen2.5-0.5B-Instruct", "system_fingerprint": "", "choices": [{"delta": {"role": "assistant", "content": "", "tool_call_id": null, "tool_calls": null}, "index": 0, "finish_reason": null, "logprobs": null}]}

(...)
```

</hfoption>
<hfoption id="python - huggingface_hub">

```python
import asyncio
from huggingface_hub import AsyncInferenceClient

messages = [{"role": "user", "content": "What is the Transformers library known for?"}]
client = AsyncInferenceClient("http://localhost:8000")

async def responses_api_test_async():
    async for chunk in (await client.chat_completion(messages, model="Qwen/Qwen2.5-0.5B-Instruct", max_tokens=256, stream=True)):
        token = chunk.choices[0].delta.content
        if token:
            print(token, end='')

asyncio.run(responses_api_test_async())
asyncio.run(client.close())
```

From which you should get an iterative string printed:

```shell
The Transformers library is primarily known for its ability to create and manipulate large-scale language models [...]
```

</hfoption>
<hfoption id="python - openai">

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
    stream=True
)

for chunk in completion:
    token = chunk.choices[0].delta.content
    if token:
        print(token, end='')
```

From which you should get an iterative string printed:

```shell
The Transformers library is primarily known for its ability to create and manipulate large-scale language models [...]
```

</hfoption>
</hfoptions>

## Chat Completions - VLMs

The Chat Completion API also supports images; see below for examples for text-and-image-based requests.

<hfoptions id="chat-completion-http-images">
<hfoption id="curl">

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
            "text": "What is in this image?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            }
          }
        ]
      }
    ],
    "max_tokens": 300
  }'

```

from which you'll receive multiple chunks in the Completions API format

```shell
data: {"id":"req_0","choices":[{"delta":{"role":"assistant"},"index":0}],"created":1753366665,"model":"Qwen/Qwen2.5-VL-7B-Instruct@main","object":"chat.completion.chunk","system_fingerprint":""}

data: {"id":"req_0","choices":[{"delta":{"content":"The "},"index":0}],"created":1753366701,"model":"Qwen/Qwen2.5-VL-7B-Instruct@main","object":"chat.completion.chunk","system_fingerprint":""}

data: {"id":"req_0","choices":[{"delta":{"content":"image "},"index":0}],"created":1753366701,"model":"Qwen/Qwen2.5-VL-7B-Instruct@main","object":"chat.completion.chunk","system_fingerprint":""}
```

</hfoption>
<hfoption id="python - huggingface_hub">

```python
import asyncio
from huggingface_hub import AsyncInferenceClient

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
client = AsyncInferenceClient("http://localhost:8000")

async def responses_api_test_async():
    async for chunk in (await client.chat_completion(messages, model="Qwen/Qwen2.5-VL-7B-Instruct", max_tokens=256, stream=True)):
        token = chunk.choices[0].delta.content
        if token:
            print(token, end='')

asyncio.run(responses_api_test_async())
asyncio.run(client.close())
```

From which you should get an iterative string printed:

```xmp
The image depicts an astronaut in a space suit standing on what appears to be the surface of the moon, given the barren, rocky landscape and the dark sky in the background. The astronaut is holding a large egg that has cracked open, revealing a small creature inside. The scene is imaginative and playful, combining elements of space exploration with a whimsical twist involving the egg and the creature.
```

</hfoption>
<hfoption id="python - openai">

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
    stream=True
)

for chunk in completion:
    token = chunk.choices[0].delta.content
    if token:
        print(token, end='')
```

From which you should get an iterative string printed:

```xmp
The image depicts an astronaut in a space suit standing on what appears to be the surface of the moon, given the barren, rocky landscape and the dark sky in the background. The astronaut is holding a large egg that has cracked open, revealing a small creature inside. The scene is imaginative and playful, combining elements of space exploration with a whimsical twist involving the egg and the creature.
```

</hfoption>
</hfoptions>

## Responses API

The Responses API is the newest addition to the supported APIs of `transformers serve`.

> [!TIP]
> This API is still experimental: expect bug patches and additition of new features in the coming weeks.
> If you run into any issues, please let us know and we'll work on fixing them ASAP.

Instead of the previous `/v1/chat/completions` path, the Responses API lies behind the `/v1/responses` path.
See below for examples interacting with our Responses endpoint with `curl`, as well as the Python OpenAI client.

So far, this endpoint only supports text and therefore only LLMs. VLMs to come!

<hfoptions id="responses">
<hfoption id="curl">

```shell
curl http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "stream": true,
    "input": "Tell me a three sentence bedtime story about a unicorn."
  }'
```

from which you'll receive multiple chunks in the Responses API format

```shell
data: {"response":{"id":"resp_req_0","created_at":1754059817.783648,"model":"Qwen/Qwen2.5-0.5B-Instruct@main","object":"response","output":[],"parallel_tool_calls":false,"tool_choice":"auto","tools":[],"status":"queued","text":{"format":{"type":"text"}}},"sequence_number":0,"type":"response.created"}

data: {"response":{"id":"resp_req_0","created_at":1754059817.783648,"model":"Qwen/Qwen2.5-0.5B-Instruct@main","object":"response","output":[],"parallel_tool_calls":false,"tool_choice":"auto","tools":[],"status":"in_progress","text":{"format":{"type":"text"}}},"sequence_number":1,"type":"response.in_progress"}

data: {"item":{"id":"msg_req_0","content":[],"role":"assistant","status":"in_progress","type":"message"},"output_index":0,"sequence_number":2,"type":"response.output_item.added"}

data: {"content_index":0,"item_id":"msg_req_0","output_index":0,"part":{"annotations":[],"text":"","type":"output_text"},"sequence_number":3,"type":"response.content_part.added"}

data: {"content_index":0,"delta":"","item_id":"msg_req_0","output_index":0,"sequence_number":4,"type":"response.output_text.delta"}

data: {"content_index":0,"delta":"Once ","item_id":"msg_req_0","output_index":0,"sequence_number":5,"type":"response.output_text.delta"}

data: {"content_index":0,"delta":"upon ","item_id":"msg_req_0","output_index":0,"sequence_number":6,"type":"response.output_text.delta"}

data: {"content_index":0,"delta":"a ","item_id":"msg_req_0","output_index":0,"sequence_number":7,"type":"response.output_text.delta"}
```

</hfoption>
<hfoption id="python - openai">

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="<KEY>")

response = client.responses.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    instructions="You are a helpful assistant.",
    input="Hello!",
    stream=True,
    metadata={"foo": "bar"},
)

for event in response:
    print(event)
```

From which you should get events printed out successively.

```shell
ResponseCreatedEvent(response=Response(id='resp_req_0', created_at=1754060400.3718212, error=None, incomplete_details=None, instructions='You are a helpful assistant.', metadata={'foo': 'bar'}, model='Qwen/Qwen2.5-0.5B-Instruct@main', object='response', output=[], parallel_tool_calls=False, temperature=None, tool_choice='auto', tools=[], top_p=None, background=None, max_output_tokens=None, max_tool_calls=None, previous_response_id=None, prompt=None, reasoning=None, service_tier=None, status='queued', text=ResponseTextConfig(format=ResponseFormatText(type='text')), top_logprobs=None, truncation=None, usage=None, user=None), sequence_number=0, type='response.created')
ResponseInProgressEvent(response=Response(id='resp_req_0', created_at=1754060400.3718212, error=None, incomplete_details=None, instructions='You are a helpful assistant.', metadata={'foo': 'bar'}, model='Qwen/Qwen2.5-0.5B-Instruct@main', object='response', output=[], parallel_tool_calls=False, temperature=None, tool_choice='auto', tools=[], top_p=None, background=None, max_output_tokens=None, max_tool_calls=None, previous_response_id=None, prompt=None, reasoning=None, service_tier=None, status='in_progress', text=ResponseTextConfig(format=ResponseFormatText(type='text')), top_logprobs=None, truncation=None, usage=None, user=None), sequence_number=1, type='response.in_progress')
ResponseOutputItemAddedEvent(item=ResponseOutputMessage(id='msg_req_0', content=[], role='assistant', status='in_progress', type='message'), output_index=0, sequence_number=2, type='response.output_item.added')
ResponseContentPartAddedEvent(content_index=0, item_id='msg_req_0', output_index=0, part=ResponseOutputText(annotations=[], text='', type='output_text', logprobs=None), sequence_number=3, type='response.content_part.added')
ResponseTextDeltaEvent(content_index=0, delta='', item_id='msg_req_0', output_index=0, sequence_number=4, type='response.output_text.delta')
ResponseTextDeltaEvent(content_index=0, delta='', item_id='msg_req_0', output_index=0, sequence_number=5, type='response.output_text.delta')
ResponseTextDeltaEvent(content_index=0, delta='Hello! ', item_id='msg_req_0', output_index=0, sequence_number=6, type='response.output_text.delta')
ResponseTextDeltaEvent(content_index=0, delta='How ', item_id='msg_req_0', output_index=0, sequence_number=7, type='response.output_text.delta')
ResponseTextDeltaEvent(content_index=0, delta='can ', item_id='msg_req_0', output_index=0, sequence_number=8, type='response.output_text.delta')
ResponseTextDeltaEvent(content_index=0, delta='I ', item_id='msg_req_0', output_index=0, sequence_number=9, type='response.output_text.delta')
ResponseTextDeltaEvent(content_index=0, delta='assist ', item_id='msg_req_0', output_index=0, sequence_number=10, type='response.output_text.delta')
ResponseTextDeltaEvent(content_index=0, delta='you ', item_id='msg_req_0', output_index=0, sequence_number=11, type='response.output_text.delta')
ResponseTextDeltaEvent(content_index=0, delta='', item_id='msg_req_0', output_index=0, sequence_number=12, type='response.output_text.delta')
ResponseTextDeltaEvent(content_index=0, delta='', item_id='msg_req_0', output_index=0, sequence_number=13, type='response.output_text.delta')
ResponseTextDeltaEvent(content_index=0, delta='today?', item_id='msg_req_0', output_index=0, sequence_number=14, type='response.output_text.delta')
ResponseTextDoneEvent(content_index=0, item_id='msg_req_0', output_index=0, sequence_number=15, text='Hello! How can I assist you today?', type='response.output_text.done')
ResponseContentPartDoneEvent(content_index=0, item_id='msg_req_0', output_index=0, part=ResponseOutputText(annotations=[], text='Hello! How can I assist you today?', type='output_text', logprobs=None), sequence_number=16, type='response.content_part.done')
ResponseOutputItemDoneEvent(item=ResponseOutputMessage(id='msg_req_0', content=[ResponseOutputText(annotations=[], text='Hello! How can I assist you today?', type='output_text', logprobs=None)], role='assistant', status='completed', type='message', annotations=[]), output_index=0, sequence_number=17, type='response.output_item.done')
ResponseCompletedEvent(response=Response(id='resp_req_0', created_at=1754060400.3718212, error=None, incomplete_details=None, instructions='You are a helpful assistant.', metadata={'foo': 'bar'}, model='Qwen/Qwen2.5-0.5B-Instruct@main', object='response', output=[ResponseOutputMessage(id='msg_req_0', content=[ResponseOutputText(annotations=[], text='Hello! How can I assist you today?', type='output_text', logprobs=None)], role='assistant', status='completed', type='message', annotations=[])], parallel_tool_calls=False, temperature=None, tool_choice='auto', tools=[], top_p=None, background=None, max_output_tokens=None, max_tool_calls=None, previous_response_id=None, prompt=None, reasoning=None, service_tier=None, status='completed', text=ResponseTextConfig(format=ResponseFormatText(type='text')), top_logprobs=None, truncation=None, usage=None, user=None), sequence_number=18, type='response.completed')
```

</hfoption>
</hfoptions>


## MCP integration

The `transformers serve` server is also an MCP client, so it can interact with MCP tools in agentic use cases. This, of course, requires the use of an LLM that is designed to use tools.

> [!TIP]
> At the moment, MCP tool usage in `transformers` is limited to the `qwen` family of models.

<!-- TODO: example with a minimal python example, and explain that it is possible to pass a full generation config in the request -->




