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

The `transformers serve` CLI is a lightweight option for local or self-hosted servers. It avoids the extra runtime and operational overhead of dedicated inference engines like vLLM. Use it for evaluation, experimentation, and moderate load deployments. Features like [continuous batching](../continuous_batching) increases throughput and lowers latency.

> [!TIP]
> For large scale production deployments, use vLLM, SGLang or TGI with a Transformer model as the backend. Learn more in the [Inference backends](../community_integrations/transformers_as_backend) guide.

The `transformers serve` command spawns a local server compatible with the [OpenAI SDK](https://platform.openai.com/docs/overview). The server works with many third-party applications and supports the REST APIs below.

- `/v1/chat/completions` for text and image requests
- `/v1/responses` supports the [Responses API](https://platform.openai.com/docs/api-reference/responses)
- `/v1/audio/transcriptions` for audio transcriptions
- `/v1/models` lists available models for third-party integrations

Install the serving dependencies.

```bash
pip install transformers[serving]
```

Run `transformers serve` to launch a server. The default server address is http://localhost:8000.

```shell
transformers serve
```

## v1/chat/completions

The `v1/chat/completions` API is based on the [Chat Completions API](https://platform.openai.com/docs/api-reference/chat). It supports text and image-based requests for LLMs and VLMs. Use it with `curl`, the [`~huggingface_hub.AsyncInferenceClient`], or the [OpenAI](https://platform.openai.com/docs/quickstart) client.

### Text-based completions

<hfoptions id="chat-completion-http">
<hfoption id="curl">

```shell
curl -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"messages": [{"role": "system", "content": "hello"}], "temperature": 0.9, "max_tokens": 1000, "stream": true, "model": "Qwen/Qwen2.5-0.5B-Instruct"}'
```

The command returns the following response.

```shell
data: {"object": "chat.completion.chunk", "id": "req_0", "created": 1751377863, "model": "Qwen/Qwen2.5-0.5B-Instruct", "system_fingerprint": "", "choices": [{"delta": {"role": "assistant", "content": "", "tool_call_id": null, "tool_calls": null}, "index": 0, "finish_reason": null, "logprobs": null}]}

data: {"object": "chat.completion.chunk", "id": "req_0", "created": 1751377863, "model": "Qwen/Qwen2.5-0.5B-Instruct", "system_fingerprint": "", "choices": [{"delta": {"role": "assistant", "content": "", "tool_call_id": null, "tool_calls": null}, "index": 0, "finish_reason": null, "logprobs": null}]}

(...)
```

</hfoption>
<hfoption id="huggingface_hub">

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

The [`~huggingface_hub.AsyncInferenceClient`] returns a printed string.

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
    stream=True
)

for chunk in completion:
    token = chunk.choices[0].delta.content
    if token:
        print(token, end='')
```

The [OpenAI](https://platform.openai.com/docs/quickstart) client returns a printed string.

```shell
The Transformers library is primarily known for its ability to create and manipulate large-scale language models [...]
```

</hfoption>
</hfoptions>

### Text and image-based completions

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

The command returns the following response.

```shell
data: {"id":"req_0","choices":[{"delta":{"role":"assistant"},"index":0}],"created":1753366665,"model":"Qwen/Qwen2.5-VL-7B-Instruct@main","object":"chat.completion.chunk","system_fingerprint":""}

data: {"id":"req_0","choices":[{"delta":{"content":"The "},"index":0}],"created":1753366701,"model":"Qwen/Qwen2.5-VL-7B-Instruct@main","object":"chat.completion.chunk","system_fingerprint":""}

data: {"id":"req_0","choices":[{"delta":{"content":"image "},"index":0}],"created":1753366701,"model":"Qwen/Qwen2.5-VL-7B-Instruct@main","object":"chat.completion.chunk","system_fingerprint":""}
```

</hfoption>
<hfoption id="huggingface_hub">

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

The [`~huggingface_hub.AsyncInferenceClient`] returns a printed string.

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
    stream=True
)

for chunk in completion:
    token = chunk.choices[0].delta.content
    if token:
        print(token, end='')
```

The [OpenAI](https://platform.openai.com/docs/quickstart) client returns a printed string.

```xmp
The image depicts an astronaut in a space suit standing on what appears to be the surface of the moon, given the barren, rocky landscape and the dark sky in the background. The astronaut is holding a large egg that has cracked open, revealing a small creature inside. The scene is imaginative and playful, combining elements of space exploration with a whimsical twist involving the egg and the creature.
```

</hfoption>
</hfoptions>

## v1/responses

> [!WARNING]
> The `v1/responses` API is still experimental and there may be bugs. Please open an issue if you encounter any errors.

The [Responses API](https://platform.openai.com/docs/api-reference/responses) is OpenAI's latest API endpoint for generation. It supports stateful interactions and integrates built-in tools to extend a model's capabilities. OpenAI [recommends](https://platform.openai.com/docs/guides/migrate-to-responses) using the Responses API over the Chat Completions API for new projects.

The `v1/responses` API supports text-based requests for LLMs through the `curl` command and [OpenAI](https://platform.openai.com/docs/quickstart) client.

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

The command returns the following response.

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
<hfoption id="openai">

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

The [OpenAI](https://platform.openai.com/docs/quickstart) client returns multiple printed strings.

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

Use the command below to download a model before running `transformers serve`.

```bash
transformers download Qwen/Qwen2.5-0.5B-Instruct
```

The model is now discoverable by the `/v1/models` endpoint.

```shell
curl http://localhost:8000/v1/models
```

This command returns a JSON object containing the list of models.

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

Pass a dictionary of parameters from [`GenerationConfig`] to the `extra_body` argument in [create](https://platform.openai.com/docs/api-reference/responses/create) to customize model generation.

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

The `transformers serve` server supports port forwarding. This lets you serve models from a remote server. Make sure you have ssh access from your device to the server. Run the following command on your device to set up port forwarding.

```bash
ssh -N -f -L 8000:localhost:8000 your_server_account@your_server_IP -p port_to_ssh_into_your_server
```

## Reproducibility

Add the `--force-model <repo_id>` argument to avoid per-request model hints. This produces stable, repeatable runs.

```sh
transformers serve \
  --force-model Qwen/Qwen2.5-0.5B-Instruct \
  --continuous-batching \
  --dtype "bfloat16"
```