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

Apart from that you can also serve transformer models easily using the `transformers serve` CLI. This is ideal for experimentation purposes, or to run models locally for personal and private use.

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

or by sending an HTTP request with `cURL`, e.g.

```shell
curl -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"messages": [{"role": "system", "content": "hello"}], "temperature": 0.9, "max_tokens": 1000, "stream": true, "model": "Qwen/Qwen2.5-0.5B-Instruct"}'
```

from which you'll receive multiple chunks in the Completions API format

```shell
data: {"object": "chat.completion.chunk", "id": "req_0", "created": 1751377863, "model": "Qwen/Qwen2.5-0.5B-Instruct", "system_fingerprint": "", "choices": [{"delta": {"role": "assistant", "content": "", "tool_call_id": null, "tool_calls": null}, "index": 0, "finish_reason": null, "logprobs": null}]}

data: {"object": "chat.completion.chunk", "id": "req_0", "created": 1751377863, "model": "Qwen/Qwen2.5-0.5B-Instruct", "system_fingerprint": "", "choices": [{"delta": {"role": "assistant", "content": "", "tool_call_id": null, "tool_calls": null}, "index": 0, "finish_reason": null, "logprobs": null}]}

(...)
```

The server is also an MCP client, so it can interact with MCP tools in agentic use cases. This, of course, requires the use of an LLM that is designed to use tools.

> [!TIP]
> At the moment, MCP tool usage in `transformers` is limited to the `qwen` family of models.

<!-- TODO: example with a minimal python example, and explain that it is possible to pass a full generation config in the request -->




