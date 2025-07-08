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

Transformer models can be efficiently deployed using libraries such as vLLM, Text Generation Inference (TGI), and others. These libraries are designed for production-grade user-facing services, and can scale to multiple servers and millions of concurrent users.

You can also serve transformer models easily using the `transformers serve` CLI. This is ideal for experimentation purposes, or to run models locally for personal and private use.

## TGI

[TGI](https://huggingface.co/docs/text-generation-inference/index) can serve models that aren't [natively implemented](https://huggingface.co/docs/text-generation-inference/supported_models) by falling back on the Transformers implementation of the model. Some of TGIs high-performance features aren't available in the Transformers implementation, but other features like continuous batching and streaming are still supported.

> [!TIP]
> Refer to the [Non-core model serving](https://huggingface.co/docs/text-generation-inference/basic_tutorials/non_core_models) guide for more details.

Serve a Transformers implementation the same way you'd serve a TGI model.

```docker
docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest --model-id gpt2
```

Add `--trust-remote_code` to the command to serve a custom Transformers model.

```docker
docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest --model-id <CUSTOM_MODEL_ID> --trust-remote-code
```

## vLLM

[vLLM](https://docs.vllm.ai/en/latest/index.html) can also serve a Transformers implementation of a model if it isn't [natively implemented](https://docs.vllm.ai/en/latest/models/supported_models.html#list-of-text-only-language-models) in vLLM.

Many features like quantization, LoRA adapters, and distributed inference and serving are supported for the Transformers implementation.

> [!TIP]
> Refer to the [Transformers fallback](https://docs.vllm.ai/en/latest/models/supported_models.html#transformers-fallback) section for more details.

By default, vLLM serves the native implementation and if it doesn't exist, it falls back on the Transformers implementation. But you can also set `--model-impl transformers` to explicitly use the Transformers model implementation.

```shell
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --task generate \
    --model-impl transformers
```

Add the `trust-remote-code` parameter to enable loading a remote code model.

```shell
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --task generate \
    --model-impl transformers \
    --trust-remote-code
```

## Serve CLI

> [!WARNING]
> This section is experimental and subject to change in future versions

<!-- TODO: LLMs -> models, after we add audio/image input/output support -->
You can serve LLMs supported by `transformers` with the `transformers serve` CLI. It spawns a local server that offers a chat Completions API compatible with the OpenAI SDK, which is the _de facto_ standard for LLM conversations. This way, you can use the server from many third party applications, or test it using the `transformers chat` CLI ([docs](conversations.md#chat-cli)).

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


### Usage example 1: apps with local requests (feat. Jan)

This example shows how to use `transformers serve` as a local LLM provider for the [Jan](https://jan.ai/) app. Jan is a ChatGPT-alternative graphical interface, fully running on your machine. The requests to `transformers serve` come directly from the local app -- while this section focuses on Jan, you can extrapolate some instructions to other apps that make local requests.

To connect `transformers serve` with Jan, you'll need to set up a new model provider ("Settings" > "Model Providers"). Click on "Add Provider", and set a new name. In your new model provider page, all you need to set is the "Base URL" to the following pattern:

```shell
http://[host]:[port]/v1
```

where `host` and `port` are the `transformers serve` CLI parameters (`localhost:8000` by default). After setting this up, you should be able to see some models in the "Models" section, hitting "Refresh". Make sure you add some text in the "API key" text field too -- this data is not actually used, but the field can't be empty. Your custom model provider page should look like this:

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_serve_jan_model_providers.png"/>
</h3>

You are now ready to chat!

> [!TIP]
> You can add any `transformers`-compatible model to Jan through `transformers serve`. In the custom model provider you created, click on the "+" button in the "Models" section and add its Hub repository name, e.g. `Qwen/Qwen3-4B`.

To conclude this example, let's look into a more advanced use-case. If you have a beefy machine to serve models with, but prefer using Jan on a different device, you need to add port forwarding. If you have `ssh` access from your Jan machine into your server, this can be accomplished by typing the following to your Jan machine's terminal

```
ssh -N -f -L 8000:localhost:8000 your_server_account@your_server_IP -p port_to_ssh_into_your_server
```

Port forwarding is not Jan-specific: you can use it to connect `transformers serve` running in a different machine with an app of your choice.


### Usage example 2: apps with external requests (feat. Cursor)

This example shows how to use `transformers serve` as a local LLM provider for [Cursor](https://cursor.com/), the popular IDE. Unlike in the previous example, requests to `transformers serve` will come from an external IP (Cursor's server IPs), which requires some additional setup. Furthermore, some of Cursor's requests require [CORS](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/CORS), which is disabled by default for security reasons.

To launch our server with CORS enabled, run

```shell
transformers serve --enable-cors
```

We'll also need to expose our server to external IPs. A potential solution is to use [`ngrok`](https://ngrok.com/), which has a permissive free tier. After setting up your `ngrok` account and authenticating on your server machine, you run

```shell
ngrok http [port]
```

where `port` is the port used by `transformers serve` (`8000` by default). On the terminal where you launched `ngrok`, you'll see an https address in the "Forwarding" row, as in the image below. This is the address to send requests to.

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_serve_ngrok.png"/>
</h3>

We're now ready to set things up on the app side! In Cursor, while we can't set a new provider, we can change the endpoint for OpenAI requests in the model selection settings. First, navigate to "Settings" > "Cursor Settings", "Models" tab, and expand the "API Keys" collapsible. To set our `transformers serve` endpoint, follow this order:
1. Unselect ALL models in the list above (e.g. `gpt4`, ...);
2. Add and select the model you want to use (e.g. `Qwen/Qwen3-4B`)
3. Add some random text to OpenAI API Key. This field won't be used, but it can’t be empty;
4. Add the https address from `ngrok` to the "Override OpenAI Base URL" field, appending `/v1` to the address (i.e. `https://(...).ngrok-free.app/v1`);
5. Hit "Verify".

After you follow these steps, your "Models" tab should look like the image below. Your server should also have received a few requests from the verification step.

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_serve_cursor.png"/>
</h3>

You are now ready to use your local model in Cursor! For instance, if you toggle the AI Pane, you can select the model you added and ask it questions about your local files.

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_serve_cursor_chat.png"/>
</h3>


### Usage example 3: `tiny-agents` CLI and MCP Tools

To showcase the use of MCP tools, let's see how to integrate the `transformers serve` server with the [`tiny-agents`](https://huggingface.co/blog/python-tiny-agents) CLI.

> [!TIP]
> Many Hugging Face Spaces can be used as MCP servers, as in this example. You can find all compatible Spaces [here](https://huggingface.co/spaces?filter=mcp-server).

The first step to use MCP tools is to let the model know which tools are available. As an example, let's consider a `tiny-agents` configuration file with a reference to an [image generation MCP server](https://evalstate-flux1-schnell.hf.space/).

```json
{
    "model": "Menlo/Jan-nano",
    "endpointUrl": "http://localhost:8000",
    "servers": [
        {
            "type": "sse",
            "url": "https://evalstate-flux1-schnell.hf.space/gradio_api/mcp/sse"
        }
    ]
}
```

You can then launch your `tiny-agents` chat interface with the following command.

```bash
tiny-agents run path/to/your/config.json
```

If you have `transformers serve` running in the background, you're ready to use MCP tools from a local model! For instance, here's the example of a chat session with `tiny-agents`:

```bash
Agent loaded with 1 tools:
 • flux1_schnell_infer
»  Generate an image of a cat on the moon
<Tool req_0_tool_call>flux1_schnell_infer {"prompt": "a cat on the moon", "seed": 42, "randomize_seed": true, "width": 1024, "height": 1024, "num_inference_steps": 4}

Tool req_0_tool_call
[Binary Content: Image image/webp, 57732 bytes]
The task is complete and the content accessible to the User
Image URL: https://evalstate-flux1-schnell.hf.space/gradio_api/file=/tmp/gradio/3dbddc0e53b5a865ed56a4e3dbdd30f3f61cf3b8aabf1b456f43e5241bd968b8/image.webp
380576952

I have generated an image of a cat on the moon using the Flux 1 Schnell Image Generator. The image is 1024x1024 pixels and was created with 4 inference steps. Let me know if you would like to make any changes or need further assistance!
```
