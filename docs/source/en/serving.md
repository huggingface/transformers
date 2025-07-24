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


### Usage example 1: chat with local requests (feat. Jan)

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


### Usage example 2: chat with external requests (feat. Cursor)

This example shows how to use `transformers serve` as a local LLM provider for [Cursor](https://cursor.com/), the popular IDE. Unlike in the previous example, requests to `transformers serve` will come from an external IP (Cursor's server IPs), which requires some additional setup. Furthermore, some of Cursor's requests require [CORS](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/CORS), which is disabled by default for security reasons.

To launch a server with CORS enabled, run

```shell
transformers serve --enable-cors
```

You'll also need to expose your server to external IPs. A potential solution is to use [`ngrok`](https://ngrok.com/), which has a permissive free tier. After setting up your `ngrok` account and authenticating on your server machine, you run

```shell
ngrok http [port]
```

where `port` is the port used by `transformers serve` (`8000` by default). On the terminal where you launched `ngrok`, you'll see an https address in the "Forwarding" row, as in the image below. This is the address to send requests to.

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_serve_ngrok.png"/>
</h3>

You're now ready to set things up on the app side! In Cursor, while you can't set a new provider, you can change the endpoint for OpenAI requests in the model selection settings. First, navigate to "Settings" > "Cursor Settings", "Models" tab, and expand the "API Keys" collapsible. To set your `transformers serve` endpoint, follow this order:
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

### Usage example 4: speech to text transcription (feat. Open WebUI)

This guide shows how to do audio transcription for chat purposes, using `transformers serve` and [Open WebUI](https://openwebui.com/). This guide assumes you have Open WebUI installed on your machine and ready to run. Please refer to the examples above to use the text functionalities of `transformer serve` with Open WebUI -- the instructions are the same.

To start, let's launch the server. Some of Open WebUI's requests require [CORS](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/CORS), which is disabled by default for security reasons, so you need to enable it:

```shell
transformers serve --enable-cors
```

Before you can speak into Open WebUI, you need to update its settings to use your server for speech to text (STT) tasks. Launch Open WebUI, and navigate to the audio tab inside the admin settings. If you're using Open WebUI with the default ports, [this link (default)](http://localhost:3000/admin/settings/audio) or [this link (python deployment)](http://localhost:8080/admin/settings/audio) will take you there. Do the following changes there:
1. Change the type of "Speech-to-Text Engine" to "OpenAI";
2. Update the address to your server's address -- `http://localhost:8000/v1` by default;
3. Type your model of choice into the "STT Model" field, e.g. `openai/whisper-large-v3` ([available models](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=trending)).

If you've done everything correctly, the audio tab should look like this

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_openwebui_stt_settings.png"/>
</h3>

You're now ready to speak! Open a new chat, utter a few words after hitting the microphone button, and you should see the corresponding text on the chat input after the model transcribes it.
