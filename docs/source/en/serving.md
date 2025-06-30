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

Transformer models can be served for inference natively with the `transformers serve` CLI, or with specialized libraries such as Text Generation Inference (TGI) and vLLM. These libraries are specifically designed to optimize performance with LLMs and include many unique optimization features that may not be included in Transformers. If you're looking for experimental features instead, serving natively might be better suited to you.

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
> This section is experimental and subject to changes in future versions

<!-- TODO: LLMs -> models, after we add audio/image input/output support -->
You can serve `transformers`-compatible LLMs with `transformers serve`. The server has a chat completion API compatible with the OpenAI SDK, so you can also quickly experiment with `transformers` models on existing aplications. To launch a server, use the `transformers serve` CLI:

```shell
transformers serve
```

<!-- TODO: either fully align the two APIs, or link to the `transformers` version instead -->
This server takes an extended version of the [`ChatCompletionInput`](https://huggingface.co/docs/huggingface_hub/v0.33.1/en/package_reference/inference_types#huggingface_hub.ChatCompletionInput), accepting a serialized `GenerationConfig` in its `extra_body` field for full `generate` parameterization. The CLI will dynamically load a new model as needed, following the `model` field in the request.

This server is also an MCP client, which can receive information available MCP servers (i.e. tools), massage their information into the model prompt, and prepare calls to these tools when the model commands to do so. Naturally, this requires a model that is trained to use tools.

> [!TIP]
> At the moment, MCP tool usage in `transformers` is limited to the `qwen` family of models.

<!-- TODO: section with a minimal python example -->

### Example 1: `tiny-agents` and MCP Tools

This subsection showcases how to use `transformers serve` as a local LLM server to the [`tiny-agents`](https://huggingface.co/blog/python-tiny-agents) CLI, and how to configure MCP tools.

The first step to use MCP tools is to let the model know which tools are available. As an example, let's consider a `tiny-agents` configuration file with a reference to an [image generation MCP server](https://evalstate-flux1-schnell.hf.space/).

```json
{
    "model": "Menlo/Jan-nano",
    "endpointUrl": "http://localhost:8000",
    "servers": [
        {
            "type": "sse",
            "config": {
                "url": "https://evalstate-flux1-schnell.hf.space/gradio_api/mcp/sse"
            }
        }
    ]
}
```

> [!TIP]
> Many Hugging Face Spaces can be used as MCP servers. You can find all compatible Spaces [here](https://huggingface.co/spaces?filter=mcp-server).

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

### Example 2: Jan and serving in a different machine

This subsection showcases how to use `transformers serve` as a local LLM server within the [Jan](https://jan.ai/) app. Jan is a ChatGPT-alternative graphical interface interface, with a focus on local models.

To connect `transformers server` with Jan, you'll need to set up a new model provider ("Settings" > "Model Providers"). Click on "Add Provider", and set a new name. In your new model provider page, all you need to set is the "Base URL" to following pattern

```
http://[host]:[port]/v1
```

where `host` and `port` are the `transformers serve` CLI parameters (`localhost:8000` by default). After setting this up, you should be able to see some models in the "Models" section, hitting "Refresh". Make sure you add a Hugging Face Hub API key too. Your custom model provider page should look like this:

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_serve_jan_model_providers.png"/>
</h3>

You are now ready to chat!

> [!TIP]
> You can add any `transformers`-compatible model to Jan through `transformers server`. In the custom model provider you created, click on the "+" button in the "Models" section and add its Hub repository name.

To conclude this example, let's look into a more advanced use-case. If you have a beefy machine to serve models with, but prefer using Jan on a different device, you need to add port forwarding. If you have `ssh` access from your Jan machine into your server, this can be accomplished by typing the following to your Jan machine's terminal

```
ssh -N -f -L localhost:8000:localhost:8000 your_server_account@your_server_IP -p port_to_ssh_into_your_server
```

Port forwarding is not Jan-specific: you can use it to connect `transformers serve` running in a different machine with an app of your choice.
