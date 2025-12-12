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

# Serve CLI integrations

Run `transformers serve` as a local LLM provider. Access Transformers models directly from your favorite apps and development tools.

## Cursor

[Cursor](https://cursor.com) is an AI-powered code editor that offers models like Sonnet, Gemini, and GPT. Enable [CORS](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/CORS) to serve local models (it's disabled by default for security).

```bash
transformers serve --enable-cors
```

Cursor requires a public address to reach your local server. Use [ngrok](https://ngrok.com/) to create a tunnel. Sign up, authenticate, and run this command.

```bash
ngrok http localhost:8000
```

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_serve_ngrok.png"/>
</h3>

Open Cursor and go to **Settings > Cursor Settings > Models > API Keys**.

1. Deselect all models in the list.
2. Add your desired model (Qwen/Qwen3-4B).
3. Enter any text in the **OpenAI API Key** field (required).
4. Paste the ngrok Forwarding address into **Override OpenAI Base URL**. Append `/v1` to the URL (https://...ngrok-free.app/v1).
5. Click **Verify**.

Your model is ready to use.

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_serve_cursor.png"/>
</h3>

## Jan

[Jan](https://jan.ai/) is an open-source ChatGPT alternative. It supports `transformers serve` natively without extra tunneling.

Go to **Settings > Model Providers**.

1. Set **Base URL** to http://localhost:8000/v1.
2. Enter any text in **API Key** (required).

Check the **Models** section (click **Refresh** if empty). Click **+** to add a specific model like Qwen/Qwen3-4B.

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_serve_jan_model_providers.png"/>
</h3>

## Open WebUI

[Open WebUI](https://openwebui.com/) provides a self-hosted interface for offline models. Enable [CORS](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/CORS) to serve local models (it's disabled by default for security).

```bash
transformers serve --enable-cors
```

To use the server for speech-to-text, go to **Settings > Audio**.

1. Set **Speech-to-Text Engine** to **OpenAI**.
2. Set the URL to http://localhost:8000/v1.
3. Enter a transcription model in **SST Model** (openai/whisper-large-v3). Find more models on the [Hub](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=trending).

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_openwebui_stt_settings.png"/>
</h3>

Start a chat and speak. The model transcribes your audio into the input field.

## tiny-agents

[tiny-agents](https://huggingface.co/blog/python-tiny-agents) offers a minimal pattern for building tool-using agents. A small Python loop connects an MCP client to tools from MCP servers. Hugging Face Spaces work as MCP servers. Find more compatible Spaces on the [Hub](https://huggingface.co/spaces?filter=mcp-server).

Create a config file that points to your local model and the tool server. This example uses an image generation [Space](https://evalstate-flux1-schnell.hf.space/).

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

Run the agent with your config.

```bash
tiny-agents run path/to/your/config.json
```

Ensure `transformers serve` is running in the background. The agent will use your local model to run the image generation tool.

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
