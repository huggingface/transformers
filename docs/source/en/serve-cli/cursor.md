<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Cursor

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