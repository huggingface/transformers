<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Open WebUI

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