<!--
Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Fireflies

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

**Fireflies** is a minimal encoder-only Transformer model designed for experimentation and lightweight deployment. It was built from scratch as a custom architecture and is currently focused on the Indonesian NLP domain, but flexible enough for general-purpose text modeling.

The model supports standard input/output patterns similar to BERT and is intended for continued development toward pretraining, downstream fine-tuning, and GGUF quantization.

The implementation supports:
- Custom architecture defined with `FirefliesConfig`
- Hugging Face `AutoModel` and `AutoConfig` integration
- GGUF conversion through compatibility with `transformers` and `transformers.onnx`

## FirefliesConfig

[[autodoc]] FirefliesConfig

## FirefliesModel

[[autodoc]] FirefliesModel
- forward
