<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for the doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# GS1

GS1 is a family of pretrained models from BIGAI for structured decision tasks. Given a state and a question, GS1
scores the available candidates for Choice, Noul, and Score questions instead of generating a text answer.

The models use a Qwen3.5 hybrid text/vision backbone with GS1-specific bidirectional decision-context processing,
attention-residual routing, and a cross-attention decision head. The head reads the question and candidate spans from
the compiled request and returns one score per candidate.

| Model | Checkpoint |
|---|---|
| GS1-4B | [bigai/GS1-4B](https://huggingface.co/bigai/GS1-4B) |
| GS1-9B | [bigai/GS1-9B](https://huggingface.co/bigai/GS1-9B) |

GS1 is developed by [BIGAI](https://gs1.mybigai.ac.cn). See the [open-source implementation](https://github.com/bigai-nlco/GS1)
for the model and inference reference.
