<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Agent skills

Skills package specialized knowledge into a `SKILL.md` file. This provides an agent with additional context to reliably complete a task in a repeatable way. Use Transformers skills to get started faster.

## Available skills

`/fine-tuning` is designed to help an agent create a fine-tuning script for any task, model, and hardware with the [`Trainer`] API. It covers all modalities across 15 tasks and handles the full setup like model loading, preprocessing, [`TrainingArguments`], and more. It also knows when to reach for LoRA/QLoRA, how to pick the appropriate distributed training strategy based on your hardware, and how to apply memory and throughput optimizations.
