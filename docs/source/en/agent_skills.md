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

# Agent skills

[Hugging Face Skills](https://github.com/huggingface/skills) teach your coding agent how to work with Transformers. Instead of writing a training script yourself, ask your agent to fine-tune a vision model and let the skill guide it.

## Install

Hugging Face skills live in [huggingface/skills](https://github.com/huggingface/skills) and work across all major coding agents.

<hfoptions id="install">
<hfoption id="Claude Code">

```sh
/plugin marketplace add huggingface/skills
/plugin install huggingface-vision-trainer@huggingface/skills
```

</hfoption>
<hfoption id="Codex">

```sh
codex plugin marketplace add huggingface/skills
```

Then run `/plugins` in Codex and install `huggingface-vision-trainer` from the `huggingface/skills` repository.

</hfoption>
</hfoptions>

See the [Installation](https://github.com/huggingface/skills#installation) guide for more supported platforms like Cursor and Gemini and for a list of all available skills.

Once a skill is installed, include it in your instructions to your coding agent.

```md
Use the HF Trainer skill to fine-tune RT-DETRv2 on [Lekim89/sportsmot](https://huggingface.co/datasets/Lekim89/sportsmot) for basketball player tracking.
```
