<!--Copyright 2026 The Upstage Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2025-12-31 and added to Hugging Face Transformers on 2026-01-13.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# SolarOpen

## Overview

The SolarOpen model was proposed in [Solar Open Technical Report](https://huggingface.co/upstage/Solar-Open-100B/blob/main/solar-open-technical-report.pdf) by Upstage Team.

The abstract from the paper is the following:

We introduce Solar Open, a 102B-parameter bilingual Mixture-of-Experts language model for underserved languages.
Solar Open demonstrates a systematic methodology for building competitive LLMs by addressing three interconnected challenges.
First, to train effectively despite data scarcity for underserved languages, we synthesize 4.5T tokens of high-quality, domain-specific, and RL-oriented
data.
Second, we coordinate this data through a progressive curriculum jointly optimizing composition, quality thresholds, and domain coverage across 20 trillion tokens.
Third, to enable reasoning capabilities through scalable RL, we apply our proposed framework SnapPO for efficient optimization.
Across benchmarks in English and Korean, Solar Open achieves competitive performance, demonstrating the effectiveness of this methodology for underserved language AI development.

## Usage Tips

Recommended inference parameters for optimal performance:

```
temperature=0.8
top_p=0.95
top_k=50
```

This model was contributed by [SSON9](https://huggingface.co/SSON9) from [Upstage](https://huggingface.co/upstage).

## SolarOpenConfig

[[autodoc]] SolarOpenConfig

## SolarOpenModel

[[autodoc]] SolarOpenModel
    - forward

## SolarOpenForCausalLM

[[autodoc]] SolarOpenForCausalLM
    - forward