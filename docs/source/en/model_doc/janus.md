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

# Janus

## Overview 

The Janus Model was originally proposed in [Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation](https://arxiv.org/abs/2410.13848) by DEEPSEEK AI team. Janus is a Vision language Model can generate both Images and Text output. The model can take both images and text as input. Note: The models doesn't generate both images and text in an interleaved format rather it could generate either text or image at a time.

The abstract from the paper is the following:

*In this paper, we introduce Janus, an autoregressive framework that unifies multimodal understanding and generation. Prior research often relies on a single visual encoder for both tasks, such as Chameleon. However, due to the differing levels of information granularity required by multimodal understanding and generation, this approach can lead to suboptimal performance, particularly in multimodal understanding. To address this issue, we decouple visual encoding into separate pathways, while still leveraging a single, unified transformer architecture for processing. The decoupling not only alleviates the conflict between the visual encoder's roles in understanding and generation, but also enhances the framework's flexibility. For instance, both the multimodal understanding and generation components can independently select their most suitable encoding methods. Experiments show that Janus surpasses previous unified model and matches or exceeds the performance of task-specific models. The simplicity, high flexibility, and effectiveness of Janus make it a strong candidate for next-generation unified multimodal models.*

Subsequently they have released [Janus-Pro: Unified Multimodal Understanding and
Generation with Data and Model Scaling](https://arxiv.org/abs/2501.17811) which is an advanced version of previous version of Janus. 

The abstract from following `Janus-Pro` paper:

*In this work, we introduce Janus-Pro, an advanced version of the previous work Janus. Specifically, Janus-Pro incorporates (1) an optimized training strategy, (2) expanded training data,
and (3) scaling to larger model size. With these improvements, Janus-Pro achieves significant
advancements in both multimodal understanding and text-to-image instruction-following capabilities, while also enhancing the stability of text-to-image generation. We hope this work will
inspire further exploration in the field. Code and models are publicly available.*

This model was contributed by [Yaswanth Gali](https://huggingface.co/yaswanthgali) and []().
The original code can be found [here](https://github.com/deepseek-ai/Janus).


## JanusConfig

[[autodoc]] JanusConfig

## JanusForConditionalGeneration

[[autodoc]] JanusForConditionalGeneration
    - forward
