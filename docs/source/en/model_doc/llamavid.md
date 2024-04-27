<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# LLaMA-VID

## Overview

The LLaMA-VID model was proposed in [<LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models>](<https://arxiv.org/abs/2311.17043>)  by Yanwei Li, Chengyao Wang, Jiaya Jia. LLaMA-VID factorises the video into content and context embedding, before feeding it to the LLM. That way LLaMA-VID will have better contextual understanding of the video, by leveraging teh Qformer embeddings tp LLM.

The abstract from the paper is the following:

*In this work, we present a novel method to tackle the
token generation challenge in Vision Language Models
(VLMs) for video and image understanding, called LLaMAVID. Current VLMs, while proficient in tasks like image
captioning and visual question answering, face computational burdens when processing long videos due to the excessive visual tokens. LLaMA-VID addresses this issue by
representing each frame with two distinct tokens, namely
context token and content token. The context token encodes
the overall image context based on user input, whereas
the content token encapsulates visual cues in each frame.
This dual-token strategy significantly reduces the overload
of long videos while preserving critical information. Generally, LLaMA-VID empowers existing frameworks to support
hour-long videos and pushes their upper limit with an extra
context token. It is proved to surpass previous methods on
most of video- or image-based benchmarks.
*

Tips:

<INSERT TIPS ABOUT MODEL HERE>

This model was contributed by [Nilesh360](https://huggingface.co/Nilesh360 HERE). The original code can be found [here](https://github.com/dvlab-research/LLaMA-VID).

## LLaMAVIDLlavaConfig

[[autodoc]] LLaMAVIDLlavaConfig


## LLaMAVIDLlavaImageProcessor

[[autodoc]] LLaMAVIDLlavaImageProcessor

LLaMAVIDLlavaProcessor

## LLaMAVIDLlavaProcessor

[[autodoc]] LLaMAVIDLlavaProcessor


## LLaMAVIDLlavaForConditionalGeneration

[[autodoc]] LLaMAVIDLlavaForConditionalGeneration
    - forward

