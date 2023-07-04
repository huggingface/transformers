<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# CLIP ViP

## Overview

The CLIP ViP model was proposed in [CLIP-ViP: Adapting Pre-trained Image-Text Model to Video-Language Representation Alignment](https://arxiv.org/abs/2209.06430) by Hongwei Xue, Yuchong Sun, Bei Liu, Jianlong Fu, Ruihua Song, Houqiang Li, Jiebo Luo.

It is a dual encoder setup that CLIP as a base model, that adds a novel Video-Proxy mechanism to adapt CLIP to video retrieval. 

The abstract from the paper is the following:

The pre-trained image-text models, like CLIP, have demonstrated the strong power of vision-language representation learned from a large scale of web-collected image-text data. In light of the well-learned visual features, some existing works transfer image representation to video domain and achieve good results. However, how to utilize image-language pre-trained model (e.g., CLIP) for video-language pre-training (post-pretraining) is still under explored. In this paper, we investigate two questions: 1) what are the factors hindering post-pretraining CLIP to further improve the performance on video-language tasks? and 2) how to mitigate the impact of these factors? Through a series of comparative experiments and analyses, we find that the data scale and domain gap between language sources have great impacts. Motivated by these, we propose a Omnisource Cross-modal Learning method equipped with a Video Proxy mechanism on the basis of CLIP, namely CLIP-ViP. Extensive results show that our approach improves the performance of CLIP on video-text retrieval by a large margin. Our model also achieves SOTA results on a variety of datasets, including MSR-VTT, DiDeMo, LSMDC, and ActivityNet. 

Tips:
    This model could be applied to search for videos based on a text query, or to search for text based on a video query.
    Some ideas for how this could be setup could come from the original implementation [here](https://github.com/microsoft/XPretrain/blob/main/CLIP-ViP/src/tasks/run_video_retrieval.py).

This model was contributed by [tensorpro](https://huggingface.co/tensorpro).
The original code can be found [here](https://github.com/microsoft/XPretrain).


## CLIPViPImageProcessor

[[autodoc]] CLIPViPImageProcessor

## CLIPViPProcessor

[[autodoc]] CLIPViPProcessor

## CLIPViPConfig

[[autodoc]] CLIPViPConfig
    - from_text_vision_configs

## CLIPViPTextConfig

[[autodoc]] CLIPViPTextConfig

## CLIPViPVisionConfig

[[autodoc]] CLIPViPVisionConfig

## CLIPViPModel

[[autodoc]] CLIPViPModel
    - forward
    - get_text_features
    - get_image_features

## CLIPViPTextModel

[[autodoc]] CLIPViPTextModel
    - forward

## CLIPViPVisionModel

[[autodoc]] CLIPViPVisionModel
    - forward
