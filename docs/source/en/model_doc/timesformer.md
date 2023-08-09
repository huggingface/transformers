<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# TimeSformer

## Overview

The TimeSformer model was proposed in [TimeSformer: Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/abs/2102.05095) by Facebook Research.
This work is a milestone in action-recognition field being the first video transformer. It inspired many transformer based video understanding and classification papers.

The abstract from the paper is the following:

*We present a convolution-free approach to video classification built exclusively on self-attention over space and time. Our method, named "TimeSformer," adapts the standard Transformer architecture to video by enabling spatiotemporal feature learning directly from a sequence of frame-level patches. Our experimental study compares different self-attention schemes and suggests that "divided attention," where temporal attention and spatial attention are separately applied within each block, leads to the best video classification accuracy among the design choices considered. Despite the radically new design, TimeSformer achieves state-of-the-art results on several action recognition benchmarks, including the best reported accuracy on Kinetics-400 and Kinetics-600. Finally, compared to 3D convolutional networks, our model is faster to train, it can achieve dramatically higher test efficiency (at a small drop in accuracy), and it can also be applied to much longer video clips (over one minute long). Code and models are available at: [this https URL](https://github.com/facebookresearch/TimeSformer).*

Tips:

There are many pretrained variants. Select your pretrained model based on the dataset it is trained on. Moreover, the number of input frames per clip changes based on the model size so you should consider this parameter while selecting your pretrained model.

This model was contributed by [fcakyon](https://huggingface.co/fcakyon).
The original code can be found [here](https://github.com/facebookresearch/TimeSformer).

## Documentation resources

- [Video classification task guide](../tasks/video_classification)

## TimesformerConfig

[[autodoc]] TimesformerConfig

## TimesformerModel

[[autodoc]] TimesformerModel
    - forward

## TimesformerForVideoClassification

[[autodoc]] TimesformerForVideoClassification
    - forward