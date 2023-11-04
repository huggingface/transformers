<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# CED

## Overview

CED are simple ViT-Transformer-based models, which were proposed in [CED: Consistent ensemble distillation for audio tagging](https://arxiv.org/abs/2308.11957) by Heinrich Dinkel, Yongqing Wang, Zhiyong Yan, Junbo Zhang and Yujun Wang.

Notable differences from other available models include:
1. Simplification for finetuning: Batchnormalization of Mel-Spectrograms. During finetuning one does not need to first compute mean/variance over the dataset, which is common for AST.
1. Support for variable length inputs. Most other models use a static time-frequency position embedding, which hinders the model's generalization to segments shorter than 10s. Many previous transformers simply pad their input to 10s in order to avoid the performance impact, which in turn slows down training/inference drastically.
1. Training/Inference speedup: 64-dimensional mel-filterbanks and 16x16 patches without overlap, leading to 248 patches from a 10s spectrogram. In comparison, AST uses 128 mel-filterbanks with 16x16 (10x10 overlap) convolution, leading to 1212 patches during training/inference. CED-Tiny runs on a common CPU as fast as a comparable MobileNetV3.
1. Performance: CED-Mini with 10M parameters outperforms the majority of previous approaches (~80M).

The abstract from the paper is the following:

Augmentation and knowledge distillation (KD) are well-established techniques employed in audio classification tasks, aimed at enhancing performance and reducing model sizes on the widely recognized Audioset (AS) benchmark. Although both techniques are effective individually, their combined use, called consistent teaching, hasn't been explored before. This paper proposes CED, a simple training framework that distils student models from large teacher ensembles with consistent teaching. To achieve this, CED efficiently stores logits as well as the augmentation methods on disk, making it scalable to large-scale datasets. Central to CED's efficacy is its label-free nature, meaning that only the stored logits are used for the optimization of a student model only requiring 0.3\% additional disk space for AS. The study trains various transformer-based models, including a 10M parameter model achieving a 49.0 mean average precision (mAP) on AS.

This model was contributed by [Junbo Zhang](https://huggingface.co/jimbozhang).
The original code can be found [here](https://github.com/RicherMans/CED).


## CedConfig

[[autodoc]] CedConfig

## CedFeatureExtractor

[[autodoc]] CedFeatureExtractor

## CedModel

[[autodoc]] CedModel
    - forward

## CedForAudioClassification

[[autodoc]] CedForAudioClassification
    - forward