<!--Copyright 2026 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-08-26.*


# GLM-5.3-Flash

## Overview

The **GLM-5.3-Flash** model use this class. The implementation in transformers does not include an MTP layer.


### GLM-5.3-Flash


GLM-5.3-Flash, the first **natively multimodal model** in the GLM-5 series. With 320B total parameters and just 18B active parameters, it outperforms GLM-5.2 across benchmarks and real-world workloads at one-tenth the price, while approaching Claude Opus 4.8 on coding and agentic benchmarks.

GLM-5.3-Flash starts from a newly trained base model, with its architecture and training recipe redesigned around capability and efficiency. For the first time in the GLM series, we introduce a hybrid architecture combining sparse and linear attention, sharply reducing long-context serving costs while preserving precise long-context capabilities. The model also adopts Manifold-Constrained Hyper-Connections (mHC) to further improve scaling efficiency. Together with our latest **30T-token** multimodal pre-training corpus, these changes enable GLM-5.3-Flash to deliver more intelligence with less compute.

![bench_53](https://raw.githubusercontent.com/zai-org/GLM-5/refs/heads/main/resources/bench_53.png)

## Glm5NextConfig

[[autodoc]] Glm5NextConfig

## Glm5NextTextConfig

[[autodoc]] Glm5NextTextConfig

## Glm5NextVisionConfig

[[autodoc]] Glm5NextVisionConfig

## Glm5NextPreTrainedModel

[[autodoc]] Glm5NextPreTrainedModel
    - forward

## Glm5NextTextModel

[[autodoc]] Glm5NextTextModel
    - forward

## Glm5NextModel
## Glm5NextVisionModel

[[autodoc]] Glm5NextVisionModel
    - forward


[[autodoc]] Glm5NextModel
    - forward

## Glm5NextForConditionalGeneration

[[autodoc]] Glm5NextForConditionalGeneration
    - forward

## Glm5NextProcessor

[[autodoc]] Glm5NextProcessor

## Glm5NextImageProcessor

[[autodoc]] Glm5NextImageProcessor

## Glm5NextImageProcessorPil

[[autodoc]] Glm5NextImageProcessorPil

## Glm5NextVideoProcessor

[[autodoc]] Glm5NextVideoProcessor
