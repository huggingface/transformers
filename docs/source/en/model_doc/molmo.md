<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Molmo

# Molmo

## Overview

The Molmo model was proposed in [Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models
]([<INSERT PAPER LINK HERE>](https://arxiv.org/abs/2409.17146)) by Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, Jiasen Lu, Taira Anderson, Erin Bransom, Kiana Ehsani, Huong Ngo, YenSung Chen, Ajay Patel, Mark Yatskar, Chris Callison-Burch, Andrew Head, Rose Hendrix, Favyen Bastani, Eli VanderBilt, Nathan Lambert, Yvonne Chou, Arnavi Chheda, Jenna Sparks, Sam Skjonsberg, Michael Schmitz, Aaron Sarnat, Byron Bischoff, Pete Walsh, Chris Newell, Piper Wolters, Tanmay Gupta, Kuo-Hao Zeng, Jon Borchardt, Dirk Groeneveld, Jen Dumas, Crystal Nam, Sophie Lebrecht, Caitlin Wittlif, Carissa Schoenick, Oscar Michel, Ranjay Krishna, Luca Weihs, Noah A. Smith, Hannaneh Hajishirzi, Ross Girshick, Ali Farhadi, Aniruddha Kembhavi.

Molmo, developed by AllenAI team, is an open-source multimodal AI model capable of processing text and images within a unified framework. It outperforms larger models in efficiency and accuracy, leveraging high-quality datasets like PixMo for tasks such as captioning, question answering, and visual pointing.

The abstract from the paper is the following:

*Today's most advanced multimodal models remain proprietary. The strongest open-weight models rely heavily on synthetic data from proprietary VLMs to achieve good performance, effectively distilling these closed models into open ones. As a result, the community is still missing foundational knowledge about how to build performant VLMs from scratch. We present Molmo, a new family of VLMs that are state-of-the-art in their class of openness. Our key innovation is a novel, highly detailed image caption dataset collected entirely from human annotators using speech-based descriptions. To enable a wide array of user interactions, we also introduce a diverse dataset mixture for fine-tuning that includes in-the-wild Q&A and innovative 2D pointing data. The success of our approach relies on careful choices for the model architecture details, a well-tuned training pipeline, and, most critically, the quality of our newly collected datasets, all of which will be released. The best-in-class 72B model within the Molmo family not only outperforms others in the class of open weight and data models but also compares favorably against proprietary systems like GPT-4o, Claude 3.5, and Gemini 1.5 on both academic benchmarks and human evaluation.
*

Tips:

<INSERT TIPS ABOUT MODEL HERE>

This model was contributed by [Molbap](https://huggingface.co/Molbap).


## MolmoConfig

[[autodoc]] MolmoConfig

## MolmoTextConfig

[[autodoc]] MolmoTextConfig

## MolmoVisionConfig

[[autodoc]] MolmoVisionConfig

## MolmoPoolingConfig

[[autodoc]] MolmoPoolingConfig

## MolmoImageProcessor

[[autodoc]] MolmoImageProcessor

## MolmoProcessor

[[autodoc]] MolmoProcessor

## MolmoTextModel

[[autodoc]] MolmoTextModel
    - forward

## MolmoForCausalLM

[[autodoc]] MolmoForCausalLM
    - forward
    
## MolmoForConditionalGeneration

[[autodoc]] MolmoForConditionalGeneration
    - forward
