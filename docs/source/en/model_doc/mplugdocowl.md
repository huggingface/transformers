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

# mPLUG-DocOwl1.5

## Overview

The mPLUG-DocOwl1.5 model was proposed in [mPLUG-DocOwl 1.5: Unified Structure Learning for OCR-free Document Understanding](https://arxiv.org/pdf/2403.12895) by Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan
Liang Zhang, Bo Zhang, Chen Li, Ji Zhang, Qin Jin, Fei Huang, Jingren Zhou.

MPLUG-DocOwl1.5 is a multimodal model designed for text-rich images. It features the H-Reducer vision-to-text module, which preserves spatial relationships and efficiently processes high-resolution document images by merging visual features horizontally.

The model employs Unified Structure Learning with structure-aware parsing tasks and multi-grained text localization tasks, teaching it to parse text using line feeds, spaces, and extended Markdown syntax, which enhances the model's ability to correlate text with specific positions in the image.

DocOwl 1.5 undergoes a two-stage training process: Unified Structure Learning followed by Multi-task Tuning among Downstream Tasks. The high-quality DocReason25K dataset boosts reasoning abilities, allowing DocOwl 1.5-Chat to balance concise answers and detailed explanations.

The abstract from the paper is the following:

*Structure information is critical for understanding the semantics of text-rich images, such as documents, tables, and charts. Existing Multimodal Large Language Models (MLLMs) for Visual Document Understanding are equipped with text recognition ability but lack general structure understanding abilities for text-rich document images. In this work, we emphasize the importance of structure information in Visual Document Understanding and propose the Unified Structure Learning to boost the performance of MLLMs. Our Unified Structure Learning comprises structure-aware parsing tasks and multi-grained text localization tasks across 5 domains: document, webpage, table, chart, and natural image. To better encode structure information, we design a simple and effective vision-to-text module H-Reducer, which can not only maintain the layout information but also reduce the length of visual features by merging horizontal adjacent patches through convolution, enabling the LLM to understand high-resolution images more efficiently. Furthermore, by constructing structure-aware text sequences and multi-grained pairs of texts and bounding boxes for publicly available text-rich images, we build a comprehensive training set DocStruct4M to support structure learning. Finally, we construct a small but high-quality reasoning tuning dataset DocReason25K to trigger the detailed explanation ability in the document domain. Our model DocOwl 1.5 achieves state-of-the-art performance on 10 visual document understanding benchmarks, improving the SOTA performance of MLLMs with a 7B LLM by more than 10 points in 5/10 benchmarks.*

Tips:

DocOowl-Chat: For more accurate and stable generation, set do_sample=False. Performs better on most of the samples compared to the DocOwl-Omni checkpoint. 
DocOwl-Omni: For optimal performance, use do_sample=True and top_p=0.7 as recommended in the original code.

This model was contributed by [danaaubakirova](https://huggingface.co/danaaubakirova).
The original code can be found [here](https://github.com/X-PLUG/mPLUG-DocOwl/tree/main/DocOwl1.5).


## MPLUGDocOwlConfig

[[autodoc]] MPLUGDocOwlConfig

## MPLUGDocOwlImageProcessor
[[autodoc]] MPLUGDocOwlImageProcessor

## MPLUGDocOwlProcessor
[[autodoc]] MPLUGDocOwlProcessor

## MPLUGDocOwlHReducer
[[autodoc]] MPLUGDocOwlHReducer

## MPLUGDocOwlForCausalLM
[[autodoc]] MPLUGDocOwlForCausalLM
    - forward

## MPLUGDocOwlLanguageModel
[[autodoc]] MPLUGDocOwlLanguageModel

## MPLUGDocOwlPreTrainedLanguageModel
[[autodoc]] MPLUGDocOwlPreTrainedLanguageModel

## MPLUGDocOwlVisionModel
[[autodoc]] MPLUGDocOwlVisionModel

## MPLUGDocOwlVisionTransformer
[[autodoc]] MPLUGDocOwlVisionTransformer

## MPLUGDocOwlForConditionalGeneration

[[autodoc]] MPLUGDocOwlForConditionalGeneration
    - forward