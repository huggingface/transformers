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

# Idefics3

## Overview

The Idefics3 model was proposed in [<INSERT PAPER NAME HERE>](<INSERT PAPER LINK HERE>) by <INSERT AUTHORS HERE>.

Idefics3 is an adaptation of the Idefics2 model with three main differences: 
- the use of Llama3 for the text model
- an updated processing logic for the images.
- The removal of the perceiver. 

The resolutions of input images can be directly controlled, and they are decomposed into
patches, or not, depending on the resolution. See [Idefics2] for more details on the model architecture.

The abstract from the paper is the following:

*<INSERT PAPER ABSTRACT HERE>*

Tips:

<INSERT TIPS ABOUT MODEL HERE>

This model was contributed by [amyeroberts](https://huggingface.co/amyeroberts) and [andimarafioti](https://huggingface.co/andito).
The original code can be found [here](<INSERT LINK TO GITHUB REPO HERE>).


## Idefics3ImageProcessor
[[autodoc]] Idefics3ImageProcessor
    - preprocess


## Idefics3Processor
[[autodoc]] Idefics3Processor
    - __call__
