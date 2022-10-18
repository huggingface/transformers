<!---
Copyright 2021 The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Examples

This folder contains actively maintained examples of use of 🤗 Transformers organized into different ML tasks. All examples in this folder are **TensorFlow** examples, and are written using native Keras rather than classes like `TFTrainer`, which we now consider deprecated. If you've previously only used 🤗 Transformers via `TFTrainer`, we highly recommend taking a look at the new style - we think it's a big improvement!

In addition, all scripts here now support the [🤗 Datasets](https://github.com/huggingface/datasets) library - you can grab entire datasets just by changing one command-line argument!

## A note on code folding

Most of these examples have been formatted with #region blocks. In IDEs such as PyCharm and VSCode, these blocks mark
named regions of code that can be folded for easier viewing. If you find any of these scripts overwhelming or difficult
to follow, we highly recommend beginning with all regions folded and then examining regions one at a time!

## The Big Table of Tasks

Here is the list of all our examples:

| Task | Example datasets |
|---|---|
| [**`language-modeling`**](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling) | WikiText-2
| [**`multiple-choice`**](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/multiple-choice) | SWAG 
| [**`question-answering`**](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/question-answering) | SQuAD
| [**`summarization`**](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/summarization) | XSum 
| [**`text-classification`**](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/text-classification) | GLUE
| [**`token-classification`**](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/token-classification) | CoNLL NER
| [**`translation`**](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/translation) | WMT

## Coming soon

- **Colab notebooks** to easily run through these scripts! 
