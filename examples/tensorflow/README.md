<!---
Copyright 2020 The HuggingFace Team. All rights reserved.
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

This folder contains actively maintained examples of use of 🤗 Transformers using the TensorFlow backend, organized along NLP tasks. It is under construction so we thank you for your patience!

Files containing `tf` such as `run_tf_glue.py` are the old-style files that will be rewritten very soon! Files without this such as `run_text_classification.py` are the newer ones. This message will be removed when the revamp is complete.

## The Big Table of Tasks

Here is the list of all our examples:
- with information on whether they are **built on top of `Keras`** (if not, they still work, they might
  just lack some features),
- whether or not they leverage the [🤗 Datasets](https://github.com/huggingface/datasets) library.
- links to **Colab notebooks** to walk through the scripts and run them easily,
<!--
Coming soon!
- links to **Cloud deployments** to be able to deploy large-scale trainings in the Cloud with little to no setup.
-->

| Task | Example datasets | Keras support | 🤗 Datasets | Colab
|---|---|:---:|:---:|:---:|
| **`language-modeling`** | WikiText-2 | - | - | -
| [**`multiple-choice`**](https://github.com/huggingface/transformers/tree/master/examples/tensorflow/multiple-choice) | SWAG | - | - | -
| [**`question-answering`**](https://github.com/huggingface/transformers/tree/master/examples/tensorflow/question-answering) | SQuAD | - | - | -
| **`summarization`** | XSum | - | -  | -
| [**`text-classification`**](https://github.com/huggingface/transformers/tree/master/examples/tensorflow/text-classification) | GLUE | - | - | -
| **`text-generation`** | n/a | - | n/a | -
| **`token-classification`** | CoNLL NER | - | - | - 
| **`translation`** | WMT | -  | - | -
