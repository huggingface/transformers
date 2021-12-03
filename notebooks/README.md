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

# 🤗 Transformers Notebooks

You can find here a list of the official notebooks provided by Hugging Face.

Also, we would like to list here interesting content created by the community. 
If you wrote some notebook(s) leveraging 🤗 Transformers and would like be listed here, please open a 
Pull Request so it can be included under the Community notebooks. 


## Hugging Face's notebooks 🤗

### Documentation notebooks

You can open any page of the documentation as a notebook in colab (there is a button directly on said pages) but they are also listed here if you need to:

| Notebook     |      Description      |   |
|:----------|:-------------|------:|
| [Quicktour of the library](https://github.com/huggingface/notebooks/blob/master/transformers_doc/quicktour.ipynb)  | A presentation of the various APIs in Transformers | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/transformers_doc/quicktour.ipynb) |
| [Summary of the tasks](https://github.com/huggingface/notebooks/blob/master/transformers_doc/task_summary.ipynb)  | How to run the models of the Transformers library task by task | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/transformers_doc/task_summary.ipynb) |
| [Preprocessing data](https://github.com/huggingface/notebooks/blob/master/transformers_doc/preprocessing.ipynb)  | How to use a tokenizer to preprocess your data | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/transformers_doc/preprocessing.ipynb) |
| [Fine-tuning a pretrained model](https://github.com/huggingface/notebooks/blob/master/transformers_doc/training.ipynb)  | How to use the Trainer to fine-tune a pretrained model | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/transformers_doc/training.ipynb) |
| [Summary of the tokenizers](https://github.com/huggingface/notebooks/blob/master/transformers_doc/tokenizer_summary.ipynb)  | The differences between the tokenizers algorithm | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/transformers_doc/tokenizer_summary.ipynb) |
| [Multilingual models](https://github.com/huggingface/notebooks/blob/master/transformers_doc/multilingual.ipynb)  | How to use the multilingual models of the library | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/transformers_doc/multilingual.ipynb) |
| [Fine-tuning with custom datasets](https://github.com/huggingface/notebooks/blob/master/transformers_doc/custom_datasets.ipynb)  | How to fine-tune a pretrained model on various tasks | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/transformers_doc/custom_datasets.ipynb) |

### PyTorch Examples

| Notebook     |      Description      |   |
|:----------|:-------------|------:|
| [Train your tokenizer](https://github.com/huggingface/notebooks/blob/master/examples/tokenizer_training.ipynb)  | How to train and use your very own tokenizer  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/tokenizer_training.ipynb) |
| [Train your language model](https://github.com/huggingface/notebooks/blob/master/examples/language_modeling_from_scratch.ipynb)   | How to easily start using transformers  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/language_modeling_from_scratch.ipynb) |
| [How to fine-tune a model on text classification](https://github.com/huggingface/notebooks/blob/master/examples/text_classification.ipynb) | Show how to preprocess the data and fine-tune a pretrained model on any GLUE task. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification.ipynb)|
| [How to fine-tune a model on language modeling](https://github.com/huggingface/notebooks/blob/master/examples/language_modeling.ipynb) | Show how to preprocess the data and fine-tune a pretrained model on a causal or masked LM task. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/language_modeling.ipynb)|
| [How to fine-tune a model on token classification](https://github.com/huggingface/notebooks/blob/master/examples/token_classification.ipynb) | Show how to preprocess the data and fine-tune a pretrained model on a token classification task (NER, PoS). | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/token_classification.ipynb)|
| [How to fine-tune a model on question answering](https://github.com/huggingface/notebooks/blob/master/examples/question_answering.ipynb) | Show how to preprocess the data and fine-tune a pretrained model on SQUAD. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/question_answering.ipynb)|
| [How to fine-tune a model on multiple choice](https://github.com/huggingface/notebooks/blob/master/examples/multiple_choice.ipynb) | Show how to preprocess the data and fine-tune a pretrained model on SWAG. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/multiple_choice.ipynb)|
| [How to fine-tune a model on translation](https://github.com/huggingface/notebooks/blob/master/examples/translation.ipynb) | Show how to preprocess the data and fine-tune a pretrained model on WMT. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/translation.ipynb)|
| [How to fine-tune a model on summarization](https://github.com/huggingface/notebooks/blob/master/examples/summarization.ipynb) | Show how to preprocess the data and fine-tune a pretrained model on XSUM. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/summarization.ipynb)|
| [How to fine-tune a speech recognition model in English](https://github.com/huggingface/notebooks/blob/master/examples/speech_recognition.ipynb)| Show how to preprocess the data and fine-tune a pretrained Speech model on TIMIT | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/speech_recognition.ipynb)|
| [How to fine-tune a speech recognition model in any language](https://github.com/huggingface/notebooks/blob/master/examples/multi_lingual_speech_recognition.ipynb)| Show how to preprocess the data and fine-tune a multi-lingually pretrained speech model on Common Voice | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/multi_lingual_speech_recognition.ipynb)|
| [How to fine-tune a model on audio classification](https://github.com/huggingface/notebooks/blob/master/examples/audio_classification.ipynb)| Show how to preprocess the data and fine-tune a pretrained Speech model on Keyword Spotting | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/audio_classification.ipynb)|
| [How to train a language model from scratch](https://github.com/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb)| Highlight all the steps to effectively train Transformer model on custom data | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb)|
| [How to generate text](https://github.com/huggingface/blog/blob/master/notebooks/02_how_to_generate.ipynb)| How to use different decoding methods for language generation with transformers | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/02_how_to_generate.ipynb)|
| [How to export model to ONNX](https://github.com/huggingface/notebooks/blob/master/examples/onnx-export.ipynb) | Highlight how to export and run inference workloads through ONNX |
| [How to use Benchmarks](https://github.com/huggingface/transformers/notebooks/blob/master/examples/benchmark.ipynb) | How to benchmark models with transformers | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/benchmark.ipynb)|
| [Reformer](https://github.com/huggingface/blog/blob/master/notebooks/03_reformer.ipynb) | How Reformer pushes the limits of language modeling | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/patrickvonplaten/blog/blob/master/notebooks/03_reformer.ipynb)|

### TensorFlow Examples

| Notebook     |      Description      |   |
|:----------|:-------------|------:|
| [Train your tokenizer](https://github.com/huggingface/notebooks/blob/master/examples/tokenizer_training.ipynb)  | How to train and use your very own tokenizer  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/tokenizer_training.ipynb) |
| [Train your language model](https://github.com/huggingface/notebooks/blob/master/examples/language_modeling_from_scratch-tf.ipynb)   | How to easily start using transformers  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/language_modeling_from_scratch-tf.ipynb) |
| [How to fine-tune a model on text classification](https://github.com/huggingface/notebooks/blob/master/examples/text_classification-tf.ipynb) | Show how to preprocess the data and fine-tune a pretrained model on any GLUE task. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification-tf.ipynb)|
| [How to fine-tune a model on language modeling](https://github.com/huggingface/notebooks/blob/master/examples/language_modeling-tf.ipynb) | Show how to preprocess the data and fine-tune a pretrained model on a causal or masked LM task. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/language_modeling-tf.ipynb)|
| [How to fine-tune a model on token classification](https://github.com/huggingface/notebooks/blob/master/examples/token_classification-tf.ipynb) | Show how to preprocess the data and fine-tune a pretrained model on a token classification task (NER, PoS). | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/token_classification-tf.ipynb)|
| [How to fine-tune a model on question answering](https://github.com/huggingface/notebooks/blob/master/examples/question_answering-tf.ipynb) | Show how to preprocess the data and fine-tune a pretrained model on SQUAD. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/question_answering-tf.ipynb)|
| [How to fine-tune a model on multiple choice](https://github.com/huggingface/notebooks/blob/master/examples/multiple_choice-tf.ipynb) | Show how to preprocess the data and fine-tune a pretrained model on SWAG. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/multiple_choice-tf.ipynb)|
| [How to fine-tune a model on translation](https://github.com/huggingface/notebooks/blob/master/examples/translation-tf.ipynb) | Show how to preprocess the data and fine-tune a pretrained model on WMT. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/translation-tf.ipynb)|
| [How to fine-tune a model on summarization](https://github.com/huggingface/notebooks/blob/master/examples/summarization-tf.ipynb) | Show how to preprocess the data and fine-tune a pretrained model on XSUM. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/summarization-tf.ipynb)|

### Optimum notebooks

🤗  [Optimum](https://github.com/huggingface/optimum) is an extension of 🤗 Transformers, providing a set of performance optimization tools enabling maximum efficiency to train and run models on targeted hardwares.

| Notebook     |      Description      |   |
|:----------|:-------------|------:|
| [How to quantize a model for text classification](https://github.com/huggingface/notebooks/blob/master/examples/text_classification_quantization_inc.ipynb) | Show how to apply [Intel Neural Compressor (INC)](https://github.com/intel/neural-compressor) quantization on a model for any GLUE task. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification_quantization_inc.ipynb)|

## Community notebooks:

More notebooks developed by the community are available [here](community#community-notebooks).
