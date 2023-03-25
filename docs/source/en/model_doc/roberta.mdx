<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# RoBERTa

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=roberta">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-roberta-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/roberta-base">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## Overview

The RoBERTa model was proposed in [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer
Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov. It is based on Google's BERT model released in 2018.

It builds on BERT and modifies key hyperparameters, removing the next-sentence pretraining objective and training with
much larger mini-batches and learning rates.

The abstract from the paper is the following:

*Language model pretraining has led to significant performance gains but careful comparison between different
approaches is challenging. Training is computationally expensive, often done on private datasets of different sizes,
and, as we will show, hyperparameter choices have significant impact on the final results. We present a replication
study of BERT pretraining (Devlin et al., 2019) that carefully measures the impact of many key hyperparameters and
training data size. We find that BERT was significantly undertrained, and can match or exceed the performance of every
model published after it. Our best model achieves state-of-the-art results on GLUE, RACE and SQuAD. These results
highlight the importance of previously overlooked design choices, and raise questions about the source of recently
reported improvements. We release our models and code.*

Tips:

- This implementation is the same as [`BertModel`] with a tiny embeddings tweak as well as a setup
  for Roberta pretrained models.
- RoBERTa has the same architecture as BERT, but uses a byte-level BPE as a tokenizer (same as GPT-2) and uses a
  different pretraining scheme.
- RoBERTa doesn't have `token_type_ids`, you don't need to indicate which token belongs to which segment. Just
  separate your segments with the separation token `tokenizer.sep_token` (or `</s>`)
- Same as BERT with better pretraining tricks:

    * dynamic masking: tokens are masked differently at each epoch, whereas BERT does it once and for all
    * together to reach 512 tokens (so the sentences are in an order than may span several documents)
    * train with larger batches
    * use BPE with bytes as a subunit and not characters (because of unicode characters)
- [CamemBERT](camembert) is a wrapper around RoBERTa. Refer to this page for usage examples.

This model was contributed by [julien-c](https://huggingface.co/julien-c). The original code can be found [here](https://github.com/pytorch/fairseq/tree/master/examples/roberta).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with RoBERTa. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

<PipelineTag pipeline="text-classification"/>

- A blog on [Getting Started with Sentiment Analysis on Twitter](https://huggingface.co/blog/sentiment-analysis-twitter) using RoBERTa and the [Inference API](https://huggingface.co/inference-api).
- A blog on [Opinion Classification with Kili and Hugging Face AutoTrain](https://huggingface.co/blog/opinion-classification-with-kili) using RoBERTa.
- A notebook on how to [finetune RoBERTa for sentiment analysis](https://colab.research.google.com/github/DhavalTaunk08/NLP_scripts/blob/master/sentiment_analysis_using_roberta.ipynb). 🌎
- [`RobertaForSequenceClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb).
- [`TFRobertaForSequenceClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/text-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb).
- [`FlaxRobertaForSequenceClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/flax/text-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification_flax.ipynb).
- [Text classification task guide](../tasks/sequence_classification)

<PipelineTag pipeline="token-classification"/>

- [`RobertaForTokenClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb).
- [`TFRobertaForTokenClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/token-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb).
- [`FlaxRobertaForTokenClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/flax/token-classification).
- [Token classification](https://huggingface.co/course/chapter7/2?fw=pt) chapter of the 🤗 Hugging Face Course.
- [Token classification task guide](../tasks/token_classification)

<PipelineTag pipeline="fill-mask"/>

- A blog on [How to train a new language model from scratch using Transformers and Tokenizers](https://huggingface.co/blog/how-to-train) with RoBERTa.
- [`RobertaForMaskedLM`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#robertabertdistilbert-and-masked-language-modeling) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb).
- [`TFRobertaForMaskedLM`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_mlmpy) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb).
- [`FlaxRobertaForMaskedLM`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#masked-language-modeling) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/masked_language_modeling_flax.ipynb).
- [Masked language modeling](https://huggingface.co/course/chapter7/3?fw=pt) chapter of the 🤗 Hugging Face Course.
- [Masked language modeling task guide](../tasks/masked_language_modeling)

<PipelineTag pipeline="question-answering"/>

- A blog on [Accelerated Inference with Optimum and Transformers Pipelines](https://huggingface.co/blog/optimum-inference) with RoBERTa for question answering.
- [`RobertaForQuestionAnswering`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb).
- [`TFRobertaForQuestionAnswering`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/question-answering) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb).
- [`FlaxRobertaForQuestionAnswering`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/flax/question-answering).
- [Question answering](https://huggingface.co/course/chapter7/7?fw=pt) chapter of the 🤗 Hugging Face Course.
- [Question answering task guide](../tasks/question_answering)

**Multiple choice**
- [`RobertaForMultipleChoice`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb).
- [`TFRobertaForMultipleChoice`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/multiple-choice) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice-tf.ipynb).
- [Multiple choice task guide](../tasks/multiple_choice)

## RobertaConfig

[[autodoc]] RobertaConfig

## RobertaTokenizer

[[autodoc]] RobertaTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## RobertaTokenizerFast

[[autodoc]] RobertaTokenizerFast
    - build_inputs_with_special_tokens

## RobertaModel

[[autodoc]] RobertaModel
    - forward

## RobertaForCausalLM

[[autodoc]] RobertaForCausalLM
    - forward

## RobertaForMaskedLM

[[autodoc]] RobertaForMaskedLM
    - forward

## RobertaForSequenceClassification

[[autodoc]] RobertaForSequenceClassification
    - forward

## RobertaForMultipleChoice

[[autodoc]] RobertaForMultipleChoice
    - forward

## RobertaForTokenClassification

[[autodoc]] RobertaForTokenClassification
    - forward

## RobertaForQuestionAnswering

[[autodoc]] RobertaForQuestionAnswering
    - forward

## TFRobertaModel

[[autodoc]] TFRobertaModel
    - call

## TFRobertaForCausalLM

[[autodoc]] TFRobertaForCausalLM
    - call

## TFRobertaForMaskedLM

[[autodoc]] TFRobertaForMaskedLM
    - call

## TFRobertaForSequenceClassification

[[autodoc]] TFRobertaForSequenceClassification
    - call

## TFRobertaForMultipleChoice

[[autodoc]] TFRobertaForMultipleChoice
    - call

## TFRobertaForTokenClassification

[[autodoc]] TFRobertaForTokenClassification
    - call

## TFRobertaForQuestionAnswering

[[autodoc]] TFRobertaForQuestionAnswering
    - call

## FlaxRobertaModel

[[autodoc]] FlaxRobertaModel
    - __call__

## FlaxRobertaForCausalLM

[[autodoc]] FlaxRobertaForCausalLM
    - __call__

## FlaxRobertaForMaskedLM

[[autodoc]] FlaxRobertaForMaskedLM
    - __call__

## FlaxRobertaForSequenceClassification

[[autodoc]] FlaxRobertaForSequenceClassification
    - __call__

## FlaxRobertaForMultipleChoice

[[autodoc]] FlaxRobertaForMultipleChoice
    - __call__

## FlaxRobertaForTokenClassification

[[autodoc]] FlaxRobertaForTokenClassification
    - __call__

## FlaxRobertaForQuestionAnswering

[[autodoc]] FlaxRobertaForQuestionAnswering
    - __call__
