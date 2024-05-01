<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# XLM-RoBERTa

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=xlm-roberta">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-xlm--roberta-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/xlm-roberta-base">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## Overview

The XLM-RoBERTa model was proposed in [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) by Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume
Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer and Veselin Stoyanov. It is based on Facebook's
RoBERTa model released in 2019. It is a large multi-lingual language model, trained on 2.5TB of filtered CommonCrawl
data.

The abstract from the paper is the following:

*This paper shows that pretraining multilingual language models at scale leads to significant performance gains for a
wide range of cross-lingual transfer tasks. We train a Transformer-based masked language model on one hundred
languages, using more than two terabytes of filtered CommonCrawl data. Our model, dubbed XLM-R, significantly
outperforms multilingual BERT (mBERT) on a variety of cross-lingual benchmarks, including +13.8% average accuracy on
XNLI, +12.3% average F1 score on MLQA, and +2.1% average F1 score on NER. XLM-R performs particularly well on
low-resource languages, improving 11.8% in XNLI accuracy for Swahili and 9.2% for Urdu over the previous XLM model. We
also present a detailed empirical evaluation of the key factors that are required to achieve these gains, including the
trade-offs between (1) positive transfer and capacity dilution and (2) the performance of high and low resource
languages at scale. Finally, we show, for the first time, the possibility of multilingual modeling without sacrificing
per-language performance; XLM-Ris very competitive with strong monolingual models on the GLUE and XNLI benchmarks. We
will make XLM-R code, data, and models publicly available.*

This model was contributed by [stefan-it](https://huggingface.co/stefan-it). The original code can be found [here](https://github.com/pytorch/fairseq/tree/master/examples/xlmr).

## Usage tips

- XLM-RoBERTa is a multilingual model trained on 100 different languages. Unlike some XLM multilingual models, it does
  not require `lang` tensors to understand which language is used, and should be able to determine the correct
  language from the input ids.
- Uses RoBERTa tricks on the XLM approach, but does not use the translation language modeling objective. It only uses masked language modeling on sentences coming from one language.

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with XLM-RoBERTa. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

<PipelineTag pipeline="text-classification"/>

- A blog post on how to [finetune XLM RoBERTa for multiclass classification with Habana Gaudi on AWS](https://www.philschmid.de/habana-distributed-training)
- [`XLMRobertaForSequenceClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb).
- [`TFXLMRobertaForSequenceClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/text-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb).
- [`FlaxXLMRobertaForSequenceClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/flax/text-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification_flax.ipynb).
- [Text classification](https://huggingface.co/docs/transformers/tasks/sequence_classification) chapter of the 🤗 Hugging Face Task Guides.
- [Text classification task guide](../tasks/sequence_classification)

<PipelineTag pipeline="token-classification"/>

- [`XLMRobertaForTokenClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb).
- [`TFXLMRobertaForTokenClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/token-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb).
- [`FlaxXLMRobertaForTokenClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/flax/token-classification).
- [Token classification](https://huggingface.co/course/chapter7/2?fw=pt) chapter of the 🤗 Hugging Face Course.
- [Token classification task guide](../tasks/token_classification)

<PipelineTag pipeline="text-generation"/>

- [`XLMRobertaForCausalLM`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb).
- [Causal language modeling](https://huggingface.co/docs/transformers/tasks/language_modeling) chapter of the 🤗 Hugging Face Task Guides.
- [Causal language modeling task guide](../tasks/language_modeling)

<PipelineTag pipeline="fill-mask"/>

- [`XLMRobertaForMaskedLM`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#robertabertdistilbert-and-masked-language-modeling) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb).
- [`TFXLMRobertaForMaskedLM`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_mlmpy) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb).
- [`FlaxXLMRobertaForMaskedLM`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#masked-language-modeling) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/masked_language_modeling_flax.ipynb).
- [Masked language modeling](https://huggingface.co/course/chapter7/3?fw=pt) chapter of the 🤗 Hugging Face Course.
- [Masked language modeling](../tasks/masked_language_modeling)

<PipelineTag pipeline="question-answering"/>

- [`XLMRobertaForQuestionAnswering`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb).
- [`TFXLMRobertaForQuestionAnswering`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/question-answering) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb).
- [`FlaxXLMRobertaForQuestionAnswering`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/flax/question-answering).
- [Question answering](https://huggingface.co/course/chapter7/7?fw=pt) chapter of the 🤗 Hugging Face Course.
- [Question answering task guide](../tasks/question_answering)

**Multiple choice**

- [`XLMRobertaForMultipleChoice`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb).
- [`TFXLMRobertaForMultipleChoice`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/multiple-choice) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice-tf.ipynb).
- [Multiple choice task guide](../tasks/multiple_choice)

🚀 Deploy

- A blog post on how to [Deploy Serverless XLM RoBERTa on AWS Lambda](https://www.philschmid.de/multilingual-serverless-xlm-roberta-with-huggingface).

<Tip> 

This implementation is the same as RoBERTa. Refer to the [documentation of RoBERTa](roberta) for usage examples as well as the information relative to the inputs and outputs.
</Tip>

## XLMRobertaConfig

[[autodoc]] XLMRobertaConfig

## XLMRobertaTokenizer

[[autodoc]] XLMRobertaTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## XLMRobertaTokenizerFast

[[autodoc]] XLMRobertaTokenizerFast

<frameworkcontent>
<pt>

## XLMRobertaModel

[[autodoc]] XLMRobertaModel
    - forward

## XLMRobertaForCausalLM

[[autodoc]] XLMRobertaForCausalLM
    - forward

## XLMRobertaForMaskedLM

[[autodoc]] XLMRobertaForMaskedLM
    - forward

## XLMRobertaForSequenceClassification

[[autodoc]] XLMRobertaForSequenceClassification
    - forward

## XLMRobertaForMultipleChoice

[[autodoc]] XLMRobertaForMultipleChoice
    - forward

## XLMRobertaForTokenClassification

[[autodoc]] XLMRobertaForTokenClassification
    - forward

## XLMRobertaForQuestionAnswering

[[autodoc]] XLMRobertaForQuestionAnswering
    - forward

</pt>
<tf>

## TFXLMRobertaModel

[[autodoc]] TFXLMRobertaModel
    - call

## TFXLMRobertaForCausalLM

[[autodoc]] TFXLMRobertaForCausalLM
    - call

## TFXLMRobertaForMaskedLM

[[autodoc]] TFXLMRobertaForMaskedLM
    - call

## TFXLMRobertaForSequenceClassification

[[autodoc]] TFXLMRobertaForSequenceClassification
    - call

## TFXLMRobertaForMultipleChoice

[[autodoc]] TFXLMRobertaForMultipleChoice
    - call

## TFXLMRobertaForTokenClassification

[[autodoc]] TFXLMRobertaForTokenClassification
    - call

## TFXLMRobertaForQuestionAnswering

[[autodoc]] TFXLMRobertaForQuestionAnswering
    - call

</tf>
<jax>

## FlaxXLMRobertaModel

[[autodoc]] FlaxXLMRobertaModel
    - __call__

## FlaxXLMRobertaForCausalLM

[[autodoc]] FlaxXLMRobertaForCausalLM
    - __call__

## FlaxXLMRobertaForMaskedLM

[[autodoc]] FlaxXLMRobertaForMaskedLM
    - __call__

## FlaxXLMRobertaForSequenceClassification

[[autodoc]] FlaxXLMRobertaForSequenceClassification
    - __call__

## FlaxXLMRobertaForMultipleChoice

[[autodoc]] FlaxXLMRobertaForMultipleChoice
    - __call__

## FlaxXLMRobertaForTokenClassification

[[autodoc]] FlaxXLMRobertaForTokenClassification
    - __call__

## FlaxXLMRobertaForQuestionAnswering

[[autodoc]] FlaxXLMRobertaForQuestionAnswering
    - __call__

</jax>
</frameworkcontent>