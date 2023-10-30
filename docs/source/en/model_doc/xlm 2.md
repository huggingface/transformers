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

# XLM

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=xlm">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-xlm-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/xlm-mlm-en-2048">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## Overview

The XLM model was proposed in [Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291) by
Guillaume Lample, Alexis Conneau. It's a transformer pretrained using one of the following objectives:

- a causal language modeling (CLM) objective (next token prediction),
- a masked language modeling (MLM) objective (BERT-like), or
- a Translation Language Modeling (TLM) object (extension of BERT's MLM to multiple language inputs)

The abstract from the paper is the following:

*Recent studies have demonstrated the efficiency of generative pretraining for English natural language understanding.
In this work, we extend this approach to multiple languages and show the effectiveness of cross-lingual pretraining. We
propose two methods to learn cross-lingual language models (XLMs): one unsupervised that only relies on monolingual
data, and one supervised that leverages parallel data with a new cross-lingual language model objective. We obtain
state-of-the-art results on cross-lingual classification, unsupervised and supervised machine translation. On XNLI, our
approach pushes the state of the art by an absolute gain of 4.9% accuracy. On unsupervised machine translation, we
obtain 34.3 BLEU on WMT'16 German-English, improving the previous state of the art by more than 9 BLEU. On supervised
machine translation, we obtain a new state of the art of 38.5 BLEU on WMT'16 Romanian-English, outperforming the
previous best approach by more than 4 BLEU. Our code and pretrained models will be made publicly available.*

Tips:

- XLM has many different checkpoints, which were trained using different objectives: CLM, MLM or TLM. Make sure to
  select the correct objective for your task (e.g. MLM checkpoints are not suitable for generation).
- XLM has multilingual checkpoints which leverage a specific `lang` parameter. Check out the [multi-lingual](../multilingual) page for more information.
- A transformer model trained on several languages. There are three different type of training for this model and the library provides checkpoints for all of them:

    * Causal language modeling (CLM) which is the traditional autoregressive training (so this model could be in the previous section as well). One of the languages is selected for each training sample, and the model input is a sentence of 256 tokens, that may span over several documents in one of those languages.
    * Masked language modeling (MLM) which is like RoBERTa. One of the languages is selected for each training sample, and the model input is a sentence of 256 tokens, that may span over several documents in one of those languages, with dynamic masking of the tokens.
    * A combination of MLM and translation language modeling (TLM). This consists of concatenating a sentence in two different languages, with random masking. To predict one of the masked tokens, the model can use both, the surrounding context in language 1 and the context given by language 2.

This model was contributed by [thomwolf](https://huggingface.co/thomwolf). The original code can be found [here](https://github.com/facebookresearch/XLM/).

## Documentation resources

- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Question answering task guide](../tasks/question_answering)
- [Causal language modeling task guide](../tasks/language_modeling)
- [Masked language modeling task guide](../tasks/masked_language_modeling)
- [Multiple choice task guide](../tasks/multiple_choice)

## XLMConfig

[[autodoc]] XLMConfig

## XLMTokenizer

[[autodoc]] XLMTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## XLM specific outputs

[[autodoc]] models.xlm.modeling_xlm.XLMForQuestionAnsweringOutput

## XLMModel

[[autodoc]] XLMModel
    - forward

## XLMWithLMHeadModel

[[autodoc]] XLMWithLMHeadModel
    - forward

## XLMForSequenceClassification

[[autodoc]] XLMForSequenceClassification
    - forward

## XLMForMultipleChoice

[[autodoc]] XLMForMultipleChoice
    - forward

## XLMForTokenClassification

[[autodoc]] XLMForTokenClassification
    - forward

## XLMForQuestionAnsweringSimple

[[autodoc]] XLMForQuestionAnsweringSimple
    - forward

## XLMForQuestionAnswering

[[autodoc]] XLMForQuestionAnswering
    - forward

## TFXLMModel

[[autodoc]] TFXLMModel
    - call

## TFXLMWithLMHeadModel

[[autodoc]] TFXLMWithLMHeadModel
    - call

## TFXLMForSequenceClassification

[[autodoc]] TFXLMForSequenceClassification
    - call

## TFXLMForMultipleChoice

[[autodoc]] TFXLMForMultipleChoice
    - call

## TFXLMForTokenClassification

[[autodoc]] TFXLMForTokenClassification
    - call

## TFXLMForQuestionAnsweringSimple

[[autodoc]] TFXLMForQuestionAnsweringSimple
    - call
