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

# JinaBERT

## Overview

The JinaBERT model was proposed in [<INSERT PAPER NAME HERE>](<INSERT PAPER LINK HERE>) by <INSERT AUTHORS HERE>.
<INSERT SHORT SUMMARY HERE>

The abstract from the paper is the following:

*<INSERT PAPER ABSTRACT HERE>*

Tips:

<INSERT TIPS ABOUT MODEL HERE>

This model was contributed by [INSERT YOUR HF USERNAME HERE](https://huggingface.co/<INSERT YOUR HF USERNAME HERE>).
The original code can be found [here](<INSERT LINK TO GITHUB REPO HERE>).


## JinaBertConfig

[[autodoc]] JinaBertConfig
    - all

## JinaBert specific outputs

[[autodoc]] models.jina_bert.modeling_jina_bert.JinaBertForPreTrainingOutput

[[autodoc]] models.jina_bert.modeling_tf_jina_bert.TFJinaBertForPreTrainingOutput

[[autodoc]] models.jina_bert.modeling_flax_jina_bert.FlaxJinaBertForPreTrainingOutput


<frameworkcontent>
<pt>

## JinaBertModel

[[autodoc]] JinaBertModel
    - forward

## JinaBertForPreTraining

[[autodoc]] JinaBertForPreTraining
    - forward

## JinaBertLMHeadModel

[[autodoc]] JinaBertLMHeadModel
    - forward

## JinaBertForMaskedLM

[[autodoc]] JinaBertForMaskedLM
    - forward

## JinaBertForNextSentencePrediction

[[autodoc]] JinaBertForNextSentencePrediction
    - forward

## JinaBertForSequenceClassification

[[autodoc]] JinaBertForSequenceClassification
    - forward

## JinaBertForMultipleChoice

[[autodoc]] JinaBertForMultipleChoice
    - forward

## JinaBertForTokenClassification

[[autodoc]] JinaBertForTokenClassification
    - forward

## JinaBertForQuestionAnswering

[[autodoc]] JinaBertForQuestionAnswering
    - forward

</pt>
<tf>
