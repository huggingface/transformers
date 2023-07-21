<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# GeoLM

## Overview

The GeoLM model was proposed in [<INSERT PAPER NAME HERE>](<INSERT PAPER LINK HERE>) by <INSERT AUTHORS HERE>.
<INSERT SHORT SUMMARY HERE>

The abstract from the paper is the following:

*<INSERT PAPER ABSTRACT HERE>*

Tips:

<INSERT TIPS ABOUT MODEL HERE>

This model was contributed by [INSERT YOUR HF USERNAME HERE](https://huggingface.co/<INSERT YOUR HF USERNAME HERE>).
The original code can be found [here](<INSERT LINK TO GITHUB REPO HERE>).


## GeoLMConfig

[[autodoc]] GeoLMConfig
    - all

## GeoLM specific outputs

[[autodoc]] models.geolm.modeling_geolm.GeoLMForPreTrainingOutput

[[autodoc]] models.geolm.modeling_tf_geolm.TFGeoLMForPreTrainingOutput

[[autodoc]] models.geolm.modeling_flax_geolm.FlaxGeoLMForPreTrainingOutput

## GeoLMModel

[[autodoc]] GeoLMModel
    - forward

## GeoLMForPreTraining

[[autodoc]] GeoLMForPreTraining
    - forward

## GeoLMLMHeadModel

[[autodoc]] GeoLMLMHeadModel
    - forward

## GeoLMForMaskedLM

[[autodoc]] GeoLMForMaskedLM
    - forward

## GeoLMForNextSentencePrediction

[[autodoc]] GeoLMForNextSentencePrediction
    - forward

## GeoLMForSequenceClassification

[[autodoc]] GeoLMForSequenceClassification
    - forward

## GeoLMForMultipleChoice

[[autodoc]] GeoLMForMultipleChoice
    - forward

## GeoLMForTokenClassification

[[autodoc]] GeoLMForTokenClassification
    - forward

## GeoLMForQuestionAnswering

[[autodoc]] GeoLMForQuestionAnswering
    - forward

## TFGeoLMModel

[[autodoc]] TFGeoLMModel
    - call

## TFGeoLMForPreTraining

[[autodoc]] TFGeoLMForPreTraining
    - call

## TFGeoLMModelLMHeadModel

[[autodoc]] TFGeoLMLMHeadModel
    - call

## TFGeoLMForMaskedLM

[[autodoc]] TFGeoLMForMaskedLM
    - call

## TFGeoLMForNextSentencePrediction

[[autodoc]] TFGeoLMForNextSentencePrediction
    - call

## TFGeoLMForSequenceClassification

[[autodoc]] TFGeoLMForSequenceClassification
    - call

## TFGeoLMForMultipleChoice

[[autodoc]] TFGeoLMForMultipleChoice
    - call

## TFGeoLMForTokenClassification

[[autodoc]] TFGeoLMForTokenClassification
    - call

## TFGeoLMForQuestionAnswering

[[autodoc]] TFGeoLMForQuestionAnswering
    - call

## FlaxGeoLMModel

[[autodoc]] FlaxGeoLMModel
    - __call__

## FlaxGeoLMForPreTraining

[[autodoc]] FlaxGeoLMForPreTraining
    - __call__

## FlaxGeoLMForCausalLM

[[autodoc]] FlaxGeoLMForCausalLM
    - __call__

## FlaxGeoLMForMaskedLM

[[autodoc]] FlaxGeoLMForMaskedLM
    - __call__

## FlaxGeoLMForNextSentencePrediction

[[autodoc]] FlaxGeoLMForNextSentencePrediction
    - __call__

## FlaxGeoLMForSequenceClassification

[[autodoc]] FlaxGeoLMForSequenceClassification
    - __call__

## FlaxGeoLMForMultipleChoice

[[autodoc]] FlaxGeoLMForMultipleChoice
    - __call__

## FlaxGeoLMForTokenClassification

[[autodoc]] FlaxGeoLMForTokenClassification
    - __call__

## FlaxGeoLMForQuestionAnswering

[[autodoc]] FlaxGeoLMForQuestionAnswering
    - __call__
