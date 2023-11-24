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

# ELECTRA

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=electra">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-electra-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/electra_large_discriminator_squad2_512">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## Overview

The ELECTRA model was proposed in the paper [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than
Generators](https://openreview.net/pdf?id=r1xMH1BtvB). ELECTRA is a new pretraining approach which trains two
transformer models: the generator and the discriminator. The generator's role is to replace tokens in a sequence, and
is therefore trained as a masked language model. The discriminator, which is the model we're interested in, tries to
identify which tokens were replaced by the generator in the sequence.

The abstract from the paper is the following:

*Masked language modeling (MLM) pretraining methods such as BERT corrupt the input by replacing some tokens with [MASK]
and then train a model to reconstruct the original tokens. While they produce good results when transferred to
downstream NLP tasks, they generally require large amounts of compute to be effective. As an alternative, we propose a
more sample-efficient pretraining task called replaced token detection. Instead of masking the input, our approach
corrupts it by replacing some tokens with plausible alternatives sampled from a small generator network. Then, instead
of training a model that predicts the original identities of the corrupted tokens, we train a discriminative model that
predicts whether each token in the corrupted input was replaced by a generator sample or not. Thorough experiments
demonstrate this new pretraining task is more efficient than MLM because the task is defined over all input tokens
rather than just the small subset that was masked out. As a result, the contextual representations learned by our
approach substantially outperform the ones learned by BERT given the same model size, data, and compute. The gains are
particularly strong for small models; for example, we train a model on one GPU for 4 days that outperforms GPT (trained
using 30x more compute) on the GLUE natural language understanding benchmark. Our approach also works well at scale,
where it performs comparably to RoBERTa and XLNet while using less than 1/4 of their compute and outperforms them when
using the same amount of compute.*

This model was contributed by [lysandre](https://huggingface.co/lysandre). The original code can be found [here](https://github.com/google-research/electra).

## Usage tips

- ELECTRA is the pretraining approach, therefore there is nearly no changes done to the underlying model: BERT. The
  only change is the separation of the embedding size and the hidden size: the embedding size is generally smaller,
  while the hidden size is larger. An additional projection layer (linear) is used to project the embeddings from their
  embedding size to the hidden size. In the case where the embedding size is the same as the hidden size, no projection
  layer is used.
- ELECTRA is a transformer model pretrained with the use of another (small) masked language model. The inputs are corrupted by that language model, which takes an input text that is randomly masked and outputs a text in which ELECTRA has to predict which token is an original and which one has been replaced. Like for GAN training, the small language model is trained for a few steps (but with the original texts as objective, not to fool the ELECTRA model like in a traditional GAN setting) then the ELECTRA model is trained for a few steps.
- The ELECTRA checkpoints saved using [Google Research's implementation](https://github.com/google-research/electra)
  contain both the generator and discriminator. The conversion script requires the user to name which model to export
  into the correct architecture. Once converted to the HuggingFace format, these checkpoints may be loaded into all
  available ELECTRA models, however. This means that the discriminator may be loaded in the
  [`ElectraForMaskedLM`] model, and the generator may be loaded in the
  [`ElectraForPreTraining`] model (the classification head will be randomly initialized as it
  doesn't exist in the generator).

## Resources

- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Question answering task guide](../tasks/question_answering)
- [Causal language modeling task guide](../tasks/language_modeling)
- [Masked language modeling task guide](../tasks/masked_language_modeling)
- [Multiple choice task guide](../tasks/multiple_choice)

## ElectraConfig

[[autodoc]] ElectraConfig

## ElectraTokenizer

[[autodoc]] ElectraTokenizer

## ElectraTokenizerFast

[[autodoc]] ElectraTokenizerFast

## Electra specific outputs

[[autodoc]] models.electra.modeling_electra.ElectraForPreTrainingOutput

[[autodoc]] models.electra.modeling_tf_electra.TFElectraForPreTrainingOutput

<frameworkcontent>
<pt>

## ElectraModel

[[autodoc]] ElectraModel
    - forward

## ElectraForPreTraining

[[autodoc]] ElectraForPreTraining
    - forward

## ElectraForCausalLM

[[autodoc]] ElectraForCausalLM
    - forward

## ElectraForMaskedLM

[[autodoc]] ElectraForMaskedLM
    - forward

## ElectraForSequenceClassification

[[autodoc]] ElectraForSequenceClassification
    - forward

## ElectraForMultipleChoice

[[autodoc]] ElectraForMultipleChoice
    - forward

## ElectraForTokenClassification

[[autodoc]] ElectraForTokenClassification
    - forward

## ElectraForQuestionAnswering

[[autodoc]] ElectraForQuestionAnswering
    - forward

</pt>
<tf>

## TFElectraModel

[[autodoc]] TFElectraModel
    - call

## TFElectraForPreTraining

[[autodoc]] TFElectraForPreTraining
    - call

## TFElectraForMaskedLM

[[autodoc]] TFElectraForMaskedLM
    - call

## TFElectraForSequenceClassification

[[autodoc]] TFElectraForSequenceClassification
    - call

## TFElectraForMultipleChoice

[[autodoc]] TFElectraForMultipleChoice
    - call

## TFElectraForTokenClassification

[[autodoc]] TFElectraForTokenClassification
    - call

## TFElectraForQuestionAnswering

[[autodoc]] TFElectraForQuestionAnswering
    - call

</tf>
<jax>

## FlaxElectraModel

[[autodoc]] FlaxElectraModel
    - __call__

## FlaxElectraForPreTraining

[[autodoc]] FlaxElectraForPreTraining
    - __call__

## FlaxElectraForCausalLM

[[autodoc]] FlaxElectraForCausalLM
    - __call__

## FlaxElectraForMaskedLM

[[autodoc]] FlaxElectraForMaskedLM
    - __call__

## FlaxElectraForSequenceClassification

[[autodoc]] FlaxElectraForSequenceClassification
    - __call__

## FlaxElectraForMultipleChoice

[[autodoc]] FlaxElectraForMultipleChoice
    - __call__

## FlaxElectraForTokenClassification

[[autodoc]] FlaxElectraForTokenClassification
    - __call__

## FlaxElectraForQuestionAnswering

[[autodoc]] FlaxElectraForQuestionAnswering
    - __call__

</jax>
</frameworkcontent>
