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

# Blenderbot Small

Note that [`BlenderbotSmallModel`] and
[`BlenderbotSmallForConditionalGeneration`] are only used in combination with the checkpoint
[facebook/blenderbot-90M](https://huggingface.co/facebook/blenderbot-90M). Larger Blenderbot checkpoints should
instead be used with [`BlenderbotModel`] and
[`BlenderbotForConditionalGeneration`]

## Overview

The Blender chatbot model was proposed in [Recipes for building an open-domain chatbot](https://arxiv.org/pdf/2004.13637.pdf) Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu,
Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston on 30 Apr 2020.

The abstract of the paper is the following:

*Building open-domain chatbots is a challenging area for machine learning research. While prior work has shown that
scaling neural models in the number of parameters and the size of the data they are trained on gives improved results,
we show that other ingredients are important for a high-performing chatbot. Good conversation requires a number of
skills that an expert conversationalist blends in a seamless way: providing engaging talking points and listening to
their partners, and displaying knowledge, empathy and personality appropriately, while maintaining a consistent
persona. We show that large scale models can learn these skills when given appropriate training data and choice of
generation strategy. We build variants of these recipes with 90M, 2.7B and 9.4B parameter models, and make our models
and code publicly available. Human evaluations show our best models are superior to existing approaches in multi-turn
dialogue in terms of engagingness and humanness measurements. We then discuss the limitations of this work by analyzing
failure cases of our models.*

This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten). The authors' code can be
found [here](https://github.com/facebookresearch/ParlAI).

## Usage tips

Blenderbot Small is a model with absolute position embeddings so it's usually advised to pad the inputs on the right rather than 
the left.


## Resources

- [Causal language modeling task guide](../tasks/language_modeling)
- [Translation task guide](../tasks/translation)
- [Summarization task guide](../tasks/summarization)

## BlenderbotSmallConfig

[[autodoc]] BlenderbotSmallConfig

## BlenderbotSmallTokenizer

[[autodoc]] BlenderbotSmallTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## BlenderbotSmallTokenizerFast

[[autodoc]] BlenderbotSmallTokenizerFast

<frameworkcontent>
<pt>

## BlenderbotSmallModel

[[autodoc]] BlenderbotSmallModel
    - forward

## BlenderbotSmallForConditionalGeneration

[[autodoc]] BlenderbotSmallForConditionalGeneration
    - forward

## BlenderbotSmallForCausalLM

[[autodoc]] BlenderbotSmallForCausalLM
    - forward

</pt>
<tf>

## TFBlenderbotSmallModel

[[autodoc]] TFBlenderbotSmallModel
    - call

## TFBlenderbotSmallForConditionalGeneration

[[autodoc]] TFBlenderbotSmallForConditionalGeneration
    - call

</tf>
<jax>

## FlaxBlenderbotSmallModel

[[autodoc]] FlaxBlenderbotSmallModel
    - __call__
    - encode
    - decode

## FlaxBlenderbotForConditionalGeneration

[[autodoc]] FlaxBlenderbotSmallForConditionalGeneration
    - __call__
    - encode
    - decode

</jax>
</frameworkcontent>
