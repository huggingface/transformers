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

# Falcon

## Overview

Falcon is a class of causal decoder-only models built by [TII](https://www.tii.ae/). The largest Falcon checkpoints
have been trained on >=1T tokens of text, with a particular emphasis on the [RefinedWeb](https://arxiv.org/abs/2306.01116)
corpus. They are made available under the Apache 2.0 license.


Falcon's architecture is modern and optimized for inference, with multi-query attention and support for efficient
attention variants like `FlashAttention`. Both 'base' models trained only as causal language models as well as
'instruct' models that have received further fine-tuning are available.


Falcon models are (as of 2023) some of the largest and most powerful open-source language models,
and consistently rank highly in the [OpenLLM leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).

## Converting custom checkpoints 

<Tip>

Falcon models were initially added to the Hugging Face Hub as custom code checkpoints. However, Falcon is now fully
supported in the Transformers library. If you fine-tuned a model from a custom code checkpoint, we recommend converting
your checkpoint to the new in-library format, as this should give significant improvements to stability and
performance, especially for generation, as well as removing the need to use `trust_remote_code=True`!

</Tip>

You can convert custom code checkpoints to full Transformers checkpoints using the `convert_custom_code_checkpoint.py` 
script located in the
[Falcon model directory](https://github.com/huggingface/transformers/tree/main/src/transformers/models/falcon)
of the Transformers library. To use this script, simply call it with 
`python convert_custom_code_checkpoint.py --checkpoint_dir my_model`. This will convert your checkpoint in-place, and
you can immediately load it from the directory afterwards with e.g. `from_pretrained()`. If your model hasn't been
uploaded to the Hub, we recommend making a backup before attempting the conversion, just in case!


## FalconConfig

[[autodoc]] FalconConfig
    - all

## FalconModel

[[autodoc]] FalconModel
    - forward

## FalconForCausalLM

[[autodoc]] FalconForCausalLM
    - forward

## FalconForSequenceClassification

[[autodoc]] FalconForSequenceClassification
    - forward

## FalconForTokenClassification

[[autodoc]] FalconForTokenClassification
    - forward

## FalconForQuestionAnswering

[[autodoc]] FalconForQuestionAnswering
    - forward


