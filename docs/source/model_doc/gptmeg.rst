.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

GPTMeg
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The GPTMeg model was proposed by BigScience Modeling work group. It is cloning HF's GPT2 to create GPTMeg with a few tiny changes for fp16 adaptation.

The 3 sources of divergence are:

- `layer_norm` override to be forced to be done in fp32 - as `MixedFusedLayerNorm (meg)` performs it in fp32 and then casts back to fp16.
- overrides `gelu_fast` to use meg's version which uses torch.jit.script which under fp16 returns diverging output (the bwd function was needed to support jit and it's not the issue)
- `_attn` override to use `torch.baddbmm` instead of `torch.matmul` - the divergence happens due to the alpha factor as it gets applied differently in the 2 ways.


GPTMegConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.GPTMegConfig
    :members:


GPTMegTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.GPTMegTokenizer
    :members: save_vocabulary


GPTMegTokenizerFast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.GPTMegTokenizerFast
    :members:


GPTMeg specific outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.models.gptmeg.modeling_gptmeg.GPTMegDoubleHeadsModelOutput
    :members:


GPTMegModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.GPTMegModel
    :members: forward, parallelize, deparallelize


GPTMegLMHeadModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.GPTMegLMHeadModel
    :members: forward, parallelize, deparallelize


GPTMegDoubleHeadsModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.GPTMegDoubleHeadsModel
    :members: forward


GPTMegForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.GPTMegForSequenceClassification
    :members: forward


GPTMegForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.GPTMegForTokenClassification
    :members: forward
