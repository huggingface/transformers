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
*This model was released on 2020-04-28 and added to Hugging Face Transformers on 2020-11-16.*

# Blenderbot

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The Blender chatbot model was proposed in [Recipes for building an open-domain chatbot](https://huggingface.co/papers/2004.13637) Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu,
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

This model was contributed by [sshleifer](https://huggingface.co/sshleifer). The authors' code can be found [here](https://github.com/facebookresearch/ParlAI) .

## Usage tips and example

Blenderbot is a model with absolute position embeddings so it's usually advised to pad the inputs on the right
rather than the left.

An example:

```python
>>> from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

>>> mname = "facebook/blenderbot-400M-distill"
>>> model = BlenderbotForConditionalGeneration.from_pretrained(mname)
>>> tokenizer = BlenderbotTokenizer.from_pretrained(mname)
>>> UTTERANCE = "My friends are cool but they eat too many carbs."
>>> inputs = tokenizer([UTTERANCE], return_tensors="pt")
>>> reply_ids = model.generate(**inputs)
>>> print(tokenizer.batch_decode(reply_ids))
["<s> That's unfortunate. Are they trying to lose weight or are they just trying to be healthier?</s>"]
```

## Implementation Notes

- Blenderbot uses a standard [seq2seq model transformer](https://huggingface.co/papers/1706.03762) based architecture.
- Available checkpoints can be found in the [model hub](https://huggingface.co/models?search=blenderbot).
- This is the *default* Blenderbot model class. However, some smaller checkpoints, such as
  `facebook/blenderbot_small_90M`, have a different architecture and consequently should be used with
  [BlenderbotSmall](blenderbot-small).


## Resources

- [Causal language modeling task guide](../tasks/language_modeling)
- [Translation task guide](../tasks/translation)
- [Summarization task guide](../tasks/summarization)

## BlenderbotConfig

[[autodoc]] BlenderbotConfig

## BlenderbotTokenizer

[[autodoc]] BlenderbotTokenizer
    - build_inputs_with_special_tokens

## BlenderbotTokenizerFast

[[autodoc]] BlenderbotTokenizerFast
    - build_inputs_with_special_tokens

## BlenderbotModel

See [`~transformers.BartModel`] for arguments to *forward* and *generate*

[[autodoc]] BlenderbotModel
    - forward

## BlenderbotForConditionalGeneration

See [`~transformers.BartForConditionalGeneration`] for arguments to *forward* and *generate*

[[autodoc]] BlenderbotForConditionalGeneration
    - forward

## BlenderbotForCausalLM

[[autodoc]] BlenderbotForCausalLM
    - forward
