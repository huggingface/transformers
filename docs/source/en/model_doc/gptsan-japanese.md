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

# GPTSAN-japanese

## Overview

The GPTSAN-japanese model was released in the repository by Toshiyuki Sakamoto (tanreinama).

GPTSAN is a Japanese language model using Switch Transformer. It has the same structure as the model introduced as Prefix LM
in the T5 paper, and support both Text Generation and Masked Language Modeling tasks. These basic tasks similarly can
fine-tune for translation or summarization.

### Usage example

The `generate()` method can be used to generate text using GPTSAN-Japanese model.

```python
>>> from transformers import AutoModel, AutoTokenizer
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
>>> model = AutoModel.from_pretrained("Tanrei/GPTSAN-japanese").cuda()
>>> x_tok = tokenizer("は、", prefix_text="織田信長", return_tensors="pt")
>>> torch.manual_seed(0)
>>> gen_tok = model.generate(x_tok.input_ids.cuda(), token_type_ids=x_tok.token_type_ids.cuda(), max_new_tokens=20)
>>> tokenizer.decode(gen_tok[0])
'織田信長は、2004年に『戦国BASARA』のために、豊臣秀吉'
```

## GPTSAN Features

GPTSAN has some unique features. It has a model structure of Prefix-LM. It works as a shifted Masked Language Model for Prefix Input tokens. Un-prefixed inputs behave like normal generative models.
The Spout vector is a GPTSAN specific input. Spout is pre-trained with random inputs, but you can specify a class of text or an arbitrary vector during fine-tuning. This allows you to indicate the tendency of the generated text.
GPTSAN has a sparse Feed Forward based on Switch-Transformer. You can also add other layers and train them partially. See the original GPTSAN repository for details.

### Prefix-LM Model

GPTSAN has the structure of the model named Prefix-LM in the `T5` paper. (The original GPTSAN repository calls it `hybrid`)
In GPTSAN, the `Prefix` part of Prefix-LM, that is, the input position that can be referenced by both tokens, can be specified with any length.
Arbitrary lengths can also be specified differently for each batch.
This length applies to the text entered in `prefix_text` for the tokenizer.
The tokenizer returns the mask of the `Prefix` part of Prefix-LM as `token_type_ids`.
The model treats the part where `token_type_ids` is 1 as a `Prefix` part, that is, the input can refer to both tokens before and after.

## Usage tips

Specifying the Prefix part is done with a mask passed to self-attention.
When token_type_ids=None or all zero, it is equivalent to regular causal mask

for example:

>>> x_token = tokenizer("ｱｲｳｴ")
input_ids:      | SOT | SEG | ｱ | ｲ | ｳ | ｴ |
token_type_ids: | 1   | 0   | 0 | 0 | 0 | 0 |
prefix_lm_mask:
SOT | 1 0 0 0 0 0 |
SEG | 1 1 0 0 0 0 |
ｱ   | 1 1 1 0 0 0 |
ｲ   | 1 1 1 1 0 0 |
ｳ   | 1 1 1 1 1 0 |
ｴ   | 1 1 1 1 1 1 |

>>> x_token = tokenizer("", prefix_text="ｱｲｳｴ")
input_ids:      | SOT | ｱ | ｲ | ｳ | ｴ | SEG |
token_type_ids: | 1   | 1 | 1 | 1 | 1 | 0  |
prefix_lm_mask:
SOT | 1 1 1 1 1 0 |
ｱ   | 1 1 1 1 1 0 |
ｲ   | 1 1 1 1 1 0 |
ｳ   | 1 1 1 1 1 0 |
ｴ   | 1 1 1 1 1 0 |
SEG | 1 1 1 1 1 1 |

>>> x_token = tokenizer("ｳｴ", prefix_text="ｱｲ")
input_ids:      | SOT | ｱ | ｲ | SEG | ｳ | ｴ |
token_type_ids: | 1   | 1 | 1 | 0   | 0 | 0 |
prefix_lm_mask:
SOT | 1 1 1 0 0 0 |
ｱ   | 1 1 1 0 0 0 |
ｲ   | 1 1 1 0 0 0 |
SEG | 1 1 1 1 0 0 |
ｳ   | 1 1 1 1 1 0 |
ｴ   | 1 1 1 1 1 1 |

### Spout Vector

A Spout Vector is a special vector for controlling text generation.
This vector is treated as the first embedding in self-attention to bring extraneous attention to the generated tokens.
In the pre-trained model published from `Tanrei/GPTSAN-japanese`, the Spout Vector is a 128-dimensional vector that passes through 8 fully connected layers in the model and is projected into the space acting as external attention.
The Spout Vector projected by the fully connected layer is split to be passed to all self-attentions.

## GPTSanJapaneseConfig

[[autodoc]] GPTSanJapaneseConfig

## GPTSanJapaneseTokenizer

[[autodoc]] GPTSanJapaneseTokenizer

## GPTSanJapaneseModel

[[autodoc]] GPTSanJapaneseModel

## GPTSanJapaneseForConditionalGeneration

[[autodoc]] GPTSanJapaneseForConditionalGeneration
    - forward
