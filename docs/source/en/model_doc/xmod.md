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
*This model was released on 2022-05-12 and added to Hugging Face Transformers on 2023-02-10.*

# X-MOD

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The X-MOD model was proposed in [Lifting the Curse of Multilinguality by Pre-training Modular Transformers](https://arxiv.org/abs/2205.06266) by Jonas Pfeiffer, Naman Goyal, Xi Lin, Xian Li, James Cross, Sebastian Riedel, and Mikel Artetxe.
X-MOD extends multilingual masked language models like [XLM-R](xlm-roberta) to include language-specific modular components (_language adapters_) during pre-training. For fine-tuning, the language adapters in each transformer layer are frozen.

The abstract from the paper is the following:

*Multilingual pre-trained models are known to suffer from the curse of multilinguality, which causes per-language performance to drop as they cover more languages. We address this issue by introducing language-specific modules, which allows us to grow the total capacity of the model, while keeping the total number of trainable parameters per language constant. In contrast with prior work that learns language-specific components post-hoc, we pre-train the modules of our Cross-lingual Modular (X-MOD) models from the start. Our experiments on natural language inference, named entity recognition and question answering show that our approach not only mitigates the negative interference between languages, but also enables positive transfer, resulting in improved monolingual and cross-lingual performance. Furthermore, our approach enables adding languages post-hoc with no measurable drop in performance, no longer limiting the model usage to the set of pre-trained languages.*

This model was contributed by [jvamvas](https://huggingface.co/jvamvas).
The original code can be found [here](https://github.com/facebookresearch/fairseq/tree/58cc6cca18f15e6d56e3f60c959fe4f878960a60/fairseq/models/xmod) and the original documentation is found [here](https://github.com/facebookresearch/fairseq/tree/58cc6cca18f15e6d56e3f60c959fe4f878960a60/examples/xmod).

## Usage tips

Tips:
- X-MOD is similar to [XLM-R](xlm-roberta), but a difference is that the input language needs to be specified so that the correct language adapter can be activated.
- The main models – base and large – have adapters for 81 languages.

## Adapter Usage

### Input language

There are two ways to specify the input language:
1. By setting a default language before using the model:

```python
from transformers import XmodModel

model = XmodModel.from_pretrained("facebook/xmod-base")
model.set_default_language("en_XX")
```

2. By explicitly passing the index of the language adapter for each sample:

```python
import torch

input_ids = torch.tensor(
    [
        [0, 581, 10269, 83, 99942, 136, 60742, 23, 70, 80583, 18276, 2],
        [0, 1310, 49083, 443, 269, 71, 5486, 165, 60429, 660, 23, 2],
    ]
)
lang_ids = torch.LongTensor(
    [
        0,  # en_XX
        8,  # de_DE
    ]
)
output = model(input_ids, lang_ids=lang_ids)
```

### Fine-tuning
The paper recommends that the embedding layer and the language adapters are frozen during fine-tuning. A method for doing this is provided:

```python
model.freeze_embeddings_and_language_adapters()
# Fine-tune the model ...
```

### Cross-lingual transfer
After fine-tuning, zero-shot cross-lingual transfer can be tested by activating the language adapter of the target language:

```python
model.set_default_language("de_DE")
# Evaluate the model on German examples ...
```

## Resources

- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Question answering task guide](../tasks/question_answering)
- [Causal language modeling task guide](../tasks/language_modeling)
- [Masked language modeling task guide](../tasks/masked_language_modeling)
- [Multiple choice task guide](../tasks/multiple_choice)

## XmodConfig

[[autodoc]] XmodConfig

## XmodModel

[[autodoc]] XmodModel
    - forward

## XmodForCausalLM

[[autodoc]] XmodForCausalLM
    - forward

## XmodForMaskedLM

[[autodoc]] XmodForMaskedLM
    - forward

## XmodForSequenceClassification

[[autodoc]] XmodForSequenceClassification
    - forward

## XmodForMultipleChoice

[[autodoc]] XmodForMultipleChoice
    - forward

## XmodForTokenClassification

[[autodoc]] XmodForTokenClassification
    - forward

## XmodForQuestionAnswering

[[autodoc]] XmodForQuestionAnswering
    - forward
