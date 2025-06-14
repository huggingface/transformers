<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
           <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# RoCBert

[RoCBert] is a BERT model that is specifically designed for the Chinese language. It is built to resist tricks and attacks, like misspellings or similar-looking words, that usually confuse language models.

You can find all the original [Model name] checkpoints under the [RoCBert](link) collection.

> [!TIP]
> Click on the RoCBert models in the right sidebar for more examples of how to apply RoCBert to different chinese NLP tasks.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
from transformers import pipeline

pipeline = pipeline(
    task="text-classification",
    model="hfl/rocbert-base"
)
pipeline("称呼") #Example Chinese input
```

</hfoption>
<hfoption id="AutoModel">

```py
# pip install datasets
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoTokenizer

model_name = "hfl/rocbert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
text = "大家好，无论谁正在阅读这篇文章"
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "水在零度时会[MASK]" | transformers-cli run --task fill-mask --model junnyu/roformer_chinese_base --device 0
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](link to quantization method) to only quantize the weights to __.

```py

```
## Resources

- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Question answering task guide](../tasks/question_answering)
- [Causal language modeling task guide](../tasks/language_modeling)
- [Masked language modeling task guide](../tasks/masked_language_modeling)
- [Multiple choice task guide](../tasks/multiple_choice)

## RoCBertConfig

[[autodoc]] RoCBertConfig
    - all

## RoCBertTokenizer

[[autodoc]] RoCBertTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## RoCBertModel

[[autodoc]] RoCBertModel
    - forward

## RoCBertForPreTraining

[[autodoc]] RoCBertForPreTraining
    - forward

## RoCBertForCausalLM

[[autodoc]] RoCBertForCausalLM
    - forward

## RoCBertForMaskedLM

[[autodoc]] RoCBertForMaskedLM
    - forward

## RoCBertForSequenceClassification

[[autodoc]] transformers.RoCBertForSequenceClassification
    - forward

## RoCBertForMultipleChoice

[[autodoc]] transformers.RoCBertForMultipleChoice
    - forward

## RoCBertForTokenClassification

[[autodoc]] transformers.RoCBertForTokenClassification
    - forward

## RoCBertForQuestionAnswering

[[autodoc]] RoCBertForQuestionAnswering
    - forward
