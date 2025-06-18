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

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
           <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white" >
           <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
    </div>
</div>


# DeBERTa-v2

[DeBERTa-v2](https://huggingface.co/papers/2006.03654) is an advanced language model that builds on the original DeBERTa architecture, which itself is an evolution of BERT and RoBERTa. What makes DeBERTa-v2 special is its unique way of handling the meaning and position of words separately, using a disentangled attention mechanism—this helps it better understand context and relationships in language. The v2 version introduces a new, larger vocabulary, a sentencepiece-based tokenizer, nGram Induced Input Encoding (nGiE) for better local dependency learning, and more efficient parameter sharing in the attention layers. It also comes in larger sizes (900M and 1.5B parameters), which means it can tackle even more complex language tasks with higher accuracy.

You can find all the original [DeBERTa-v2] checkpoints under the [DeBERTa-v2](https://huggingface.co/microsoft?search_models=deberta-v2) collection.


> [!TIP]
> Click on the [DeBERTa-v2] models in the right sidebar for more examples of how to apply [DeBERTa-v2] to different text classification, token classification, question answering, and multiple choice tasks.

The example below demonstrates how to generate text based on an image with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

# Example: Text classification with DeBERTa-v2
classifier = pipeline(
    task="text-classification",
    model="microsoft/deberta-v2-xlarge",
    device=0,
    torch_dtype=torch.float16
)
result = classifier("DeBERTa-v2 is great at understanding context!")
print(result)
```

</hfoption>
<hfoption id="AutoModel">

```py
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge")
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path="microsoft/deberta-v2-xlarge",
    device_map=None
)

inputs = tokenizer("DeBERTa-v2 is great at understanding context!", return_tensors="pt")
outputs = model(**inputs)
print(outputs.last_hidden_state)

```

</hfoption>

<hfoption id="transformers-cli">

```bash
echo -e "Plants create [MASK] through a process known as photosynthesis." | transformers run --task fill-mask --model microsoft/deberta-v2-xlarge --device 0
```
</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes quantization](https://huggingface.co/docs/transformers/main/en/main_classes/quantization) to only quantize the weights to 8-bit.

## Notes

- DeBERTa-v2 introduces a sentencepiece-based tokenizer and a larger vocabulary (128K tokens), improving its ability to handle diverse text.
- The nGiE (nGram Induced Input Encoding) layer helps the model better capture local dependencies in text.
- Parameter sharing in the attention layer reduces model size without sacrificing performance.
- Relative position encoding uses log buckets, similar to T5, for more efficient handling of long sequences.
- Larger model sizes (900M and 1.5B parameters) are available, providing improved performance on downstream tasks but requiring significantly more computational resources.

## Resources

- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Question answering task guide](../tasks/question_answering)
- [Masked language modeling task guide](../tasks/masked_language_modeling)
- [Multiple choice task guide](../tasks/multiple_choice)

## DebertaV2Config

[[autodoc]] DebertaV2Config

## DebertaV2Tokenizer

[[autodoc]] DebertaV2Tokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## DebertaV2TokenizerFast

[[autodoc]] DebertaV2TokenizerFast
    - build_inputs_with_special_tokens
    - create_token_type_ids_from_sequences

<frameworkcontent>
<pt>

## DebertaV2Model

[[autodoc]] DebertaV2Model
    - forward

## DebertaV2PreTrainedModel

[[autodoc]] DebertaV2PreTrainedModel
    - forward

## DebertaV2ForMaskedLM

[[autodoc]] DebertaV2ForMaskedLM
    - forward

## DebertaV2ForSequenceClassification

[[autodoc]] DebertaV2ForSequenceClassification
    - forward

## DebertaV2ForTokenClassification

[[autodoc]] DebertaV2ForTokenClassification
    - forward

## DebertaV2ForQuestionAnswering

[[autodoc]] DebertaV2ForQuestionAnswering
    - forward

## DebertaV2ForMultipleChoice

[[autodoc]] DebertaV2ForMultipleChoice
    - forward

</pt>
<tf>

## TFDebertaV2Model

[[autodoc]] TFDebertaV2Model
    - call

## TFDebertaV2PreTrainedModel

[[autodoc]] TFDebertaV2PreTrainedModel
    - call

## TFDebertaV2ForMaskedLM

[[autodoc]] TFDebertaV2ForMaskedLM
    - call

## TFDebertaV2ForSequenceClassification

[[autodoc]] TFDebertaV2ForSequenceClassification
    - call

## TFDebertaV2ForTokenClassification

[[autodoc]] TFDebertaV2ForTokenClassification
    - call

## TFDebertaV2ForQuestionAnswering

[[autodoc]] TFDebertaV2ForQuestionAnswering
    - call

## TFDebertaV2ForMultipleChoice

[[autodoc]] TFDebertaV2ForMultipleChoice
    - call

</tf>
</frameworkcontent>
