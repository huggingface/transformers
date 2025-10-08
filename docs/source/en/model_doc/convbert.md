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
*This model was released on 2020-08-06 and added to Hugging Face Transformers on 2021-01-27 and contributed by [abhishek](https://huggingface.co/abhishek).*

# ConvBERT

[ConvBERT: Improving BERT with Span-based Dynamic Convolution](https://huggingface.co/papers/2008.02496) proposes a novel span-based dynamic convolution to enhance BERT by replacing some self-attention heads with convolution heads, forming a mixed attention block. This design improves efficiency in learning both global and local contexts. ConvBERT outperforms BERT and its variants in various tasks, achieving an 86.4 GLUE score with less training cost and fewer parameters.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="fill-mask", model="YituTech/conv-bert-base", dtype="auto")
pipeline("Plants create [MASK] through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("YituTech/conv-bert-base", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("YituTech/conv-bert-base")

inputs = tokenizer("Plants create [MASK] through a process known as photosynthesis.", return_tensors="pt")
outputs = model(**inputs)
mask_token_id = tokenizer.mask_token_id
mask_position = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_word = tokenizer.decode(outputs.logits[0, mask_position].argmax(dim=-1))
print(f"Predicted word: {predicted_word}")
```

</hfoption>
</hfoptions>

## Resources

## ConvBertConfig

[[autodoc]] ConvBertConfig

## ConvBertTokenizer

[[autodoc]] ConvBertTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## ConvBertTokenizerFast

[[autodoc]] ConvBertTokenizerFast

## ConvBertModel

[[autodoc]] ConvBertModel
    - forward

## ConvBertForMaskedLM

[[autodoc]] ConvBertForMaskedLM
    - forward

## ConvBertForSequenceClassification

[[autodoc]] ConvBertForSequenceClassification
    - forward

## ConvBertForMultipleChoice

[[autodoc]] ConvBertForMultipleChoice
    - forward

## ConvBertForTokenClassification

[[autodoc]] ConvBertForTokenClassification
    - forward

## ConvBertForQuestionAnswering

[[autodoc]] ConvBertForQuestionAnswering
    - forward

