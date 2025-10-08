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
*This model was released on 2022-05-27 and added to Hugging Face Transformers on 2022-11-08 and contributed by [weiweishi](https://huggingface.co/weiweishi).*

# RoCBert

[RoCBert](https://huggingface.co/papers/2022.acl-long.65) is a robust pretrained Chinese BERT model designed to withstand various adversarial attacks such as word perturbation, synonyms, and typos. It employs contrastive learning to maintain label consistency across different adversarial examples. By incorporating multimodal features including semantic, phonetic, and visual information, ROCBERT enhances its robustness against attacks in multiple forms. The model outperforms strong baselines on five Chinese NLU tasks under three blackbox adversarial algorithms while maintaining high performance on clean datasets. Additionally, it excels in toxic content detection under human-made attacks.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="fill-mask", model="weiweishi/roc-bert-base-zh", dtype="auto")
pipeline("植物通过[MASK]合作用产生能量")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("weiweishi/roc-bert-base-zh", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("weiweishi/roc-bert-base-zh")

inputs = tokenizer("植物通过[MASK]合作用产生能量", return_tensors="pt")
outputs = model(**inputs)
mask_token_id = tokenizer.mask_token_id
mask_position = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_word = tokenizer.decode(outputs.logits[0, mask_position].argmax(dim=-1))
print(f"Predicted word: {predicted_word}")
```

</hfoption>
</hfoptions>

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

