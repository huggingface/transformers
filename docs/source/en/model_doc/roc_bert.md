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

[RoCBert](https://aclanthology.org/2022.acl-long.65.pdf) is a pretrained Chinese [BERT](./bert) model designed against adversarial attacks like typos and synonyms. It is pretrained with a contrastive learning objective to align normal and adversarial text examples. The examples include different semantic, phonetic, and visual features of Chinese. This makes RoCBert more robust against manipulation.

You can find all the original RoCBert checkpoints under the [weiweishi](https://huggingface.co/weiweishi) profile.

> [!TIP]
> This model was contributed by [weiweishi](https://huggingface.co/weiweishi).
> 
> Click on the RoCBert models in the right sidebar for more examples of how to apply RoCBert to different Chinese language tasks.

The example below demonstrates how to predict the [MASK] token with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
   task="fill-mask",
   model="weiweishi/roc-bert-base-zh",
   dtype=torch.float16,
   device=0
)
pipeline("這家餐廳的拉麵是我[MASK]過的最好的拉麵之")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
   "weiweishi/roc-bert-base-zh",
)
model = AutoModelForMaskedLM.from_pretrained(
   "weiweishi/roc-bert-base-zh",
   dtype=torch.float16,
   device_map="auto",
)
inputs = tokenizer("這家餐廳的拉麵是我[MASK]過的最好的拉麵之", return_tensors="pt").to("cuda")

with torch.no_grad():
   outputs = model(**inputs)
   predictions = outputs.logits

masked_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
predicted_token_id = predictions[0, masked_index].argmax(dim=-1)
predicted_token = tokenizer.decode(predicted_token_id)

print(f"The predicted token is: {predicted_token}")
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "這家餐廳的拉麵是我[MASK]過的最好的拉麵之" | transformers-cli run --task fill-mask --model weiweishi/roc-bert-base-zh --device 0
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
