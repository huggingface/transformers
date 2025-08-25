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
*This model was released on 2019-07-26 and added to Hugging Face Transformers on 2020-11-16.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# RoBERTa

[RoBERTa](https://huggingface.co/papers/1907.11692) improves BERT with new pretraining objectives, demonstrating [BERT](./bert) was undertrained and training design is important. The pretraining objectives include dynamic masking, sentence packing, larger batches and a byte-level BPE tokenizer.

You can find all the original RoBERTa checkpoints under the [Facebook AI](https://huggingface.co/FacebookAI) organization.


> [!TIP]
> Click on the RoBERTa models in the right sidebar for more examples of how to apply RoBERTa to different language tasks.

The example below demonstrates how to predict the `<mask>` token with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="fill-mask",
    model="FacebookAI/roberta-base",
    dtype=torch.float16,
    device=0
)
pipeline("Plants create <mask> through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "FacebookAI/roberta-base",
)
model = AutoModelForMaskedLM.from_pretrained(
    "FacebookAI/roberta-base",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)
inputs = tokenizer("Plants create <mask> through a process known as photosynthesis.", return_tensors="pt").to(model.device)

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
echo -e "Plants create <mask> through a process known as photosynthesis." | transformers-cli run --task fill-mask --model FacebookAI/roberta-base --device 0
```

</hfoption>
</hfoptions>

## Notes

- RoBERTa doesn't have `token_type_ids` so you don't need to indicate which token belongs to which segment. Separate your segments with the separation token `tokenizer.sep_token` or `</s>`.

## RobertaConfig

[[autodoc]] RobertaConfig

## RobertaTokenizer

[[autodoc]] RobertaTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## RobertaTokenizerFast

[[autodoc]] RobertaTokenizerFast
    - build_inputs_with_special_tokens

## RobertaModel

[[autodoc]] RobertaModel
    - forward

## RobertaForCausalLM

[[autodoc]] RobertaForCausalLM
    - forward

## RobertaForMaskedLM

[[autodoc]] RobertaForMaskedLM
    - forward

## RobertaForSequenceClassification

[[autodoc]] RobertaForSequenceClassification
    - forward

## RobertaForMultipleChoice

[[autodoc]] RobertaForMultipleChoice
    - forward

## RobertaForTokenClassification

[[autodoc]] RobertaForTokenClassification
    - forward

## RobertaForQuestionAnswering

[[autodoc]] RobertaForQuestionAnswering
    - forward
