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
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# XLM-RoBERTa-XL

[XLM-RoBERTa-XL](https://huggingface.co/papers/2105.00572) is a 3.5B parameter multilingual masked language model pretrained on 100 languages. It shows that by scaling model capacity, multilingual models demonstrates strong performance on high-resource languages and can even zero-shot low-resource languages.

You can find all the original XLM-RoBERTa-XL checkpoints under the [AI at Meta](https://huggingface.co/facebook?search_models=xlm) organization.

> [!TIP]
> Click on the XLM-RoBERTa-XL models in the right sidebar for more examples of how to apply XLM-RoBERTa-XL to different cross-lingual tasks like classification, translation, and question answering.

The example below demonstrates how to predict the `<mask>` token with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch  
from transformers import pipeline  

pipeline = pipeline(  
    task="fill-mask",  
    model="facebook/xlm-roberta-xl",  
    dtype=torch.float16,  
    device=0  
)  
pipeline("Bonjour, je suis un modèle <mask>.")  
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch  
from transformers import AutoModelForMaskedLM, AutoTokenizer  

tokenizer = AutoTokenizer.from_pretrained(  
    "facebook/xlm-roberta-xl",  
)  
model = AutoModelForMaskedLM.from_pretrained(  
    "facebook/xlm-roberta-xl",  
    dtype=torch.float16,  
    device_map="auto",  
    attn_implementation="sdpa"  
)  
inputs = tokenizer("Bonjour, je suis un modèle <mask>.", return_tensors="pt").to("cuda")  

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
echo -e "Plants create <mask> through a process known as photosynthesis." | transformers-cli run --task fill-mask --model facebook/xlm-roberta-xl --device 0
```
</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [torchao](../quantization/torchao) to only quantize the weights to int4.

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, TorchAoConfig

quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
tokenizer = AutoTokenizer.from_pretrained(
    "facebook/xlm-roberta-xl",
)
model = AutoModelForMaskedLM.from_pretrained(
    "facebook/xlm-roberta-xl",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa",
    quantization_config=quantization_config
)
inputs = tokenizer("Bonjour, je suis un modèle <mask>.", return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

masked_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
predicted_token_id = predictions[0, masked_index].argmax(dim=-1)
predicted_token = tokenizer.decode(predicted_token_id)

print(f"The predicted token is: {predicted_token}")
```

## Notes

- Unlike some XLM models, XLM-RoBERTa-XL doesn't require `lang` tensors to understand which language is used. It automatically determines the language from the input ids.

## XLMRobertaXLConfig

[[autodoc]] XLMRobertaXLConfig

## XLMRobertaXLModel

[[autodoc]] XLMRobertaXLModel
    - forward

## XLMRobertaXLForCausalLM

[[autodoc]] XLMRobertaXLForCausalLM
    - forward

## XLMRobertaXLForMaskedLM

[[autodoc]] XLMRobertaXLForMaskedLM
    - forward

## XLMRobertaXLForSequenceClassification

[[autodoc]] XLMRobertaXLForSequenceClassification
    - forward

## XLMRobertaXLForMultipleChoice

[[autodoc]] XLMRobertaXLForMultipleChoice
    - forward

## XLMRobertaXLForTokenClassification

[[autodoc]] XLMRobertaXLForTokenClassification
    - forward

## XLMRobertaXLForQuestionAnswering

[[autodoc]] XLMRobertaXLForQuestionAnswering
    - forward
