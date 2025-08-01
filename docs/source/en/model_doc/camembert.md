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
		<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
		<img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
    <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
	</div>
</div>

# CamemBERT

[CamemBERT](https://huggingface.co/papers/1911.03894) is a language model based on [RoBERTa](./roberta), but trained specifically on French text from the OSCAR dataset, making it more effective for French language tasks.

What sets CamemBERT apart is that it learned from a huge, high quality collection of French data, as opposed to mixing lots of languages. This helps it really understand French better than many multilingual models.

Common applications of CamemBERT include masked language modeling (Fill-mask prediction), text classification (sentiment analysis), token classification (entity recognition) and sentence pair classification (entailment tasks).

You can find all the original CamemBERT checkpoints under the [ALMAnaCH](https://huggingface.co/almanach/models?search=camembert) organization.

> [!TIP]
> This model was contributed by the [ALMAnaCH (Inria)](https://huggingface.co/almanach) team.
>
> Click on the CamemBERT models in the right sidebar for more examples of how to apply CamemBERT to different NLP tasks.

The examples below demonstrate how to predict the `<mask>` token with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">

<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipeline = pipeline("fill-mask", model="camembert-base", dtype=torch.float16, device=0)
pipeline("Le camembert est un délicieux fromage <mask>.")
```
</hfoption> 

<hfoption id="AutoModel">

```python
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModelForMaskedLM.from_pretrained("camembert-base", dtype="auto", device_map="auto", attn_implementation="sdpa")
inputs = tokenizer("Le camembert est un délicieux fromage <mask>.", return_tensors="pt").to("cuda")

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
echo -e "Le camembert est un délicieux fromage <mask>." | transformers run --task fill-mask --model camembert-base --device 0
```

</hfoption> 

</hfoptions> 


Quantization reduces the memory burden of large models by representing weights in lower precision. Refer to the [Quantization](../quantization/overview) overview for available options.

The example below uses [bitsandbytes](../quantization/bitsandbytes) quantization to quantize the weights to 8-bits.
  
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM, BitsAndBytesConfig
import torch

quant_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForMaskedLM.from_pretrained(
    "almanach/camembert-large",
    quantization_config=quant_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("almanach/camembert-large")

inputs = tokenizer("Le camembert est un délicieux fromage <mask>.", return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

masked_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
predicted_token_id = predictions[0, masked_index].argmax(dim=-1)
predicted_token = tokenizer.decode(predicted_token_id)

print(f"The predicted token is: {predicted_token}")
```

## CamembertConfig

[[autodoc]] CamembertConfig

## CamembertTokenizer

[[autodoc]] CamembertTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## CamembertTokenizerFast

[[autodoc]] CamembertTokenizerFast

<frameworkcontent>
<pt>

## CamembertModel

[[autodoc]] CamembertModel

## CamembertForCausalLM

[[autodoc]] CamembertForCausalLM

## CamembertForMaskedLM

[[autodoc]] CamembertForMaskedLM

## CamembertForSequenceClassification

[[autodoc]] CamembertForSequenceClassification

## CamembertForMultipleChoice

[[autodoc]] CamembertForMultipleChoice

## CamembertForTokenClassification

[[autodoc]] CamembertForTokenClassification

## CamembertForQuestionAnswering

[[autodoc]] CamembertForQuestionAnswering

</pt>
<tf>

## TFCamembertModel

[[autodoc]] TFCamembertModel

## TFCamembertForCausalLM

[[autodoc]] TFCamembertForCausalLM

## TFCamembertForMaskedLM

[[autodoc]] TFCamembertForMaskedLM

## TFCamembertForSequenceClassification

[[autodoc]] TFCamembertForSequenceClassification

## TFCamembertForMultipleChoice

[[autodoc]] TFCamembertForMultipleChoice

## TFCamembertForTokenClassification

[[autodoc]] TFCamembertForTokenClassification

## TFCamembertForQuestionAnswering

[[autodoc]] TFCamembertForQuestionAnswering

</tf>
</frameworkcontent>