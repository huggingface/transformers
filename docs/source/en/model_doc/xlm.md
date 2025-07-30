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
    </div>
</div>

# XLM

[XLM](https://huggingface.co/papers/1901.07291) demonstrates cross-lingual pretraining with two approaches, unsupervised training on a single language and supervised training on more than one language with a cross-lingual language model objective. The XLM model supports the causal language modeling objective, masked language modeling, and translation language modeling (an extension of the [BERT](./bert)) masked language modeling objective to multiple language inputs).

You can find all the original XLM checkpoints under the [Facebook AI community](https://huggingface.co/FacebookAI?search_models=xlm-mlm) organization.

> [!TIP]
> Click on the XLM models in the right sidebar for more examples of how to apply XLM to different cross-lingual tasks like classification, translation, and question answering.

The example below demonstrates how to predict the `<mask>` token with [`Pipeline`], [`AutoModel`] and from the command line.

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
    "FacebookAI/xlm-mlm-en-2048",  
)  
model = AutoModelForMaskedLM.from_pretrained(  
    "FacebookAI/xlm-mlm-en-2048",  
    dtype=torch.float16,  
    device_map="auto",  
)  
inputs = tokenizer("Hello, I'm a <mask> model.", return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1)

predicted_token = tokenizer.decode(predictions[0][inputs["input_ids"][0] == tokenizer.mask_token_id])
print(f"Predicted token: {predicted_token}")
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "Plants create <mask> through a process known as photosynthesis." | transformers-cli run --task fill-mask --model FacebookAI/xlm-mlm-en-2048 --device 0
```
</hfoption>
</hfoptions>

## XLMConfig

[[autodoc]] XLMConfig

## XLMTokenizer

[[autodoc]] XLMTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## XLM specific outputs

[[autodoc]] models.xlm.modeling_xlm.XLMForQuestionAnsweringOutput

<frameworkcontent>
<pt>

## XLMModel

[[autodoc]] XLMModel
    - forward

## XLMWithLMHeadModel

[[autodoc]] XLMWithLMHeadModel
    - forward

## XLMForSequenceClassification

[[autodoc]] XLMForSequenceClassification
    - forward

## XLMForMultipleChoice

[[autodoc]] XLMForMultipleChoice
    - forward

## XLMForTokenClassification

[[autodoc]] XLMForTokenClassification
    - forward

## XLMForQuestionAnsweringSimple

[[autodoc]] XLMForQuestionAnsweringSimple
    - forward

## XLMForQuestionAnswering

[[autodoc]] XLMForQuestionAnswering
    - forward

</pt>
<tf>

## TFXLMModel

[[autodoc]] TFXLMModel
    - call

## TFXLMWithLMHeadModel

[[autodoc]] TFXLMWithLMHeadModel
    - call

## TFXLMForSequenceClassification

[[autodoc]] TFXLMForSequenceClassification
    - call

## TFXLMForMultipleChoice

[[autodoc]] TFXLMForMultipleChoice
    - call

## TFXLMForTokenClassification

[[autodoc]] TFXLMForTokenClassification
    - call

## TFXLMForQuestionAnsweringSimple

[[autodoc]] TFXLMForQuestionAnsweringSimple
    - call

</tf>
</frameworkcontent>


