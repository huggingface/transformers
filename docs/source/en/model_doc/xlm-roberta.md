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
*This model was released on 2019-11-05 and added to Hugging Face Transformers on 2020-11-16 and contributed by [stefan-it](https://huggingface.co/stefan-it).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# XLM-RoBERTa

[XLM-R](https://huggingface.co/papers/1911.02116) is a Transformer-based masked language model trained on over two terabytes of filtered CommonCrawl data across one hundred languages. It outperforms multilingual BERT (mBERT) on various cross-lingual benchmarks, showing improvements such as +13.8% average accuracy on XNLI, +12.3% average F1 score on MLQA, and +2.1% average F1 score on NER. XLM-R particularly excels in low-resource languages, enhancing XNLI accuracy by 11.8% for Swahili and 9.2% for Urdu compared to the previous XLM model. The model demonstrates the feasibility of multilingual modeling without compromising per-language performance, competing effectively with strong monolingual models on GLUE and XNLI benchmarks.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="fill-mask", model="FacebookAI/xlm-roberta-base", dtype="auto")
pipeline("Les plantes créent <mask> grâce à un processus appelé photosynthèse.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("FacebookAI/xlm-roberta-base", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")

inputs = tokenizer("Les plantes créent <mask> grâce à un processus appelé photosynthèse.", return_tensors="pt")
outputs = model(**inputs)
mask_token_id = tokenizer.mask_token_id
mask_position = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_word = tokenizer.decode(outputs.logits[0, mask_position].argmax(dim=-1))
print(f"Predicted word: {predicted_word}")
```

</hfoption>
</hfoptions>


## XLMRobertaConfig

[[autodoc]] XLMRobertaConfig

## XLMRobertaTokenizer

[[autodoc]] XLMRobertaTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## XLMRobertaTokenizerFast

[[autodoc]] XLMRobertaTokenizerFast

## XLMRobertaModel

[[autodoc]] XLMRobertaModel
    - forward

## XLMRobertaForCausalLM

[[autodoc]] XLMRobertaForCausalLM
    - forward

## XLMRobertaForMaskedLM

[[autodoc]] XLMRobertaForMaskedLM
    - forward

## XLMRobertaForSequenceClassification

[[autodoc]] XLMRobertaForSequenceClassification
    - forward

## XLMRobertaForMultipleChoice

[[autodoc]] XLMRobertaForMultipleChoice
    - forward

## XLMRobertaForTokenClassification

[[autodoc]] XLMRobertaForTokenClassification
    - forward

## XLMRobertaForQuestionAnswering

[[autodoc]] XLMRobertaForQuestionAnswering
    - forward

