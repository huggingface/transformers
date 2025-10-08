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
*This model was released on 2019-11-10 and added to Hugging Face Transformers on 2020-11-16 and contributed by [almanach](https://huggingface.co/almanach).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# CamemBERT

[CamemBERT: a Tasty French Language Model](https://huggingface.co/papers/1911.03894) is a French version of the BERT model, trained on 138GB of French text. It addresses the limitation of existing models that are either English-centric or multilingual, offering improved performance in French-specific tasks such as part-of-speech tagging, dependency parsing, named-entity recognition, and natural language inference. The pretrained CamemBERT model is released to encourage further research and applications in French NLP.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="fill-mask", model="almanach/camembert-base", dtype="auto")
pipeline("Les plantes créent <mask> grâce à un processus appelé photosynthèse.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("almanach/camembert-base", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("almanach/camembert-base")

inputs = tokenizer("Les plantes créent <mask> grâce à un processus appelé photosynthèse.", return_tensors="pt")
outputs = model(**inputs)
mask_token_id = tokenizer.mask_token_id
mask_position = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_word = tokenizer.decode(outputs.logits[0, mask_position].argmax(dim=-1))
print(f"Predicted word: {predicted_word}")
```

</hfoption>
</hfoptions>

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

