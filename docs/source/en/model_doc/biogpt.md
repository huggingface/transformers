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
*This model was released on {release_date} and added to Hugging Face Transformers on 2022-12-05 and contributed by [kamalkraj](https://huggingface.co/kamalkraj).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# BioGPT

[BioGPT](https://huggingface.co/papers/bbac409) is a domain-specific generative Transformer language model designed for biomedical text generation and mining. Trained on 15M PubMed abstracts, BioGPT excels in various biomedical NLP tasks, outperforming previous models. It achieves notable F1 scores of 44.98%, 38.42%, and 40.76% on BC5CDR, KD-DTI, and DDI end-to-end relation extraction tasks, respectively, and sets a new record with 78.2% accuracy on PubMedQA. Additionally, BioGPT demonstrates superior text generation capabilities, producing fluent descriptions for biomedical terms.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="microsoft/biogpt", dtype="auto")
pipeline("Ibuprofen is best used for ")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("microsoft/biogpt", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")

inputs = tokenizer("Ibuprofen is best used for ", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## BioGptConfig

[[autodoc]] BioGptConfig

## BioGptTokenizer

[[autodoc]] BioGptTokenizer
    - save_vocabulary

## BioGptModel

[[autodoc]] BioGptModel
    - forward

## BioGptForCausalLM

[[autodoc]] BioGptForCausalLM
    - forward
    
## BioGptForTokenClassification

[[autodoc]] BioGptForTokenClassification
    - forward

## BioGptForSequenceClassification

[[autodoc]] BioGptForSequenceClassification
    - forward

