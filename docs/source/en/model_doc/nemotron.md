<!--Copyright 2024 The HuggingFace Team. All rights reserved.
Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->
*This model was released on 2024-02-26 and added to Hugging Face Transformers on 2024-08-06.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>


# Nemotron

[Minitron](https://huggingface.co/papers/2407.14679) explores an efficient alternative to training large language models (LLMs) from scratch by pruning a pretrained model and retraining it with less than 3% of the original data. The authors develop practical compression strategies combining depth, width, attention, and MLP pruning with knowledge distillation, optimizing the architecture through empirical experiments. Applying this approach to the Nemotron-4 family, they produce smaller 8B and 4B models from a 15B base using up to 40× fewer training tokens, achieving 1.8× compute savings for the full model family. The resulting Minitron models match or exceed the performance of similarly sized models, show up to 16% higher MMLU scores, and outperform prior compression techniques, with all model weights and code openly available.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="nvidia/Minitron-4B-Base", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("nvidia/Minitron-4B-Base")
model = AutoModelForCausalLM.from_pretrained("nvidia/Minitron-4B-Base", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## NemotronConfig

[[autodoc]] NemotronConfig

## NemotronModel

[[autodoc]] NemotronModel
    - forward

## NemotronForCausalLM

[[autodoc]] NemotronForCausalLM
    - forward

## NemotronForSequenceClassification

[[autodoc]] NemotronForSequenceClassification
    - forward

## NemotronForQuestionAnswering

[[autodoc]] NemotronForQuestionAnswering
    - forward

## NemotronForTokenClassification

[[autodoc]] NemotronForTokenClassification
    - forward