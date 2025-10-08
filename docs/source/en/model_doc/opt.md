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
*This model was released on 2022-05-02 and added to Hugging Face Transformers on 2022-05-12 and contributed by [ArthurZ](https://huggingface.co/ArthurZ), [ybelkada](https://huggingface.co/ybelkada), and [patrickvonplaten](https://huggingface.co/patrickvonplaten).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# OPT

[OPT](https://huggingface.co/papers/2205.01068) is a suite of decoder-only pre-trained transformers with parameter sizes ranging from 125M to 175B. OPT-175B matches the capabilities of GPT-3 but with a significantly lower carbon footprint during development. The models are fully shared, along with infrastructure challenges and experimental code, enabling broader research access. OPT uses the same architecture as BartDecoder and uniquely adds the EOS token `</s>` at the start of each prompt.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="facebook/opt-125mfacebook/opt-125m", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## OPTConfig

[[autodoc]] OPTConfig

## OPTModel

[[autodoc]] OPTModel
    - forward

## OPTForCausalLM

[[autodoc]] OPTForCausalLM
    - forward

## OPTForSequenceClassification

[[autodoc]] OPTForSequenceClassification
    - forward

## OPTForQuestionAnswering

[[autodoc]] OPTForQuestionAnswering
    - forward

