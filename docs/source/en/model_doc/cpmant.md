<!--Copyright 2022 The HuggingFace Team and The OpenBMB Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2022-09-16 and added to Hugging Face Transformers on 2023-04-12.*

# CPMAnt

[CPMAnt](https://github.com/OpenBMB/CPM-Live/tree/cpm-ant/cpm-live) is a 10B-parameter open-source Chinese pre-trained language model and the first milestone of the CPM-Live open training project. It achieves strong results with delta tuning on the CUGE benchmark, and compressed variants are available for different hardware configurations.

The example below demonstrates how to generate text with [`Pipeline`] or the [`CpmAntForCausalLM`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="openbmb/cpm-ant-10b",
    dtype=torch.bfloat16,
)
pipe("今天天气很好，")
```

</hfoption>
<hfoption id="CpmAntForCausalLM">

```py
import torch
from transformers import CpmAntForCausalLM, CpmAntTokenizer

tokenizer = CpmAntTokenizer.from_pretrained("openbmb/cpm-ant-10b")
model = CpmAntForCausalLM.from_pretrained(
    "openbmb/cpm-ant-10b",
    dtype=torch.bfloat16,
    device_map="auto",
)
input_ids = tokenizer("今天天气很好，", return_tensors="pt").to(model.device)

output = model.generate(**input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## CpmAntConfig

[[autodoc]] CpmAntConfig
    - all

## CpmAntTokenizer

[[autodoc]] CpmAntTokenizer
    - all

## CpmAntModel

[[autodoc]] CpmAntModel
    - all

## CpmAntForCausalLM

[[autodoc]] CpmAntForCausalLM
    - all
