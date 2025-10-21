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
*This model was released on {release_date} and added to Hugging Face Transformers on 2023-04-12 and contributed by [openbmb](https://huggingface.co/openbmb).*

# CPMAnt

[CPM-Ant](https://github.com/OpenBMB/CPM-Live/tree/cpm-ant/cpm-live) is developed from CPM-Live, an open-source framework for training and serving large language models. It supports distributed training across multiple GPUs and nodes with model, data, and pipeline parallelism, enabling efficient scaling to billions of parameters. The framework provides features like dynamic micro-batching, mixed precision training, and checkpointing for fault tolerance. It also includes APIs for interactive inference, making it practical for both research and real-world deployment of large Transformer-based models.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="openbmb/cpm-ant-10b", dtype="auto")
pipeline("植物通过光合作用产生能量。")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("openbmb/cpm-ant-10b", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("openbmb/cpm-ant-10b")

inputs = tokenizer("植物通过光合作用产生能量。", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50, do_sample=True)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
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