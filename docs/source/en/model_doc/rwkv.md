<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2022-08-17 and added to Hugging Face Transformers on 2023-05-09 and contributed by [sgugger](https://huggingface.co/sgugger).*

# RWKV

[RWKV](https://huggingface.co/papers/2305.13048) addresses the trade-off between Transformers’ high memory and computational cost and RNNs’ limited performance. It combines a linear attention mechanism with a flexible architecture that can operate as either a Transformer or an RNN, enabling parallelized training and constant-memory, efficient inference. This design allows RWKV to scale to tens of billions of parameters while matching the performance of similarly sized Transformers. The model demonstrates a path toward highly efficient sequence processing without sacrificing accuracy.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="RWKV/v5-Eagle-7B-HF", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("RWKV/v5-Eagle-7B-HF")
model = AutoModelForCausalLM.from_pretrained("RWKV/v5-Eagle-7B-HF", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## RwkvConfig

[[autodoc]] RwkvConfig

## RwkvModel

[[autodoc]] RwkvModel
    - forward

## RwkvLMHeadModel

[[autodoc]] RwkvForCausalLM
    - forward