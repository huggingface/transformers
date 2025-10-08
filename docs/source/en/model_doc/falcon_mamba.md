<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2024-10-07 and added to Hugging Face Transformers on 2024-08-12.*

# FalconMamba

[FalconMamba](https://huggingface.co/papers/2410.05355) is a large language model based on the Mamba architecture, trained on 5.8 trillion tokens from a diverse data mixture. It outperforms leading open-weight models like Mistral 7B, Llama3 8B, and Falcon2 11B, and is on par with Gemma 7B. FalconMamba surpasses both existing Mamba and hybrid Mamba-Transformer models in performance. The model is faster at inference and requires less memory for long sequence generation, challenging the notion that hybrid models are superior to pure architecture designs. The weights are publicly available under a permissive license.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline("text-generation", model="tiiuae/falcon-mamba-7b", dtype="auto")
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-mamba-7b", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-mamba-7b")

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>


## FalconMambaConfig

[[autodoc]] FalconMambaConfig

## FalconMambaModel

[[autodoc]] FalconMambaModel
    - forward

## FalconMambaLMHeadModel

[[autodoc]] FalconMambaForCausalLM
    - forward

