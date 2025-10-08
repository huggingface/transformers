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
*This model was released on 2024-05-26 and added to Hugging Face Transformers on 2024-10-04 and contributed by [pglo](https://huggingface.co/pglo).*

# Zamba

[Zamba](https://huggingface.co/papers/2405.16712) is a 7B parameter hybrid model that combines a Mamba backbone with a lightweight shared attention module, achieving transformer-level benefits with fewer parameters. It is trained on 1 trillion tokens across two phases: broad web datasets followed by an annealing stage with high-quality instruct and synthetic data under rapid learning rate decay. This architecture enables faster inference and lower memory use compared to similarly sized transformers, while maintaining strong competitive performance. The model, along with all checkpoints from both training phases, is released as open source.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="Zyphra/Zamba-7B-v1", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Zyphra/Zamba-7B-v1")
model = AutoModelForCausalLM.from_pretrained("Zyphra/Zamba-7B-v1", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## ZambaConfig

[[autodoc]] ZambaConfig

## ZambaModel

[[autodoc]] ZambaModel
    - forward

## ZambaForCausalLM

[[autodoc]] ZambaForCausalLM
    - forward

## ZambaForSequenceClassification

[[autodoc]] transformers.ZambaForSequenceClassification
    - forward
