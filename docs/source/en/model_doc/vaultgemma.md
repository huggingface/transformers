<!--Copyright 2025 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was released on 2016-07-01 and added to Hugging Face Transformers on 2025-09-12.*

# VaultGemma

[VaultGemma](https://services.google.com/fh/files/blogs/vaultgemma_tech_report.pdf) is a 1-billion-parameter model in the Gemma family that has been fully trained with differential privacy, ensuring strong privacy guarantees during training. It was pretrained on the same data mixture used for the Gemma 2 series, maintaining consistency with prior work while adding privacy-preserving capabilities. This marks a notable advancement in developing large language models that balance utility with privacy protection. The model has been openly released to the research community for broader use and evaluation.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/vaultgemma-1b", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/vaultgemma-1b")
model = AutoModelForCausalLM.from_pretrained("google/vaultgemma-1b", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## VaultGemmaConfig

[[autodoc]] VaultGemmaConfig

## VaultGemmaModel

[[autodoc]] VaultGemmaModel
    - forward

## VaultGemmaForCausalLM

[[autodoc]] VaultGemmaForCausalLM
