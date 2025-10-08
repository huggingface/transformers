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
*This model was released on 2024-08-23 and added to Hugging Face Transformers on 2024-08-27 and contributed by [mayank-mishra](https://huggingface.co/mayank-mishra).*

# Granite

[Granite](https://huggingface.co/papers/2408.13359) proposes the Power scheduler, addressing the challenge of finding optimal learning rates for language model pretraining. Through extensive experiments, a power-law relationship between learning rate, batch size, and training tokens was identified, demonstrating transferability across different model sizes. The Power scheduler, combined with Maximum Update Parameterization (\mup), achieves consistent performance without needing to adjust hyperparameters for varying training conditions. PowerLM-3B, a 3B-parameter model trained with this scheduler, matches state-of-the-art performance in small language models across benchmarks like natural language multi-choice, code generation, and math reasoning.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="ibm-granite/granite-3.3-2b-base", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.3-2b-base")
model = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-3.3-2b-base", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## GraniteConfig

[[autodoc]] GraniteConfig

## GraniteModel

[[autodoc]] GraniteModel
    - forward

## GraniteForCausalLM

[[autodoc]] GraniteForCausalLM
    - forward

