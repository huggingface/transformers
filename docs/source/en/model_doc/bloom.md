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
*This model was released on 2022-11-09 and added to Hugging Face Transformers on 2022-06-09.*

# BLOOM

[BLOOM](https://huggingface.co/papers/2211.05100) is a 176-billion parameter open-access large language model built collaboratively by hundreds of researchers to promote wider accessibility of LLM technology. It is a decoder-only Transformer trained on the ROOTS corpus, which includes text from hundreds of sources across 46 natural and 13 programming languages. BLOOM demonstrates competitive performance across diverse benchmarks, with further gains achieved through multitask prompted finetuning. The model and code are publicly released under the Responsible AI License to support open research and applications.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="bigscience/bloom-560m", dtype="auto")
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## BloomConfig

[[autodoc]] BloomConfig
    - all

## BloomTokenizerFast

[[autodoc]] BloomTokenizerFast
    - all

## BloomModel

[[autodoc]] BloomModel
    - forward

## BloomForCausalLM

[[autodoc]] BloomForCausalLM
    - forward

## BloomForSequenceClassification

[[autodoc]] BloomForSequenceClassification
    - forward

## BloomForTokenClassification

[[autodoc]] BloomForTokenClassification
    - forward

## BloomForQuestionAnswering

[[autodoc]] BloomForQuestionAnswering
    - forward

