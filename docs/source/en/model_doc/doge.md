<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

*This model was released on 2024-12-27 and added to Hugging Face Transformers on 2025-07-08.*

# Doge

[Doge-20M](https://huggingface.co/papers/PAPER_ID) is utilized for text generation, demonstrating its capability to produce coherent and contextually relevant responses. For question answering, Doge-20M-Instruct is employed, showcasing enhanced performance in understanding and generating answers through a structured conversational format. The model leverages specific generation configurations, including temperature and top-p sampling, to ensure varied and engaging outputs.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="SmallDoge/Doge-20M", dtype="auto")
pipeline("Plants generate energy through a process known as  ")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("SmallDoge/Doge-20M", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("SmallDoge/Doge-20M")

inputs = tokenizer("Plants generate energy through a process known as  ", return_tensors='pt', return_token_type_ids=False)
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
```

</hfoption>
</hfoptions>

## DogeConfig

[[autodoc]] DogeConfig

## DogeModel

[[autodoc]] DogeModel
    - forward

## DogeForCausalLM

[[autodoc]] DogeForCausalLM
    - forward

## DogeForSequenceClassification

[[autodoc]] DogeForSequenceClassification
    - forward

