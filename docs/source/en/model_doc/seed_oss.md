<!--
 Copyright 2025 Bytedance-Seed Ltd and the HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2025-08-22.*

# Seed-OSS

[Seed-OSS](https://seed.bytedance.com/en/blog/seed-oss-open-source-models-release) is a series of open-source language models, with Seed-OSS-36B trained on 12 trillion tokens and available under the Apache-2.0 license. These models support native long contexts up to 512K tokens, flexible reasoning length control, and enhanced reasoning and agentic capabilities, making them suitable for tasks like tool use and problem-solving. The release includes both pre-trained models with synthetic instruction data and versions without, offering research flexibility. The post-trained Seed-OSS-36B-Instruct model achieves state-of-the-art performance among open-source models of similar size across mathematics, coding, reasoning, agent tasks, and long-text processing.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="ByteDance-Seed/SeedOss-36B", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("ByteDance-Seed/SeedOss-36B")
model = AutoModelForCausalLM.from_pretrained("ByteDance-Seed/SeedOss-36B", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## SeedOssConfig

[[autodoc]] SeedOssConfig

## SeedOssModel

[[autodoc]] SeedOssModel
    - forward

## SeedOssForCausalLM

[[autodoc]] SeedOssForCausalLM
    - forward

## SeedOssForSequenceClassification

[[autodoc]] SeedOssForSequenceClassification
    - forward

## SeedOssForTokenClassification

[[autodoc]] SeedOssForTokenClassification
    - forward

## SeedOssForQuestionAnswering

[[autodoc]] SeedOssForQuestionAnswering
    - forward
