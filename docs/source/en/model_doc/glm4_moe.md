<!--Copyright 2025 The ZhipuAI Inc. and The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2025-07-28 and added to Hugging Face Transformers on 2025-07-21.*

# Glm4Moe

[Glm4Moe](https://z.ai/blog/glm-4.6) is an upgraded large language model with a 200K-token context window (up from 128K), enabling it to handle more complex and extended tasks. It delivers stronger coding performance—especially in front-end generation and real-world applications—and shows marked gains in reasoning, writing quality, and tool-using capability for agentic workflows. Evaluations across eight benchmarks confirm consistent improvements over GLM-4.5 and competitive performance against leading models like Claude Sonnet 4, while maintaining better efficiency by completing tasks with about 15% fewer tokens. In extended real-world testing via the CC-Bench framework, GLM-4.6 achieved near-parity with Claude Sonnet 4 and outperformed other open-source baselines.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="https://huggingface.co/zai-org/GLM-4.6", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("https://huggingface.co/zai-org/GLM-4.6")
model = AutoModelForCausalLM.from_pretrained("https://huggingface.co/zai-org/GLM-4.6", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## Glm4MoeConfig

[[autodoc]] Glm4MoeConfig

## Glm4MoeModel

[[autodoc]] Glm4MoeModel
    - forward

## Glm4MoeForCausalLM

[[autodoc]] Glm4MoeForCausalLM
    - forward
