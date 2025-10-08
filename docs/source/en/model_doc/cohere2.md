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
*This model was released on {release_date} and added to Hugging Face Transformers on 2024-12-13.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Command R7B

[Command R7B](https://cohere.com/blog/command-r7b) is Cohere’s smallest model in the R series, optimized for speed, efficiency, and high-quality outputs on commodity GPUs and edge devices. It has 7 billion parameters and is fine-tuned for retrieval-augmented generation (RAG), enabling strong grounding in enterprise data while maintaining low latency. The model is designed to balance cost and performance, making it accessible for real-world applications like search, summarization, and knowledge management. R7B continues the R-series focus on practical deployment, emphasizing scalability and adaptability for business use cases

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="CohereLabs/c4ai-command-r7b-12-2024", dtype="auto")
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("CohereLabs/c4ai-command-r7b-12-2024", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("CohereLabs/c4ai-command-r7b-12-2024")

messages = [{"role": "user", "content": "How do plants generate energy?"}]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

outputs = model.generate(input_ids, max_new_tokens=100, do_sample=True, temperature=0.3,)
print(tokenizer.decode(outputs[0]))
```

## Cohere2Config

[[autodoc]] Cohere2Config

## Cohere2Model

[[autodoc]] Cohere2Model
    - forward

## Cohere2ForCausalLM

[[autodoc]] Cohere2ForCausalLM
    - forward

