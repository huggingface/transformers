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
*This model was released on 2022-05-10 and added to Hugging Face Transformers on 2023-06-20 and contributed by [Seledorn](https://huggingface.co/Seledorn).*

# UL2

[UL2](https://huggingface.co/papers/2205.05131v1) introduces a unified pre-training framework that separates model architectures from pre-training objectives and recasts various self-supervised NLP objectives into a single generalized perspective. It proposes Mixture-of-Denoisers (MoD), a novel objective that blends multiple pre-training paradigms, along with mode switching, which aligns fine-tuning tasks with specific pre-training schemes. Extensive experiments show that this approach advances the performance frontier, outperforming T5 and GPT-like baselines across diverse tasks. When scaled to 20B parameters, the model achieves state-of-the-art results on 50 NLP benchmarks, including language generation, reasoning, and retrieval, while surpassing GPT-3 (175B) in zero-shot SuperGLUE and significantly improving one-shot summarization over T5-XXL.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text2text-generation", model="google/ul2", dtype="auto",)
pipeline("translate English to French: Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/ul2")
model = AutoModelForSeq2SeqLM.from_pretrained("google/ul2", dtype="auto",)

inputs = tokenizer("translate English to French: Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## Usage tips

- UL2 has the same architecture as T5v1.1 but uses Gated-SiLU activation instead of Gated-GELU.