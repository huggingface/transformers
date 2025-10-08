<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2023-03-03 and added to Hugging Face Transformers on 2023-06-20.*

# FLAN-UL2

[FLAN-UL2](https://huggingface.co/papers/2210.11416) investigates instruction finetuning of language models, focusing on scaling task diversity, model size, and chain-of-thought data. The authors show that instruction finetuning substantially improves performance across model families (PaLM, T5, U-PaLM), prompting strategies (zero-shot, few-shot, CoT), and benchmarks (MMLU, BBH, TyDiQA, MGSM). Notably, Flan-PaLM 540B finetuned on 1.8K tasks surpasses PaLM 540B by 9.4% on average and achieves state-of-the-art results, including 75.2% on five-shot MMLU. Additionally, publicly released Flan-T5 checkpoints demonstrate strong few-shot abilities rivaling much larger models, underscoring instruction finetuning as a broadly effective method for enhancing pretrained language models.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline("text-generation", model="google/flan-ul2", dtype="auto")
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("ggoogle/flan-ul2", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>