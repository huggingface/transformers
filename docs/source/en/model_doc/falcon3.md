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
*This model was released on 2023-11-28 and added to Hugging Face Transformers on 2024-12-17.*

# Falcon3

[Falcon3](https://huggingface.co/papers/2311.16867) introduces decoder-only language models with 7B, 40B, and 180B parameters, trained primarily on high-quality web data. Falcon-180B was trained on 3.5 trillion tokens, representing the largest openly documented pretraining run to date, and achieves performance close to PaLM-2-Large while being more cost-efficient. It surpasses models such as PaLM, Chinchilla, LLaMA 2, and Inflection-1, placing it among the top three language models globally alongside GPT-4 and PaLM-2-Large. The project also details its custom distributed training system capable of scaling to 4,096 A100 GPUs on AWS, and openly releases both the models and a 600B-token dataset extract under a permissive license to support open research.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline("text-generation", model="tiiuae/Falcon3-1B-Base", dtype="auto")
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("tiiuae/Falcon3-1B-Base", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("tiiuae/Falcon3-1B-Base")

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>
