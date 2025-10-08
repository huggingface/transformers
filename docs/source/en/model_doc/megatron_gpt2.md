<!--Copyright 2021 NVIDIA Corporation and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2019-09-17 and added to Hugging Face Transformers on 2021-10-01 and contributed by [jdemouth](https://huggingface.co/jdemouth).*

# MegatronGPT2

[MegatronGPT2](https://huggingface.co/papers/1909.08053) presents techniques for training very large transformer models using intra-layer model parallelism, enabling the training of models with billions of parameters on 512 GPUs. This approach achieves 15.1 PetaFLOPs with 76% scaling efficiency. The model demonstrates state-of-the-art results on datasets like WikiText103 and LAMBADA, and a 3.9 billion parameter BERT-like model achieves top performance on the RACE dataset. Careful placement of layer normalization is highlighted as crucial for BERT-like models as they scale.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="robowaifudev/megatron-gpt2-345m", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("robowaifudev/megatron-gpt2-345m")
model = AutoModelForCausalLM.from_pretrained("robowaifudev/megatron-gpt2-345m", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>
