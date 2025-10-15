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
*This model was released on 2023-12-01 and added to Hugging Face Transformers on 2024-03-05 and contributed by [ArthurZ](https://huggingface.co/ArthurZ).*

# Mamba

[Mamba](https://huggingface.co/papers/2312.00752) is a state-space-model architecture designed to address the computational inefficiency of Transformers on long sequences. It incorporates selective state spaces that allow the model to selectively propagate or forget information based on the input, enhancing content-based reasoning. Despite this, Mamba maintains linear scaling in sequence length and achieves fast inference, with a throughput 5× higher than Transformers. The model integrates mixer layers, similar to attention layers, and is optimized for hardware efficiency. Mamba outperforms Transformers of the same size and matches larger Transformers in performance across various modalities, including language, audio, and genomics.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="state-spaces/mamba-130m-hf", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## Usage tips

- The current implementation uses the original CUDA kernels. The FlashAttention equivalent implementation is hosted in the `mamba-ssm` and `causal_conv1d` repositories. Install them if your hardware supports it.
- Mamba stacks mixer layers which are equivalent to attention layers. Find the main logic of Mamba in the [`MambaMixer`] class.

## MambaConfig

[[autodoc]] MambaConfig

## MambaModel

[[autodoc]] MambaModel
    - forward

## MambaLMHeadModel

[[autodoc]] MambaForCausalLM
    - forward

## MambaCache

[[autodoc]] MambaCache

