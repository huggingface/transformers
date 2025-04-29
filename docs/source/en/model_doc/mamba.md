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

<div style="float: right;">
  <div class="flex flex-wrap space-x-1">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
  </div>
</div>

# Mamba

[Mamba](https://huggingface.co/papers/2312.00752) is a new `state space model` architecture that rivals the classic Transformers. The architecture stacks `mixer` layers, which are the equivalent of `Attention` layers. The core logic of `mamba` is held in the `MambaMixer` class.

You can find all the original Mamba checkpoints under the [state-spaces](https://huggingface.co/state-spaces) organization.

# Usage

> [!TIP]
> Click on the Mamba models in the right sidebar for more examples of how to apply Mamba to different language tasks.

The example below demonstrates how to generate text with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="text-generation",
    model="state-spaces/mamba-130m-hf",
    torch_dtype=torch.float16,
    device=0
)
pipeline("Plants create energy through a process known as")
```

</hfoption>
<hfoption id="AutoModel">

```py
from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
input_ids = tokenizer("Plants create energy through a process known as", return_tensors= "pt")["input_ids"]

out = model.generate(input_ids, max_new_tokens=10)
print(tokenizer.batch_decode(out))
```

</hfoption>
<hfoption id="transformers-cli">

```bash
echo -e "Plants create energy through a process known as" | transformers-cli run --task text-generation --model state-spaces/mamba-130m-hf --device 0
```

</hfoption>
</hfoptions>

## Notes

- The current implementation leverages the original cuda kernels: the equivalent of flash attention for Mamba are hosted in the [`mamba-ssm`](https://github.com/state-spaces/mamba) and the [`causal_conv1d`](https://github.com/Dao-AILab/causal-conv1d) repositories. Make sure to install them if your hardware supports them!

## MambaConfig

[[autodoc]] MambaConfig

## MambaModel

[[autodoc]] MambaModel
    - forward

## MambaLMHeadModel

[[autodoc]] MambaForCausalLM
    - forward
