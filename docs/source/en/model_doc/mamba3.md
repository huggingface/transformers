<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-11-18.*

# Mamba 3

[Mamba 3](https://github.com/state-spaces/mamba) is a selective state space model. Compared to [Mamba 2](./mamba2) it drops the
short convolution, uses a data-dependent decay with a trapezoidal discretization, and applies a data-dependent rotary
embedding to B and C. Each block pairs a Mamba-3 mixer with a gated MLP.

Only the SISO variant is supported. The original checkpoints are under the
[State Space Models](https://huggingface.co/state-spaces) organization and can be converted with
`convert_mamba3_ssm_checkpoint_to_pytorch.py`. The [mamba-ssm](https://github.com/state-spaces/mamba) Triton kernel is
used when installed, otherwise a PyTorch implementation runs.

```py
from transformers import AutoTokenizer, Mamba3ForCausalLM

tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/mamba3-siso-187m")
model = Mamba3ForCausalLM.from_pretrained("hf-internal-testing/mamba3-siso-187m")
inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=20)[0]))
```

## Mamba3Config

[[autodoc]] Mamba3Config

## Mamba3Model

[[autodoc]] Mamba3Model
    - forward

## Mamba3ForCausalLM

[[autodoc]] Mamba3ForCausalLM
    - forward
