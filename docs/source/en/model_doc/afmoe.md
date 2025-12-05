<!--Copyright 2025 Arcee AI and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2025-12-04.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# AFMoE

AFMoE (Arcee Foundational Mixture of Experts) is a decoder-only transformer model that extends the Llama architecture with a sparse Mixture of Experts (MoE) approach. The model combines token-choice routing with shared experts and employs several architectural innovations for efficient inference and improved performance.

## Key Architecture Features

AFMoE introduces several key modifications to the standard transformer architecture:

- **Mixture of Experts with Shared Experts**: Combines routed experts (activated per-token via learned routing) with always-active shared experts for stable base computation
- **Token-Choice Routing**: Uses sigmoid or softmax-based routing with normalization and scaling for expert selection
- **Q/K Normalization and Gating**: Applies RMSNorm to query and key projections and uses sigmoid gating on attention outputs for improved stability
- **Hybrid Attention Patterns**: Alternates between sliding window attention and full attention across layers for efficiency with long contexts
- **Dual Normalization**: Uses pre- and post-normalization around both attention and MLP blocks for training stability
- **Configurable Dense Layers**: Allows initial layers to use dense MLPs before transitioning to sparse MoE layers

The model supports extended context lengths with RoPE embeddings and includes all standard Transformers features including Flash Attention 2, SDPA, gradient checkpointing, and quantization support.

> [!TIP]
> AFMoE is particularly well-suited for scenarios requiring efficient scaling through sparsity while maintaining strong performance. The shared experts provide a stable computation baseline while routed experts enable model capacity scaling.

The example below demonstrates how to generate text with AFMoE using [`Pipeline`] or the [`AutoModel`].

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="text-generation",
    model="arcee-ai/Trinity-Mini",
    torch_dtype=torch.bfloat16,
    device=0
)

output = pipeline("The key innovation in mixture of experts is")
print(output[0]["generated_text"])
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoTokenizer, AfmoeForCausalLM

tokenizer = AutoTokenizer.from_pretrained("arcee-ai/Trinity-Mini")
model = AfmoeForCausalLM.from_pretrained(
    "arcee-ai/Trinity-Mini",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

inputs = tokenizer("The key innovation in mixture of experts is", return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## Model Architecture Details

### Expert Routing

AFMoE uses token-choice routing where each token independently selects top-k experts based on router logits. The routing mechanism includes:

- Configurable scoring function (sigmoid or softmax)
- Optional route normalization for balanced expert utilization
- Route scaling to control expert contribution strength
- Bias correction for expert selection

### Shared Experts

Unlike standard MoE models, AFMoE includes shared experts that are always activated for every token, providing:

- A stable computation baseline across all tokens
- Reduced variance in model outputs
- Better handling of out-of-distribution inputs

### Attention Mechanism

The hybrid attention pattern alternates between:

- **Sliding Window Attention**: For efficiency on long sequences, with configurable window size
- **Full Attention**: Applied every N layers (configurable via `global_attn_every_n_layers`) for global context

All attention layers include Q/K normalization and output gating for improved training dynamics.

## AfmoeConfig

[[autodoc]] AfmoeConfig

## AfmoeModel

[[autodoc]] AfmoeModel
    - forward

## AfmoeForCausalLM

[[autodoc]] AfmoeForCausalLM
    - forward
