<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->
*This model was contributed to Hugging Face Transformers on 2026-10-05.*

# K2 Horizon

[K2-Horizon-0.9B](https://huggingface.co/IFM/K2-Horizon-0.9B) is a dense, decoder-only language model from the K2 Horizon family. It uses grouped-query attention, RMS normalization, a gated SiLU feed-forward network, and YaRN rotary position embeddings for a context window of 131,072 tokens.

This implementation also supports [K2-Horizon-375B-A23B](https://huggingface.co/IFM/K2-Horizon-375B-A23B), with routed and shared MLP experts, and [K2-Horizon-MoVA-36B-A4B](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B), which additionally routes tokens through value-projection experts. Their existing checkpoint tensor names load directly without conversion.

## Usage

Load the checkpoint with [`AutoTokenizer`] and [`AutoModelForCausalLM`]. The model and tokenizer load directly with Transformers; no `trust_remote_code` argument is needed.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "IFM/K2-Horizon-MoVA-36B-A4B"  # or IFM/K2-Horizon-375B-A23B / IFM/K2-Horizon-0.9B
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
    trust_remote_code=False,
)

messages = [{"role": "user", "content": "Explain why the sky is blue."}]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    reasoning_effort="high",
    return_dict=True,
    return_tensors="pt",
).to(model.device)
inputs.pop("token_type_ids", None)

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
    )

generated_ids = output[0, inputs["input_ids"].shape[-1]:]
print(tokenizer.decode(generated_ids, skip_special_tokens=True))
```

The short generation budget above is for demonstration. Reasoning responses can require substantially more tokens; consult the selected checkpoint's model card for evaluation settings. The large BF16 checkpoints require enough aggregate device and host memory for all expert weights plus runtime state. Native support includes eager and SDPA attention, KV caching, optional router logits, and the router auxiliary loss during training.

## FP8 checkpoints

The native model also loads `IFM/K2-Horizon-375B-A23B-FP8`,
`IFM/K2-Horizon-MoVA-36B-A4B-FP8`, and `IFM/K2-Horizon-7B-FP8` with
`trust_remote_code=False`. The loader reads the quantization backend from each
checkpoint's configuration. Install `compressed-tensors` for 375B and 7B, and
`kernels` for fine-grained MoVA FP8 execution on a compatible GPU.

375B and MoVA quantize routed MLP expert projections. Attention, routers, shared
experts, and MoVA value projections remain BF16. The dense 7B release quantizes
attention and MLP projections while retaining a BF16 language-model head.

The existing compressed-tensors backend expands 375B/7B weights for computation;
this does not provide persistent block-FP8 kernel execution. Its optional
`use_optimized_inference=True` row-wise path does not support these block scales.
MoVA uses the existing fine-grained FP8 backend, which dequantizes on CPU.
To explicitly dequantize a checkpoint during loading:

```python
from transformers import AutoConfig, AutoModelForCausalLM

model_id = "IFM/K2-Horizon-7B-FP8"
config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
config.quantization_config["dequantize"] = True
model = AutoModelForCausalLM.from_pretrained(
    model_id, config=config, dtype="bfloat16", device_map="auto", trust_remote_code=False
)
```

Dequantization requires enough memory for the expanded weights. All FP8
adaptations are scoped to the K2 configuration and use the existing quantizers.

## K2HorizonConfig

[[autodoc]] K2HorizonConfig

## K2HorizonModel

[[autodoc]] K2HorizonModel
    - forward

## K2HorizonForCausalLM

[[autodoc]] K2HorizonForCausalLM
    - forward
