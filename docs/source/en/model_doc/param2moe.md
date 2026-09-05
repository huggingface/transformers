<!--Copyright 2026 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-08-25.*

# Param2Moe

## Overview

Param2Moe was released by the [BharatGen AI](https://huggingface.co/bharatgenai) team as **Param-2-17B-MoE-A2.4B**, a Hybrid Mixture-of-Experts language model with 17B total parameters and only 2.4B active per token. It is pretrained from scratch on ~22 trillion tokens across two phases, with an emphasis on linguistic diversity — supporting English, Hindi, and 21 Indian languages. The model ships as an early post-training checkpoint with reasoning, tool calling, math, and code capabilities.

The original model can be found [here](https://huggingface.co/bharatgenai/Param2-17B-A2.4B-Thinking).

Tips:

- **Expert routing**: 2 shared experts are always active alongside 6 dynamically routed experts selected from a pool of 64 per token.
- **Memory**: Loading the full model requires ~34 GB VRAM in bfloat16.
- **Context**: Maximum supported length is 4096 tokens.
- **Custom code**: Pass `trust_remote_code=True` when loading with `AutoModelForCausalLM`.
- **Decoding**: Set `skip_special_tokens=False` to preserve `<think>...</think>` reasoning tags in Thinking checkpoint outputs. Use `do_sample=False` for deterministic/evaluation runs.
- **Safety**: The model has not undergone RLHF or safety alignment — fine-tune and evaluate before production use.

## Architecture

Param2Moe's modular implementation is composed almost entirely of existing building blocks, with only config-level differences:

- **MoE block, router, and experts** are inherited unchanged from **DeepSeek-V3** (`DeepseekV3MoE`, `DeepseekV3TopkRouter`, `DeepseekV3Experts`). This gives Param2Moe the same sigmoid-scored, grouped top-k routing with an expert-bias correction term and shared experts added on top of the routed output. The only differences from DeepSeek-V3 are in the config values: Param2Moe routes to 6 of 64 experts per token (vs. DeepSeek-V3's 8 of 256) with `n_group=1` (i.e. no grouping restricts which experts can be chosen — DeepSeek-V3 uses `n_group=8`/`topk_group=4`), and uses 2 shared experts (vs. DeepSeek-V3's 1).
- **Attention** is inherited unchanged from **Qwen3-MoE** (`Qwen3MoeAttention`), not from DeepSeek-V3. This means Param2Moe uses standard multi-head attention with per-head QK RMSNorm (`q_norm`/`k_norm`), rather than DeepSeek-V3's Multi-head Latent Attention (MLA) with its low-rank Q/KV projections. There is no `q_lora_rank`/`kv_lora_rank` in `Param2MoeConfig` because of this.
- **Decoder layer, RMSNorm, rotary embedding, dense MLP, and causal-LM wrapper** follow the standard **Llama/Mixtral** pattern used across the MoE model family (`LlamaDecoderLayer`, `LlamaRMSNorm`, `LlamaRotaryEmbedding`, `MixtralForCausalLM`, `MixtralModel`), the same as Qwen3-MoE and DeepSeek-V3 itself.

In short: Param2Moe = DeepSeek-V3's MoE/routing + Qwen3-MoE's attention, glued together with the shared Llama/Mixtral scaffolding, at a smaller scale (17B total / 2.4B active).

## Usage examples

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from parsers import parse_model_output

model_name = "bharatgenai/Param2-17B-A2.4B-Thinking"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=False
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto"
)

conversation = [
    {"role": "system", "content": "You are helpful assistant."},
    {"role": "user", "content": "What is the BharatGen Mission?"}
]

inputs = tokenizer.apply_chat_template(
    conversation=conversation,
    return_tensors="pt",
    add_generation_prompt=True
).to(model.device)

with torch.no_grad():
    output = model.generate(
        inputs,
        max_new_tokens=300,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=False,
    )

generated_tokens = output[0][inputs.shape[-1]:]

# IMPORTANT: skip_special_tokens=False
generated_text = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=False
)

parsed = parse_model_output(generated_text)

print("\n========== RAW ==========\n", generated_text)
print("\n========== REASONING ==========\n", parsed["reasoning"])
print("\n========== TOOL CALLS ==========\n", parsed["tool_calls"])
print("\n========== FINAL ANSWER ==========\n", parsed["final_answer"])
```

## Param2MoeConfig

[[autodoc]] Param2MoeConfig

## Param2MoePreTrainedModel

[[autodoc]] Param2MoePreTrainedModel
    - forward

## Param2MoeModel

[[autodoc]] Param2MoeModel
    - forward

## Param2MoeForCausalLM

[[autodoc]] Param2MoeForCausalLM
    - forward
