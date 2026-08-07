<!--Copyright 2026 The Upstage Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was published in HF papers on 2026-07-24 and contributed to Hugging Face Transformers on 2026-08-07.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# SolarOpen2

## Overview

The SolarOpen2 model was proposed in [Solar Open 2 Technical Report](https://huggingface.co/papers/2607.20062) by Upstage Team.

SolarOpen2 is the second model in the Solar Open series of open-weights LLMs created by
[Upstage](https://huggingface.co/upstage), following
[Solar Open 1](https://huggingface.co/upstage/Solar-Open-100B). It is released as
[upstage/Solar-Open2-250B](https://huggingface.co/upstage/Solar-Open2-250B).

The abstract from the paper is the following:

*We present Solar Open 2, a 250B-A15B Mixture-of-Experts language model built for long-horizon agentic tasks, scaled
up from Solar Open 1 (Solar Open 100B). To hold entire agent trajectories in a single context, Solar Open 2 reaches a
1M-token window through a hybrid attention stack that interleaves one softmax layer among every three linear-attention
layers, using no positional encoding and a gated delta rule extended to negative eigenvalues. To train at this scale
under a fixed compute budget, we make training efficient in two ways: a stronger starting point, and higher-value
data. For the starting point, we initialize Solar Open 2 from Solar Open 1, transferring the 5.69B-parameter shared
skeleton that survives the architectural change and learning everything else through full pre-training. For the data,
we curate for value per token: quality- and rarity-aware data curation and mixture-ratio optimization refine a 20T
pool into a 10T mixture that, at equal token budget, outperforms the Solar Open 1 recipe. To build its agent skills,
we train twelve domain specialists across purpose-built scenarios, then consolidate them into a single model by
Multi-teacher On-Policy Distillation (MOPD). Against comparably sized open-weight models on English benchmarks, Solar
Open 2 leads on MMLU-Pro, LiveCodeBench, and the APEX-Agents agentic suite, and stays competitive with the strongest
(DeepSeek-V4-Flash and MiMo-V2.5) elsewhere. On Korean benchmarks, Solar Open 2 records the highest average of any
model compared, including fast-tier closed APIs, and on Ko-GDPval, an in-house Korean officework-agent benchmark, it
is competitive with DeepSeek-V4-Pro (1.6T) at less than a sixth of its size.*

This model was contributed by [SSON9](https://huggingface.co/SSON9) from [Upstage](https://huggingface.co/upstage).

## Usage Tips

Recommended generation parameters:

```
temperature=1.0
top_p=1.0
```

Set `max_new_tokens` high enough, as it covers both the reasoning trace and the final output — if it is too low, long reasoning traces can truncate the answer.
We recommend up to 128K output tokens for non-reasoning requests and up to 256K for reasoning requests.

The linear-attention layers use fast kernels from the [FLA library](https://github.com/fla-org/flash-linear-attention) when `fla-core` is installed and a CUDA device is available, and fall back to a pure PyTorch reference implementation otherwise.

**Examples**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "upstage/Solar-Open2-250B"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=MODEL_ID,
    dtype="bfloat16",
    device_map="auto",
)

# Prepare input
messages = [{"role": "user", "content": "who are you?"}]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
    reasoning_effort="high",  # reasoning mode (default); use "none" for non-reasoning mode
)
inputs = inputs.to(model.device)

# Generate response
generated_ids = model.generate(
    **inputs,
    max_new_tokens=16384,
    temperature=1.0,
    top_p=1.0,
    do_sample=True,
)
generated_text = tokenizer.decode(generated_ids[0][inputs.input_ids.shape[1] :])
print(generated_text)
```

## SolarOpen2Config

[[autodoc]] SolarOpen2Config

## SolarOpen2Model

[[autodoc]] SolarOpen2Model
    - forward

## SolarOpen2ForCausalLM

[[autodoc]] SolarOpen2ForCausalLM
    - forward
