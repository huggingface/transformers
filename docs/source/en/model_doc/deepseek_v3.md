<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# DeepSeek-V3

## Overview

The DeepSeek-V3 model was proposed in [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) by DeepSeek-AI Team.

The abstract from the paper is the following:
We present DeepSeek-V3, a strong Mixture-of-Experts (MoE) language model with 671B total parameters with 37B activated for each token. To achieve efficient inference and cost-effective training, DeepSeek-V3 adopts Multi-head Latent Attention (MLA) and DeepSeekMoE architectures, which were thoroughly validated in DeepSeek-V2. Furthermore, DeepSeek-V3 pioneers an auxiliary-loss-free strategy for load balancing and sets a multi-token prediction training objective for stronger performance. We pre-train DeepSeek-V3 on 14.8 trillion diverse and high-quality tokens, followed by Supervised Fine-Tuning and Reinforcement Learning stages to fully harness its capabilities. Comprehensive evaluations reveal that DeepSeek-V3 outperforms other open-source models and achieves performance comparable to leading closed-source models. Despite its excellent performance, DeepSeek-V3 requires only 2.788M H800 GPU hours for its full training. In addition, its training process is remarkably stable. Throughout the entire training process, we did not experience any irrecoverable loss spikes or perform any rollbacks. The model checkpoints are available at https://github.com/deepseek-ai/DeepSeek-V3.

## Limitations and call for contribution!

We are super happy to make this code community-powered, and would love to see how you can best optimize the following: 

- current implementation uses the "naive" attention compution (so not really MLA)
- current implementation loops through the experts. This should be replaced. Pointers to use `get_packed_weights` from `intetrations/tensor_parallel`. 
- current implementation uses the eleuther formula for ROPE, using the orginal one would be more efficient! (should still follow our API)
- static cache is not supported (this should be just a generation config issue / config shape issues)

### Usage tips
The model uses Multi-head Latent Attention (MLA) and DeepSeekMoE architectures for efficient inference and cost-effective training. It employs an auxiliary-loss-free strategy for load balancing and multi-token prediction training objective. The model can be used for various language tasks after being pre-trained on 14.8 trillion tokens and going through Supervised Fine-Tuning and Reinforcement Learning stages.

You can run the model in `FP8` automatically, using 2 nodes of 8 H100 should be more than enough! 

```python
# `run_deepseek_v1.py`
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
torch.manual_seed(30)

tokenizer = AutoTokenizer.from_pretrained("deepseek-r1")

chat = [
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "I'd like to show off how chat templating works!"},
]


model = AutoModelForCausalLM.from_pretrained("deepseek-r1", device_map="auto", torch_dtype=torch.bfloat16)
inputs = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
import time
start = time.time()
outputs = model.generate(inputs, max_new_tokens=50)
print(tokenizer.batch_decode(outputs))
print(time.time()-start)
```
Use the following to run it
```bash
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0|1 --rdzv-id an_id --rdzv-backend c10d --rdzv-endpoint master_addr:master_port run_deepseek_r1.py
```

If you have: 
```bash
[rank0]: ncclInternalError: Internal check failed.
[rank0]: Last error:
[rank0]: Bootstrap : no socket interface found
```
error, it means NCCL was probably not loaded. 


## DeepseekV3Config

[[autodoc]] DeepseekV3Config

## DeepseekV3Model

[[autodoc]] DeepseekV3Model
    - forward

## DeepseekV3ForCausalLM

[[autodoc]] DeepseekV3ForCausalLM
    - forward
