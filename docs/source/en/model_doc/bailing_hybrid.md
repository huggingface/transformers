<!--Copyright 2026 the HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->
*This model was contributed to Hugging Face Transformers on 2026-10-07.*

# Ling 3.0

## Overview

Ling 3.0 Flash is a hybrid Mixture-of-Experts language model from InclusionAI. Its decoder groups five Kimi Delta
Attention (KDA) layers with one Multi-head Latent Attention (MLA) layer. The MLA blocks use interleaved rotary
embeddings and a head-wise output gate, while the feed-forward blocks combine routed and shared experts.

The native Transformers architecture is exposed as `BailingHybridForCausalLM` and uses the checkpoint model type
`bailing_hybrid`. Native support means the model can be loaded without `trust_remote_code=True`.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer


model_id = "inclusionAI/Ling-3.0-flash"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", dtype="auto")

messages = [{"role": "user", "content": "Introduce the Ling model family."}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to(model.device)
output = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True))
```

The KDA layers include a pure PyTorch fallback. If `flash-linear-attention` and `causal-conv1d` are installed, their
optimized kernels are used automatically on CUDA. Installing `kernels` and passing `use_kernels=True` to
`from_pretrained` can enable compatible Hub kernels when available.

## BailingHybridConfig

[[autodoc]] BailingHybridConfig

## BailingHybridModel

[[autodoc]] BailingHybridModel
    - forward

## BailingHybridForCausalLM

[[autodoc]] BailingHybridForCausalLM
    - forward
