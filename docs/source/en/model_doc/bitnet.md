<!--Copyright 2025 The BitNet Team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# BitNet

## Overview

Trained on a corpus of 4 trillion tokens, this model demonstrates that native 1-bit LLMs can achieve performance comparable to leading open-weight, full-precision models of similar size, while offering substantial advantages in computational efficiency (memory, energy, latency).

➡️ **Technical Report:** [BitNet b1.58 2B4T Technical Report](https://huggingface.co/papers/2504.12285)

➡️ **Official Inference Code:** [microsoft/BitNet (bitnet.cpp)](https://github.com/microsoft/BitNet)

## Model Variants

Several versions of the model weights are available on Hugging Face:

* [**`microsoft/bitnet-b1.58-2B-4T`**](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T): Contains the packed 1.58-bit weights optimized for efficient inference. **Use this for deployment.**

* [**`microsoft/bitnet-b1.58-2B-4T-bf16`**](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-bf16): Contains the master weights in BF16 format. **Use this only for training or fine-tuning purposes.**

* [**`microsoft/bitnet-b1.58-2B-4T-gguf`**](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf): Contains the model weights in GGUF format, compatible with the `bitnet.cpp` library for CPU inference.


### Model Details


* **Architecture:** Transformer-based, modified with `BitLinear` layers (BitNet framework).
    * Uses Rotary Position Embeddings (RoPE).
    * Uses squared ReLU (ReLU²) activation in FFN layers.
    * Employs [`subln`](https://proceedings.mlr.press/v202/wang23u.html) normalization.
    * No bias terms in linear or normalization layers.
* **Quantization:** Native 1.58-bit weights and 8-bit activations (W1.58A8).
    * Weights are quantized to ternary values {-1, 0, +1} using absmean quantization during the forward pass.
    * Activations are quantized to 8-bit integers using absmax quantization (per-token).
    * **Crucially, the model was *trained from scratch* with this quantization scheme, not post-training quantized.**
* **Parameters:** ~2 Billion
* **Training Tokens:** 4 Trillion
*   **Context Length:** Maximum sequence length of **4096 tokens**.
    *   *Recommendation:* For optimal performance on tasks requiring very long contexts (beyond the pre-training length or for specialized long-reasoning tasks), we recommend performing intermediate long-sequence adaptation/training before the final fine-tuning stage.
* **Training Stages:**
    1.  **Pre-training:** Large-scale training on public text/code and synthetic math data using a two-stage learning rate and weight decay schedule.
    2.  **Supervised Fine-tuning (SFT):** Fine-tuned on instruction-following and conversational datasets using sum loss aggregation and specific hyperparameter tuning.
    3.  **Direct Preference Optimization (DPO):** Aligned with human preferences using preference pairs.
* **Tokenizer:** LLaMA 3 Tokenizer (vocab size: 128,256).


## Usage tips


**VERY IMPORTANT NOTE ON EFFICIENCY**

> Please do NOT expect performance efficiency gains (in terms of speed, latency, or energy consumption) when using this model with the standard transformers library.
>
> The current execution paths within transformers do not contain the specialized, highly optimized computational kernels required to leverage the advantages of the BitNet architecture. Running the model via transformers will likely result in inference speeds and energy usage comparable to, or potentially worse than, standard full-precision models within this framework on both CPU and GPU.
>
> While you might observe reduced memory usage due to the quantized weights, the primary computational efficiency benefits are not accessible through this standard transformers usage path.
>
> For achieving the efficiency benefits demonstrated in the technical paper, you MUST use the dedicated C++ implementation: [bitnet.cpp](https://github.com/microsoft/BitNet).

### Requirements

```bash
pip install transformers
```

### Example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "microsoft/bitnet-b1.58-2B-4T"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16
)

# Apply the chat template
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "How are you?"},
]
chat_input = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)

# Generate response
chat_outputs = model.generate(chat_input, max_new_tokens=50)
response = tokenizer.decode(chat_outputs[0][chat_input.shape[-1]:], skip_special_tokens=True) # Decode only the response part
print("\nAssistant Response:", response)
```


## BitNetConfig

[[autodoc]] BitNetConfig

## BitNetModel

[[autodoc]] BitNetModel
    - forward

## BitNetForCausalLM

[[autodoc]] BitNetForCausalLM
    - forward
