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

*This model was released on 2024-12-13 and added to Hugging Face Transformers on 2025-10-07 and contributed by [itazap](https://huggingface.co/itazap).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Byte Latent Transformer (BLT)

[Byte Latent Transformer](https://huggingface.co/papers/2412.09871) is a byte-level LLM architecture that matches tokenization-based LLM performance at scale. It encodes bytes into dynamically sized patches based on entropy, optimizing compute and model capacity where data complexity is higher. This approach improves inference efficiency and robustness, with the first flop-controlled scaling study up to 8B parameters and 4T training bytes. BLT demonstrates better scaling than tokenization-based models by dynamically selecting long patches for predictable data, enhancing reasoning and long-tail generalization.

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="microsoft/BLT-1B", dtype="auto")
pipeline("The future of artificial intelligence is")
```

## Usage Tips

- **Dual Model Architecture**: BLT consists of two separate trained models:
  - **Patcher (Entropy Model)**: A smaller transformer model that predicts byte-level entropy to determine patch boundaries and segment input.
  - **Main Transformer Model**: The primary model that processes the patches through a Local Encoder, Global Transformer, and Local Decoder.

- **Dynamic Patching**: The model uses entropy-based dynamic patching where:
  - High-entropy regions (complex data) get shorter patches with more computational attention
  - Low-entropy regions (predictable data) get longer patches for efficiency
  - This allows the model to allocate compute resources where they're most needed

- **Local Encoder**: Processes byte sequences with cross-attention to patch embeddings
- **Global Transformer**: Processes patch-level representations with full attention across patches
- **Local Decoder**: Generates output with cross-attention back to the original byte sequence

- **Byte-Level Tokenizer**: Unlike traditional tokenizers that use learned vocabularies, BLT's tokenizer simply converts text to UTF-8 bytes and maps each byte to a token ID. There is no need for a vocabulary.

The model can be loaded via:

<hfoption id="AutoModel">

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("itazap/blt-1b-hf")
model = AutoModelForCausalLM.from_pretrained(
    "itazap/blt-1b-hf",
    device_map="auto",
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

prompt = "my name is"
generated_ids = model.generate(
    **inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, use_cache=False
)

print(tokenizer.decode(generated_ids[0]))
```

</hfoption>

This model was contributed by [itazap](https://huggingface.co/<itazap>).
The original code can be found [here](<https://github.com/facebookresearch/blt>).

## BltConfig

[[autodoc]] BltConfig

[[autodoc]] BltModel
    - forward

## BltForCausalLM

[[autodoc]] BltForCausalLM
    - forward

