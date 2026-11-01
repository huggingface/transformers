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
*This model was contributed to Hugging Face Transformers on 2026-10-23.*


# LFM3

## Overview

LFM3 is the third generation of Liquid Foundation Models by [Liquid AI](https://liquid.ai/). Like LFM2 it is a hybrid
of gated short convolutions and grouped-query attention, here split between full-attention and sliding-window layers.

## Architecture

LFM3 stacks three kinds of layers: gated short-convolution layers, full-attention layers and sliding-window attention
layers. The attention layers apply an RMSNorm to the query and key heads and gate their output with a sigmoid per head.
Full-attention layers use partial rotary embeddings, sliding-window layers use their own RoPE base, and every layer
applies a sandwich norm to the operator and feed-forward outputs before the residual addition.

Packed documents follow the LFM2 contract: the flattening collator's `seq_idx` keeps the short-conv window inside each
document and the attention masks come from the mask utilities. The short convolution uses the `causal-conv1d` kernel
from the Hub when kernels are enabled and the PyTorch implementation otherwise.

## Example

The following example shows how to generate an answer using the `AutoModelForCausalLM` class.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_id = "LiquidAI/LFM3-1.7B-Preview"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    dtype="bfloat16",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Generate answer
prompt = "What is C. elegans?"
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    return_tensors="pt",
    tokenize=True,
).to(model.device)

output = model.generate(
    **inputs,
    do_sample=True,
    temperature=0.3,
    min_p=0.15,
    repetition_penalty=1.05,
    max_new_tokens=512,
)

print(tokenizer.decode(output[0], skip_special_tokens=False))
```

## Lfm3Config

[[autodoc]] Lfm3Config

## Lfm3Model

[[autodoc]] Lfm3Model
    - forward

## Lfm3ForCausalLM

[[autodoc]] Lfm3ForCausalLM
    - forward
