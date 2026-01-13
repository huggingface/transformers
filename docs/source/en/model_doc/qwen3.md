<!--Copyright 2024 The Qwen Team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2025-04-29 and added to Hugging Face Transformers on 2025-03-31.*

# Qwen3

## Overview

[Qwen3](https://huggingface.co/papers/2505.09388) refers to the dense model architecture Qwen3-32B which was released with its mixture of experts variant [Qwen3MoE](qwen3_moe) ([blog post](https://qwenlm.github.io/blog/qwen3/)).

### Model Details

Qwen3 is the dense 32B parameter variant in the Qwen3 series. Key architectural improvements include:

- **Extended Context Length**: Supports up to 128K tokens context window
- **Enhanced Architecture**: Uses GQA (Grouped Query Attention) for improved efficiency
- **Dual Attention Mechanism**: Similar to Qwen2.5, alternates between local sliding window attention and global attention layers
- **Improved Training**: Post-trained with RLHF and advanced instruction tuning

Qwen3-32B is available in both base (`Qwen/Qwen3-32B`) and instruction-tuned (`Qwen/Qwen3-32B-Instruct`) variants.

## Usage tips

### Basic Text Generation

Here's how to use Qwen3 for text generation:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-32B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B-Instruct")

# Basic generation
prompt = "Explain quantum computing in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.8,
    do_sample=True
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Chat Format

For the instruction-tuned variant, use the chat template for multi-turn conversations:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-32B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B-Instruct")

messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "What are the main differences between Python and JavaScript?"}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer([text], return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7
)

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

### Memory Optimization

For systems with limited GPU memory, use quantization:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# 4-bit quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-32B-Instruct",
    quantization_config=quantization_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B-Instruct")

# Use as normal - memory footprint is significantly reduced
```

### Long Context Usage

Qwen3 supports up to 128K tokens. For long documents:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-32B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2"  # Recommended for long contexts
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B-Instruct")

# Process long document
long_document = "..." * 10000  # Your long text here
prompt = f"Summarize the following document:\n\n{long_document}"

inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(model.device)

# Note: Ensure your input doesn't exceed 128K tokens
print(f"Input tokens: {inputs.input_ids.shape[1]}")

outputs = model.generate(**inputs, max_new_tokens=500)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(summary)
```

### Tips for Best Performance

- **Use `torch.bfloat16`**: Provides best balance of speed and quality
- **Enable Flash Attention 2**: Significantly faster for long contexts (`attn_implementation="flash_attention_2"`)
- **Batch Processing**: Process multiple inputs together for better throughput
- **Temperature Tuning**: Lower (0.1-0.5) for factual tasks, higher (0.7-1.0) for creative tasks
- **System Prompt**: Use clear system prompts for instruction-tuned variant to guide behavior

## Qwen3Config

[[autodoc]] Qwen3Config

## Qwen3Model

[[autodoc]] Qwen3Model
    - forward

## Qwen3ForCausalLM

[[autodoc]] Qwen3ForCausalLM
    - forward

## Qwen3ForSequenceClassification

[[autodoc]] Qwen3ForSequenceClassification
    - forward

## Qwen3ForTokenClassification

[[autodoc]] Qwen3ForTokenClassification
    - forward

## Qwen3ForQuestionAnswering

[[autodoc]] Qwen3ForQuestionAnswering
    - forward
