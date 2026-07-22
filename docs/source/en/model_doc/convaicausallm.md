# ConvaiCausalLM

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-EE4C2C?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

ConvaiCausalLM is a decoder-only transformer model developed by Convai Innovations, specifically pre-trained for Hindi causal language modeling. It is designed to generate coherent and contextually relevant Hindi text.

This implementation leverages the **Llama architecture** within Hugging Face Transformers, incorporating features like **Rotary Positional Embeddings (RoPE)** using `LlamaRotaryEmbedding`, **RMS Normalization** via `LlamaRMSNorm`, and **Grouped-Query Attention (GQA)** based on `LlamaAttention`. The core building blocks like the MLP layer (`LlamaMLP`) are also inherited.

The reference checkpoint for this model can be found on the Hugging Face Hub at [convaiinnovations/hindi-causal-lm](https://huggingface.co/convaiinnovations/hindi-causal-lm).

This model was contributed by [NandhaKishorM](https://huggingface.co/NandhaKishorM).

Key architectural features:
-   **Base Architecture:** Inherits from Llama implementation.
-   **Normalization:** RMSNorm applied before attention and MLP blocks.
-   **Positional Embeddings:** Rotary Positional Embeddings (RoPE).
-   **Attention:** Grouped-Query Attention (GQA). The reference checkpoint uses 16 query heads and 4 key-value heads.
-   **Activation Function:** SiLU (SwiGLU).
-   **Vocabulary:** SentencePiece tokenizer with 16,000 tokens.

## Usage Tips

The model and tokenizer can be loaded via the standard `Auto*` classes:

```python
# pip install transformers sentencepiece torch accelerate

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "convaiinnovations/hindi-causal-lm"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

# Example generation
prompt_hindi = "भारत की अर्थव्यवस्था" # "India's economy"
messages = [{"role": "user", "content": prompt_hindi}] # Use a simple prompt structure

# Note: This model might not have a specific chat template.
# Using a basic prompt structure directly. Adjust if a template exists.
# inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
# Alternative for non-chat models:
inputs = tokenizer(prompt_hindi, return_tensors="pt").to(device)


# Generate text continuation
gen_tokens = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id # Important for generation
)

gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
print(f"Prompt: {prompt_hindi}")
print(f"Generated: {gen_text}")
# Example output might be: भारत की अर्थव्यवस्था तेजी से बढ़ रही है और यह दुनिया की पांचवीं सबसे बड़ी अर्थव्यवस्था बन गई है। कृषि, उद्योग और सेवा क्षेत्र इसके मुख्य आधार हैं।
```
*Note: If running locally from a modified clone, ensure the library is installed (`pip install -e .`) after placing the model files correctly.*

## Resources
-   Model Hub: [convaiinnovations/hindi-causal-lm](https://huggingface.co/convaiinnovations/hindi-causal-lm)
-   Llama Paper (for architectural reference): [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)

## ConvaiCausalLMConfig

[[autodoc]] ConvaiCausalLMConfig

## ConvaiCausalLMTokenizer

[[autodoc]] ConvaiCausalLMTokenizer

## ConvaiCausalLMModel

[[autodoc]] ConvaiCausalLMModel
    - forward

## ConvaiCausalLMForCausalLM

[[autodoc]] ConvaiCausalLMForCausalLM
    - forward
