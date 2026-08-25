# Lumma-0.6B-Base

### A multilingual language model optimized for efficient deployment and English–Indic language understanding.

**600M Parameters • 1 Trillion Training Tokens • 12,288 Context Length • Shared KV**

</p>

## Supported Languages

The model is trained on English and a diverse set of Indic languages, including:

English, Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia

---

# Overview

Lumma-0.6B-Base is a multilingual decoder-only language model trained from scratch on 1 trillion tokens. It is designed for efficient deployment, long-context inference, and strong multilingual performance across English and Indic languages, featuring a compact transformer architecture, memory-efficient attention mechanisms, and an optimized multilingual tokenizer.

---

# Key Features

- Trained from scratch on **1 trillion tokens**
- 600 million parameter decoder-only Transformer
- Native English and Indic language pretraining
- Shared KV Attention for memory-efficient inference
- 12,288 token context length
- Grouped Query Attention (GQA)
- RMSNorm with QK Normalization
- SwiGLU feed-forward network
- Factorized tied embeddings
- Large multilingual tokenizer optimized for Indic languages

---

> [!NOTE]
> We do not recommend using base language models for conversations. Instead, you can apply post-training, e.g., SFT, RLHF, continued pretraining, etc., on this model.


# Shared KV 

Lumma introduces **Shared KV**, an alternative key-value caching strategy designed to reduce inference memory requirements without significantly impacting model quality.

Instead of computing independent Key and Value projections, both are derived from a shared latent representation. During attention computation, lightweight Key normalization and RoPE transformations are applied dynamically.

This approach reduces KV-cache memory usage by approximately **50%**, making Lumma better suited for long-context inference and memory-constrained deployments.

<p align="center">
  <img src="./shared_kv_cache_comparison_improved.png" width="650"/>
</p>

---

# KV Cache Modes

Lumma supports two inference modes depending on deployment requirements.

## Shared KV

```python
model.config.kv_cache_mode = "shared"
```

Recommended when memory is the primary bottleneck.

- Approximately 50% lower KV-cache memory
- Slightly higher compute overhead
- Better suited for long-context inference

---

## Vanilla KV

```python
model.config.kv_cache_mode = "vanilla"
```

Recommended for standard deployments.

- Standard KV-cache implementation
- Lower compute overhead
- Maximum compatibility across inference frameworks

---


# Benchmark Results

The following results correspond to the released Lumma-0.6B model trained on **1 trillion tokens**.

## General Benchmarks

<div align="center">

<table>
  <thead>
    <tr>
      <th>Model Name</th>
      <th>Tokens Budget<br>(Trillion)</th>
      <th>HellaSwag</th>
      <th>Winogrande</th>
      <th>OBQA</th>
      <th>ARC-e</th>
      <th>ARC-c</th>
      <th>Average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>MobiLlama-0.5B-Base</td>
      <td align="center">1.3</td>
      <td align="center">39.65</td>
      <td align="center">53.67</td>
      <td align="center">30.60</td>
      <td align="center">52.82</td>
      <td align="center">23.63</td>
      <td align="center">40.07</td>
    </tr>
    <tr>
      <td>Qwen-2-0.5-Base</td>
      <td align="center">12</td>
      <td align="center">49.01</td>
      <td align="center">57.69</td>
      <td align="center">33.20</td>
      <td align="center">54.79</td>
      <td align="center">25.42</td>
      <td align="center">44.02</td>
    </tr>
    <tr>
      <td>Qwen2.5-0.5B-Base</td>
      <td align="center">18</td>
      <td align="center">52.16</td>
      <td align="center">56.82</td>
      <td align="center">35.40</td>
      <td align="center">64.64</td>
      <td align="center">29.86</td>
      <td align="center">47.78</td>
    </tr>
    <tr>
      <td><strong>Lumma-0.6B-Base</strong></td>
      <td align="center"><strong>1</strong></td>
      <td align="center">46.25</td>
      <td align="center">54.14</td>
      <td align="center">32.80</td>
      <td align="center">60.60</td>
      <td align="center">28.58</td>
      <td align="center">44.47</td>
    </tr>
  </tbody>
</table>

</div>

---

# Multilingual Tokenization

Efficient tokenization is particularly important for multilingual language models.

Lower fertility indicates fewer tokens are required to represent text, improving both training efficiency and inference cost.

| Language | SmolLM3-3B | Qwen3-0.6B | Sarvam-1 | Lumma-0.6B |
|-----------|------------|------------|-----------|------------|
| English | 1.17 | 1.16 | 1.32 | **1.18** |
| Bengali | 8.66 | 7.51 | 1.55 | **1.44** |
| Gujarati | 10.47 | 9.37 | 1.55 | **1.53** |
| Hindi | 2.71 | 5.14 | **1.25** | 1.32 |
| Kannada | 16.43 | 12.96 | 2.10 | **1.90** |
| Malayalam | 17.77 | 14.56 | 2.49 | **2.05** |
| Marathi | 3.73 | 6.70 | 1.55 | **1.55** |
| Odia | 19.07 | 15.75 | **2.18** | 2.68 |
| Punjabi | 9.23 | 8.66 | 1.47 | **1.42** |
| Tamil | 13.56 | 10.93 | 2.06 | **2.05** |
| Telugu | 15.40 | 13.38 | 2.09 | **1.77** |
| Assamese | 9.26 | 8.13 | 4.31 | **1.51** |

---

# Usage

```python
!pip install transformers=='5.4.0'


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "FrontiersMind/Lumma-0.6B-Base"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    dtype=torch.bfloat16
).eval()

# Memory-efficient mode
model.config.kv_cache_mode = "shared"

# Standard mode
# model.config.kv_cache_mode = "vanilla"

prompt = "The world is a strange place"

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.3,
    top_p=0.95,
    top_k=20,
    repetition_penalty=1.1,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

# Citation

```bibtex
@misc{lumma2026,
  title={Lumma-0.6B},
  author={FrontiersMind},
  year={2026},
  url={https://huggingface.co/FrontiersMind/Lumma-0.6B}
}
```