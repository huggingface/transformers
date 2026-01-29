# HumanV

## Overview

HumanV is a lightweight Transformer-based architecture integrated into the Hugging Face Transformers library. The initial implementation focuses on a decoder-only language model (causal LM) with rotary position embeddings and a modular attention setup. Task-specific heads for sequence classification, token classification, and question answering are also provided for future research and experimentation.

## Usage

### Load a pretrained model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "nebularesearchtrain/nilla-story"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

inputs = tokenizer("Once upon a time", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
````

### Initialize from config

```python
from transformers import HumanVConfig, HumanVForCausalLM

config = HumanVConfig(
    vocab_size=50257,
    hidden_size=256,
    intermediate_size=1024,
    num_hidden_layers=8,
    num_attention_heads=8,
    num_key_value_heads=8,
    head_dim=32,
    max_position_embeddings=1024,
    layer_types=["full_attention"] * 8,
    use_cache=False,
)

model = HumanVForCausalLM(config)
```

## HumanVConfig

[[autodoc]] HumanVConfig

## HumanVModel

[[autodoc]] HumanVModel

## HumanVForCausalLM

[[autodoc]] HumanVForCausalLM

## HumanVForSequenceClassification

[[autodoc]] HumanVForSequenceClassification

## HumanVForTokenClassification

[[autodoc]] HumanVForTokenClassification

## HumanVForQuestionAnswering

[[autodoc]] HumanVForQuestionAnswering

```
::contentReference[oaicite:0]{index=0}