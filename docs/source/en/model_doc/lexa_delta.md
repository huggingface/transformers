# Lexa-Delta

## Overview

Lexa-Delta is a large language model developed by **Robi Labs**. It is based on the GPT-OSS architecture and provides state-of-the-art performance for causal language modeling tasks.

## Usage

### Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("RobiLabs/Lexa-Delta")
tokenizer = AutoTokenizer.from_pretrained("RobiLabs/Lexa-Delta")

inputs = tokenizer("Hello, Lexa!", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
```

### Configuration

The LexaDeltaConfig class allows you to customize the model architecture:

```python
from transformers import LexaDeltaConfig

config = LexaDeltaConfig(
    vocab_size=201088,
    hidden_size=2880,
    num_hidden_layers=24,
    num_attention_heads=64,
    intermediate_size=2880,
    max_position_embeddings=131072,
    rms_norm_eps=1e-05,
)
```

## Model Architecture

Lexa-Delta uses a transformer-based architecture with:
- **Hidden Size**: 2880
- **Number of Layers**: 24
- **Number of Attention Heads**: 64
- **Vocabulary Size**: 201088
- **Maximum Position Embeddings**: 131072

## Training

Lexa-Delta was trained on a large corpus of text data using advanced training techniques to achieve optimal performance.

## Citation

If you use Lexa-Delta in your research, please cite:

```bibtex
@misc{lexa-delta-2025,
  title={Lexa-Delta: A Large Language Model by Robi Labs},
  author={Robi Labs},
  year={2025},
  url={https://huggingface.co/RobiLabs/Lexa-Delta}
}
```
