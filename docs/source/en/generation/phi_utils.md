# Phi-Recursive Generation

## Overview
`PhiRecursiveGenerator` uses golden ratio (φ)-optimized sampling parameters for self-improving text generation.

## Usage
```python
from transformers import PhiRecursiveGenerator

generator = PhiRecursiveGenerator(model, tokenizer)
output = generator.generate("The future of AI is", max_iterations=3)
Parameters
max_iterations: Number of improvement cycles (default: 3)

verbose: Print progress (default: False)

temperature: Initial temperature (default: 0.7)

Mathematical Background
φ = 1.618... (golden ratio)

Parameters are optimized to maximize diversity and coherence.
