# Rish AI

## Model Description

Rish AI is a cutting-edge Mixture of Experts (MoE) transformer model designed for efficient and scalable language understanding and generation. It features sparse routing with 7 experts per token, advanced rotary position embeddings, and optimized attention mechanisms.

## Key Features

- **Sparse Mixture of Experts**: 7 experts with 5 experts activated per token for optimal efficiency
- **Rotary Position Embeddings**: Dynamic RoPE scaling for better long-context handling
- **Grouped Query Attention**: Efficient attention with reduced key/value heads
- **RMSNorm**: Improved normalization for stable training
- **Load Balancing**: Automatic expert load balancing during training

## Usage

### Installation

```bash
pip install transformers
```

### Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_name = "RishAILabs/RLLM-Base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Prepare input
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")

# Generate response
outputs = model.generate(**inputs, max_length=50, do_sample=True, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Advanced Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model with specific configuration
model = AutoModelForCausalLM.from_pretrained(
    "RishAILabs/RLLM-Base",
    torch_dtype=torch.bfloat16,  # For memory efficiency
    device_map="auto"  # Automatic device placement
)

tokenizer = AutoTokenizer.from_pretrained("your-org/RishAI-1B-7B")

# Multi-turn conversation
conversation = [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is a subset of AI..."},
    {"role": "user", "content": "Can you give a practical example?"}
]

# Format conversation
formatted_input = tokenizer.apply_chat_template(conversation, tokenize=False)
inputs = tokenizer(formatted_input, return_tensors="pt")

# Generate with controlled parameters
outputs = model.generate(
    **inputs,
    max_length=200,
    temperature=0.8,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Model Configuration

```python
from transformers import RishAIConfig

# Create custom configuration
config = RishAIConfig(
    vocab_size=100352,
    hidden_size=4096,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_experts=7,           # Number of experts
    num_experts_per_tok=5,   # Experts activated per token
    max_position_embeddings=4096,
    rope_scaling={"rope_type": "dynamic", "factor": 1.0}
)

# Initialize model with config
from transformers import RishAIModel
model = RishAIModel(config)
```

## Model Architecture

### Sparse Mixture of Experts (MoE)
- **Experts**: 7 specialized sub-networks
- **Routing**: Top-5 expert selection per token
- **Load Balancing**: Automatic expert utilization optimization

### Attention Mechanism
- **Grouped Query Attention**: Efficient key/value head reduction
- **Rotary Embeddings**: Position-aware attention with dynamic scaling
- **RMSNorm**: Stable layer normalization

### Training Features
- **Gradient Checkpointing**: Memory-efficient training
- **Flash Attention**: Optimized attention computation
- **Expert Parallelism**: Distributed expert training

## Performance

### Speed
- **Inference**: Optimized for fast generation
- **Training**: Efficient MoE routing and load balancing
- **Memory**: Sparse activation reduces memory footprint

### Quality
- **Perplexity**: Competitive with state-of-the-art models
- **Long Context**: Effective handling of 4K+ token sequences
- **Multitask**: Strong performance across diverse tasks

## Limitations

- Requires significant computational resources for training
- Memory usage scales with number of active experts
- Best performance on modern GPUs with ample VRAM

## Citation

```bibtex
@misc{rishailabs_2026,
    author       = { RishAILabs },
    title        = { RLLM-Base (Revision 552ee30) },
    year         = 2026,
    url          = { https://huggingface.co/RishAILabs/RLLM-Base },
    doi          = { 10.57967/hf/7560 },
    publisher    = { Hugging Face }
}
```

## License

This model is released under the Apache 2.0 license.
