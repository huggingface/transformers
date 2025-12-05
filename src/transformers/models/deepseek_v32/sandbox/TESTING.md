# Testing DeepSeek V3.2 Implementation

This document describes how to test the DeepSeek V3.2 model implementation.

## Prerequisites

```bash
# Activate the virtual environment
source .venv/bin/activate

# Ensure transformers is installed in development mode
pip install -e .
```

## Test 1: Configuration Loading

Verify the configuration class can load the official DeepSeek V3.2 config:

```python
from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
import json

# Load the official config.json
with open('src/transformers/models/deepseek_v32/deepseek_files/config.json') as f:
    official_config = json.load(f)

# Instantiate config
config = DeepseekV32Config(**official_config)

# Verify key parameters
print(f'vocab_size: {config.vocab_size}')           # Expected: 129280
print(f'hidden_size: {config.hidden_size}')         # Expected: 7168
print(f'num_hidden_layers: {config.num_hidden_layers}')  # Expected: 61
print(f'num_attention_heads: {config.num_attention_heads}')  # Expected: 128
print(f'n_routed_experts: {config.n_routed_experts}')  # Expected: 256
print(f'scoring_func: {config.scoring_func}')       # Expected: sigmoid
print(f'index_topk: {config.index_topk}')           # Expected: 2048
print(f'rope_scaling type: {config.rope_scaling.get("type")}')  # Expected: yarn
```

## Test 2: Small Model Forward Pass

Test model instantiation and forward pass with a small configuration:

```python
import torch
from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32ForCausalLM

# Create a small test configuration (fits in memory)
config = DeepseekV32Config(
    vocab_size=1000,
    hidden_size=256,
    intermediate_size=512,
    moe_intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    q_lora_rank=64,
    kv_lora_rank=32,
    qk_nope_head_dim=32,
    qk_rope_head_dim=16,
    v_head_dim=32,
    n_routed_experts=4,
    n_shared_experts=1,
    num_experts_per_tok=2,
    n_group=2,
    topk_group=1,
    index_n_heads=4,
    index_head_dim=32,
    index_topk=128,
    first_k_dense_replace=1,
)

# Create model
model = DeepseekV32ForCausalLM(config)
print(f'Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

# Test forward pass
input_ids = torch.randint(0, 1000, (1, 16))
with torch.no_grad():
    outputs = model(input_ids)

print(f'Output logits shape: {outputs.logits.shape}')  # Expected: [1, 16, 1000]
```

## Test 3: Generation (Optional)

Test text generation with the small model:

```python
import torch
from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32ForCausalLM

config = DeepseekV32Config(
    vocab_size=1000,
    hidden_size=256,
    intermediate_size=512,
    moe_intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    q_lora_rank=64,
    kv_lora_rank=32,
    qk_nope_head_dim=32,
    qk_rope_head_dim=16,
    v_head_dim=32,
    n_routed_experts=4,
    n_shared_experts=1,
    num_experts_per_tok=2,
    n_group=2,
    topk_group=1,
    index_n_heads=4,
    index_head_dim=32,
    index_topk=128,
    first_k_dense_replace=1,
)

model = DeepseekV32ForCausalLM(config)

# Generate tokens
input_ids = torch.randint(0, 1000, (1, 8))
with torch.no_grad():
    outputs = model.generate(input_ids, max_new_tokens=10, do_sample=False)

print(f'Generated shape: {outputs.shape}')  # Expected: [1, 18]
```

## Test 4: Regenerate from Modular

Verify the modular converter produces consistent output:

```bash
source .venv/bin/activate
python3 utils/modular_model_converter.py src/transformers/models/deepseek_v32/modular_deepseek_v32.py
```

Expected output:
```
Converting src/transformers/models/deepseek_v32/modular_deepseek_v32.py to a single model single file format
LoC: XXXX (modular) vs XXXX (generated) - saved -XX LoC (-X.X%)
```

## Optional Dependencies

### Hadamard Transform

For optimal indexer performance, install the fast Hadamard transform:

```bash
pip install fast-hadamard-transform
```

Without this package, the indexer falls back to identity (no transform), which still produces correct results but may be less efficient.

## Full Model Testing (Requires GPU)

To test with the full DeepSeek V3.2 model (685B parameters), you'll need:
- Multiple high-memory GPUs (8x A100 80GB or similar)
- Model weights from the official release

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-V3",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=False,  # Use our implementation
)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")

inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

## Known Limitations

1. **FP8 Indexer**: The `use_fp8_indexer` option is not fully implemented. The model will run without FP8 quantization in the indexer.

2. **Hadamard Transform**: Optional dependency - falls back to identity if not installed.

3. **Memory**: The full model requires significant GPU memory for inference.
