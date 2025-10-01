<!--Copyright 2025 The HRM Team and HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# HRM

## Overview

The Hierarchical Reasoning Model (HRM) is a novel recurrent neural network architecture designed for sequential reasoning tasks. Unlike traditional large language models that rely on Chain-of-Thought (CoT) techniques, HRM employs a hierarchical and multi-timescale processing approach inspired by the human brain.

The model was proposed in [Hierarchical Reasoning Model](https://arxiv.org/abs/2506.21734) by Guan Wang, Jin Li, Yuhao Sun, Xing Chen, Changling Liu, Yue Wu, Meng Lu, Sen Song, and Yasin Abbasi Yadkori.

The abstract from the paper is the following:

*Reasoning, the process of devising and executing complex goal-oriented action sequences, remains a critical challenge in AI. Current large language models (LLMs) primarily employ Chain-of-Thought (CoT) techniques, which suffer from brittle task decomposition, extensive data requirements, and high latency. Inspired by the hierarchical and multi-timescale processing in the human brain, we propose the Hierarchical Reasoning Model (HRM), a novel recurrent architecture that attains significant computational depth while maintaining both training stability and efficiency. HRM executes sequential reasoning tasks in a single forward pass without explicit supervision of the intermediate process, through two interdependent recurrent modules: a high-level module responsible for slow, abstract planning, and a low-level module handling rapid, detailed computations. With only 27 million parameters, HRM achieves exceptional performance on complex reasoning tasks using only 1000 training samples. The model operates without pre-training or CoT data, yet achieves nearly perfect performance on challenging tasks including complex Sudoku puzzles and optimal path finding in large mazes. Furthermore, HRM outperforms much larger models with significantly longer context windows on the Abstraction and Reasoning Corpus (ARC), a key benchmark for measuring artificial general intelligence capabilities.*

This model was contributed by [Zachary Bloss](https://huggingface.co/zbloss). The original code can be found [here](https://github.com/liujch1998/HRM).

## Key Features

HRM introduces several innovative features:

1. **Two-Level Hierarchical Processing**:
   - **High-Level (H) Module**: Performs slow, abstract planning and reasoning
   - **Low-Level (L) Module**: Handles fast, detailed computations
   - The modules interact through input injection, enabling communication between abstraction levels

2. **Adaptive Computation Time (ACT)**:
   - Q-learning based mechanism for dynamic halting
   - Sequences adaptively decide when to stop computation
   - Balances computational efficiency with task performance

3. **Efficient Training Strategy**:
   - Executes multiple reasoning cycles without gradients
   - Applies gradients only on the final iteration (1-step gradient)
   - Maintains computational depth while enabling stable training

4. **Small-Sample Learning**:
   - Achieves strong performance with only 1000 training examples
   - No pre-training required
   - No Chain-of-Thought data needed

## Model Architecture

### Hierarchical Modules

The H-level and L-level modules each consist of multiple transformer blocks with:
- Multi-head self-attention (non-causal)
- SwiGLU feed-forward networks
- RMS normalization (post-norm architecture)
- Rotary Position Embeddings (RoPE) or learned positional encodings

### Forward Pass

```
for h_cycle in range(H_cycles):
    for l_cycle in range(L_cycles):
        z_L = L_level(z_L, z_H + input_embeddings)
    z_H = H_level(z_H, z_L)
```

The final cycle uses gradients for backpropagation, while earlier cycles run without gradients for efficiency.

## Usage Examples

### Basic Inference

```python
from transformers import HrmConfig, HrmForCausalLM
import torch

# Create model configuration for Sudoku solving
config = HrmConfig(
    vocab_size=11,  # 0-9 digits + padding
    hidden_size=512,
    num_hidden_layers=4,
    h_layers=4,  # High-level reasoning layers
    l_layers=4,  # Low-level computation layers
    num_attention_heads=8,
    max_position_embeddings=81,  # 9x9 Sudoku grid
    h_cycles=2,  # High-level reasoning cycles
    l_cycles=2,  # Low-level computation cycles per H cycle
    halt_max_steps=16,  # Maximum ACT steps
    halt_exploration_prob=0.1,  # Exploration during training
    pos_encodings="rope",  # Use RoPE
    expansion=4.0,  # MLP expansion ratio
    torch_dtype="bfloat16",
)

# Initialize model
model = HrmForCausalLM(config)

# Prepare input (partial Sudoku puzzle)
input_ids = torch.randint(0, 11, (1, 81))

# Forward pass
outputs = model(input_ids=input_ids)
logits = outputs["logits"]  # (batch_size, seq_len, vocab_size)
```

### Training with Carry State

```python
from transformers import HrmForCausalLM, HrmConfig
import torch

config = HrmConfig(
    vocab_size=11,
    hidden_size=512,
    max_position_embeddings=81,
    h_cycles=2,
    l_cycles=2,
    halt_max_steps=16,
)

model = HrmForCausalLM(config)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop with ACT
for batch in dataloader:
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    # Initialize carry state
    carry = model.model.initial_carry({"input_ids": input_ids})

    # Forward with ACT mechanism
    outputs = model(
        input_ids=input_ids,
        labels=labels,
        carry=carry,
    )

    loss = outputs["loss"]
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Using Puzzle-Specific Embeddings

```python
config = HrmConfig(
    vocab_size=11,
    hidden_size=512,
    max_position_embeddings=81,
    puzzle_emb_ndim=512,  # Enable puzzle embeddings
    num_puzzle_identifiers=100,  # Number of unique puzzles
    h_cycles=2,
    l_cycles=2,
)

model = HrmForCausalLM(config)

# Provide puzzle identifiers
input_ids = torch.randint(0, 11, (4, 81))
puzzle_identifiers = torch.tensor([0, 1, 0, 2])  # Different puzzle types

outputs = model(
    input_ids=input_ids,
    puzzle_identifiers=puzzle_identifiers,
)
```

### Generation

```python
from transformers import HrmForCausalLM, HrmConfig

config = HrmConfig(
    vocab_size=11,
    hidden_size=256,
    max_position_embeddings=81,
)

model = HrmForCausalLM(config)
model.eval()

# Start with partial puzzle
input_ids = torch.tensor([[1, 2, 0, 0, 0]])  # 0 represents empty cells

# Generate solution
with torch.no_grad():
    generated = model.generate(
        input_ids,
        max_new_tokens=50,
    )

print(generated)
```

## Configuration

### Key Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `vocab_size` | Size of token vocabulary | 11 |
| `hidden_size` | Dimension of hidden states | 512 |
| `h_layers` | Number of high-level transformer layers | 4 |
| `l_layers` | Number of low-level transformer layers | 4 |
| `h_cycles` | High-level reasoning cycles per forward pass | 2 |
| `l_cycles` | Low-level computation cycles per H cycle | 2 |
| `num_attention_heads` | Number of attention heads | 8 |
| `max_position_embeddings` | Maximum sequence length | 81 |
| `halt_max_steps` | Maximum ACT steps before forcing halt | 16 |
| `halt_exploration_prob` | Exploration probability for ACT training | 0.1 |
| `pos_encodings` | Type of positional encoding (`"rope"` or `"learned"`) | `"rope"` |
| `expansion` | MLP expansion ratio for SwiGLU | 4.0 |
| `puzzle_emb_ndim` | Dimension of puzzle-specific embeddings (0 to disable) | 0 |

## Use Cases

HRM is particularly effective for:

1. **Sudoku Solving**: Achieves near-perfect accuracy on 9x9 extreme difficulty puzzles
2. **Maze Navigation**: Finds optimal paths in 30x30 mazes
3. **ARC-AGI Tasks**: Excels at abstract reasoning and pattern recognition
4. **General Sequential Reasoning**: Any task requiring step-by-step logical deduction

## Performance

With only 27M parameters and 1000 training examples:
- **Sudoku (9x9 Extreme)**: ~100% accuracy
- **Maze (30x30 Hard)**: Optimal path finding
- **ARC-AGI**: Outperforms much larger models

## Pre-trained Models

Available checkpoints on the Hugging Face Hub:

- [sapientinc/HRM-checkpoint-sudoku-extreme](https://huggingface.co/sapientinc/HRM-checkpoint-sudoku-extreme)
- [sapientinc/HRM-checkpoint-maze-30x30-hard](https://huggingface.co/sapientinc/HRM-checkpoint-maze-30x30-hard)
- [sapientinc/HRM-checkpoint-ARC-2](https://huggingface.co/sapientinc/HRM-checkpoint-ARC-2)

## Tips and Best Practices

1. **Small Sample Learning**: HRM works well with 1000-10000 training examples. More data helps but isn't strictly necessary.

2. **Cycle Configuration**:
   - Start with `h_cycles=2, l_cycles=2`
   - Increase for more complex reasoning tasks
   - More cycles = deeper computation but slower

3. **ACT Configuration**:
   - `halt_max_steps=16` works well for most tasks
   - Lower values force faster decisions
   - Higher values allow more deliberation

4. **Positional Encodings**:
   - `"rope"` generally performs better
   - `"learned"` can work for fixed-size tasks

5. **Training**:
   - Use learning rate ~1e-4 to 7e-5
   - Apply weight decay ~0.1 to 1.0
   - Monitor Q-values to ensure ACT is learning

6. **Puzzle Embeddings**:
   - Enable (`puzzle_emb_ndim > 0`) when solving multiple distinct puzzles
   - Each puzzle gets its own learned embedding
   - Helps model distinguish between problem instances

## Limitations

- **Task-Specific**: Best for reasoning tasks with clear structure
- **Tokenization**: Requires custom tokenization for non-text domains
- **FlashAttention**: Performance benefits from FlashAttention 2/3 (optional)
- **Recurrent Nature**: Cannot leverage KV caching like standard transformers

## Citation

```bibtex
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model},
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734},
}
```

## HrmConfig

[[autodoc]] HrmConfig

## HrmModel

[[autodoc]] HrmModel
    - forward

## HrmForCausalLM

[[autodoc]] HrmForCausalLM
    - forward
    - generate
