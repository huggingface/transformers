# DeepSeek V3.2 Implementation Handoff

## Summary

DeepSeek V3.2 implementation in HuggingFace Transformers:
- **Repository**: `lyfegame/transformers`
- **Branch**: `shuyingl/deepseek-v3.2-test`
- **PR**: https://github.com/lyfegame/transformers/pull/2

**Documentation**:
- **API Reference**: See `docs/DEEPSEEK_V32_API.md` for full API documentation

---

## Key Learnings for Next Agent

### 1. Modal GPU Image Building with CUDA Compilation

**Problem**: `fast-hadamard-transform` requires CUDA compilation at install time, which fails on Modal by default.

**Solution discovered**: Use `gpu="T4"` parameter in `run_commands()`:
```python
.run_commands(
    "git clone https://github.com/Dao-AILab/fast-hadamard-transform.git /tmp/fht && cd /tmp/fht && pip install -v .",
    gpu="T4",  # This enables GPU during image build!
)
```

Also requires: `clang` in apt_install for linking.

### 2. Modal Volume Performance

**Problem**: Loading 1.25TB model from Modal Volume takes 60+ min vs expected ~6 min.

**Root cause**: Volumes are network-mounted FUSE filesystems with high latency for random I/O.

**Strategies being tested**:
1. Direct load from volume (current baseline)
2. Pre-cache: Copy to local NVMe first (`ephemeral_disk=2000*1024`)
3. Volumes v2 (beta): `modal.Volume.from_name("name", version=2)`

### 3. RoPE Dimension Bug Fixed

**Problem**: Indexer uses non-interleaved RoPE (different from MLA's interleaved).

**Fix**: In `apply_rotary_pos_emb_non_interleave`:
- Changed `unsqueeze_dim` from 1 to 2
- Added cos/sin slicing to half dimension

### 4. HuggingFace Modular Architecture

The transformers fork uses HF's new modular architecture:
- `modular_deepseek_v32.py` is the source of truth
- `modeling_deepseek_v32.py` and `configuration_deepseek_v32.py` are auto-generated
- Pass-through classes (e.g., `DeepseekV32RMSNorm(DeepseekV3RMSNorm): pass`) are needed for proper class name generation
- Config class in modular file gets extracted to configuration file automatically

### 5. W&B Logging with Heartbeat

For long-running Modal jobs, use a heartbeat thread to prevent W&B timeout:
```python
import threading
def heartbeat_loop(run, stop_event):
    while not stop_event.is_set():
        run.log({"heartbeat": 1, "elapsed_minutes": ...})
        stop_event.wait(60)  # Log every minute
```

### 6. Model Architecture Details

- **671B parameters** with MoE (256 routed experts, 1 shared)
- **851 safetensor shards** for BF16 weights (~1.25TB total)
- **Lightning Indexer**: Selects top-k tokens for sparse attention
- **Hadamard transform**: Applied to Q/K in indexer (requires `fast-hadamard-transform` or PyTorch fallback)

---

## Quick Start

```bash
pip install git+https://github.com/lyfegame/transformers.git@shuyingl/deepseek-v3.2-test
pip install fast-hadamard-transform
```

```python
from transformers import DeepseekV32ForCausalLM

model = DeepseekV32ForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V3.2-Exp")
```

---

## What This Repo Provides (Capabilities)

1. **`use_sparse_attention` config** - Toggle sparse/dense attention mode
2. **`output_indexer_scores=True`** - Return raw indexer predictions `[batch, seq, seq]`
3. **`output_indexer_kl_target=True`** - Return KL target distribution `[batch, seq, seq]`
4. **`compute_indexer_kl_loss()`** - Helper for KL-divergence loss computation

**Training recipes and configs are managed in research-infra, not here.**

---

## Bugs Fixed During Implementation

### 1. RoPE Dimension Mismatch

**File**: `src/transformers/models/deepseek_v32/modular_deepseek_v32.py`

**Problem**: The indexer uses non-interleaved RoPE (different from MLA's interleaved RoPE). The original code had:
- Wrong `unsqueeze_dim` (1 instead of 2)
- Didn't slice cos/sin to half dimension

**Fix** (lines 139-192):
```python
def apply_rotary_pos_emb_non_interleave(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 2,  # Changed from 1 to 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    q1, q2 = q.chunk(2, dim=-1)
    k1, k2 = k.chunk(2, dim=-1)

    # Slice cos/sin to half dimension for non-interleaved RoPE
    half_dim = q1.shape[-1]
    if cos.shape[-1] != half_dim:
        cos = cos[..., :half_dim]
        sin = sin[..., :half_dim]

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_embed = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)

    return q_embed, k_embed
```

### 2. Missing HAS_FAST_HADAMARD Import

**File**: `src/transformers/models/deepseek_v32/modeling_deepseek_v32.py`

**Problem**: The auto-generated modeling file didn't include the `HAS_FAST_HADAMARD` import.

**Fix** (add after line ~50, after other imports):
```python
# Try to import fast_hadamard_transform, fall back to pure PyTorch if not available
try:
    from fast_hadamard_transform import hadamard_transform
    HAS_FAST_HADAMARD = True
except ImportError:
    HAS_FAST_HADAMARD = False
    logger.warning_once(
        "fast-hadamard-transform not installed. Using slower PyTorch fallback. "
        "For better performance, install with: pip install fast-hadamard-transform"
    )
```

---

## Installing fast-hadamard-transform on Modal

**Problem**: The `fast-hadamard-transform` package requires CUDA compilation at install time, which fails on Modal without special configuration.

**Solution** (discovered 2025-12-08):

The key insight is that Modal's `run_commands` supports a `gpu` parameter that enables GPU access during image build:

```python
.run_commands(
    "git clone https://github.com/Dao-AILab/fast-hadamard-transform.git /tmp/fht && cd /tmp/fht && pip install -v .",
    gpu="T4",  # This is the key!
)
```

**Full working configuration**:
```python
modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential", "clang")  # clang for linking
    .pip_install("wheel", "setuptools", "packaging", "ninja")  # build deps
    .pip_install("torch>=2.4.0", ...)
    .run_commands(
        "git clone https://github.com/Dao-AILab/fast-hadamard-transform.git /tmp/fht && cd /tmp/fht && pip install -v .",
        gpu="T4",
    )
```

**Previous failed attempts**:
1. ❌ `pip install fast-hadamard-transform` - Build isolation prevents torch access
2. ❌ `pip install --no-build-isolation` - Missing `wheel` module
3. ❌ Source build without `gpu=` - CUDA compilation fails
4. ❌ Source build without `clang` - `clang++: No such file or directory`

**References**:
- [Modal docs: Installing CUDA Toolkit](https://modal.com/docs/examples/install_cuda)
- [Modal docs: GPU during image build](https://modal.com/docs/guide/images)
- [fast-hadamard-transform GitHub](https://github.com/Dao-AILab/fast-hadamard-transform)

---

## Modal Test Commands

The test script is at `scripts/modal_verify_deepseek_v32.py`.

### Prerequisites

```bash
# Install Modal
pip install modal

# Authenticate (to 'fairies' workspace)
modal setup
```

### Run Tests

**IMPORTANT**: Always use `--detach` for long-running jobs to prevent disconnection:

```bash
# Run comprehensive verification (small config)
modal run --detach scripts/modal_verify_deepseek_v32.py --config small

# Run with full model checkpoint
modal run --detach scripts/modal_verify_deepseek_v32.py --checkpoint deepseek-ai--DeepSeek-V3.2_bf16

# Check running apps
modal app list

# View logs
modal app logs <APP_ID>
```

### What Tests Verify

1. **Forward Pass**: Model produces valid logits (no NaN/Inf)
2. **Backward Pass**: Gradients flow through all parameters
3. **Loss Decreases**: Model learns over training steps
4. **SFT Test**: Indexer stays frozen, loss still decreases
5. **Indexer KL Training**: Indexer-only training with KL loss
6. **Dual LoRA Training**: Two separate gradient paths (LLM LoRA + Indexer LoRA)
7. **DeepSpeed ZeRO-2**: Distributed training compatibility

---

## Integration Patterns for External Repos

Since DeepSeek V3.2 is not yet in upstream transformers, you need to install from our fork. Here are patterns for different integration scenarios.

### Pattern 1: requirements.txt

```txt
# requirements.txt
torch>=2.4.0
accelerate
safetensors
sentencepiece

# Install transformers fork with DeepSeek V3.2 support
# Pin to specific commit for reproducibility
transformers @ git+https://github.com/lyfegame/transformers.git@shuyingl/deepseek-v3.2-test

# Optional but recommended for performance
fast-hadamard-transform
```

### Pattern 2: pyproject.toml

```toml
[project]
dependencies = [
    "torch>=2.4.0",
    "accelerate",
    "safetensors",
    "sentencepiece",
    "transformers @ git+https://github.com/lyfegame/transformers.git@shuyingl/deepseek-v3.2-test",
]

[project.optional-dependencies]
perf = ["fast-hadamard-transform"]
```

### Pattern 3: Dockerfile

```dockerfile
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3.12 python3-pip git

# Install dependencies
RUN pip install torch>=2.4.0 accelerate safetensors sentencepiece

# Install transformers fork with DeepSeek V3.2
RUN pip install git+https://github.com/lyfegame/transformers.git@shuyingl/deepseek-v3.2-test

# Optional: fast-hadamard-transform (may need build tools)
RUN pip install fast-hadamard-transform || echo "Using PyTorch fallback for Hadamard transform"
```

### Pattern 4: Modal Image (with fast-hadamard-transform)

**IMPORTANT**: Installing `fast-hadamard-transform` on Modal requires special handling because it needs CUDA compilation during build.

```python
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    # REQUIRED: clang for linking during fast-hadamard-transform build
    .apt_install("git", "build-essential", "clang")
    # Build dependencies for fast-hadamard-transform
    .pip_install(
        "wheel",
        "setuptools",
        "packaging",
        "ninja",
    )
    .pip_install(
        "torch>=2.4.0",
        "accelerate",
        "safetensors",
        "sentencepiece",
    )
    # KEY: gpu="T4" enables CUDA compilation during image build!
    # Without this, the CUDA kernels won't compile
    .run_commands(
        "git clone https://github.com/Dao-AILab/fast-hadamard-transform.git /tmp/fht && cd /tmp/fht && pip install -v .",
        gpu="T4",
    )
    .pip_install(
        "git+https://github.com/lyfegame/transformers.git@shuyingl/deepseek-v3.2-test"
    )
)
```

**Why this works**:
1. `gpu="T4"` in `run_commands` gives GPU access during image build (Modal feature)
2. `clang` is needed for the final C++ linking step
3. Build deps must be pre-installed before the CUDA compilation

**Without fast-hadamard-transform** (simpler but slower):
```python
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .pip_install(
        "torch>=2.4.0",
        "accelerate",
        "safetensors",
        "sentencepiece",
    )
    .pip_install(
        "git+https://github.com/lyfegame/transformers.git@shuyingl/deepseek-v3.2-test"
    )
)
# Model will use PyTorch fallback for Hadamard transform (functional but slower)
```

### Pattern 5: Pin to Specific Commit (Recommended for Production)

For reproducibility, pin to a specific commit hash:

```bash
# Get current commit hash
git ls-remote https://github.com/lyfegame/transformers.git shuyingl/deepseek-v3.2-test

# Use in requirements.txt
transformers @ git+https://github.com/lyfegame/transformers.git@COMMIT_HASH
```

### Why AutoModel Works

The fork registers `deepseek_v32` as a model type in transformers' model registry. This means:

```python
# These all work automatically:
from transformers import AutoModelForCausalLM, AutoConfig

# Load by model type (if config.json has model_type="deepseek_v32")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V3.2-Exp")

# Or import directly
from transformers import DeepseekV32ForCausalLM
model = DeepseekV32ForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V3.2-Exp")
```

No manual model registration is needed - installing the fork is sufficient.

### Version Conflicts

If you have an existing transformers installation:

```bash
# Uninstall existing transformers first
pip uninstall transformers

# Install fork
pip install git+https://github.com/lyfegame/transformers.git@shuyingl/deepseek-v3.2-test
```

The fork is based on transformers `5.0.0.dev0` and includes all standard transformers functionality plus DeepSeek V3.2.

---

## Model Loading

```python
from transformers import DeepseekV32ForCausalLM, DeepseekV32Config

# Standard loading
model = DeepseekV32ForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-V3.2-Exp",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Or with config override
config = DeepseekV32Config.from_pretrained("deepseek-ai/DeepSeek-V3.2-Exp")
config.use_sparse_attention = False  # Dense mode
model = DeepseekV32ForCausalLM.from_pretrained(..., config=config)
```

See `docs/DEEPSEEK_V32_API.md` for full API details including:
- Parameter freezing utilities
- Indexer training outputs
- KL loss computation

---

## Config Reference

### Test Configs

**Tiny** (for quick validation):
```python
DeepseekV32Config(
    vocab_size=100,
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    n_routed_experts=4,
    index_n_heads=4,
    index_topk=8,
)
```

**Small** (for standard testing):
```python
DeepseekV32Config(
    vocab_size=10000,
    hidden_size=1024,
    num_hidden_layers=8,
    num_attention_heads=16,
    n_routed_experts=16,
    index_n_heads=16,
    index_topk=256,
)
```

### Full Model Config (Default)

```python
DeepseekV32Config(
    vocab_size=129280,
    hidden_size=7168,
    intermediate_size=18432,
    moe_intermediate_size=2048,
    num_hidden_layers=61,
    num_attention_heads=128,
    num_key_value_heads=128,
    n_shared_experts=1,
    n_routed_experts=256,
    kv_lora_rank=512,
    q_lora_rank=1536,
    qk_rope_head_dim=64,
    v_head_dim=128,
    qk_nope_head_dim=128,
    index_n_heads=64,
    index_head_dim=128,
    index_topk=2048,
)
```

---

## Files Reference

### In transformers fork (`lyfegame/transformers@shuyingl/deepseek-v3.2-test`)

| File | Description |
|------|-------------|
| `src/transformers/models/deepseek_v32/modular_deepseek_v32.py` | Source of truth for model implementation |
| `src/transformers/models/deepseek_v32/modeling_deepseek_v32.py` | Auto-generated (with manual HAS_FAST_HADAMARD fix) |
| `src/transformers/models/deepseek_v32/configuration_deepseek_v32.py` | Model configuration class |

### In this workspace

| File | Description |
|------|-------------|
| `scripts/modal_verify_deepseek_v32.py` | Comprehensive GPU verification (7 tests) |
| `scripts/test_deepseek_v32.py` | Local CPU test script |
| `docs/DEEPSEEK_V32_HANDOFF.md` | This handoff document |
| `docs/DEEPSEEK_V32_API.md` | API reference |

---

## Code Quality Issues to Address

### 1. Config Class Location (modular_deepseek_v32.py:195)

**Issue**: `DeepseekV32Config` is defined in `modular_deepseek_v32.py` but inherits from `DeepseekV3Config`.

**Context**: In HuggingFace's modular architecture, the Config class is typically defined in the modular file and auto-extracted to `configuration_*.py`. This is the new convention - see the auto-generated header in `configuration_deepseek_v32.py`.

**Decision needed**: Should we keep following the modular convention (current approach) or separate configs manually?

### 2. Indexer Class Inheritance (modular_deepseek_v32.py:416)

**Issue**: `DeepseekV32Indexer` inherits directly from `nn.Module` instead of a HuggingFace base class.

**Official implementation**: In `deepseek-ai/DeepSeek-V3.2-Exp/inference/model.py`, they also use `class Indexer(torch.nn.Module)` - so we're consistent with the official implementation.

**HuggingFace convention**: Typically, new components inherit from existing base classes where possible (e.g., `LlamaAttention` inherits from `nn.Module` directly because it's a core component).

**Analysis**: The Indexer is a new, standalone component that doesn't have an equivalent in other HF models. Using `nn.Module` directly is acceptable and matches the official implementation.

### 3. Pass-through Classes (modular_deepseek_v32.py:396-414)

**Current code**:
```python
class DeepseekV32RMSNorm(DeepseekV3RMSNorm):
    pass

class DeepseekV32RotaryEmbedding(DeepseekV3RotaryEmbedding):
    pass
# ... etc
```

**Purpose**: These pass-through classes are needed for the modular architecture to generate proper imports in the modeling file. They allow V3.2 to reuse V3 implementations while maintaining separate class names.

**Alternative**: Could use `__all__` exports and direct imports, but the pass-through pattern is cleaner for modular generation.

---

## Remaining Tasks

1. **Test official checkpoint**: Load full weights and verify inference quality
2. **Integrate into research-infra**: Add model loader and modify SFT orchestrator
3. **Run actual SFT training**: Train on your data with frozen indexer

---

## Modal-Specific Notes

### Environment Setup

**Workspace**: `fairies`
**Environment**: `training2`

```bash
# Authenticate to Modal
modal setup

# Run verification script
modal run --detach scripts/modal_verify_deepseek_v32.py --config small

# Check running apps
modal app list

# View logs
modal app logs <APP_ID>
```

### Volume Mounts

| Volume | Mount Path | Contents |
|--------|------------|----------|
| `models` | `/models` | Model weights (DeepSeek V3.2 BF16 at `/models/deepseek-ai--DeepSeek-V3.2_bf16`) |
| `datasets` | `/datasets` | Eval datasets (GSM8K at `/datasets/gsm8k/`) |

### W&B Integration

Logs go to project `deepseek-v32-test`. Create the secret:
```bash
modal secret create wandb WANDB_API_KEY=your_key_here
```

### GPU Recommendations

| Test | Recommended GPU | Notes |
|------|-----------------|-------|
| Tiny/Small config | H100 x1 | Quick validation |
| Full 671B model | H100 x8 or H200 x4 | BF16 weights need ~1.3TB memory |
| Image builds with FHT | T4 | Just for CUDA compilation |

---

## Troubleshooting

### "fast-hadamard-transform not installed"

**On local machines**:
```bash
pip install fast-hadamard-transform
```

**On Modal** (requires special handling):

The package needs CUDA compilation at install time. Use this pattern:
```python
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential", "clang")  # clang required!
    .pip_install("wheel", "setuptools", "packaging", "ninja", "torch>=2.4.0")
    # KEY: gpu="T4" enables CUDA during build
    .run_commands(
        "git clone https://github.com/Dao-AILab/fast-hadamard-transform.git /tmp/fht && cd /tmp/fht && pip install -v .",
        gpu="T4",
    )
)
```

**Common Modal build errors and solutions**:
| Error | Solution |
|-------|----------|
| `clang++: No such file or directory` | Add `clang` to `.apt_install()` |
| `ModuleNotFoundError: packaging` | Pre-install build deps (`wheel`, `setuptools`, `packaging`) |
| CUDA compilation fails silently | Add `gpu="T4"` to `.run_commands()` |

The model will fall back to pure PyTorch if not installed, but performance will be slower.

### OOM (Out of Memory)

1. Reduce batch size
2. Enable gradient checkpointing: `model.gradient_checkpointing_enable()`
3. Use smaller config for testing
4. Use `device_map="auto"` for multi-GPU distribution

### NaN in loss

1. Check input token IDs are within vocab_size
2. Ensure proper attention mask
3. Try lower learning rate (1e-5 instead of 1e-4)
4. Check for proper dtype (bfloat16 recommended)

### Import errors

Make sure you're using the correct branch:
```bash
pip install git+https://github.com/lyfegame/transformers.git@shuyingl/deepseek-v3.2-test
```

---

## Contact

For questions about the implementation, see PR #2 on the transformers fork:
https://github.com/lyfegame/transformers/pull/2
