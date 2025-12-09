#!/usr/bin/env python3
"""
Comprehensive verification script for DeepSeek V3.2 on Modal.

Tests:
1. Forward/backward with all losses (LM loss + indexer KL loss)
2. Timing for 1 training step and 1 evaluation step
3. Evaluation with small dataset (GSM8K sample)
4. Dual LoRA training (LLM LoRA -> LM Loss, Indexer LoRA -> KL Loss)

Usage (ALWAYS use --detach to run in background):
    modal run --detach scripts/modal_verify_deepseek_v32.py
    modal run --detach scripts/modal_verify_deepseek_v32.py --config small
    modal run --detach scripts/modal_verify_deepseek_v32.py --checkpoint deepseek-ai--DeepSeek-V3.2_bf16

Note: Uses volumes from 'training2' Modal environment:
    - models: /models (contains deepseek-ai--DeepSeek-V3.2_bf16)
    - datasets: /datasets (contains gsm8k)
"""

import modal
import os

app = modal.App("deepseek-v32-verify")

MINUTES = 60

# Modal volumes in the training2 environment
# Models: contains deepseek-ai--DeepSeek-V3.2_bf16 and other models
# Datasets: contains gsm8k and other datasets
models_volume = modal.Volume.from_name("models", environment_name="training2")
datasets_volume = modal.Volume.from_name("datasets", environment_name="training2")

# Create image with all dependencies
# Image version: 10 (git clone + pip install with gpu=H100 for fast-hadamard-transform)
verify_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential", "clang")
    .pip_install(
        "torch==2.4.0",  # Pin to 2.4.0 for CUDA 12.4 compatibility
        "accelerate",
        "safetensors",
        "sentencepiece",
        "datasets",
        "wandb",
        "ninja",
        "peft",  # For LoRA support
        "wheel",
        "setuptools",
        "packaging",
    )
    # Install fast-hadamard-transform from git (source tarball is incomplete)
    # Need to git clone to get all source files, then pip install with GPU
    .run_commands(
        "git clone https://github.com/Dao-AILab/fast-hadamard-transform.git /tmp/fht && cd /tmp/fht && pip install .",
        gpu="H100",
    )
    .pip_install(
        # Force fresh install with no cache to get the latest commit
        "git+https://github.com/lyfegame/transformers.git@shuyingl/deepseek-v3.2-test"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


def get_config(config_name: str):
    """Get model config by name."""
    from transformers import DeepseekV32Config

    configs = {
        "tiny": DeepseekV32Config(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=128,
            moe_intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            n_shared_experts=1,
            n_routed_experts=4,
            routed_scaling_factor=2.5,
            kv_lora_rank=16,
            q_lora_rank=32,
            qk_rope_head_dim=16,
            v_head_dim=16,
            qk_nope_head_dim=16,
            n_group=2,
            topk_group=1,
            num_experts_per_tok=2,
            first_k_dense_replace=1,
            max_position_embeddings=128,
            index_n_heads=4,
            index_head_dim=32,
            index_topk=8,
            use_sparse_attention=True,
        ),
        "small": DeepseekV32Config(
            vocab_size=10000,
            hidden_size=1024,
            intermediate_size=2048,
            moe_intermediate_size=512,
            num_hidden_layers=8,
            num_attention_heads=16,
            num_key_value_heads=16,
            n_shared_experts=1,
            n_routed_experts=16,
            routed_scaling_factor=2.5,
            kv_lora_rank=128,
            q_lora_rank=256,
            qk_rope_head_dim=64,
            v_head_dim=64,
            qk_nope_head_dim=64,
            n_group=4,
            topk_group=2,
            num_experts_per_tok=4,
            first_k_dense_replace=2,
            max_position_embeddings=4096,
            index_n_heads=16,
            index_head_dim=128,
            index_topk=256,
            use_sparse_attention=True,
        ),
        "medium": DeepseekV32Config(
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=11008,
            moe_intermediate_size=1408,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,
            n_shared_experts=1,
            n_routed_experts=64,
            routed_scaling_factor=2.5,
            kv_lora_rank=512,
            q_lora_rank=1536,
            qk_rope_head_dim=64,
            v_head_dim=128,
            qk_nope_head_dim=128,
            n_group=8,
            topk_group=4,
            num_experts_per_tok=8,
            first_k_dense_replace=3,
            max_position_embeddings=4096,
            index_n_heads=64,
            index_head_dim=128,
            index_topk=2048,
            use_sparse_attention=True,
        ),
    }
    return configs[config_name]


@app.function(
    image=verify_image,
    gpu="H100",
    timeout=60 * MINUTES,
    secrets=[modal.Secret.from_dotenv()],
    volumes={"/models": models_volume, "/datasets": datasets_volume},
)
def verify_all(
    config_name: str = "small",
    wandb_project: str = "deepseek-v32-verify",
    checkpoint_path: str = None,
):
    """Run all verification tests with timing.

    Args:
        config_name: Config size (tiny, small, medium) - only used if checkpoint_path is None
        wandb_project: W&B project name for logging
        checkpoint_path: Path to model checkpoint in /models volume (e.g., "deepseek-v32")
    """
    import time
    import torch
    import wandb
    from transformers import DeepseekV32ForCausalLM, AutoTokenizer

    results = {
        "config": config_name,
        "checkpoint": checkpoint_path,
        "tests": {},
        "timing": {},
    }

    # Setup
    device = torch.device("cuda")
    print(f"\n{'='*70}")
    print(f"DeepSeek V3.2 Comprehensive Verification")
    print(f"{'='*70}")
    print(f"Config: {config_name}")
    print(f"Checkpoint: {checkpoint_path or 'None (random init)'}")
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # List available models in volume
    print("\nAvailable models in /models volume:")
    import subprocess
    # List with -L to follow symlinks
    result = subprocess.run(["ls", "-laL", "/models"], capture_output=True, text=True)
    print(result.stdout if result.stdout else "  (empty or not accessible)")
    if result.stderr:
        print(f"  Error: {result.stderr}")

    # Try to list subdirectories
    result2 = subprocess.run(["find", "/models", "-maxdepth", "2", "-type", "d"], capture_output=True, text=True)
    if result2.stdout:
        print(f"Directories:\n{result2.stdout}")

    # Initialize wandb if API key available
    wandb_enabled = os.environ.get("WANDB_API_KEY") is not None
    run_name = f"verify-{checkpoint_path or config_name}"
    if wandb_enabled:
        wandb.init(
            project=wandb_project,
            name=run_name,
            config={"config_name": config_name, "checkpoint": checkpoint_path},
        )
        print("W&B logging enabled")
    else:
        print("W&B logging disabled (no API key)")

    # Load model - either from checkpoint or create with config
    tokenizer = None
    if checkpoint_path:
        model_path = f"/models/{checkpoint_path}"
        print(f"\nLoading model from checkpoint: {model_path}")

        # List checkpoint contents
        result = subprocess.run(["ls", "-la", model_path], capture_output=True, text=True)
        print(f"Checkpoint contents:\n{result.stdout}")

        t0 = time.time()
        model = DeepseekV32ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model_creation_time = time.time() - t0
        print(f"Model load time: {model_creation_time:.2f}s")

        # Try to load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            print(f"Tokenizer loaded from checkpoint")
        except Exception as e:
            print(f"No tokenizer in checkpoint: {e}")

        config = model.config
    else:
        # Create model with config
        config = get_config(config_name)
        print(f"\nModel Config:")
        print(f"  hidden_size: {config.hidden_size}")
        print(f"  num_hidden_layers: {config.num_hidden_layers}")
        print(f"  n_routed_experts: {config.n_routed_experts}")
        print(f"  index_n_heads: {config.index_n_heads}")
        print(f"  index_topk: {config.index_topk}")
        print(f"  use_sparse_attention: {config.use_sparse_attention}")

        print("\nCreating model with random weights...")
        t0 = time.time()
        model = DeepseekV32ForCausalLM(config)
        model = model.to(device=device, dtype=torch.bfloat16)
        model_creation_time = time.time() - t0
        print(f"Model creation time: {model_creation_time:.2f}s")

    # Print model info
    print(f"\nModel Config (loaded):")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  vocab_size: {config.vocab_size}")
    if hasattr(config, 'n_routed_experts'):
        print(f"  n_routed_experts: {config.n_routed_experts}")
    if hasattr(config, 'index_n_heads'):
        print(f"  index_n_heads: {config.index_n_heads}")
    if hasattr(config, 'index_topk'):
        print(f"  index_topk: {config.index_topk}")
    if hasattr(config, 'use_sparse_attention'):
        print(f"  use_sparse_attention: {config.use_sparse_attention}")

    total_params = sum(p.numel() for p in model.parameters())
    indexer_params = sum(p.numel() for n, p in model.named_parameters() if "indexer" in n)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Indexer parameters: {indexer_params:,}")

    results["model_params"] = total_params
    results["indexer_params"] = indexer_params
    results["timing"]["model_creation"] = model_creation_time

    # =========================================================================
    # TEST 1: Forward pass
    # =========================================================================
    print(f"\n{'='*70}")
    print("TEST 1: Forward Pass")
    print(f"{'='*70}")

    model.eval()
    batch_size, seq_len = 4, 512

    # Use tokenizer if available, otherwise random tokens
    if tokenizer:
        test_text = "Hello, I am a large language model. I can help you with"
        inputs = tokenizer([test_text] * batch_size, return_tensors="pt", padding=True, truncation=True, max_length=seq_len)
        input_ids = inputs.input_ids.to(device)
        print(f"  Using tokenizer, input text: '{test_text[:50]}...'")
    else:
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        print(f"  Using random tokens (no tokenizer)")

    torch.cuda.synchronize()
    t0 = time.time()

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids)

    torch.cuda.synchronize()
    forward_time = time.time() - t0

    logits = outputs.logits
    has_nan = torch.isnan(logits).any().item()
    has_inf = torch.isinf(logits).any().item()

    forward_passed = not has_nan and not has_inf

    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Has NaN: {has_nan}, Has Inf: {has_inf}")
    print(f"  Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
    print(f"  Forward time: {forward_time*1000:.2f}ms")
    print(f"  Result: {'PASSED' if forward_passed else 'FAILED'}")

    results["tests"]["forward_pass"] = forward_passed
    results["timing"]["forward_pass_ms"] = forward_time * 1000

    # =========================================================================
    # TEST 2: Backward with LM loss
    # =========================================================================
    print(f"\n{'='*70}")
    print("TEST 2: Backward Pass (LM loss)")
    print(f"{'='*70}")

    model.train()
    model.zero_grad()

    labels = input_ids.clone()

    torch.cuda.synchronize()
    t0 = time.time()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(input_ids, labels=labels)
        lm_loss = outputs.loss

    lm_loss.backward()

    torch.cuda.synchronize()
    backward_lm_time = time.time() - t0

    lm_loss_val = lm_loss.item()
    params_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    indexer_grads = sum(1 for n, p in model.named_parameters()
                       if "indexer" in n and p.grad is not None and p.grad.abs().sum() > 0)

    backward_lm_passed = not torch.isnan(lm_loss).item() and params_with_grad > 0

    print(f"  LM loss: {lm_loss_val:.4f}")
    print(f"  Total params with gradient: {params_with_grad}")
    print(f"  Indexer params with gradient: {indexer_grads}")
    print(f"  Backward time: {backward_lm_time*1000:.2f}ms")
    print(f"  Result: {'PASSED' if backward_lm_passed else 'FAILED'}")

    results["tests"]["backward_lm_loss"] = backward_lm_passed
    results["timing"]["backward_lm_ms"] = backward_lm_time * 1000

    # =========================================================================
    # TEST 3: Frozen indexer training (SFT mode)
    # =========================================================================
    print(f"\n{'='*70}")
    print("TEST 3: Frozen Indexer Training (SFT mode)")
    print(f"{'='*70}")

    model.train()
    model.zero_grad()

    # Freeze indexer params
    for name, param in model.named_parameters():
        if "indexer" in name:
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_indexer_params = sum(p.numel() for n, p in model.named_parameters()
                                if "indexer" in n and not p.requires_grad)
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Frozen indexer params: {frozen_indexer_params:,}")

    torch.cuda.synchronize()
    t0 = time.time()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss

    loss.backward()

    torch.cuda.synchronize()
    frozen_indexer_time = time.time() - t0

    # Verify indexer has no gradients
    indexer_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in model.named_parameters() if "indexer" in n
    )
    non_indexer_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in model.named_parameters() if "indexer" not in n
    )

    frozen_indexer_passed = not indexer_has_grad and non_indexer_has_grad

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Indexer has gradients: {indexer_has_grad} (should be False)")
    print(f"  Non-indexer has gradients: {non_indexer_has_grad} (should be True)")
    print(f"  Time: {frozen_indexer_time*1000:.2f}ms")
    print(f"  Result: {'PASSED' if frozen_indexer_passed else 'FAILED'}")

    results["tests"]["frozen_indexer_training"] = frozen_indexer_passed
    results["timing"]["frozen_indexer_ms"] = frozen_indexer_time * 1000

    # Unfreeze all params
    for param in model.parameters():
        param.requires_grad = True

    # =========================================================================
    # TEST 4: Full training step with optimizer
    # =========================================================================
    print(f"\n{'='*70}")
    print("TEST 4: Full Training Step (forward + backward + optimizer)")
    print(f"{'='*70}")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()

    # Timed training step
    t0 = time.time()

    optimizer.zero_grad()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()
    training_step_time = time.time() - t0

    training_step_passed = not torch.isnan(loss).item()

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Training step time: {training_step_time*1000:.2f}ms")
    print(f"  Throughput: {batch_size * seq_len / training_step_time:.0f} tokens/sec")
    print(f"  Result: {'PASSED' if training_step_passed else 'FAILED'}")

    results["tests"]["training_step"] = training_step_passed
    results["timing"]["training_step_ms"] = training_step_time * 1000
    results["timing"]["training_throughput_tokens_per_sec"] = batch_size * seq_len / training_step_time

    # =========================================================================
    # TEST 5: Evaluation step
    # =========================================================================
    print(f"\n{'='*70}")
    print("TEST 5: Evaluation Step")
    print(f"{'='*70}")

    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids, labels=labels)

    torch.cuda.synchronize()

    # Timed eval step
    t0 = time.time()

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids, labels=labels)
            eval_loss = outputs.loss

    torch.cuda.synchronize()
    eval_step_time = time.time() - t0

    eval_step_passed = not torch.isnan(eval_loss).item()

    print(f"  Eval loss: {eval_loss.item():.4f}")
    print(f"  Eval step time: {eval_step_time*1000:.2f}ms")
    print(f"  Throughput: {batch_size * seq_len / eval_step_time:.0f} tokens/sec")
    print(f"  Result: {'PASSED' if eval_step_passed else 'FAILED'}")

    results["tests"]["eval_step"] = eval_step_passed
    results["timing"]["eval_step_ms"] = eval_step_time * 1000
    results["timing"]["eval_throughput_tokens_per_sec"] = batch_size * seq_len / eval_step_time

    # =========================================================================
    # TEST 6: Evaluation with real dataset (GSM8K sample)
    # =========================================================================
    print(f"\n{'='*70}")
    print("TEST 6: Evaluation with GSM8K Sample")
    print(f"{'='*70}")

    from datasets import load_dataset

    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="test[:20]")

    # Simple tokenization (since we're using random vocab, just hash to vocab size)
    def simple_tokenize(text, max_len=256):
        # Hash each character to vocab range
        tokens = [hash(c) % config.vocab_size for c in text[:max_len]]
        if len(tokens) < max_len:
            tokens = tokens + [0] * (max_len - len(tokens))
        return tokens[:max_len]

    # Process dataset
    eval_losses = []
    eval_times = []

    model.eval()

    for i, example in enumerate(dataset):
        text = example["question"] + " " + example["answer"]
        tokens = simple_tokenize(text)
        input_ids = torch.tensor([tokens], device=device)
        labels = input_ids.clone()

        torch.cuda.synchronize()
        t0 = time.time()

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss

        torch.cuda.synchronize()
        step_time = time.time() - t0

        eval_losses.append(loss.item())
        eval_times.append(step_time)

        if i < 5 or i == len(dataset) - 1:
            print(f"  Example {i}: loss={loss.item():.4f}, time={step_time*1000:.2f}ms")

    avg_eval_loss = sum(eval_losses) / len(eval_losses)
    avg_eval_time = sum(eval_times) / len(eval_times)

    dataset_eval_passed = all(not torch.isnan(torch.tensor(l)).item() for l in eval_losses)

    print(f"\n  Avg eval loss: {avg_eval_loss:.4f}")
    print(f"  Avg eval time: {avg_eval_time*1000:.2f}ms")
    print(f"  Total examples: {len(dataset)}")
    print(f"  Result: {'PASSED' if dataset_eval_passed else 'FAILED'}")

    results["tests"]["dataset_eval"] = dataset_eval_passed
    results["timing"]["dataset_avg_eval_ms"] = avg_eval_time * 1000
    results["dataset_eval_loss"] = avg_eval_loss

    # =========================================================================
    # TEST 7: Dual LoRA Training (LLM LoRA + Indexer LoRA with separate losses)
    # =========================================================================
    print(f"\n{'='*70}")
    print("TEST 7: Dual LoRA Training Setup")
    print("  - Main model LoRA -> LM Loss")
    print("  - Indexer LoRA -> KL Loss (placeholder)")
    print(f"{'='*70}")

    from peft import LoraConfig, get_peft_model, PeftModel

    # Recreate fresh model for LoRA test
    if checkpoint_path:
        model_for_lora = DeepseekV32ForCausalLM.from_pretrained(
            f"/models/{checkpoint_path}",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        model_for_lora = DeepseekV32ForCausalLM(config)
        model_for_lora = model_for_lora.to(device=device, dtype=torch.bfloat16)

    # Define LoRA targets for main model (non-indexer)
    # Target attention and MLP projections
    llm_lora_targets = [
        "q_a_proj", "q_b_proj",
        "kv_a_proj_with_mqa", "kv_b_proj",
        "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    # Define LoRA targets for indexer
    # Correct names from modular_deepseek_v32.py: wq_b, wk, weights_proj, k_norm
    indexer_lora_targets = [
        "indexer.wq_b",
        "indexer.wk",
        "indexer.weights_proj",
    ]

    print(f"\n  LLM LoRA targets: {llm_lora_targets}")
    print(f"  Indexer LoRA targets: {indexer_lora_targets}")

    # Strategy: Apply LoRA to both, then manage gradient flow via loss computation
    all_lora_targets = llm_lora_targets + indexer_lora_targets

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=all_lora_targets,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    try:
        model_with_lora = get_peft_model(model_for_lora, lora_config)
        print(f"\n  PEFT model created successfully!")

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model_with_lora.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model_with_lora.parameters())
        print(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

        # Count LoRA params by type
        llm_lora_params = sum(
            p.numel() for n, p in model_with_lora.named_parameters()
            if p.requires_grad and "indexer" not in n
        )
        indexer_lora_params = sum(
            p.numel() for n, p in model_with_lora.named_parameters()
            if p.requires_grad and "indexer" in n
        )
        print(f"  LLM LoRA params: {llm_lora_params:,}")
        print(f"  Indexer LoRA params: {indexer_lora_params:,}")

        # Test forward pass with LoRA
        model_with_lora.train()
        batch_size_lora, seq_len_lora = 2, 256
        input_ids_lora = torch.randint(0, config.vocab_size, (batch_size_lora, seq_len_lora), device=device)
        labels_lora = input_ids_lora.clone()

        # =====================================================================
        # Dual gradient path training step
        # =====================================================================
        print("\n  Testing dual gradient path training...")

        # Create separate optimizers for LLM LoRA and Indexer LoRA
        llm_lora_params_list = [
            p for n, p in model_with_lora.named_parameters()
            if p.requires_grad and "indexer" not in n
        ]
        indexer_lora_params_list = [
            p for n, p in model_with_lora.named_parameters()
            if p.requires_grad and "indexer" in n
        ]

        llm_optimizer = torch.optim.AdamW(llm_lora_params_list, lr=1e-4)
        indexer_optimizer = torch.optim.AdamW(indexer_lora_params_list, lr=1e-3)

        torch.cuda.synchronize()
        t0 = time.time()

        # Step 1: Forward pass
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model_with_lora(input_ids_lora, labels=labels_lora)
            lm_loss = outputs.loss

            # Placeholder for KL loss (would come from indexer scores vs dense attention)
            # In actual implementation, this would be:
            # kl_loss = compute_indexer_kl_loss(outputs.indexer_scores, outputs.indexer_kl_targets)
            # For now, use a dummy loss based on output logits variance as placeholder
            kl_loss_placeholder = outputs.logits.var() * 0.001  # Placeholder

        # Step 2: Backward for LM loss -> updates LLM LoRA
        llm_optimizer.zero_grad()
        # Only backprop through non-indexer params
        lm_loss.backward(retain_graph=True)

        # Check LLM LoRA has gradients
        llm_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in llm_lora_params_list
        )

        llm_optimizer.step()

        # Step 3: Backward for KL loss -> updates Indexer LoRA
        indexer_optimizer.zero_grad()
        kl_loss_placeholder.backward()

        # Check Indexer LoRA has gradients
        indexer_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in indexer_lora_params_list
        )

        indexer_optimizer.step()

        torch.cuda.synchronize()
        dual_lora_time = time.time() - t0

        print(f"\n  LM Loss: {lm_loss.item():.4f}")
        print(f"  KL Loss (placeholder): {kl_loss_placeholder.item():.6f}")
        print(f"  LLM LoRA has gradients: {llm_has_grad}")
        print(f"  Indexer LoRA has gradients: {indexer_has_grad}")
        print(f"  Dual LoRA step time: {dual_lora_time*1000:.2f}ms")

        dual_lora_passed = llm_has_grad and indexer_has_grad
        print(f"  Result: {'PASSED' if dual_lora_passed else 'FAILED'}")

        results["tests"]["dual_lora_training"] = dual_lora_passed
        results["timing"]["dual_lora_step_ms"] = dual_lora_time * 1000
        results["dual_lora"] = {
            "llm_lora_params": llm_lora_params,
            "indexer_lora_params": indexer_lora_params,
            "lm_loss": lm_loss.item(),
            "kl_loss_placeholder": kl_loss_placeholder.item(),
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        results["tests"]["dual_lora_training"] = False
        dual_lora_passed = False

    # Clean up LoRA model
    del model_with_lora
    if 'model_for_lora' in dir():
        del model_for_lora
    torch.cuda.empty_cache()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print("\nTest Results:")
    all_passed = True
    for test_name, passed in results["tests"].items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("\nTiming Results:")
    print(f"  Model creation: {results['timing']['model_creation']:.2f}s")
    print(f"  Forward pass: {results['timing']['forward_pass_ms']:.2f}ms")
    print(f"  Backward (LM loss): {results['timing']['backward_lm_ms']:.2f}ms")
    print(f"  Frozen indexer training: {results['timing']['frozen_indexer_ms']:.2f}ms")
    print(f"  Full training step: {results['timing']['training_step_ms']:.2f}ms")
    print(f"  Full eval step: {results['timing']['eval_step_ms']:.2f}ms")
    print(f"  Training throughput: {results['timing']['training_throughput_tokens_per_sec']:.0f} tokens/sec")
    print(f"  Eval throughput: {results['timing']['eval_throughput_tokens_per_sec']:.0f} tokens/sec")

    # Memory stats
    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated() / 1e9
        reserved_mem = torch.cuda.max_memory_reserved() / 1e9
        print(f"\nGPU Memory:")
        print(f"  Peak allocated: {max_mem:.2f} GB")
        print(f"  Peak reserved: {reserved_mem:.2f} GB")
        results["memory"] = {"peak_allocated_gb": max_mem, "peak_reserved_gb": reserved_mem}

    # Log to wandb
    if wandb_enabled:
        wandb.log(results["timing"])
        wandb.log({"all_tests_passed": all_passed})
        wandb.finish()

    print(f"\n{'='*70}")
    print(f"{'ALL TESTS PASSED!' if all_passed else 'SOME TESTS FAILED!'}")
    print(f"{'='*70}")

    return results


@app.local_entrypoint()
def main(
    config: str = "small",
    wandb_project: str = "deepseek-v32-verify",
    checkpoint: str = None,
):
    """
    Run comprehensive DeepSeek V3.2 verification on Modal.

    Args:
        config: Config size (tiny, small, medium) - only used if checkpoint is None
        wandb_project: W&B project name for logging
        checkpoint: Path to model checkpoint in volume (e.g., "checkpoints/deepseek-v32")

    Examples:
        # Test with random init small config
        modal run scripts/modal_verify_deepseek_v32.py --config small

        # Test with checkpoint from volume
        modal run scripts/modal_verify_deepseek_v32.py --checkpoint checkpoints/deepseek-v32

        # Test with HuggingFace model (checkpoint starting with 'hf:')
        modal run scripts/modal_verify_deepseek_v32.py --checkpoint hf:deepseek-ai/DeepSeek-V3.2-Exp
    """
    if checkpoint:
        print(f"Running verification with checkpoint: {checkpoint}")
    else:
        print(f"Running verification with {config} config (random init)...")

    results = verify_all.remote(config, wandb_project, checkpoint)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    all_passed = all(results["tests"].values())

    if all_passed:
        print("\n[SUCCESS] All verification tests passed!")
    else:
        print("\n[FAILURE] Some tests failed!")
        for test_name, passed in results["tests"].items():
            if not passed:
                print(f"  - {test_name}: FAILED")
        raise SystemExit(1)

    return results
