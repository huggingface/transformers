#!/usr/bin/env python3
"""
Modal app to debug a single forward pass through DeepSeek V3.2.

Checks for NaN/Inf values and other numerical issues at each stage.

Usage:
    modal run sandbox/debug_forward_pass.py
"""

import modal


# Configuration
GPU_TYPE = "B200"
GPU_COUNT = 8
MODEL_VOLUME_NAME = "models"
MODEL_MOUNT_PATH = "/mnt/model"
LOCAL_MODEL_PATH = "/tmp/model"
DEFAULT_MODEL_PATH = "deepseek-ai--DeepSeek-V3.2_bf16"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.4.0",
        "accelerate>=0.34.0",
        "safetensors",
        "tiktoken",
        "sentencepiece",
    )
    .run_commands(
        "pip install --no-cache-dir git+https://github.com/jyliu24/transformers.git",
    )
)

app = modal.App("debug-deepseek-forward", image=image)

model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME)


@app.function(
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    volumes={MODEL_MOUNT_PATH: model_volume},
    timeout=60 * 120,
    ephemeral_disk=2 * 1024 * 1024,
)
def debug_forward_pass(
    model_path: str = DEFAULT_MODEL_PATH,
    prompt: str = "Hello",
    use_local_copy: bool = True,
) -> dict:
    """
    Debug a single forward pass, checking for numerical issues.
    """
    import os
    import shutil
    import time

    import torch

    volume_model_path = f"{MODEL_MOUNT_PATH}/{model_path}"
    local_model_path = f"{LOCAL_MODEL_PATH}/{model_path}"

    # Copy model to local disk
    if use_local_copy:
        print("📋 Copying model to local disk...")
        if os.path.exists(local_model_path):
            shutil.rmtree(local_model_path)
        shutil.copytree(volume_model_path, local_model_path)
        full_model_path = local_model_path
    else:
        full_model_path = volume_model_path

    print("=" * 70)
    print("DeepSeek V3.2 Forward Pass Debug")
    print("=" * 70)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    # Load tokenizer
    print("\n📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(full_model_path, trust_remote_code=True)
    print(f"   Tokenizer: {type(tokenizer).__name__}")
    print(f"   Vocab size: {tokenizer.vocab_size}")

    # Load model
    print("\n📥 Loading model...")
    from accelerate import infer_auto_device_map, init_empty_weights

    config = AutoConfig.from_pretrained(full_model_path, trust_remote_code=True)
    print(f"   Config: {type(config).__name__}")
    print(f"   num_hidden_layers: {config.num_hidden_layers}")
    print(f"   index_topk: {getattr(config, 'index_topk', 'N/A')}")

    num_gpus = torch.cuda.device_count()
    max_memory = dict.fromkeys(range(num_gpus), "175GiB")
    max_memory["cpu"] = "100GiB"

    with init_empty_weights():
        empty_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    device_map = infer_auto_device_map(
        empty_model,
        max_memory=max_memory,
        no_split_module_classes=["DeepseekV32DecoderLayer"],
    )
    del empty_model

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        full_model_path,
        trust_remote_code=True,
        device_map=device_map,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )
    print(f"   Loaded in {time.time() - t0:.1f}s")
    print(f"   Model type: {type(model).__name__}")

    # Prepare input
    print(f"\n📝 Preparing input: '{prompt}'")
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Input tokens: {input_ids.tolist()}")

    # Check embedding
    print("\n🔍 Checking embedding layer...")
    embed_weight = model.model.embed_tokens.weight
    print(f"   Embedding shape: {embed_weight.shape}")
    print(f"   Embedding dtype: {embed_weight.dtype}")
    print(f"   Embedding has NaN: {torch.isnan(embed_weight).any().item()}")
    print(f"   Embedding has Inf: {torch.isinf(embed_weight).any().item()}")
    print(f"   Embedding mean: {embed_weight.float().mean().item():.6f}")
    print(f"   Embedding std: {embed_weight.float().std().item():.6f}")

    # Check first layer attention weights
    print("\n🔍 Checking layer 0 attention weights...")
    attn = model.model.layers[0].self_attn

    for name in ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]:
        weight = getattr(attn, name).weight
        has_nan = torch.isnan(weight).any().item()
        has_inf = torch.isinf(weight).any().item()
        status = "✅" if not (has_nan or has_inf) else "❌"
        print(f"   {status} {name}: shape={weight.shape}, nan={has_nan}, inf={has_inf}")

    # Check indexer weights
    print("\n🔍 Checking layer 0 indexer weights...")
    indexer = attn.indexer

    for name in ["wq_b", "wk", "weights_proj"]:
        weight = getattr(indexer, name).weight
        has_nan = torch.isnan(weight).any().item()
        has_inf = torch.isinf(weight).any().item()
        mean = weight.float().mean().item()
        std = weight.float().std().item()
        status = "✅" if not (has_nan or has_inf) else "❌"
        print(f"   {status} {name}: nan={has_nan}, inf={has_inf}, mean={mean:.6f}, std={std:.6f}")

    # Check k_norm
    k_norm_weight = indexer.k_norm.weight
    k_norm_bias = indexer.k_norm.bias
    print(f"   k_norm.weight: nan={torch.isnan(k_norm_weight).any().item()}, mean={k_norm_weight.float().mean().item():.6f}")
    print(f"   k_norm.bias: nan={torch.isnan(k_norm_bias).any().item()}, mean={k_norm_bias.float().mean().item():.6f}")

    # Check if indexer weights are all zeros (not loaded)
    print("\n🔍 Checking if indexer weights were loaded (not all zeros)...")
    wq_b_nonzero = (indexer.wq_b.weight != 0).any().item()
    wk_nonzero = (indexer.wk.weight != 0).any().item()
    weights_proj_nonzero = (indexer.weights_proj.weight != 0).any().item()

    print(f"   indexer.wq_b has non-zero values: {wq_b_nonzero}")
    print(f"   indexer.wk has non-zero values: {wk_nonzero}")
    print(f"   indexer.weights_proj has non-zero values: {weights_proj_nonzero}")

    if not (wq_b_nonzero and wk_nonzero and weights_proj_nonzero):
        print("\n❌ PROBLEM: Some indexer weights are all zeros!")
        print("   This means the weights weren't loaded from the checkpoint.")
        print("   The indexer will select random tokens, causing gibberish.")

    # Run a forward pass with hooks to check intermediate values
    print("\n🔍 Running forward pass with debug hooks...")

    debug_info = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output

            if isinstance(out, torch.Tensor):
                debug_info[name] = {
                    "shape": list(out.shape),
                    "dtype": str(out.dtype),
                    "has_nan": torch.isnan(out).any().item(),
                    "has_inf": torch.isinf(out).any().item(),
                    "mean": out.float().mean().item() if out.numel() > 0 else 0,
                    "std": out.float().std().item() if out.numel() > 1 else 0,
                    "min": out.float().min().item() if out.numel() > 0 else 0,
                    "max": out.float().max().item() if out.numel() > 0 else 0,
                }
        return hook

    # Register hooks on key modules
    hooks = []
    hooks.append(model.model.embed_tokens.register_forward_hook(make_hook("embed_tokens")))
    hooks.append(model.model.layers[0].input_layernorm.register_forward_hook(make_hook("layer0_input_norm")))
    hooks.append(model.model.layers[0].self_attn.register_forward_hook(make_hook("layer0_attn")))
    hooks.append(model.model.layers[0].self_attn.indexer.register_forward_hook(make_hook("layer0_indexer")))
    hooks.append(model.model.layers[0].mlp.register_forward_hook(make_hook("layer0_mlp")))
    hooks.append(model.model.norm.register_forward_hook(make_hook("final_norm")))
    hooks.append(model.lm_head.register_forward_hook(make_hook("lm_head")))

    # Run forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Print debug info
    print("\n📊 Forward pass results:")
    for name, info in debug_info.items():
        status = "✅" if not (info["has_nan"] or info["has_inf"]) else "❌"
        print(f"\n   {status} {name}:")
        print(f"      shape: {info['shape']}, dtype: {info['dtype']}")
        print(f"      nan: {info['has_nan']}, inf: {info['has_inf']}")
        print(f"      mean: {info['mean']:.6f}, std: {info['std']:.6f}")
        print(f"      min: {info['min']:.6f}, max: {info['max']:.6f}")

    # Check logits
    logits = outputs.logits
    print("\n📊 Final logits:")
    print(f"   Shape: {logits.shape}")
    print(f"   Has NaN: {torch.isnan(logits).any().item()}")
    print(f"   Has Inf: {torch.isinf(logits).any().item()}")
    print(f"   Mean: {logits.float().mean().item():.6f}")
    print(f"   Std: {logits.float().std().item():.6f}")

    # Get top-5 predictions
    print("\n📊 Top-5 next token predictions:")
    last_logits = logits[0, -1, :]
    top5_values, top5_indices = torch.topk(last_logits, 5)
    for i, (idx, val) in enumerate(zip(top5_indices.tolist(), top5_values.tolist())):
        token = tokenizer.decode([idx])
        print(f"   {i+1}. token_id={idx}, logit={val:.4f}, decoded='{token}'")

    # Try generating a few tokens
    print("\n🔄 Generating 10 tokens...")
    with torch.no_grad():
        gen_outputs = model.generate(
            input_ids,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
    print(f"   Input: '{prompt}'")
    print(f"   Generated: '{generated_text}'")

    # Check if output looks like gibberish
    new_tokens = gen_outputs[0][input_ids.shape[1]:].tolist()
    print(f"   New token IDs: {new_tokens}")

    # Summary
    print("\n" + "=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)

    issues = []

    # Check for NaN/Inf
    for name, info in debug_info.items():
        if info["has_nan"]:
            issues.append(f"NaN values in {name}")
        if info["has_inf"]:
            issues.append(f"Inf values in {name}")

    # Check indexer weights
    if not wq_b_nonzero:
        issues.append("indexer.wq_b weights are all zeros (not loaded!)")
    if not wk_nonzero:
        issues.append("indexer.wk weights are all zeros (not loaded!)")
    if not weights_proj_nonzero:
        issues.append("indexer.weights_proj weights are all zeros (not loaded!)")

    # Check logits
    if torch.isnan(logits).any().item():
        issues.append("NaN in final logits")
    if torch.isinf(logits).any().item():
        issues.append("Inf in final logits")

    if issues:
        print("\n❌ ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("\n✅ No numerical issues detected in forward pass")
        print("\nIf still getting gibberish, possible causes:")
        print("   1. Index mask is masking out important tokens")
        print("   2. RoPE interleaved/non-interleaved mismatch")
        print("   3. Softmax scale calculation error")
        print("   4. Attention pattern is degenerate")

    return {
        "debug_info": debug_info,
        "issues": issues,
        "generated_text": generated_text,
        "new_tokens": new_tokens,
    }


@app.local_entrypoint()
def main(
    model_path: str = DEFAULT_MODEL_PATH,
    prompt: str = "Hello",
    use_local_copy: bool = True,
):
    """
    Debug a forward pass through DeepSeek V3.2.

    Examples:
        modal run sandbox/debug_forward_pass.py
        modal run sandbox/debug_forward_pass.py --prompt "What is 2+2?"
    """
    print("🔍 DeepSeek V3.2 Forward Pass Debugger")
    print("=" * 70)

    result = debug_forward_pass.remote(
        model_path=model_path,
        prompt=prompt,
        use_local_copy=use_local_copy,
    )

    if result.get("issues"):
        print("\n❌ Issues found - see above for details")
    else:
        print("\n✅ Forward pass looks OK")

    return result

