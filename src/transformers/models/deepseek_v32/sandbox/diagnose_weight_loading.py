#!/usr/bin/env python3
"""
Modal app to diagnose weight loading issues for DeepSeek V3.2.

Compares what weights the model expects vs what's in the checkpoint
to identify mismatches that could cause gibberish output.

Usage:
    modal run sandbox/diagnose_weight_loading.py
    modal run sandbox/diagnose_weight_loading.py --model-path "deepseek-ai--DeepSeek-V3.2_bf16"
"""

import modal


# Configuration
GPU_TYPE = "H100"
GPU_COUNT = 1  # Only need 1 GPU for empty model
MODEL_VOLUME_NAME = "models"
MODEL_MOUNT_PATH = "/mnt/model"
DEFAULT_MODEL_PATH = "deepseek-ai--DeepSeek-V3.2_bf16"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.4.0",
        "accelerate>=0.34.0",
        "safetensors",
    )
    .run_commands(
        "pip install --no-cache-dir git+https://github.com/jyliu24/transformers.git",
    )
)

app = modal.App("diagnose-deepseek-weights", image=image)

model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME)


@app.function(
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    volumes={MODEL_MOUNT_PATH: model_volume},
    timeout=60 * 30,
)
def diagnose_weight_loading(model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """
    Diagnose weight loading issues by comparing model expectations vs checkpoint.
    """
    import json
    import os

    from accelerate import init_empty_weights

    from transformers import AutoConfig, AutoModelForCausalLM

    full_model_path = f"{MODEL_MOUNT_PATH}/{model_path}"

    print("=" * 70)
    print("DeepSeek V3.2 Weight Loading Diagnosis")
    print("=" * 70)
    print(f"Model path: {full_model_path}")

    # Step 1: Load checkpoint weight names
    print("\n📥 Step 1: Loading checkpoint weight names...")
    index_path = os.path.join(full_model_path, "model.safetensors.index.json")

    if not os.path.exists(index_path):
        print(f"❌ ERROR: No index file found at {index_path}")
        return {"error": "No index file"}

    with open(index_path, "r") as f:
        index = json.load(f)

    checkpoint_weights = set(index.get("weight_map", {}).keys())
    print(f"   Checkpoint has {len(checkpoint_weights)} weights")

    # Step 2: Create empty model to get expected weight names
    print("\n📥 Step 2: Creating empty model to get expected weight names...")
    config = AutoConfig.from_pretrained(full_model_path, trust_remote_code=True)
    print(f"   Model type: {config.model_type}")
    print(f"   Architectures: {getattr(config, 'architectures', 'N/A')}")

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    model_weights = set(model.state_dict().keys())
    print(f"   Model expects {len(model_weights)} weights")

    # Step 3: Compare weight names
    print("\n📊 Step 3: Comparing weight names...")

    # Weights in checkpoint but not expected by model
    unexpected = checkpoint_weights - model_weights
    # Weights expected by model but not in checkpoint
    missing = model_weights - checkpoint_weights
    # Weights that match
    matched = checkpoint_weights & model_weights

    print(f"\n   ✅ Matched weights: {len(matched)}")
    print(f"   ⚠️  Unexpected (in checkpoint, not in model): {len(unexpected)}")
    print(f"   ❌ Missing (in model, not in checkpoint): {len(missing)}")

    # Step 4: Analyze indexer weights specifically
    print("\n🔍 Step 4: Analyzing indexer weights...")

    checkpoint_indexer = [w for w in checkpoint_weights if "indexer" in w.lower()]
    model_indexer = [w for w in model_weights if "indexer" in w.lower()]
    matched_indexer = [w for w in matched if "indexer" in w.lower()]
    unexpected_indexer = [w for w in unexpected if "indexer" in w.lower()]
    missing_indexer = [w for w in missing if "indexer" in w.lower()]

    print(f"\n   Checkpoint indexer weights: {len(checkpoint_indexer)}")
    print(f"   Model expects indexer weights: {len(model_indexer)}")
    print(f"   ✅ Matched indexer weights: {len(matched_indexer)}")
    print(f"   ⚠️  Unexpected indexer weights: {len(unexpected_indexer)}")
    print(f"   ❌ Missing indexer weights: {len(missing_indexer)}")

    if missing_indexer:
        print("\n   ❌ MISSING INDEXER WEIGHTS (model expects but checkpoint doesn't have):")
        for w in sorted(missing_indexer)[:20]:
            print(f"      {w}")
        if len(missing_indexer) > 20:
            print(f"      ... and {len(missing_indexer) - 20} more")

    if unexpected_indexer:
        print("\n   ⚠️  UNEXPECTED INDEXER WEIGHTS (checkpoint has but model doesn't expect):")
        for w in sorted(unexpected_indexer)[:20]:
            print(f"      {w}")
        if len(unexpected_indexer) > 20:
            print(f"      ... and {len(unexpected_indexer) - 20} more")

    # Step 5: Check for systematic naming differences
    print("\n🔍 Step 5: Checking for systematic naming patterns...")

    # Look for patterns in unexpected/missing weights
    if unexpected:
        print("\n   Sample UNEXPECTED weights (checkpoint has, model doesn't expect):")
        for w in sorted(unexpected)[:15]:
            print(f"      {w}")

    if missing:
        print("\n   Sample MISSING weights (model expects, checkpoint doesn't have):")
        for w in sorted(missing)[:15]:
            print(f"      {w}")

    # Step 6: Check layer 0 attention in detail
    print("\n🔍 Step 6: Layer 0 self_attn weight comparison...")

    ckpt_layer0_attn = sorted([w for w in checkpoint_weights if "layers.0.self_attn" in w])
    model_layer0_attn = sorted([w for w in model_weights if "layers.0.self_attn" in w])

    print(f"\n   Checkpoint layer 0 attn weights: {len(ckpt_layer0_attn)}")
    for w in ckpt_layer0_attn:
        status = "✅" if w in model_weights else "⚠️ "
        print(f"      {status} {w}")

    print(f"\n   Model layer 0 attn weights: {len(model_layer0_attn)}")
    for w in model_layer0_attn:
        status = "✅" if w in checkpoint_weights else "❌"
        print(f"      {status} {w}")

    # Step 7: Summary and diagnosis
    print("\n" + "=" * 70)
    print("📋 DIAGNOSIS SUMMARY")
    print("=" * 70)

    issues = []

    if len(missing_indexer) > 0:
        issues.append(f"❌ {len(missing_indexer)} indexer weights missing from checkpoint")

    if len(matched_indexer) == 0 and len(checkpoint_indexer) > 0 and len(model_indexer) > 0:
        issues.append("❌ Indexer weight names don't match between model and checkpoint!")

    if len(missing) > len(checkpoint_weights) * 0.1:
        issues.append(f"❌ {len(missing)} weights ({len(missing)/len(model_weights)*100:.1f}%) missing from checkpoint")

    # Check for scale_inv pattern (FP8 quantization scales)
    scale_inv_unexpected = [w for w in unexpected if "scale_inv" in w]
    if scale_inv_unexpected:
        print(f"\n   ℹ️  {len(scale_inv_unexpected)} scale_inv weights in checkpoint (FP8 quantization scales)")
        print("      These are expected to be ignored when loading BF16")

    if not issues:
        print("\n✅ No major issues detected!")
        print("   Weight names match between model and checkpoint.")
        if matched_indexer:
            print(f"   ✅ All {len(matched_indexer)} indexer weights should load correctly.")
    else:
        print("\n⚠️  ISSUES DETECTED:")
        for issue in issues:
            print(f"   {issue}")

    # Check if the issue might be elsewhere
    if len(matched_indexer) == len(model_indexer) and len(model_indexer) > 0:
        print("\n💡 Indexer weights look fine. If you're still getting gibberish, check:")
        print("   1. RoPE implementation (interleaved vs non-interleaved)")
        print("   2. Attention mask computation")
        print("   3. MoE routing logic")
        print("   4. Softmax scale calculation")

    return {
        "checkpoint_weights": len(checkpoint_weights),
        "model_weights": len(model_weights),
        "matched": len(matched),
        "unexpected": len(unexpected),
        "missing": len(missing),
        "checkpoint_indexer": len(checkpoint_indexer),
        "model_indexer": len(model_indexer),
        "matched_indexer": len(matched_indexer),
        "missing_indexer": len(missing_indexer),
        "unexpected_indexer": len(unexpected_indexer),
        "issues": issues,
    }


@app.local_entrypoint()
def main(model_path: str = DEFAULT_MODEL_PATH):
    """
    Diagnose weight loading issues for DeepSeek V3.2.

    Examples:
        modal run sandbox/diagnose_weight_loading.py
        modal run sandbox/diagnose_weight_loading.py --model-path "deepseek-ai--DeepSeek-V3.2"
    """
    print("🔍 DeepSeek V3.2 Weight Loading Diagnostics")
    print("=" * 70)
    print(f"Analyzing: {model_path}")

    result = diagnose_weight_loading.remote(model_path)

    print("\n" + "=" * 70)
    print("Final Summary:")
    print("=" * 70)

    if result.get("issues"):
        print("⚠️  Issues found:")
        for issue in result["issues"]:
            print(f"   {issue}")
    else:
        print("✅ Weight loading looks correct")
        print("\nIf still getting gibberish, the bug is likely in:")
        print("   - Attention computation")
        print("   - RoPE implementation")
        print("   - Indexer sparse mask application")
        print("   - MoE routing")

    return result

