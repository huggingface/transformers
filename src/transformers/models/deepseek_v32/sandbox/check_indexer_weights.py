#!/usr/bin/env python3
"""
Modal app to check if DeepSeek V3.2 checkpoint has indexer weights.

If indexer weights are missing, the model will produce gibberish because
the sparse attention will select random tokens instead of important ones.

Usage:
    modal run sandbox/check_indexer_weights.py
    modal run sandbox/check_indexer_weights.py --model-path "deepseek-ai--DeepSeek-V3.2_bf16"
"""

import modal


# Configuration
MODEL_VOLUME_NAME = "models"
MODEL_MOUNT_PATH = "/mnt/model"
DEFAULT_MODEL_PATH = "deepseek-ai--DeepSeek-V3.2_bf16"

image = modal.Image.debian_slim(python_version="3.11").pip_install("safetensors")

app = modal.App("check-deepseek-indexer-weights", image=image)

model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME)


@app.function(
    volumes={MODEL_MOUNT_PATH: model_volume},
    timeout=60 * 10,
)
def check_indexer_weights(model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """
    Check if the DeepSeek V3.2 checkpoint contains indexer weights.
    
    Returns a dict with diagnostic information.
    """
    import json
    import os

    full_model_path = f"{MODEL_MOUNT_PATH}/{model_path}"

    print("=" * 70)
    print("DeepSeek V3.2 Indexer Weight Check")
    print("=" * 70)
    print(f"Model path: {full_model_path}")

    if not os.path.exists(full_model_path):
        print(f"❌ ERROR: Model path does not exist: {full_model_path}")
        return {"error": "Model path not found", "has_indexer": False}

    # Check for safetensors index file
    index_path = os.path.join(full_model_path, "model.safetensors.index.json")

    if not os.path.exists(index_path):
        # Try single file
        single_file = os.path.join(full_model_path, "model.safetensors")
        if os.path.exists(single_file):
            print("⚠️  Single safetensors file found (no index)")
            print("   Checking weights directly...")
            from safetensors import safe_open
            with safe_open(single_file, framework="pt") as f:
                all_weights = list(f.keys())
        else:
            print("❌ ERROR: No model.safetensors.index.json or model.safetensors found")
            return {"error": "No safetensors files found", "has_indexer": False}
    else:
        print(f"✅ Found index file: {index_path}")
        with open(index_path, "r") as f:
            index = json.load(f)
        all_weights = list(index.get("weight_map", {}).keys())

    print(f"\n📊 Total weights in checkpoint: {len(all_weights)}")

    # Check for indexer weights
    indexer_weights = [w for w in all_weights if "indexer" in w.lower()]

    print(f"\n🔍 Indexer weights found: {len(indexer_weights)}")

    if not indexer_weights:
        print("\n" + "=" * 70)
        print("❌ NO INDEXER WEIGHTS FOUND!")
        print("=" * 70)
        print("""
This is likely the cause of gibberish output!

The DeepSeek V3.2 model uses sparse attention with a Lightning Indexer.
When seq_len > 1 (during prefill), the indexer selects which tokens to attend to.

If indexer weights are missing:
  - The indexer modules are randomly initialized
  - Random tokens are selected for attention
  - The model produces gibberish

Expected indexer weight patterns:
  - model.layers.X.self_attn.indexer.wq_b.weight
  - model.layers.X.self_attn.indexer.wk.weight
  - model.layers.X.self_attn.indexer.k_norm.weight
  - model.layers.X.self_attn.indexer.k_norm.bias
  - model.layers.X.self_attn.indexer.weights_proj.weight

Solutions:
  1. Use a checkpoint that includes indexer weights (original FP8 or properly converted)
  2. Disable sparse attention in your model implementation
  3. Convert the model with indexer weights included
""")
        print("=" * 70)

        # Show what weights ARE present for first layer attention
        print("\n📋 Attention weights present in layer 0:")
        layer0_attn = [w for w in all_weights if "layers.0.self_attn" in w]
        for w in sorted(layer0_attn)[:20]:
            print(f"  {w}")
        if len(layer0_attn) > 20:
            print(f"  ... and {len(layer0_attn) - 20} more")

        return {
            "has_indexer": False,
            "total_weights": len(all_weights),
            "indexer_weights": 0,
            "layer0_attn_weights": layer0_attn,
            "diagnosis": "MISSING_INDEXER_WEIGHTS",
        }

    # Indexer weights found - analyze them
    print("\n✅ Indexer weights found!")
    print("\n📋 Sample indexer weights:")
    for w in sorted(indexer_weights)[:10]:
        print(f"  {w}")
    if len(indexer_weights) > 10:
        print(f"  ... and {len(indexer_weights) - 10} more")

    # Check which layers have indexer weights
    layers_with_indexer = set()
    for w in indexer_weights:
        # Extract layer number from weight name
        parts = w.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    layer_num = int(parts[i + 1])
                    layers_with_indexer.add(layer_num)
                except ValueError:
                    pass

    print(f"\n📊 Layers with indexer weights: {len(layers_with_indexer)}")
    if layers_with_indexer:
        print(f"   Range: {min(layers_with_indexer)} to {max(layers_with_indexer)}")

    # Check for expected weight patterns
    expected_patterns = [
        "indexer.wq_b.weight",
        "indexer.wk.weight",
        "indexer.k_norm.weight",
        "indexer.k_norm.bias",
        "indexer.weights_proj.weight",
    ]

    print("\n🔍 Checking expected indexer weight patterns:")
    for pattern in expected_patterns:
        matches = [w for w in indexer_weights if pattern in w]
        status = "✅" if matches else "❌"
        print(f"  {status} {pattern}: {len(matches)} found")

    # Calculate expected vs actual
    # DeepSeek V3.2 has 61 layers, each should have 5 indexer weights
    expected_indexer_weights = 61 * 5  # 305
    actual_indexer_weights = len(indexer_weights)

    print("\n📊 Indexer weight count:")
    print(f"   Expected (61 layers × 5 weights): {expected_indexer_weights}")
    print(f"   Actual: {actual_indexer_weights}")

    if actual_indexer_weights < expected_indexer_weights * 0.9:
        print(f"   ⚠️  Only {actual_indexer_weights / expected_indexer_weights * 100:.1f}% of expected indexer weights present!")
    else:
        print("   ✅ Indexer weights look complete")

    print("\n" + "=" * 70)
    print("✅ INDEXER WEIGHTS PRESENT - Model should work correctly")
    print("=" * 70)

    return {
        "has_indexer": True,
        "total_weights": len(all_weights),
        "indexer_weights": len(indexer_weights),
        "layers_with_indexer": sorted(layers_with_indexer),
        "diagnosis": "OK",
    }


@app.function(
    volumes={MODEL_MOUNT_PATH: model_volume},
    timeout=60 * 10,
)
def compare_checkpoints(
    checkpoint1: str = "deepseek-ai--DeepSeek-V3.2_bf16",
    checkpoint2: str = "deepseek-ai--DeepSeek-V3.2",
) -> dict:
    """
    Compare two checkpoints to see weight differences.
    
    Useful for understanding what's different between FP8 and BF16 versions.
    """
    import json
    import os

    print("=" * 70)
    print("Checkpoint Comparison")
    print("=" * 70)

    results = {}

    for name, path in [("checkpoint1", checkpoint1), ("checkpoint2", checkpoint2)]:
        full_path = f"{MODEL_MOUNT_PATH}/{path}"
        index_path = os.path.join(full_path, "model.safetensors.index.json")

        print(f"\n📁 {name}: {path}")

        if not os.path.exists(index_path):
            print("   ❌ No index file found")
            results[name] = {"error": "No index file", "weights": []}
            continue

        with open(index_path, "r") as f:
            index = json.load(f)

        weights = list(index.get("weight_map", {}).keys())
        indexer_weights = [w for w in weights if "indexer" in w.lower()]

        print(f"   Total weights: {len(weights)}")
        print(f"   Indexer weights: {len(indexer_weights)}")

        results[name] = {
            "path": path,
            "total_weights": len(weights),
            "indexer_weights": len(indexer_weights),
            "weights": set(weights),
        }

    # Compare
    if results.get("checkpoint1", {}).get("weights") and results.get("checkpoint2", {}).get("weights"):
        w1 = results["checkpoint1"]["weights"]
        w2 = results["checkpoint2"]["weights"]

        only_in_1 = w1 - w2
        only_in_2 = w2 - w1

        print("\n📊 Comparison:")
        print(f"   Weights only in {checkpoint1}: {len(only_in_1)}")
        print(f"   Weights only in {checkpoint2}: {len(only_in_2)}")

        # Check specifically for indexer differences
        indexer_only_in_1 = [w for w in only_in_1 if "indexer" in w.lower()]
        indexer_only_in_2 = [w for w in only_in_2 if "indexer" in w.lower()]

        print(f"\n   Indexer weights only in {checkpoint1}: {len(indexer_only_in_1)}")
        print(f"   Indexer weights only in {checkpoint2}: {len(indexer_only_in_2)}")

        if indexer_only_in_1:
            print(f"\n   Sample indexer weights only in {checkpoint1}:")
            for w in sorted(indexer_only_in_1)[:5]:
                print(f"     {w}")

        if indexer_only_in_2:
            print(f"\n   Sample indexer weights only in {checkpoint2}:")
            for w in sorted(indexer_only_in_2)[:5]:
                print(f"     {w}")

    return results


@app.local_entrypoint()
def main(
    model_path: str = DEFAULT_MODEL_PATH,
    compare: bool = False,
    checkpoint2: str = "deepseek-ai--DeepSeek-V3.2",
):
    """
    Check if DeepSeek V3.2 checkpoint has indexer weights.
    
    Examples:
        # Check BF16 checkpoint
        modal run sandbox/check_indexer_weights.py
        
        # Check specific checkpoint
        modal run sandbox/check_indexer_weights.py --model-path "deepseek-ai--DeepSeek-V3.2"
        
        # Compare two checkpoints
        modal run sandbox/check_indexer_weights.py --compare --checkpoint2 "deepseek-ai--DeepSeek-V3.2"
    """
    print("🔍 DeepSeek V3.2 Indexer Weight Checker")
    print("=" * 70)

    if compare:
        print(f"Comparing: {model_path} vs {checkpoint2}")
        result = compare_checkpoints.remote(model_path, checkpoint2)
    else:
        print(f"Checking: {model_path}")
        result = check_indexer_weights.remote(model_path)

    print("\n" + "=" * 70)
    print("Result Summary:")
    print("=" * 70)

    if isinstance(result, dict):
        if result.get("has_indexer"):
            print("✅ Indexer weights FOUND - model should work correctly")
        elif result.get("has_indexer") is False:
            print("❌ Indexer weights MISSING - this causes gibberish output!")
            print("\nThe model will use randomly initialized indexer weights,")
            print("causing sparse attention to select random tokens.")

        if "diagnosis" in result:
            print(f"\nDiagnosis: {result['diagnosis']}")

    return result

