#!/usr/bin/env python3
"""
Modal app to check the actual dtype of weights in the DeepSeek V3.2 checkpoint.

If the checkpoint has FP8 weights but they're being loaded as BF16 without
dequantization, this causes gibberish output.

Usage:
    modal run sandbox/check_weight_dtype.py
    modal run sandbox/check_weight_dtype.py --model-path "deepseek-ai--DeepSeek-V3.2_bf16"
"""

import modal


# Configuration
MODEL_VOLUME_NAME = "models"
MODEL_MOUNT_PATH = "/mnt/model"
DEFAULT_MODEL_PATH = "deepseek-ai--DeepSeek-V3.2_bf16"

image = modal.Image.debian_slim(python_version="3.11").pip_install("safetensors", "torch")

app = modal.App("check-deepseek-weight-dtype", image=image)

model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME)


@app.function(
    volumes={MODEL_MOUNT_PATH: model_volume},
    timeout=60 * 10,
)
def check_weight_dtype(model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """
    Check the actual dtype of weights in the checkpoint.
    """
    import os

    from safetensors import safe_open

    full_model_path = f"{MODEL_MOUNT_PATH}/{model_path}"

    print("=" * 70)
    print("DeepSeek V3.2 Weight Dtype Check")
    print("=" * 70)
    print(f"Model path: {full_model_path}")

    # Find first safetensor file
    safetensor_files = sorted([
        f for f in os.listdir(full_model_path)
        if f.endswith(".safetensors") and "model-" in f
    ])

    if not safetensor_files:
        # Try single file
        if os.path.exists(os.path.join(full_model_path, "model.safetensors")):
            safetensor_files = ["model.safetensors"]
        else:
            print("❌ No safetensor files found!")
            return {"error": "No safetensor files"}

    print(f"\n📁 Found {len(safetensor_files)} safetensor files")
    first_file = safetensor_files[0]
    print(f"   Checking: {first_file}")

    # Open first safetensor and check dtypes
    file_path = os.path.join(full_model_path, first_file)

    with safe_open(file_path, framework="pt") as f:
        tensor_names = list(f.keys())
        print(f"\n📊 Tensors in {first_file}: {len(tensor_names)}")

        # Check a few key tensors
        dtype_counts = {}
        weight_examples = {}
        scale_examples = {}

        for name in tensor_names[:50]:  # Check first 50 tensors
            tensor = f.get_tensor(name)
            dtype_str = str(tensor.dtype)

            if dtype_str not in dtype_counts:
                dtype_counts[dtype_str] = 0
            dtype_counts[dtype_str] += 1

            # Collect examples
            if "weight" in name and "scale" not in name:
                if len(weight_examples) < 5:
                    weight_examples[name] = {
                        "dtype": dtype_str,
                        "shape": list(tensor.shape),
                        "mean": tensor.float().mean().item() if tensor.numel() > 0 else 0,
                        "std": tensor.float().std().item() if tensor.numel() > 1 else 0,
                        "min": tensor.float().min().item() if tensor.numel() > 0 else 0,
                        "max": tensor.float().max().item() if tensor.numel() > 0 else 0,
                    }
            elif "scale_inv" in name:
                if len(scale_examples) < 5:
                    scale_examples[name] = {
                        "dtype": dtype_str,
                        "shape": list(tensor.shape),
                        "mean": tensor.float().mean().item() if tensor.numel() > 0 else 0,
                    }

        print("\n📊 Dtype distribution:")
        for dtype, count in sorted(dtype_counts.items()):
            print(f"   {dtype}: {count} tensors")

        print("\n📋 Sample weight tensors:")
        for name, info in weight_examples.items():
            print(f"\n   {name}:")
            print(f"      dtype: {info['dtype']}")
            print(f"      shape: {info['shape']}")
            print(f"      mean: {info['mean']:.6f}")
            print(f"      std: {info['std']:.6f}")
            print(f"      range: [{info['min']:.6f}, {info['max']:.6f}]")

        if scale_examples:
            print("\n📋 Sample scale_inv tensors:")
            for name, info in scale_examples.items():
                print(f"\n   {name}:")
                print(f"      dtype: {info['dtype']}")
                print(f"      shape: {info['shape']}")
                print(f"      mean: {info['mean']:.6f}")

        # Check specifically for FP8 dtype
        fp8_dtypes = ["torch.float8_e4m3fn", "torch.float8_e5m2"]
        has_fp8 = any(dt in dtype_counts for dt in fp8_dtypes)

        print("\n" + "=" * 70)
        print("📋 DIAGNOSIS")
        print("=" * 70)

        if has_fp8:
            print("\n❌ CHECKPOINT HAS FP8 WEIGHTS!")
            print("""
This is likely the cause of gibberish!

The weights are stored in FP8 format but being loaded as-is without
proper dequantization. FP8 bit patterns interpreted as BF16 produce
garbage values.

Solutions:
1. Use FineGrainedFP8Config or FbgemmFp8Config when loading:
   
   from transformers import FineGrainedFP8Config
   model = AutoModelForCausalLM.from_pretrained(
       model_path,
       quantization_config=FineGrainedFP8Config(weight_block_size=[128, 128]),
   )

2. Use a properly dequantized BF16 checkpoint

3. Write a conversion script that dequantizes FP8 -> BF16:
   dequantized = fp8_weight.float() * scale_inv
""")
            return {
                "has_fp8": True,
                "dtype_counts": dtype_counts,
                "diagnosis": "FP8_WEIGHTS_NOT_DEQUANTIZED",
            }

        elif "torch.bfloat16" in dtype_counts:
            print("\n✅ Weights are in BF16 format")

            # Check if weights look reasonable
            all_reasonable = True
            for name, info in weight_examples.items():
                # Typical weight ranges for neural networks
                if abs(info["mean"]) > 10 or info["std"] > 10:
                    all_reasonable = False
                    print(f"\n⚠️  {name} has unusual values:")
                    print(f"   mean={info['mean']:.6f}, std={info['std']:.6f}")

            if all_reasonable:
                print("   Weight values look reasonable (mean/std in normal range)")
                print("\n   If still getting gibberish, the issue is likely in:")
                print("   - Attention computation")
                print("   - RoPE implementation")
                print("   - Indexer sparse mask application")
            else:
                print("\n⚠️  Some weights have unusual values!")
                print("   This could indicate corrupted conversion from FP8")

            return {
                "has_fp8": False,
                "dtype_counts": dtype_counts,
                "weight_examples": weight_examples,
                "diagnosis": "BF16_OK" if all_reasonable else "BF16_SUSPICIOUS",
            }

        else:
            print(f"\nℹ️  Weights are in: {list(dtype_counts.keys())}")
            return {
                "has_fp8": False,
                "dtype_counts": dtype_counts,
                "diagnosis": "UNKNOWN_DTYPE",
            }


@app.function(
    volumes={MODEL_MOUNT_PATH: model_volume},
    timeout=60 * 30,
)
def check_dequantization_needed(model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """
    Check if checkpoint needs dequantization and demonstrate correct loading.
    """
    import json
    import os

    import torch
    from safetensors import safe_open

    full_model_path = f"{MODEL_MOUNT_PATH}/{model_path}"

    print("=" * 70)
    print("DeepSeek V3.2 Dequantization Check")
    print("=" * 70)

    # Find first safetensor file with indexer weights
    index_path = os.path.join(full_model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        print("❌ No index file found")
        return {"error": "No index file"}

    with open(index_path) as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})

    # Find which file has layer 0 indexer weights
    indexer_weight_name = "model.layers.0.self_attn.indexer.wq_b.weight"
    scale_name = "model.layers.0.self_attn.indexer.wq_b.weight_scale_inv"

    if indexer_weight_name not in weight_map:
        print(f"❌ {indexer_weight_name} not in checkpoint")
        return {"error": "Indexer weight not found"}

    weight_file = weight_map[indexer_weight_name]
    print(f"\n📁 Indexer weight file: {weight_file}")

    file_path = os.path.join(full_model_path, weight_file)

    with safe_open(file_path, framework="pt") as f:
        # Load the weight
        weight = f.get_tensor(indexer_weight_name)
        print(f"\n📊 Indexer weight ({indexer_weight_name}):")
        print(f"   dtype: {weight.dtype}")
        print(f"   shape: {weight.shape}")

        # Check if FP8
        is_fp8 = weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
        print(f"   is_fp8: {is_fp8}")

        if is_fp8:
            print(f"\n   Raw FP8 values (first 10): {weight.flatten()[:10].tolist()}")

            # Check if scale exists
            if scale_name in f.keys():
                scale = f.get_tensor(scale_name)
                print(f"\n📊 Scale tensor ({scale_name}):")
                print(f"   dtype: {scale.dtype}")
                print(f"   shape: {scale.shape}")
                print(f"   mean: {scale.float().mean().item():.6f}")

                # Demonstrate dequantization
                print("\n🔧 Demonstrating dequantization...")
                block_size = 128

                # Dequantize following reference implementation
                shape = weight.shape
                w = weight.view(
                    shape[0] // block_size, block_size,
                    shape[1] // block_size, block_size
                ).transpose(1, 2).contiguous().view(-1, block_size * block_size)

                w_dequant = (w.float() * scale.view(-1, 1).float()).view(
                    shape[0] // block_size, shape[1] // block_size,
                    block_size, block_size
                ).transpose(1, 2).contiguous().view(shape)

                print("\n📊 Dequantized weight:")
                print(f"   dtype: {w_dequant.dtype}")
                print(f"   mean: {w_dequant.mean().item():.6f}")
                print(f"   std: {w_dequant.std().item():.6f}")
                print(f"   range: [{w_dequant.min().item():.6f}, {w_dequant.max().item():.6f}]")

                return {
                    "is_fp8": True,
                    "has_scale": True,
                    "needs_dequantization": True,
                    "diagnosis": "FP8_NEEDS_DEQUANT",
                }
            else:
                print(f"\n⚠️  Scale tensor {scale_name} not found!")
                return {
                    "is_fp8": True,
                    "has_scale": False,
                    "diagnosis": "FP8_MISSING_SCALE",
                }
        else:
            # Already BF16/FP32
            print(f"\n   mean: {weight.float().mean().item():.6f}")
            print(f"   std: {weight.float().std().item():.6f}")
            print(f"   range: [{weight.float().min().item():.6f}, {weight.float().max().item():.6f}]")

            # Check if scale_inv exists anyway (leftover from conversion)
            has_scale = scale_name in weight_map
            print(f"\n   Has scale_inv tensor: {has_scale}")

            return {
                "is_fp8": False,
                "has_scale": has_scale,
                "needs_dequantization": False,
                "diagnosis": "BF16_OK",
            }


@app.local_entrypoint()
def main(
    model_path: str = DEFAULT_MODEL_PATH,
    check_dequant: bool = False,
):
    """
    Check weight dtypes in DeepSeek V3.2 checkpoint.

    Examples:
        modal run sandbox/check_weight_dtype.py
        modal run sandbox/check_weight_dtype.py --check-dequant
    """
    print("🔍 DeepSeek V3.2 Weight Dtype Checker")
    print("=" * 70)

    if check_dequant:
        result = check_dequantization_needed.remote(model_path)
    else:
        result = check_weight_dtype.remote(model_path)

    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)

    diagnosis = result.get("diagnosis", "UNKNOWN")
    print(f"Diagnosis: {diagnosis}")

    if "FP8" in diagnosis:
        print("""
⚠️  The checkpoint has FP8 weights!

To load correctly, use FP8 quantization config:

    from transformers import AutoModelForCausalLM, FineGrainedFP8Config

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=FineGrainedFP8Config(weight_block_size=[128, 128]),
        device_map="auto",
        trust_remote_code=True,
    )

Or convert to BF16 first using the dequantization formula:
    dequantized = fp8_weight * scale_inv
""")

    return result

