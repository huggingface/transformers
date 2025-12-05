#!/usr/bin/env python3
"""
Modal app to debug weight name mapping issues between
DeepSeek V3.2 checkpoints and the HuggingFace implementation.

Usage:
    modal run sandbox/debug_weight_mapping.py
    modal run sandbox/debug_weight_mapping.py --model-path "deepseek-ai--DeepSeek-V3.2_bf16"
"""

import modal


# Configuration
MODEL_VOLUME_NAME = "models"
MODEL_MOUNT_PATH = "/mnt/model"
DEFAULT_MODEL_PATH = "deepseek-ai--DeepSeek-V3.2_bf16"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "safetensors",
    )
)

app = modal.App("debug-deepseek-weight-mapping", image=image)

model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME)

# Expected HF naming patterns
HF_PATTERNS = {
    "self_attn.q_a_proj",
    "self_attn.q_a_layernorm",
    "self_attn.q_b_proj",
    "self_attn.kv_a_proj_with_mqa",
    "self_attn.kv_a_layernorm",
    "self_attn.kv_b_proj",
    "self_attn.o_proj",
    "self_attn.indexer",
    "input_layernorm",
    "post_attention_layernorm",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "mlp.gate.weight",
    "mlp.gate.e_score_correction_bias",
    "mlp.experts",
    "mlp.shared_experts",
    "model.embed_tokens",
    "model.norm",
    "lm_head",
}

# Reference naming patterns
REFERENCE_PATTERNS = {
    "attn.wq_a",
    "attn.q_norm",
    "attn.wq_b",
    "attn.wkv_a",
    "attn.kv_norm",
    "attn.wkv_b",
    "attn.wo",
    "attn.indexer",
    "attn_norm",
    "ffn_norm",
    "ffn.w1",
    "ffn.w2",
    "ffn.w3",
    "ffn.gate.weight",
    "ffn.gate.bias",
    "ffn.experts",
    "ffn.shared_experts",
    "embed",
    "norm",
    "head",
}

# Weight name mappings (reference -> HF)
WEIGHT_MAPPINGS = [
    ("layers.{}.attn.wq_a", "model.layers.{}.self_attn.q_a_proj"),
    ("layers.{}.attn.q_norm", "model.layers.{}.self_attn.q_a_layernorm"),
    ("layers.{}.attn.wq_b", "model.layers.{}.self_attn.q_b_proj"),
    ("layers.{}.attn.wkv_a", "model.layers.{}.self_attn.kv_a_proj_with_mqa"),
    ("layers.{}.attn.kv_norm", "model.layers.{}.self_attn.kv_a_layernorm"),
    ("layers.{}.attn.wkv_b", "model.layers.{}.self_attn.kv_b_proj"),
    ("layers.{}.attn.wo", "model.layers.{}.self_attn.o_proj"),
    ("layers.{}.attn.indexer.wq_b", "model.layers.{}.self_attn.indexer.wq_b"),
    ("layers.{}.attn.indexer.wk", "model.layers.{}.self_attn.indexer.wk"),
    ("layers.{}.attn.indexer.k_norm", "model.layers.{}.self_attn.indexer.k_norm"),
    ("layers.{}.attn.indexer.weights_proj", "model.layers.{}.self_attn.indexer.weights_proj"),
    ("layers.{}.attn_norm", "model.layers.{}.input_layernorm"),
    ("layers.{}.ffn_norm", "model.layers.{}.post_attention_layernorm"),
    ("layers.{}.ffn.w1", "model.layers.{}.mlp.gate_proj"),
    ("layers.{}.ffn.w2", "model.layers.{}.mlp.down_proj"),
    ("layers.{}.ffn.w3", "model.layers.{}.mlp.up_proj"),
    ("layers.{}.ffn.gate.weight", "model.layers.{}.mlp.gate.weight"),
    ("layers.{}.ffn.gate.bias", "model.layers.{}.mlp.gate.e_score_correction_bias"),
    ("layers.{}.ffn.experts.{}.w1", "model.layers.{}.mlp.experts.{}.gate_proj"),
    ("layers.{}.ffn.experts.{}.w2", "model.layers.{}.mlp.experts.{}.down_proj"),
    ("layers.{}.ffn.experts.{}.w3", "model.layers.{}.mlp.experts.{}.up_proj"),
    ("layers.{}.ffn.shared_experts.w1", "model.layers.{}.mlp.shared_experts.gate_proj"),
    ("layers.{}.ffn.shared_experts.w2", "model.layers.{}.mlp.shared_experts.down_proj"),
    ("layers.{}.ffn.shared_experts.w3", "model.layers.{}.mlp.shared_experts.up_proj"),
    ("embed.weight", "model.embed_tokens.weight"),
    ("norm.weight", "model.norm.weight"),
    ("head.weight", "lm_head.weight"),
]


@app.function(
    volumes={MODEL_MOUNT_PATH: model_volume},
    timeout=60 * 30,
)
def debug_weight_mapping(
    model_path: str = DEFAULT_MODEL_PATH,
    max_files: int = 2,
) -> dict:
    """
    Debug weight name mapping by checking checkpoint keys.
    """
    from pathlib import Path

    from safetensors import safe_open

    full_model_path = f"{MODEL_MOUNT_PATH}/{model_path}"

    print("=" * 70)
    print("DeepSeek V3.2 Weight Mapping Debugger")
    print("=" * 70)

    # Find checkpoint files
    print(f"\n📂 Checking model path: {full_model_path}")

    model_dir = Path(full_model_path)
    if not model_dir.exists():
        print(f"❌ ERROR: Path does not exist: {full_model_path}")
        return {"error": "Path not found"}

    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    print(f"   Found {len(safetensor_files)} safetensor files")

    if not safetensor_files:
        print("❌ ERROR: No safetensor files found")
        return {"error": "No checkpoint files"}

    # Load keys from first few files
    print(f"\n📥 Loading keys from first {max_files} file(s)...")
    ckpt_keys = []

    for sf_file in safetensor_files[:max_files]:
        print(f"   Loading: {sf_file.name}")
        with safe_open(sf_file, framework="pt") as f:
            file_keys = list(f.keys())
            ckpt_keys.extend(file_keys)
            print(f"      Found {len(file_keys)} keys")

    print(f"\n   Total keys loaded: {len(ckpt_keys)}")

    # Detect naming style
    print("\n🔍 Detecting naming convention...")

    hf_score = 0
    ref_score = 0

    for key in ckpt_keys:
        # HF patterns
        if "self_attn" in key:
            hf_score += 1
        if "q_a_proj" in key or "kv_a_proj" in key:
            hf_score += 1
        if "input_layernorm" in key or "post_attention_layernorm" in key:
            hf_score += 1
        if "gate_proj" in key or "up_proj" in key or "down_proj" in key:
            hf_score += 1
        if "e_score_correction_bias" in key:
            hf_score += 1
        if "model.embed_tokens" in key:
            hf_score += 1

        # Reference patterns
        if ".attn." in key and "self_attn" not in key:
            ref_score += 1
        if ".wq_a" in key or ".wkv_a" in key:
            ref_score += 1
        if "attn_norm" in key or "ffn_norm" in key:
            ref_score += 1
        if ".w1." in key or ".w2." in key or ".w3." in key:
            ref_score += 1
        if "gate.bias" in key and "e_score" not in key:
            ref_score += 1
        if key == "embed.weight" or key.endswith(".embed.weight"):
            ref_score += 1

    print(f"   HuggingFace pattern score: {hf_score}")
    print(f"   Reference pattern score:   {ref_score}")

    if hf_score > ref_score:
        style = "huggingface"
    elif ref_score > hf_score:
        style = "reference"
    else:
        style = "unknown"

    print(f"\n   Detected style: {style.upper()}")

    # Show sample keys
    print("\n" + "-" * 70)
    print("SAMPLE CHECKPOINT KEYS (first 40)")
    print("-" * 70)
    for key in sorted(ckpt_keys)[:40]:
        print(f"   {key}")
    if len(ckpt_keys) > 40:
        print(f"   ... and {len(ckpt_keys) - 40} more")

    # Key pattern check
    print("\n" + "-" * 70)
    print("KEY PATTERN CHECK")
    print("-" * 70)

    checks = [
        ("self_attn", "HF-style attention"),
        (".attn.wq", "Reference-style attention"),
        ("q_a_proj", "HF-style Q projection"),
        ("wq_a", "Reference-style Q projection"),
        ("gate_proj", "HF-style MLP"),
        (".w1.", "Reference-style MLP"),
        ("input_layernorm", "HF-style layer norm"),
        ("attn_norm", "Reference-style layer norm"),
        ("e_score_correction_bias", "HF-style gate bias"),
        (".gate.bias", "Reference-style gate bias"),
        ("model.embed_tokens", "HF-style embedding"),
        ("embed.weight", "Reference-style embedding"),
    ]

    for pattern, desc in checks:
        found = any(pattern in k for k in ckpt_keys)
        status = "✅ FOUND" if found else "   not found"
        print(f"   [{status}] {pattern:<30} ({desc})")

    # Analysis
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    issues = []
    result = {
        "style": style,
        "hf_score": hf_score,
        "ref_score": ref_score,
        "num_keys": len(ckpt_keys),
        "sample_keys": sorted(ckpt_keys)[:50],
    }

    if style == "reference":
        print("""
❌ CRITICAL: Checkpoint uses REFERENCE naming convention!

   Your checkpoint has keys like:
      layers.*.attn.wq_a
      layers.*.attn.wkv_b
      layers.*.ffn.w1
      layers.*.attn_norm

   But your HF model expects:
      model.layers.*.self_attn.q_a_proj
      model.layers.*.self_attn.kv_b_proj
      model.layers.*.mlp.gate_proj
      model.layers.*.input_layernorm

   THIS IS THE CAUSE OF GIBBERISH OUTPUT!
   The weights are not being loaded into the correct parameters.
""")
        issues.append("Weight names do not match HF model expectations")

        print("-" * 70)
        print("REQUIRED WEIGHT NAME MAPPINGS")
        print("-" * 70)
        print(f"\n{'Reference (checkpoint)':<50} -> {'HuggingFace (model)':<50}")
        print("-" * 103)
        for ref_name, hf_name in WEIGHT_MAPPINGS:
            print(f"   {ref_name:<47} -> {hf_name}")

    elif style == "huggingface":
        print("""
✅ Checkpoint appears to use HuggingFace naming convention.
   Weight names should load correctly.

   If you're still getting gibberish, the issue is likely:
      1. Missing Hadamard transform (install fast-hadamard-transform)
      2. Indexer not applied during decode (seq_len == 1)
      3. Config mismatch between checkpoint and model
      4. Numerical issues (NaN/Inf)
""")

    else:
        print("""
⚠️  Could not definitively determine naming convention.
    Please check the sample keys above manually.

    Look for patterns like:
       HF style: self_attn, q_a_proj, gate_proj, input_layernorm
       Reference style: attn.wq, wq_a, w1, attn_norm
""")
        issues.append("Could not determine naming convention")

    # Check for model. prefix
    has_model_prefix = any(k.startswith("model.") for k in ckpt_keys)
    print(f"\n   Has 'model.' prefix: {has_model_prefix}")

    if style == "huggingface" and not has_model_prefix:
        print("   ⚠️  HF style but missing 'model.' prefix - may need adjustment")
        issues.append("Missing 'model.' prefix")

    result["issues"] = issues
    result["has_model_prefix"] = has_model_prefix

    return result


@app.local_entrypoint()
def main(
    model_path: str = DEFAULT_MODEL_PATH,
    max_files: int = 2,
):
    """
    Debug weight name mapping for DeepSeek V3.2.

    Examples:
        modal run sandbox/debug_weight_mapping.py
        modal run sandbox/debug_weight_mapping.py --model-path "my-model-path"
    """
    print("🔍 DeepSeek V3.2 Weight Mapping Debugger")
    print("=" * 70)

    result = debug_weight_mapping.remote(
        model_path=model_path,
        max_files=max_files,
    )

    if result.get("issues"):
        print("\n❌ Issues found:")
        for issue in result["issues"]:
            print(f"   - {issue}")
    else:
        print("\n✅ No naming issues detected")

    return result
