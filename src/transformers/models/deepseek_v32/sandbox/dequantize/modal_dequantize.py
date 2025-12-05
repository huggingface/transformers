"""
Modal app to dequantize FP8 weights to BF16.

This script provides a Modal function to convert FP8 quantized model weights
to BF16 format using Triton kernels. Processing is parallelized across multiple
GPU workers for faster conversion of large models.

Usage:
    # Using default "models" volume
    modal run utils/dequantize/modal_dequantize.py \
        --input-path deepseek-ai--DeepSeek-V3 \
        --output-path deepseek-ai--DeepSeek-V3-bf16

    # Using a custom volume (set VOLUME_NAME env var before running)
    VOLUME_NAME=my-models modal run utils/dequantize/modal_dequantize.py \
        --input-path path/to/fp8/model \
        --output-path path/to/bf16/output
"""

import os
from pathlib import Path

import modal


# Create Modal app
app = modal.App("dequantize-fp8-to-bf16")

# Volume configuration - set VOLUME_NAME env var to use a different volume
VOLUME_NAME = os.environ.get("VOLUME_NAME", "models")
MOUNT_PATH = "/mnt/data"

# Get the directory containing this file (for mounting kernel.py)
LOCAL_DIR = Path(__file__).parent

# Create image with required dependencies for Triton kernels
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "triton>=2.1.0",
        "safetensors>=0.4.0",
        "tqdm>=4.65.0",
        "numpy>=1.24.0",
        "packaging",  # Required by safetensors
    )
    .add_local_file(
        str(LOCAL_DIR / "kernel.py"),
        remote_path="/root/kernel.py",
    )
)


@app.function(
    image=image,
    gpu="H100",  # H100 required for fp8e4nv (FP8 E4M3) support
    volumes={MOUNT_PATH: modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)},
    timeout=3600,  # 1 hour timeout per file
)
def convert_single_file(
    safetensor_file: str,
    fp8_path: str,
    bf16_path: str,
    weight_map: dict,
) -> list[str]:
    """
    Convert a single safetensor file from FP8 to BF16.

    Args:
        safetensor_file: Path to the safetensor file to convert
        fp8_path: Base path to the FP8 model directory
        bf16_path: Base path for the output BF16 model directory
        weight_map: Mapping of tensor names to file names

    Returns:
        List of FP8 weight names that were converted
    """
    import os
    import sys

    import torch
    from safetensors.torch import load_file, save_file

    # Import the kernel from mounted directory
    sys.path.insert(0, "/root")
    from kernel import weight_dequant

    torch.set_default_dtype(torch.bfloat16)

    file_name = os.path.basename(safetensor_file)
    print(f"Processing: {file_name}")

    # Cache for loaded safetensor files (for cross-file scale_inv lookups)
    loaded_files = {}

    def get_tensor(tensor_name):
        """Retrieve a tensor from cache or load from disk."""
        target_file = weight_map[tensor_name]
        if target_file not in loaded_files:
            file_path = os.path.join(fp8_path, target_file)
            loaded_files[target_file] = load_file(file_path, device="cuda")
        return loaded_files[target_file][tensor_name]

    # Load current file
    current_state_dict = load_file(safetensor_file, device="cuda")
    loaded_files[file_name] = current_state_dict

    new_state_dict = {}
    fp8_weight_names = []

    for weight_name, weight in current_state_dict.items():
        if weight_name.endswith("_scale_inv"):
            # Skip scale tensors - they won't be needed after dequantization
            continue
        elif weight.element_size() == 1:  # FP8 weight (1 byte per element)
            scale_inv_name = f"{weight_name}_scale_inv"
            try:
                scale_inv = get_tensor(scale_inv_name)
                fp8_weight_names.append(weight_name)
                new_state_dict[weight_name] = weight_dequant(weight, scale_inv)
            except KeyError:
                print(f"Warning: Missing scale_inv tensor for {weight_name}, skipping conversion")
                new_state_dict[weight_name] = weight
        else:
            # Non-FP8 weights pass through unchanged
            new_state_dict[weight_name] = weight

    # Save converted file
    os.makedirs(bf16_path, exist_ok=True)
    new_safetensor_file = os.path.join(bf16_path, file_name)
    save_file(new_state_dict, new_safetensor_file)

    # Clean up GPU memory
    del loaded_files
    del current_state_dict
    del new_state_dict
    torch.cuda.empty_cache()

    print(f"✓ Completed: {file_name} ({len(fp8_weight_names)} FP8 weights converted)")
    return fp8_weight_names


def _fix_config_dtype(model_path: str) -> None:
    """
    Fix config.json to reflect BF16 dtype after dequantization.
    
    The original FP8 model's config.json may have torch_dtype set to float8
    and quantization_config settings. After dequantization to BF16, we need
    to update these to reflect the actual dtype.
    """
    import json
    import os

    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    modified = False

    # Update torch_dtype to bfloat16
    if config.get("torch_dtype") in ["float8_e4m3fn", "float8_e5m2", "float8"]:
        config["torch_dtype"] = "bfloat16"
        modified = True
        print("  ✓ Updated torch_dtype: float8 → bfloat16")

    # Remove quantization_config if present (no longer quantized)
    if "quantization_config" in config:
        del config["quantization_config"]
        modified = True
        print("  ✓ Removed quantization_config (model is now BF16)")

    if modified:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print("  ✓ config.json updated for BF16")
    else:
        print("  ℹ️  config.json already has correct dtype settings")


def _fix_config_auto_map(model_path: str) -> None:
    """
    Fix the config.json auto_map for custom model architectures.
    
    Models like DeepSeek V3.2 have custom model_type that isn't in transformers'
    CONFIG_MAPPING. For trust_remote_code=True to work, the config.json must have
    an auto_map field pointing to the custom configuration/modeling classes.
    
    This function checks if the model has custom code files and adds the appropriate
    auto_map if it's missing.
    """
    import json
    import os

    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        print("  No config.json found, skipping auto_map fix")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    model_type = config.get("model_type", "")
    auto_map = config.get("auto_map", {})

    # Define known custom model architectures and their auto_map configurations
    CUSTOM_MODEL_CONFIGS = {
        "deepseek_v32": {
            "AutoConfig": "configuration_deepseek.DeepseekV3Config",
            "AutoModelForCausalLM": "modeling_deepseek.DeepseekV3ForCausalLM",
        },
        "deepseek_v3": {
            "AutoConfig": "configuration_deepseek.DeepseekV3Config",
            "AutoModelForCausalLM": "modeling_deepseek.DeepseekV3ForCausalLM",
        },
    }

    if model_type in CUSTOM_MODEL_CONFIGS:
        expected_auto_map = CUSTOM_MODEL_CONFIGS[model_type]

        # Check if auto_map is missing or incomplete
        needs_update = False
        for key, value in expected_auto_map.items():
            if key not in auto_map:
                needs_update = True
                break

        if needs_update:
            # Verify that the custom code files exist
            config_file = os.path.join(model_path, "configuration_deepseek.py")
            modeling_file = os.path.join(model_path, "modeling_deepseek.py")

            if os.path.exists(config_file) and os.path.exists(modeling_file):
                # Update auto_map
                config["auto_map"] = {**auto_map, **expected_auto_map}

                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)

                print(f"  ✓ Fixed auto_map for model_type '{model_type}'")
                print(f"    Added: {expected_auto_map}")
            else:
                print(f"  ⚠️  Model type '{model_type}' needs custom code but files are missing:")
                if not os.path.exists(config_file):
                    print("      Missing: configuration_deepseek.py")
                if not os.path.exists(modeling_file):
                    print("      Missing: modeling_deepseek.py")
        else:
            print(f"  ✓ auto_map already configured for model_type '{model_type}'")
    else:
        print(f"  ℹ️  Model type '{model_type}' - no auto_map fix needed")


@app.function(
    image=image,
    volumes={MOUNT_PATH: modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)},
    timeout=86_400,  # 24 hour timeout for orchestration
)
def dequantize_fp8_to_bf16(
    input_path: str,
    output_path: str,
) -> str:
    """
    Convert FP8 weights to BF16 format using parallel processing.

    Args:
        input_path: Path to the FP8 model directory (relative to volume mount)
        output_path: Path where the BF16 model will be saved (relative to volume mount)

    Returns:
        Path to the converted BF16 model
    """
    import json
    import os
    import shutil
    from glob import glob
    from pathlib import Path

    # Construct full paths
    fp8_path = os.path.join(MOUNT_PATH, input_path.lstrip("/"))
    bf16_path = os.path.join(MOUNT_PATH, output_path.lstrip("/"))

    print(f"Input FP8 model path: {fp8_path}")
    print(f"Output BF16 model path: {bf16_path}")

    # Validate input path exists
    if not os.path.exists(fp8_path):
        raise FileNotFoundError(f"Input path does not exist: {fp8_path}")

    # Check for model index file
    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    if not os.path.exists(model_index_file):
        raise FileNotFoundError(f"Model index file not found: {model_index_file}")

    print("=" * 60)
    print("Starting PARALLEL FP8 to BF16 conversion")
    print("=" * 60)

    # Create output directory
    os.makedirs(bf16_path, exist_ok=True)

    # Load model index
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]

    # Find all safetensor files
    safetensor_files = sorted(glob(os.path.join(fp8_path, "*.safetensors")))
    print(f"Found {len(safetensor_files)} safetensor files to process in parallel")

    # Process files in parallel using Modal's .map()
    all_fp8_weight_names = []
    results = list(convert_single_file.map(
        safetensor_files,
        kwargs={
            "fp8_path": fp8_path,
            "bf16_path": bf16_path,
            "weight_map": weight_map,
        }
    ))

    # Collect all converted weight names
    for fp8_names in results:
        all_fp8_weight_names.extend(fp8_names)

    print(f"\n✓ Parallel processing complete: {len(results)} files processed")

    # Update and save model index (remove scale_inv references)
    print("Updating model index...")
    new_model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")
    for weight_name in all_fp8_weight_names:
        scale_inv_name = f"{weight_name}_scale_inv"
        if scale_inv_name in weight_map:
            weight_map.pop(scale_inv_name)

    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)

    # Copy other model files (config, tokenizer, etc.) and directories (encoding, inference, etc.)
    print("Copying additional model files and directories...")
    for item in Path(fp8_path).iterdir():
        dest = Path(bf16_path) / item.name
        if item.is_dir():
            # Copy directories recursively (e.g., encoding/, inference/ for custom tokenizers)
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
            print(f"  Copied directory: {item.name}/")
        elif item.is_file() and not item.name.endswith(".safetensors"):
            if item.name != "model.safetensors.index.json":  # Already handled
                shutil.copy2(item, dest)
                print(f"  Copied: {item.name}")

    # Fix config.json auto_map for custom model architectures (DeepSeek V3.2, etc.)
    _fix_config_auto_map(bf16_path)

    # Fix config.json to reflect BF16 dtype (remove FP8 quantization settings)
    _fix_config_dtype(bf16_path)

    # Commit changes to volume
    volume = modal.Volume.from_name(VOLUME_NAME)
    volume.commit()

    print("=" * 60)
    print("✓ Successfully converted FP8 model to BF16")
    print(f"✓ Output saved to: {bf16_path}")
    print(f"✓ Converted {len(all_fp8_weight_names)} FP8 weight tensors")
    print(f"✓ Processed {len(safetensor_files)} files in parallel")
    print(f"✓ Changes committed to Modal volume '{VOLUME_NAME}'")
    print("=" * 60)

    return bf16_path


@app.function(
    image=image,
    volumes={MOUNT_PATH: modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)},
    timeout=300,
)
def fix_model_config(model_path: str) -> None:
    """
    Fix the config.json for an existing dequantized model.
    
    This fixes:
    - auto_map for custom model architectures
    - torch_dtype (float8 → bfloat16)
    - Removes quantization_config
    
    Usage:
        modal run utils/dequantize/modal_dequantize.py --fix-config deepseek-ai--DeepSeek-V3.2_bf16
    """
    import os

    full_path = os.path.join(MOUNT_PATH, model_path.lstrip("/"))
    print(f"Fixing config for model at: {full_path}")

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Model path does not exist: {full_path}")

    _fix_config_auto_map(full_path)
    _fix_config_dtype(full_path)

    # Commit changes to volume
    volume = modal.Volume.from_name(VOLUME_NAME)
    volume.commit()

    print(f"✓ Config fix complete and committed to volume '{VOLUME_NAME}'")


@app.function(
    image=image,
    volumes={MOUNT_PATH: modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)},
    timeout=600,
)
def copy_missing_dirs(source_path: str, dest_path: str) -> None:
    """
    Copy missing directories from source model to destination model.
    
    Use this to copy directories like encoding/, inference/ that were missed
    during earlier dequantization runs.
    
    Usage:
        modal run utils/dequantize/modal_dequantize.py \\
            --copy-dirs-from deepseek-ai--DeepSeek-V3.2 \\
            --copy-dirs-to deepseek-ai--DeepSeek-V3.2_bf16
    """
    import os
    import shutil
    from pathlib import Path

    source_full = os.path.join(MOUNT_PATH, source_path.lstrip("/"))
    dest_full = os.path.join(MOUNT_PATH, dest_path.lstrip("/"))

    print(f"Source model: {source_full}")
    print(f"Destination model: {dest_full}")

    if not os.path.exists(source_full):
        raise FileNotFoundError(f"Source path does not exist: {source_full}")
    if not os.path.exists(dest_full):
        raise FileNotFoundError(f"Destination path does not exist: {dest_full}")

    # Find and copy directories
    copied = []
    for item in Path(source_full).iterdir():
        if item.is_dir():
            dest_item = Path(dest_full) / item.name
            if not dest_item.exists():
                print(f"  Copying missing directory: {item.name}/")
                shutil.copytree(item, dest_item)
                copied.append(item.name)
            else:
                print(f"  Directory already exists: {item.name}/")

    if copied:
        # Commit changes to volume
        volume = modal.Volume.from_name(VOLUME_NAME)
        volume.commit()
        print(f"\n✓ Copied {len(copied)} directories: {', '.join(copied)}")
        print(f"✓ Changes committed to volume '{VOLUME_NAME}'")
    else:
        print("\n✓ No missing directories found")


@app.local_entrypoint()
def main(
    input_path: str = "",
    output_path: str = "",
    fix_config: str = "",
    copy_dirs_from: str = "",
    copy_dirs_to: str = "",
):
    """
    Dequantize FP8 model weights to BF16 format.

    Examples:
        # Basic usage with default "models" volume
        modal run utils/dequantize/modal_dequantize.py \\
            --input-path deepseek-ai--DeepSeek-V3 \\
            --output-path deepseek-ai--DeepSeek-V3-bf16

        # With custom volume (set VOLUME_NAME env var before running)
        VOLUME_NAME=my-models modal run utils/dequantize/modal_dequantize.py \\
            --input-path path/to/fp8-model \\
            --output-path path/to/bf16-model

        # Fix config.json auto_map for an existing model
        modal run utils/dequantize/modal_dequantize.py \\
            --fix-config deepseek-ai--DeepSeek-V3.2_bf16

        # Copy missing directories (encoding/, inference/, etc.) from source to dest
        modal run utils/dequantize/modal_dequantize.py \\
            --copy-dirs-from deepseek-ai--DeepSeek-V3.2 \\
            --copy-dirs-to deepseek-ai--DeepSeek-V3.2_bf16
    """
    # Handle --copy-dirs-from/--copy-dirs-to mode
    if copy_dirs_from and copy_dirs_to:
        print(f"Volume: {VOLUME_NAME}")
        print(f"Copying missing directories from: {copy_dirs_from}")
        print(f"                              to: {copy_dirs_to}")
        print()
        copy_missing_dirs.remote(source_path=copy_dirs_from, dest_path=copy_dirs_to)
        return
    elif copy_dirs_from or copy_dirs_to:
        print("Error: Both --copy-dirs-from and --copy-dirs-to are required together")
        return

    # Handle --fix-config mode
    if fix_config:
        print(f"Volume: {VOLUME_NAME}")
        print(f"Fixing config for model: {fix_config}")
        print()
        fix_model_config.remote(model_path=fix_config)
        return

    if not input_path:
        print("Error: --input-path is required (or use --fix-config to fix existing model)")
        print("Usage: modal run utils/dequantize/modal_dequantize.py --input-path <path> --output-path <path>")
        return

    if not output_path:
        print("Error: --output-path is required")
        print("Usage: modal run utils/dequantize/modal_dequantize.py --input-path <path> --output-path <path>")
        return

    print(f"Volume: {VOLUME_NAME}")
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print()

    result = dequantize_fp8_to_bf16.remote(
        input_path=input_path,
        output_path=output_path,
    )

    print(f"\nModel converted and saved to: {result}")
