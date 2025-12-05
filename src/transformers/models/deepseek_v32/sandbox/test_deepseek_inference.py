#!/usr/bin/env python3
"""
Modal app to test barebones DeepSeek V3.2 inference.

Tests the custom transformers fork that adds deepseek_v32 model support.

Usage:
    modal run utils/test_deepseek_inference.py
    modal run utils/test_deepseek_inference.py --model-path "deepseek-ai--DeepSeek-V3.2"
    modal run utils/test_deepseek_inference.py --prompt "What is 2+2?"
"""

import modal


# Configuration
GPU_TYPE = "B200"
GPU_COUNT = 8
MODEL_VOLUME_NAME = "models"
MODEL_MOUNT_PATH = "/mnt/model"
LOCAL_MODEL_PATH = "/tmp/model"  # Local ephemeral disk for faster loading
DEFAULT_MODEL_PATH = "deepseek-ai--DeepSeek-V3.2_bf16"

# Build image with transformers fork
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")  # Required for pip install from git
    .pip_install(
        "torch>=2.4.0",
        "accelerate>=0.34.0",
        "safetensors",
        "tiktoken",
        "sentencepiece",
        "fbgemm-gpu",  # Required for FbgemmFp8Config
    )
    # Install transformers fork with DeepSeek V3.2 support
    .run_commands(
        "echo 'Installing transformers fork v2.7.2'",  # increment to force rebuild
        "pip install --no-cache-dir git+https://github.com/jyliu24/transformers.git",
    )
)

app = modal.App("test-deepseek-inference-copytest", image=image)

model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME)


@app.function(
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    volumes={MODEL_MOUNT_PATH: model_volume},
    timeout=60 * 240,  # 2 hour timeout (loading alone takes ~1hr)
    ephemeral_disk=2 * 1024 * 1024,  # 2 TB local disk for model copy
)
@modal.experimental.clustered(1, rdma=True)
def test_inference(
    model_path: str = DEFAULT_MODEL_PATH,
    prompt: str = "Hello! What is 2 + 2?",
    max_new_tokens: int = 1000,
    use_fp8: bool = False,
    use_local_copy: bool = True,  # Copy model to local disk for faster loading
) -> str:
    """
    Test barebones inference with DeepSeek V3.2 using transformers.
    """
    import os
    import shutil

    import torch

    volume_model_path = f"{MODEL_MOUNT_PATH}/{model_path}"
    local_model_path = f"{LOCAL_MODEL_PATH}/{model_path}"

    # Optionally copy model to local disk for faster random I/O during weight loading
    if use_local_copy:
        print("=" * 70)
        print("📋 Copying model to local NVMe disk for faster loading...")
        print("=" * 70)

        # Calculate total size first
        if not os.path.exists(volume_model_path):
            print(f"  ❌ Model path does not exist: {volume_model_path}")
            return "ERROR: Model path not found"

        import time
        t_copy = time.time()

        # Get list of files with sizes for progress tracking
        files_to_copy = []
        total_size = 0
        for f in os.listdir(volume_model_path):
            src_path = os.path.join(volume_model_path, f)
            if os.path.isfile(src_path):
                size = os.path.getsize(src_path)
                files_to_copy.append((f, size))
                total_size += size

        print(f"  Total size to copy: {total_size / 1024**3:.1f} GB ({len(files_to_copy)} files)")

        # Clean up any existing local copy
        if os.path.exists(local_model_path):
            shutil.rmtree(local_model_path)

        os.makedirs(local_model_path, exist_ok=True)

        # Copy with progress tracking
        print(f"  Copying from: {volume_model_path}")
        print(f"  Copying to:   {local_model_path}")

        copied_size = 0
        for i, (filename, filesize) in enumerate(files_to_copy):
            src = os.path.join(volume_model_path, filename)
            dst = os.path.join(local_model_path, filename)
            shutil.copy2(src, dst)
            copied_size += filesize

            # Progress update
            pct = copied_size / total_size * 100
            elapsed = time.time() - t_copy
            speed = copied_size / elapsed / 1024**3 if elapsed > 0 else 0
            eta = (total_size - copied_size) / (copied_size / elapsed) if copied_size > 0 else 0
            bar_width = 30
            filled = int(bar_width * copied_size / total_size)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(f"\r  [{bar}] {pct:5.1f}% | {copied_size/1024**3:.1f}/{total_size/1024**3:.1f} GB | {speed:.2f} GB/s | ETA: {eta:.0f}s | {filename[:40]:<40}", end="", flush=True)

        print()  # Newline after progress bar

        copy_time = time.time() - t_copy
        print(f"  ✅ Copied in {copy_time:.1f}s ({total_size / 1024**3 / copy_time:.1f} GB/s)")
        print("=" * 70)

        full_model_path = local_model_path
    else:
        full_model_path = volume_model_path

    # Add model path to PYTHONPATH for inference/ package if present
    inference_dir = os.path.join(full_model_path, "inference")
    if os.path.exists(inference_dir):
        current_pythonpath = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = f"{full_model_path}:{current_pythonpath}"
        print(f"✅ Added {full_model_path} to PYTHONPATH for inference/ package")

    print("=" * 70)
    print("DeepSeek V3.2 Inference Test")
    print("=" * 70)
    print(f"Model path: {full_model_path}")
    print(f"GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"         Memory: {props.total_memory / 1024**3:.1f} GB")
    print("=" * 70)

    import json
    import time

    # Check model directory
    print("\n📁 Model directory contents:")
    if not os.path.exists(full_model_path):
        print(f"  ❌ Model path does not exist: {full_model_path}")
        return "ERROR: Model path not found"

    # Count and size model files
    safetensor_files = []
    total_size = 0
    for item in sorted(os.listdir(full_model_path)):
        item_path = os.path.join(full_model_path, item)
        if os.path.isfile(item_path):
            size_bytes = os.path.getsize(item_path)
            total_size += size_bytes
            if item.endswith('.safetensors'):
                safetensor_files.append((item, size_bytes))

    print(f"  Total files: {len(os.listdir(full_model_path))}")
    print(f"  Safetensor shards: {len(safetensor_files)}")
    print(f"  Total model size: {total_size / 1024**3:.1f} GB")

    # Show first few safetensor files
    for name, size in safetensor_files[:5]:
        print(f"    {name} ({size / 1024**3:.2f} GB)")
    if len(safetensor_files) > 5:
        print(f"    ... and {len(safetensor_files) - 5} more shards")

    # Check config.json for model_type
    config_path = os.path.join(full_model_path, "config.json")
    quant_config = None
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        print("\n📋 Model config:")
        print(f"  model_type: {config.get('model_type', 'NOT FOUND')}")
        print(f"  architectures: {config.get('architectures', 'NOT FOUND')}")
        print(f"  torch_dtype: {config.get('torch_dtype', 'NOT FOUND')}")
        print(f"  dtype: {config.get('dtype', 'NOT FOUND')}")

        # Check quantization config
        quant_config = config.get('quantization_config')
        if quant_config:
            print("\n📋 Quantization config found:")
            print(f"  quant_method: {quant_config.get('quant_method', 'NOT FOUND')}")
            print(f"  fmt: {quant_config.get('fmt', 'NOT FOUND')}")
            print(f"  weight_block_size: {quant_config.get('weight_block_size', 'NOT FOUND')}")
            print(f"  activation_scheme: {quant_config.get('activation_scheme', 'NOT FOUND')}")
            print(f"  scale_fmt: {quant_config.get('scale_fmt', 'NOT FOUND')}")
        else:
            print("\n📋 No quantization_config in model config")

    # Check tokenizer_config.json
    tokenizer_config_path = os.path.join(full_model_path, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path, "r") as f:
            tok_config = json.load(f)
        print("\n📋 Tokenizer config:")
        print(f"  tokenizer_class: {tok_config.get('tokenizer_class', 'NOT FOUND')}")
        print(f"  model_max_length: {tok_config.get('model_max_length', 'NOT FOUND')}")
        # Show auto_map if present (for custom tokenizers)
        if 'auto_map' in tok_config:
            print(f"  auto_map: {tok_config.get('auto_map')}")
    else:
        print("\n⚠️  No tokenizer_config.json found")

    # Check for tokenizer.json
    tokenizer_json_path = os.path.join(full_model_path, "tokenizer.json")
    print(f"  tokenizer.json exists: {os.path.exists(tokenizer_json_path)}")

    print("\n🔧 Loading transformers...")
    import transformers
    print(f"  transformers version: {transformers.__version__}")

    # Check available FP8 quantization configs
    print("\n🔍 Checking FP8 support in transformers:")
    try:
        from transformers import FbgemmFp8Config
        print("  ✅ FbgemmFp8Config available")
    except ImportError:
        print("  ❌ FbgemmFp8Config not available")

    try:
        from transformers import FineGrainedFP8Config
        print("  ✅ FineGrainedFP8Config available")
    except ImportError:
        print("  ❌ FineGrainedFP8Config not available")

    try:
        from transformers import BitsAndBytesConfig
        print("  ✅ BitsAndBytesConfig available")
    except ImportError:
        print("  ❌ BitsAndBytesConfig not available")

    # Check quantization utils
    try:
        from transformers.quantizers import auto
        print(f"  Available quantization methods: {list(auto.AUTO_QUANTIZER_MAPPING.keys()) if hasattr(auto, 'AUTO_QUANTIZER_MAPPING') else 'unknown'}")
    except Exception as e:
        print(f"  Could not check quantization methods: {e}")

    # Enable transformers logging for progress
    transformers.logging.set_verbosity_info()

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print("\n📥 Loading tokenizer...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        full_model_path,
        trust_remote_code=True,
    )
    print(f"  ✅ Tokenizer loaded in {time.time() - t0:.1f}s: {type(tokenizer).__name__}")
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Load config first to inspect quantization settings
    print("\n🔍 Loading AutoConfig to check quantization...")
    auto_config = AutoConfig.from_pretrained(full_model_path, trust_remote_code=True)
    print(f"  Config class: {type(auto_config).__name__}")
    if hasattr(auto_config, 'quantization_config') and auto_config.quantization_config:
        print(f"  quantization_config type: {type(auto_config.quantization_config)}")
        print(f"  quantization_config: {auto_config.quantization_config}")
    else:
        print("  quantization_config: None or not set")

    # Try to create explicit FP8 quantization config
    fp8_quant_config = None

    # Use FP8 if requested via flag OR if model already has FP8 config
    should_use_fp8 = use_fp8 or (quant_config and quant_config.get('quant_method') == 'fp8')

    if should_use_fp8:
        print("\n🔧 Creating FP8 quantization config...", flush=True)
        if use_fp8:
            print("  (requested via --use-fp8 flag)", flush=True)

        # Try FineGrainedFP8Config first (better for MoE models like DeepSeek)
        try:
            from transformers import FineGrainedFP8Config
            block_size = quant_config.get('weight_block_size', [128, 128]) if quant_config else [128, 128]
            fp8_quant_config = FineGrainedFP8Config(weight_block_size=block_size)
            print(f"  ✅ Created FineGrainedFP8Config with block_size={block_size}", flush=True)
        except Exception as e:
            print(f"  ❌ FineGrainedFP8Config failed: {e}", flush=True)

            # Try FbgemmFp8Config as fallback
            try:
                from transformers import FbgemmFp8Config
                fp8_quant_config = FbgemmFp8Config()
                print("  ✅ Created FbgemmFp8Config as fallback", flush=True)
            except Exception as e2:
                print(f"  ❌ FbgemmFp8Config also failed: {e2}", flush=True)
                print("  ⚠️ Proceeding without FP8 quantization", flush=True)

    print("\n📥 Loading model...", flush=True)
    print(f"  Loading {total_size / 1024**3:.1f} GB across {torch.cuda.device_count()} GPUs", flush=True)

    # Configure memory allocation based on quantization
    num_gpus = torch.cuda.device_count()
    if fp8_quant_config:
        # FP8 model is ~half the size, should fit entirely on GPU
        max_memory = dict.fromkeys(range(num_gpus), "175GiB")
        max_memory["cpu"] = "0GiB"  # No CPU needed for FP8
        print("  max_memory config: GPU=175GiB each, CPU=0GiB (FP8 fits on GPU)", flush=True)
    else:
        # BF16 model - allocate GPU memory evenly, minimal CPU
        max_memory = dict.fromkeys(range(num_gpus), "175GiB")
        max_memory["cpu"] = "100GiB"  # Allow CPU for buffers/intermediates
        print("  max_memory config: GPU=175GiB each, CPU=100GiB (buffers only)", flush=True)

    if fp8_quant_config:
        print(f"  Using explicit FP8 quantization config: {type(fp8_quant_config).__name__}", flush=True)

    # Pre-compute device map for faster loading
    # This avoids the slow auto device placement during weight loading
    print("  Computing optimal device map...", flush=True)
    from accelerate import infer_auto_device_map, init_empty_weights

    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(full_model_path, trust_remote_code=True)

    with init_empty_weights():
        from transformers import AutoModelForCausalLM as AutoModelEmpty
        empty_model = AutoModelEmpty.from_config(model_config, trust_remote_code=True)

    device_map = infer_auto_device_map(
        empty_model,
        max_memory=max_memory,
        no_split_module_classes=["DeepseekV32DecoderLayer"],
    )
    del empty_model

    # Check if any layers ended up on CPU in the device map
    cpu_modules = [k for k, v in device_map.items() if v == "cpu"]
    if cpu_modules:
        print(f"  ⚠️ Device map has {len(cpu_modules)} modules on CPU", flush=True)
    else:
        print(f"  ✅ Device map: all modules on GPU ({len(device_map)} modules)", flush=True)

    print("  Loading with pre-computed device map + safetensors mmap...", flush=True)
    print("-" * 50, flush=True)

    t0 = time.time()
    load_kwargs = {
        "trust_remote_code": True,
        "device_map": device_map,  # Use pre-computed map instead of "auto"
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16,
        "use_safetensors": True,  # Enable mmap for faster loading
    }

    # If we have an explicit FP8 config, use it
    if fp8_quant_config:
        load_kwargs["quantization_config"] = fp8_quant_config
        del load_kwargs["torch_dtype"]  # Let quantization config handle dtype

    model = AutoModelForCausalLM.from_pretrained(
        full_model_path,
        **load_kwargs,
    )
    load_time = time.time() - t0
    print("-" * 50, flush=True)
    print(f"  ✅ Model loaded in {load_time:.1f}s ({total_size / 1024**3 / load_time:.1f} GB/s)", flush=True)
    print(f"  Model type: {type(model).__name__}", flush=True)

    # Print model distribution across GPUs - check for CPU offloading
    if hasattr(model, "hf_device_map"):
        devices_used = set(str(v) for v in model.hf_device_map.values())
        print(f"  Distributed across devices: {devices_used}", flush=True)

        # Check if any layers ended up on CPU (BAD for inference speed!)
        cpu_layers = [k for k, v in model.hf_device_map.items() if str(v) == "cpu"]
        if cpu_layers:
            print(f"  ⚠️ WARNING: {len(cpu_layers)} layers on CPU! Inference will be SLOW!", flush=True)
            print(f"     First few CPU layers: {cpu_layers[:5]}", flush=True)
        else:
            print("  ✅ All layers on GPU - no CPU offloading", flush=True)

    # Check actual tensor dtypes
    print("\n🔍 Checking actual tensor dtypes:", flush=True)
    for name, param in list(model.named_parameters())[:5]:
        print(f"  {name}: {param.dtype}, shape={param.shape}", flush=True)
    print("  ...", flush=True)

    print("\n🚀 Running inference...", flush=True)
    print(f"  Prompt: {prompt}", flush=True)
    print(f"  Max new tokens: {max_new_tokens}", flush=True)

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    print(f"  Input tokens: {inputs['input_ids'].shape[1]}", flush=True)

    # Generate with progress tracking
    print("  Starting generation...", flush=True)
    t_gen = time.time()

    # Use a streamer to show progress
    from transformers import TextStreamer
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for reproducibility
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,  # Stream tokens as they're generated
        )

    gen_time = time.time() - t_gen
    num_new_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    print(f"\n  Generation complete: {num_new_tokens} tokens in {gen_time:.1f}s ({num_new_tokens/gen_time:.1f} tok/s)", flush=True)

    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n" + "=" * 70)
    print("📝 RESPONSE:")
    print("=" * 70)
    print(response)
    print("=" * 70)

    print("\n✅ Inference test completed successfully!")
    return response


@app.function(
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    volumes={MODEL_MOUNT_PATH: model_volume},
    timeout=60 * 240,  # 2 hour timeout (loading alone takes ~1hr)
    ephemeral_disk=2 * 1024 * 1024,  # 2 TB local disk for model copy
)
@modal.experimental.clustered(1, rdma=True)
def test_chat_inference(
    model_path: str = DEFAULT_MODEL_PATH,
    user_message: str = "What is the capital of France?",
    max_new_tokens: int = 1000,
    use_fp8: bool = False,
    use_local_copy: bool = True,  # Copy model to local disk for faster loading
) -> str:
    """
    Test chat-style inference with DeepSeek V3.2 using apply_chat_template.
    """
    import os
    import shutil
    import time

    import torch

    volume_model_path = f"{MODEL_MOUNT_PATH}/{model_path}"
    local_model_path = f"{LOCAL_MODEL_PATH}/{model_path}"

    # Optionally copy model to local disk for faster random I/O during weight loading
    if use_local_copy:
        print("=" * 70)
        print("📋 Copying model to local NVMe disk for faster loading...")
        print("=" * 70)

        # Calculate total size first
        if not os.path.exists(volume_model_path):
            print(f"  ❌ Model path does not exist: {volume_model_path}")
            return "ERROR: Model path not found"

        t_copy = time.time()

        # Get list of files with sizes for progress tracking
        files_to_copy = []
        total_size = 0
        for f in os.listdir(volume_model_path):
            src_path = os.path.join(volume_model_path, f)
            if os.path.isfile(src_path):
                size = os.path.getsize(src_path)
                files_to_copy.append((f, size))
                total_size += size

        print(f"  Total size to copy: {total_size / 1024**3:.1f} GB ({len(files_to_copy)} files)")

        # Clean up any existing local copy
        if os.path.exists(local_model_path):
            shutil.rmtree(local_model_path)

        os.makedirs(local_model_path, exist_ok=True)

        # Copy with progress tracking
        print(f"  Copying from: {volume_model_path}")
        print(f"  Copying to:   {local_model_path}")

        copied_size = 0
        for i, (filename, filesize) in enumerate(files_to_copy):
            src = os.path.join(volume_model_path, filename)
            dst = os.path.join(local_model_path, filename)
            shutil.copy2(src, dst)
            copied_size += filesize

            # Progress update
            pct = copied_size / total_size * 100
            elapsed = time.time() - t_copy
            speed = copied_size / elapsed / 1024**3 if elapsed > 0 else 0
            eta = (total_size - copied_size) / (copied_size / elapsed) if copied_size > 0 else 0
            bar_width = 30
            filled = int(bar_width * copied_size / total_size)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(f"\r  [{bar}] {pct:5.1f}% | {copied_size/1024**3:.1f}/{total_size/1024**3:.1f} GB | {speed:.2f} GB/s | ETA: {eta:.0f}s | {filename[:40]:<40}", end="", flush=True)

        print()  # Newline after progress bar

        copy_time = time.time() - t_copy
        print(f"  ✅ Copied in {copy_time:.1f}s ({total_size / 1024**3 / copy_time:.1f} GB/s)")
        print("=" * 70)

        full_model_path = local_model_path
    else:
        full_model_path = volume_model_path

    # Add model path to PYTHONPATH for inference/ package if present
    inference_dir = os.path.join(full_model_path, "inference")
    if os.path.exists(inference_dir):
        current_pythonpath = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = f"{full_model_path}:{current_pythonpath}"

    print("=" * 70)
    print("DeepSeek V3.2 Chat Inference Test")
    print("=" * 70)
    print(f"Model path: {full_model_path}")
    print(f"GPUs available: {torch.cuda.device_count()}")

    # Calculate total model size
    total_size = sum(
        os.path.getsize(os.path.join(full_model_path, f))
        for f in os.listdir(full_model_path)
        if os.path.isfile(os.path.join(full_model_path, f))
    )
    print(f"Total model size: {total_size / 1024**3:.1f} GB")
    print("=" * 70)

    import transformers
    transformers.logging.set_verbosity_info()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n📥 Loading tokenizer...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        full_model_path,
        trust_remote_code=True,
    )
    print(f"  ✅ Tokenizer loaded in {time.time() - t0:.1f}s")

    print("\n📥 Loading model...", flush=True)
    print(f"  Loading {total_size / 1024**3:.1f} GB across {torch.cuda.device_count()} GPUs", flush=True)

    # Setup FP8 quantization if requested
    fp8_quant_config = None
    if use_fp8:
        print("  🔧 Using FP8 quantization (requested via --use-fp8)", flush=True)
        try:
            from transformers import FineGrainedFP8Config
            fp8_quant_config = FineGrainedFP8Config(weight_block_size=[128, 128])
            print("    ✅ Created FineGrainedFP8Config", flush=True)
        except Exception as e:
            print(f"    ❌ FineGrainedFP8Config failed: {e}", flush=True)
            try:
                from transformers import FbgemmFp8Config
                fp8_quant_config = FbgemmFp8Config()
                print("    ✅ Created FbgemmFp8Config as fallback", flush=True)
            except Exception as e2:
                print(f"    ❌ FbgemmFp8Config also failed: {e2}", flush=True)

    # Configure memory
    num_gpus = torch.cuda.device_count()
    max_memory = dict.fromkeys(range(num_gpus), "175GiB")
    max_memory["cpu"] = "0GiB" if fp8_quant_config else "100GiB"

    # Pre-compute device map for faster loading
    print("  Computing optimal device map...", flush=True)
    from accelerate import infer_auto_device_map, init_empty_weights

    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(full_model_path, trust_remote_code=True)

    with init_empty_weights():
        from transformers import AutoModelForCausalLM as AutoModelEmpty
        empty_model = AutoModelEmpty.from_config(model_config, trust_remote_code=True)

    device_map = infer_auto_device_map(
        empty_model,
        max_memory=max_memory,
        no_split_module_classes=["DeepseekV32DecoderLayer"],
    )
    del empty_model

    cpu_modules = [k for k, v in device_map.items() if v == "cpu"]
    if cpu_modules:
        print(f"  ⚠️ Device map has {len(cpu_modules)} modules on CPU", flush=True)
    else:
        print(f"  ✅ Device map: all modules on GPU ({len(device_map)} modules)", flush=True)

    print("-" * 50, flush=True)
    t0 = time.time()
    load_kwargs = {
        "trust_remote_code": True,
        "device_map": device_map,  # Use pre-computed map
        "low_cpu_mem_usage": True,
        "use_safetensors": True,  # Enable mmap
    }
    if fp8_quant_config:
        load_kwargs["quantization_config"] = fp8_quant_config
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        full_model_path,
        **load_kwargs,
    )
    load_time = time.time() - t0
    print("-" * 50, flush=True)
    print(f"  ✅ Model loaded in {load_time:.1f}s ({total_size / 1024**3 / load_time:.1f} GB/s)", flush=True)

    print("\n🚀 Running chat inference...", flush=True)

    # Build chat messages
    messages = [
        {"role": "user", "content": user_message}
    ]
    print(f"  User message: {user_message}", flush=True)

    # Apply chat template
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        print("  ✅ Using chat template", flush=True)
    else:
        text = user_message
        print("  ⚠️ No chat template, using raw prompt", flush=True)

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    print(f"  Input tokens: {inputs['input_ids'].shape[1]}", flush=True)

    # Generate with streaming
    print("  Starting generation...", flush=True)
    t_gen = time.time()

    from transformers import TextStreamer
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )

    gen_time = time.time() - t_gen
    num_new_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    print(f"\n  Generation complete: {num_new_tokens} tokens in {gen_time:.1f}s ({num_new_tokens/gen_time:.1f} tok/s)", flush=True)

    # Decode only new tokens
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print("\n" + "=" * 70)
    print("📝 ASSISTANT RESPONSE:")
    print("=" * 70)
    print(response)
    print("=" * 70)

    return response


@app.local_entrypoint()
def main(
    model_path: str = DEFAULT_MODEL_PATH,
    prompt: str = "Hello! What is 2 + 2?",
    max_new_tokens: int = 1000,
    chat_mode: bool = False,
    use_fp8: bool = False,
    use_local_copy: bool = True,  # Copy model to local disk for faster loading
):
    """
    Test DeepSeek V3.2 inference with custom transformers fork.
    
    Examples:
        # Basic inference test
        modal run utils/test_deepseek_inference.py
        
        # Custom prompt
        modal run utils/test_deepseek_inference.py --prompt "Explain quantum computing"
        
        # Chat mode (uses apply_chat_template)
        modal run utils/test_deepseek_inference.py --chat-mode
        
        # Custom model path
        modal run utils/test_deepseek_inference.py --model-path "deepseek-ai--DeepSeek-V3.2"
        
        # Use FP8 quantization (halves memory, fits entirely on GPU)
        modal run utils/test_deepseek_inference.py --use-fp8
        
        # Skip local copy (load directly from volume - slower but no copy time)
        modal run utils/test_deepseek_inference.py --no-use-local-copy
    """
    print("🚀 Launching DeepSeek V3.2 inference test on Modal")
    print("=" * 70)
    print("Configuration:")
    print(f"  GPU Type:        {GPU_TYPE}")
    print(f"  GPU Count:       {GPU_COUNT}")
    print(f"  Model Path:      {model_path}")
    print(f"  Mode:            {'Chat' if chat_mode else 'Completion'}")
    print(f"  Max New Tokens:  {max_new_tokens}")
    print(f"  Use FP8:         {use_fp8}")
    print(f"  Use Local Copy:  {use_local_copy}")
    print("=" * 70)

    if chat_mode:
        result = test_chat_inference.remote(
            model_path=model_path,
            user_message=prompt,
            max_new_tokens=max_new_tokens,
            use_fp8=use_fp8,
            use_local_copy=use_local_copy,
        )
    else:
        result = test_inference.remote(
            model_path=model_path,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            use_fp8=use_fp8,
            use_local_copy=use_local_copy,
        )

    print("\n" + "=" * 70)
    print("✅ Test completed!")
    print("=" * 70)

