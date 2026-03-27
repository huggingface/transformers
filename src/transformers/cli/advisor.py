# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from transformers import AutoConfig


def advisor(model_name_or_path: str):
    """
    Unique CLI tool to advise on model loading and quantization based on hardware.
    """
    print("\n--- Transformers Model Advisor ---\n")
    print(f"Analyzing model: {model_name_or_path}")

    # 1. System Check
    has_cuda = torch.cuda.is_available()
    vram_gb = 0
    if has_cuda:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Hardware Detected: {gpu_name} ({vram_gb:.2f} GB VRAM)")
    else:
        print("Hardware Detected: CPU Only")

    # 2. Model Forensics
    try:
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading model config: {e}")
        return

    # Approximate parameter count from config (very heuristic)
    # Most LLMs: num_layers * hidden_size * hidden_size * 12 (approx)
    params = 0
    if hasattr(config, "num_hidden_layers") and hasattr(config, "hidden_size"):
        params = config.num_hidden_layers * (config.hidden_size**2) * 12
    elif hasattr(config, "n_layer") and hasattr(config, "n_embd"):
        params = config.n_layer * (config.n_embd**2) * 12

    params_b = params / (10**9)
    if params_b > 0:
        print(f"Estimated parameters: {params_b:.2f}B")
    else:
        print("Estimated parameters: Unknown (Config structure complex)")

    # VRAM Estimation
    vram_fp16 = params_b * 2  # 2 bytes per param
    vram_int8 = params_b * 1
    vram_int4 = params_b * 0.7  # including overhead

    print("\nEstimated VRAM Requirements:")
    print(f"  - FP16/BF16: ~{vram_fp16:.2f} GB")
    print(f"  - INT8:      ~{vram_int8:.2f} GB")
    print(f"  - 4-bit:     ~{vram_int4:.2f} GB")

    # 3. Recommendations
    print("\n--- Optimization Strategy ---")
    from transformers.utils.hardware import get_best_loading_strategy
    strategy = get_best_loading_strategy(config, model_name_or_path)

    print(f"Osman Advisor Recommended Strategy: {strategy}")
    if "load_in_4bit" in strategy:
        print("Advice: Use 4-bit quantization (BitsAndBytes) as VRAM is sufficient for compressed loading but tight for FP16.")
    elif strategy.get("torch_dtype") == torch.float16:
        print("Advice: You have plenty of VRAM. Using full precision (FP16).")

    print("\nPro-Tip: Use `load_best_fit=True` in `from_pretrained` to automate this!")
    print("\n------------------------------\n")
