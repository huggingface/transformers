# Minimal Repro Template (PEFT/LoRA)

Use this template to produce a **copy/paste runnable** repro that someone else can run and see the same issue.

## 0. One-line goal
**Goal:** <What should happen?>

## 1. What is happening (actual)
**Actual:** <What happens instead? Include exact error message or wrong output>

## 2. Environment (must be exact)
Fill in all that apply.

- OS: Windows / Linux / macOS (include version)
- Python: `python -V`
- PEFT: `python -c "import peft; print(peft.__version__)"`
- Transformers: `python -c "import transformers; print(transformers.__version__)"`
- PyTorch: `python -c "import torch; print(torch.__version__)"`
- bitsandbytes: `python -c "import bitsandbytes; print(bitsandbytes.__version__)"`
- Device: CPU / CUDA (include GPU model + VRAM)
- CUDA version: `nvcc --version` or `torch.version.cuda`

## 3. Installation commands (exact)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install torch transformers peft bitsandbytes accelerate
```

## 4. Minimal script (single file)

Create `repro.py` with the smallest code that still fails.

```python
import os
import sys

def print_env():
    print("== ENV ==")
    print("python:", sys.version.replace("\n", " "))
    try:
        import peft
        print("peft:", peft.__version__)
    except Exception as e:
        print("peft import failed:", repr(e))
    try:
        import transformers
        print("transformers:", transformers.__version__)
    except Exception as e:
        print("transformers import failed:", repr(e))
    try:
        import torch
        print("torch:", torch.__version__)
        print("cuda available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("cuda device:", torch.cuda.get_device_name(0))
            print("cuda memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
    except Exception as e:
        print("torch import failed:", repr(e))
    try:
        import bitsandbytes as bnb
        print("bitsandbytes:", bnb.__version__)
    except Exception as e:
        print("bitsandbytes:", repr(e))
    print()

def main():
    print_env()
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    
    # TODO: Replace with your exact setup
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # TODO: Replace with your LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # TODO: Add the failing operation
    # ...

if __name__ == "__main__":
    main()
```

## 5. Run command + full output

Command used:
```bash
python repro.py
```

Paste the **full output** here (don't truncate).

## 6. Expected vs actual (explicit)

* **Expected:** <exact>
* **Actual:** <exact>

## 7. LoRA config details

```python
LoraConfig(
    r=...,
    lora_alpha=...,
    lora_dropout=...,
    target_modules=...,
    modules_to_save=...,
    task_type=...,
)
```

## 8. For quantized models (QLoRA)

```python
BitsAndBytesConfig(
    load_in_4bit=True/False,
    load_in_8bit=True/False,
    bnb_4bit_quant_type="nf4"/"fp4",
    bnb_4bit_compute_dtype=torch.bfloat16/torch.float16,
    bnb_4bit_use_double_quant=True/False,
)
```

## 9. Knobs to try (only relevant ones)

* LoRA rank: r=8 vs r=16 vs r=32
* Target modules: specific vs `"all-linear"`
* Quantization: None vs 8-bit vs 4-bit
* Compute dtype: fp16 vs bf16
* `prepare_model_for_kbit_training()`: called / not called
* Batch size: 1 / 4 / 8
* Sequence length: 128 / 256 / 512

## 10. If it's a PEFT bug

* Suspected module/file in peft repo:
  * `src/peft/...`
* Related issues on GitHub:
  * Link if found
* Does it work with older peft version?
  * Last known working: `pip install peft==X.Y.Z`
