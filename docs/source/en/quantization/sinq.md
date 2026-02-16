[![arXiv](https://img.shields.io/badge/arXiv-2509.22944-b31b1b.svg)](https://arxiv.org/abs/2509.22944)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub stars](https://img.shields.io/github/stars/huawei-csl/SINQ?label=Stars&logo=github&logoColor=white&style=flat-square)](https://github.com/huawei-csl/SINQ/stargazers)
[![hf-space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Huawei%20CSL-ffc107?color=ffc107&logoColor=white)](https://huggingface.co/huawei-csl)

# SINQ

[Sinkhorn-Normalized Quantization (SINQ)](https://github.com/huawei-csl/SINQ/tree/main) is a fast, plug-and-play, model-agnostic quantization technique delivering state-of-the-art performance for Large Language Models without sacrificing accuracy.

### 🔍 What You’ll Find Here

- [1. Quantize (and save) any LLM with SINQ](#1-quantize-any-llm-with-sinq)
- [2. How to Cite This Work](#2-how-to-cite-this-work)
- [3. Current Limitations](#3-current-limitations)

#### 📊 Feature Comparison: SINQ vs HQQ _(calibration-free)_ and A-SINQ vs AWQ _(calibrated)_

| Feature | **SINQ** | **HQQ** | **A-SINQ** | **AWQ** |
|------------|:--------:|:--------:|:----------:|:-------:|
| 🎯 Calibration | Calibration-free | Calibration-free |  Calibrated | Calibrated |
| 🧮 Quantization Type | Symmetric & Asymmetric | Asymmetric only | Symmetric & Asymmetric | Symmetric & Asymmetric |
| 📦 NF4 Support | **Yes** | No | **Yes** | No |
| ⚡ Quantization Speed | ~2× **Faster** than HQQ | Slower | ~4× **Faster** than AWQ | Slower |
| 📈 Model Quality | **Higher** | Lower | **Higher** | Lower |


📄 **Want to know more?** 
- Read our paper on [**arXiv**](http://arxiv.org/abs/2509.22944)
- Check the official [**SINQ**](https://github.com/huawei-csl/SINQ/tree/main) github repository

--- 

## 1. Quantize any LLM with SINQ

### Setup & Quick Start

First, install the package. It can be done in two ways:
- From source using the official Github repository [**SINQ**](https://github.com/huawei-csl/SINQ/tree/main) **[Recommended]**
- Using pip package:
```bash
pip install sinq
```

---

### Quantize in a few lines

Quantizing any 🤗 Hugging Face model with SINQ is simple and takes only a few lines of code. 
First, create a [`SinqConfig`] and specify the following parameters:

| Flag | Description | Type | Options | Default |
|------|-------------|---------|---------|----------|
| `--nbits` | Bit-width for weight quantization | int | 2, 3, 4, 5, 6, 8 | 4 |
| `--tiling_mode` | Weight matrix tiling strategy | str | 1D, 2D | 1D |
| `--group_size` | Weights per quantization group | int | 64, 128 | 64 |
| `--method` | Quantization method | str | sinq, asinq | sinq |
| `--modules_to_not_convert` | List of the layers that are NOT quantize | List of str | [lm_head, ...] | [lm_head] |

Then specify the model you want to quantize and pass the SinqConfig as quantization configuration option

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, SinqConfig

model_name = "Qwen/Qwen3-1.7B"

cfg = SinqConfig(
    nbits=4,
    group_size=64,
    tiling_mode="1D",
    method="sinq",
    modules_to_not_convert=["lm_head"]
)

tok = AutoTokenizer.from_pretrained(model_name)
qmodel = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=cfg,
    dtype=torch.bfloat16
)

```

✅ That’s it. Your model is now quantized with **SINQ** and ready for inference or saving.

> Check our official [**SINQ**](https://github.com/huawei-csl/SINQ/tree/main) github repository to stay updated!

---

### Save & reload

If you want to reuse a quantized model later, save it to disk or push it on the HuggingFace Hub and reload it without needing base FP weights.
If you installed SINQ from source you should call *patch_hf_pretrained_io* function when re-loading a quantized model:
```python
# Save sinq quantized model
model.save_pretrained("/path/to/save/qwen3-1.7B-sinq-4bit")
model.push_to_hub("HF_Hub_username/qwen3-1.7B-sinq-4bit")
tokenizer.push_to_hub("HF_Hub_username/qwen3-1.7B-sinq-4bit")
```
```python
from sinq.hf_io import patch_hf_pretrained_io
patch_hf_pretrained_io()
# Reload a sinq quantized model
hf_hub_model = "HF_Hub_username/qwen3-1.7B-sinq-4bit"
tokenizer  = AutoTokenizer.from_pretrained(hf_hub_model)
model = AutoModelForCausalLM.from_pretrained(hf_hub_model)
```
Otherwise, if you installed SINQ through pip, you can simply use HF built-in functions:

```python
# --- Save to a folder (sharded safetensors) ---

# 'model' must already be SINQ-quantized
# Locally save
qmodel.save_pretrained("/path/to/save/qwen3-1.7B-sinq-4bit")
# Push to the Hub
qmodel.push_to_hub("HF_Hub_username/qwen3-1.7B-sinq-4bit")
tok.push_to_hub("HF_Hub_username/qwen3-1.7B-sinq-4bit")

# --- Reload later--

save_dir = "/path/to/save/qwen3-1.7B-sinq-4bit"
hf_hub_model = "HF_Hub_username/qwen3-1.7B-sinq-4bit"

# From local directory
tok = AutoTokenizer.from_pretrained(save_dir)
qmodel = AutoModelForCausalLM.from_pretrained(save_dir)

# From HF Hub
tok = AutoTokenizer.from_pretrained(hf_hub_model)
qmodel = AutoModelForCausalLM.from_pretrained(hf_hub_model)

```

✅ Your model is now loaded and ready for inference!

> Note: If the model has been quantized in 4 bit and `gemlite` library is installed, gemlite faster kernel is used to run the inference.

---

### Compatible with [`lm-eval`](https://github.com/EleutherAI/lm-evaluation-harness) evaluation framework

Below is a minimal example showing how to evaluate a SINQ-quantized model on a benchmark dataset:

```python
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

# Wrap the already quantized model and tokenizer with HFLM
lm = HFLM(pretrained=qmodel, tokenizer=tok, device=device)
device = "cuda:0"

# Evaluate (many tasks available on lm-eval such as MMLU and HellaSwag)
results = evaluator.simple_evaluate(
    model=lm,
    tasks=["wikitext"],  # small and fast benchmark
    device=device
)
```

## 2. How to Cite This Work

If you find **SINQ** useful in your research or applications
- Support our project by putting a star ⭐️ in the [**SINQ**](https://github.com/huawei-csl/SINQ/tree/main) github repository
- Please cite our <a href="http://arxiv.org/abs/2509.22944" target="_blank"><strong>paper</strong></a>:

```bibtex
@misc{muller2025sinq,
      title={SINQ: Sinkhorn-Normalized Quantization for Calibration-Free Low-Precision LLM Weights}, 
      author={Lorenz K. Muller and Philippe Bich and Jiawei Zhuang and Ahmet Celik and Luca Benfenati and Lukas Cavigelli},
      year={2025},
      eprint={2509.22944},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={http://arxiv.org/abs/2509.22944}
}
```

## 3. Current Limitations

Currently, the A-SINQ method is not supported in Hugging Face. Please refer to the official [SINQ repository](https://github.com/huawei-csl/SINQ/tree/main) to quantize a model with this strategy.
At the moment the SINQ quantization strategy and SINQ quantized models do not support Multi-GPU option, so if your system counts multiple GPUs please specify which one should be used.