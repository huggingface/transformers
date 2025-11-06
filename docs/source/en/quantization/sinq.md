[![arXiv](https://img.shields.io/badge/arXiv-2509.22944-b31b1b.svg)](https://arxiv.org/abs/2509.22944)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub stars](https://img.shields.io/github/stars/huawei-csl/SINQ?label=Stars&logo=github&logoColor=white&style=flat-square)](https://github.com/huawei-csl/SINQ/stargazers)
[![hf-space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Huawei%20CSL-ffc107?color=ffc107&logoColor=white)](https://huggingface.co/huawei-csl)

# SINQ

[Sinkhorn-Normalized Quantization (SINQ)](https://github.com/huawei-csl/SINQ/tree/main) is a fast, plug-and-play, model-agnostic quantization technique delivering state-of-the-art performance for Large Language Models without sacrificing accuracy.

### 🔍 What You’ll Find Here

- [1. How does SINQ work?](#1-how-does-sinq-work)
- [2. Why should I use SINQ?](#2-why-should-i-use-sinq)
- <u>[3. Quantize (and save) any LLM with SINQ](#3-quantize-any-llm-with-sinq)</u>
- [4. How to Cite This Work](#4-how-to-cite-this-work)
- [5. Current Limitations](#5-current-limitations)

#### 📊 Feature Comparison: SINQ vs HQQ _(calibration-free)_ and A-SINQ vs AWQ _(calibrated)_

| Feature | **SINQ** | **HQQ** | **A-SINQ** | **AWQ** |
|------------|:--------:|:--------:|:----------:|:-------:|
| 🎯 Calibration | Calibration-free | Calibration-free |  Calibrated | Calibrated |
| 🧮 Quantization Type | Symmetric & Asymmetric | Asymmetric only | Symmetric & Asymmetric | Symmetric & Asymmetric |
| 📦 NF4 Support | **Yes** | No | **Yes** | No |
| ⚡ Quantization Speed | ~2× **Faster** than HQQ | Slower | ~4× **Faster** than AWQ | Slower |
| 📈 Model Quality | **Higher** | Lower | **Higher** | Lower |


📄 **Want to know more?** Read our paper on [**arXiv**](http://arxiv.org/abs/2509.22944)!

--- 

## 1. How does SINQ work?

<details>
<summary>Click to expand a quick explanation of SINQ’s core idea</summary>

#### 1️⃣ Dual-Scaling for Better Quantization

Conventional quantization uses **one scale per weight dimension**, which makes models vulnerable to **outliers**: large weights that distort scaling and cause significant errors.

**SINQ** solves this by introducing **dual scaling**: separate scale factors for **rows and columns**. This flexibility redistributes outlier influence and keeps quantization errors smaller and more balanced.

---


#### 2️⃣ More Even Error Distribution

With standard single-scale quantization, errors tend to **cluster around outliers**.  
With **SINQ**, they become **spread out and less severe**, preserving model accuracy even at **3 bit precision**. This improvement is driven by SINQ’s **Sinkhorn-normalized optimization**, which iteratively rescales rows and columns to balance their variance - a process inspired by Sinkhorn matrix normalization. By reducing the overall **_matrix imbalance_** (refer to the paper for more info), weights become inherently easier to quantize, leading to more stable behavior across layers and consistently higher accuracy even at very low bit-widths.


</details>

---
## 2. Why should I use SINQ?
<details>
<summary>Click to expand a quick explanation on why you should use SINQ to quantize your LLM</summary>


#### **SINQ (calibration-free)**  
- **Higher LLM quality** and **~2× faster** quantization than **HQQ** 
- **>31× faster** quantization process and comparable or better LLM quality compared to **AWQ / GPTQ**
- **Model-agnostic**: works without knowing the specific LLM architecture, unlike **QuaRot**  
- **Training-free**: it does not require end-to-end training, unlike **SpinQuant** or **KurTail** 
- **Additionally, A-SINQ (calibrated)** further **beats AWQ, GPTQ, and Hadamard+GPTQ** on quality while achieving **>4× faster** quantization time.

</details>

## 3. Quantize any LLM with SINQ

### Setup & Quick Start

First, install the package:

```bash
pip install sinq
```

---

### Quantize in a few lines

Quantizing any 🤗 Hugging Face model with SINQ is simple and takes only a few lines of code. 
First, create a [`SinqConfig`] and specifying the following parameters:

| Flag | Description | Options | Default |
|------|-------------|---------|----------|
| `--nbits` | Bit-width for weight quantization | 2, 3, 4, 5, 6, 8 | 4 |
| `--tiling_mode` | Weight matrix tiling strategy | 1D, 2D | 1D |
| `--group_size` | Weights per quantization group | 64, 128 | 64 |
| `--method` | Quantization method | sinq, asinq | sinq |
| `--dtype` | Data type of the original model | auto, float16, float32, etc | auto (bfloaf16) |
| `--modules_to_not_convert` | List of the layers that are NOT quantize | [lm_head, ...] | [lm_head] |
| `--auto_patch_io` | Flag to use custom save and reload functions | True, False | True |

Then specify the model you want to quantize and pass the SinqConfig as quantization configuration option

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, SinqConfig

model_name = "Qwen/Qwen3-1.7B"
device = "cuda:0"

cfg = SinqConfig(
    nbits=4,
    group_size=64,
    tiling_mode="1D",
    method="sinq",
    dtype="auto",
    modules_to_not_convert=["lm_head"],
    auto_patch_io=True
)

tok = AutoTokenizer.from_pretrained(model_name)
qmodel = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=cfg,
    torch_dtype=torch.bfloat16
)

```

✅ That’s it. Your model is now quantized with **SINQ** and ready for inference or saving.

---

### Save & reload

If you want to reuse a quantized model later, save it to disk or push it on the HuggingFace Hub and reload it without needing base FP weights.

```python
# --- Save to a folder (sharded safetensors) ---

# 'model' must already be SINQ-quantized
# Locally save
qmodel.save_pretrained("/path/to/save/qwen3-1.7B-sinq-4bit")
# Push to the Hub
qmodel.push_to_hub("HF_Hub_username/qwen3-1.7B-sinq-4bit", safe_serialization=True)
tok.push_to_hub("HF_Hub_username/qwen3-1.7B-sinq-4bit")

```

```python
# --- Reload later--

save_dir = "/path/to/save/qwen3-1.7B-sinq-4bit"
hf_hub_model = "HF_Hub_username/qwen3-1.7B-sinq-4bit"

# From local directory
tokenizer = AutoTokenizer.from_pretrained(save_dir)
qmodel = AutoModelForCausalLM.from_pretrained(save_dir)

# From HF Hub
tokenizer = AutoTokenizer.from_pretrained(hf_hub_model)
qmodel = AutoModelForCausalLM.from_pretrained(hf_hub_model)

```

✅ Your model is now loaded and ready for inference!

> Note: If the model has been quantized in 4 bit and 'gemlite' library is installed, SINQ utilize gemlite kernel to run the inference.

---

### Compatible with [`lm-eval`](https://github.com/EleutherAI/lm-evaluation-harness) evaluation framework

Below is a minimal example showing how to evaluate a SINQ-quantized model on a benchmark dataset:

```python
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

# Wrap the already quantized model and tokenizer with HFLM
lm = HFLM(pretrained=qmodel, tokenizer=tokenizer, device=device)

# Evaluate (many tasks available on lm-eval such as MMLU and HellaSwag)
results = evaluator.simple_evaluate(
    model=lm,
    tasks=["lambada_openai"],  # small and fast benchmark
    device=device
)
```

## 4. How to Cite This Work

If you find **SINQ** useful in your research or applications, please cite our <a href="http://arxiv.org/abs/2509.22944" target="_blank"><strong>paper</strong></a>:

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

## 5. Current Limitations

SINQ quantization strategy and SINQ quantized models do not support Multi-GPU option, so if your system counts multiple GPUs please specify which one should be used.