<!---
Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Transformers Installation Guide (বাংলা)

> এই গাইডে Hugging Face Transformers লাইব্রেরি ইন্সটল, সেটআপ, সোর্স ইন্সটল, এডিটেবল ইন্সটল, ক্যাশ কনফিগারেশন এবং অফলাইন মোড বিস্তারিতভাবে ব্যাখ্যা করা হয়েছে।

---

# Installation

Transformers সাধারণত PyTorch-এর সাথে কাজ করে। এটি Python 3.10+ এবং PyTorch 2.4+ ভার্সনে টেস্ট করা হয়েছে।

---

# ভার্চুয়াল এনভায়রনমেন্ট (Virtual Environment)

[`uv`](https://docs.astral.sh/uv/) হলো Rust-ভিত্তিক অত্যন্ত দ্রুতগতির একটি Python package এবং project manager। এটি আলাদা আলাদা প্রজেক্টের dependency isolate রাখতে সাহায্য করে এবং default ভাবেই virtual environment ব্যবহার করতে উৎসাহ দেয়।

তুমি চাইলে `uv`-কে `pip`-এর বিকল্প হিসেবেও ব্যবহার করতে পারো।

## uv Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows-এর জন্য:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

# Virtual Environment তৈরি করা

```bash
uv venv .env
```

Linux/macOS:

```bash
source .env/bin/activate
```

Windows CMD:

```cmd
.env\Scripts\activate
```

Windows PowerShell:

```powershell
.env\Scripts\Activate.ps1
```

---

# Transformers Install

```bash
uv pip install transformers
```

---

# PyTorch Install

## CPU Version

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

তারপর Transformers install করো:

```bash
uv pip install transformers
```

---

## GPU Version (CUDA)

GPU acceleration পাওয়ার জন্য CUDA compatible PyTorch install করতে হবে।

PyTorch official website থেকে সঠিক command generate করা যায়:

https://pytorch.org/get-started/locally/

Example:

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

# NVIDIA GPU Check

তোমার system NVIDIA GPU detect করছে কি না তা check করতে:

```bash
nvidia-smi
```

যদি GPU information show করে তাহলে CUDA সঠিকভাবে কাজ করছে।

---

# Installation Verify

Transformers ঠিকভাবে install হয়েছে কি না check করতে:

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('hugging face is the best'))"
```

Expected Output:

```python
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

---

# Source Install

Source থেকে install করলে Transformers-এর latest development version পাওয়া যায়।

এটি useful যখন:

- নতুন feature test করতে চাও
- unreleased bug fix ব্যবহার করতে চাও
- latest updates দরকার

## Install from GitHub

```bash
uv pip install git+https://github.com/huggingface/transformers
```

Verify:

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('hugging face is the best'))"
```

Expected Output:

```python
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

---

# Editable Install

লোকালি Transformers-এর source code modify বা contribute করতে চাইলে editable install ব্যবহার করা হয়।

## Clone Repository

```bash
git clone https://github.com/huggingface/transformers.git
```

```bash
cd transformers
```

## Editable Mode Install

```bash
uv pip install -e .
```

> ⚠️ Warning:
>
> Editable install ব্যবহার করলে local repository delete করা যাবে না। কারণ Python সরাসরি ওই folder-এর সাথে linked থাকে।

---

# Latest Changes Pull করা

```bash
cd ~/transformers/
```

```bash
git pull
```

---

# Conda দিয়ে Install

`conda` হলো language-agnostic package manager।

নতুন virtual environment-এ conda-forge channel থেকে Transformers install করতে:

```bash
conda install conda-forge::transformers
```

---

# Setup

Install হয়ে গেলে কিছু optional configuration করা যায়:

- Cache directory change
- Offline mode enable
- Model pre-download

---

# Cache Directory

যখন `PreTrainedModel.from_pretrained()` ব্যবহার করে model load করা হয় তখন model automatically Hugging Face Hub থেকে download হয়ে local cache-এ save হয়।

পরবর্তীতে একই model load করলে cache থেকে load হয়।

---

## Default Cache Location

### Linux/macOS

```text
~/.cache/huggingface/hub
```

### Windows

```text
C:\Users\username\.cache\huggingface\hub
```

---

# Cache Directory Change

নিচের environment variable ব্যবহার করে cache location change করা যায়।

Priority order:

1. `HF_HUB_CACHE`
2. `HF_HOME`
3. `XDG_CACHE_HOME/huggingface`

---

## Linux/macOS Example

```bash
export HF_HUB_CACHE=/path/to/cache
```

## Windows CMD Example

```cmd
set HF_HUB_CACHE=D:\huggingface_cache
```

## Windows PowerShell Example

```powershell
$env:HF_HUB_CACHE="D:\huggingface_cache"
```

---

# Offline Mode

যদি internet ছাড়া environment-এ Transformers ব্যবহার করতে চাও তাহলে আগে থেকেই model download করে রাখতে হবে।

---

# Model Download করে রাখা

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="meta-llama/Llama-2-7b-hf",
    repo_type="model"
)
```

---

# Offline Mode Enable

Linux/macOS:

```bash
HF_HUB_OFFLINE=1 python script.py
```

Example:

```bash
HF_HUB_OFFLINE=1 \
python examples/pytorch/language-modeling/run_clm.py \
--model_name_or_path meta-llama/Llama-2-7b-hf \
--dataset_name wikitext
```

---

# Local Files Only Mode

শুধুমাত্র local cached files ব্যবহার করতে:

```python
from transformers import LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained(
    "./path/to/local/directory",
    local_files_only=True
)
```

---

# Transformers Basic Usage

## Sentiment Analysis

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

result = classifier("I love Transformers library")

print(result)
```

---

# Text Generation

```python
from transformers import pipeline

generator = pipeline("text-generation")

result = generator(
    "Machine learning is",
    max_length=30,
    num_return_sequences=1
)

print(result)
```

---

# Translation Example

```python
from transformers import pipeline

translator = pipeline(
    "translation_en_to_fr",
    model="Helsinki-NLP/opus-mt-en-fr"
)

result = translator("Hello, how are you?")

print(result)
```

---

# Common Pipelines

| Pipeline | Description |
|---|---|
| sentiment-analysis | Sentiment detect করে |
| text-generation | Text generate করে |
| translation | Language translation |
| summarization | Text summarize করে |
| question-answering | প্রশ্নের উত্তর দেয় |
| image-classification | Image classify করে |
| object-detection | Object detect করে |
| automatic-speech-recognition | Speech to text |

---

# Common Errors & Fixes

## Torch Not Installed

Error:

```text
ModuleNotFoundError: No module named 'torch'
```

Fix:

```bash
uv pip install torch
```

---

## CUDA Not Available

Check:

```python
import torch
print(torch.cuda.is_available())
```

If False:

- CUDA version mismatch হতে পারে
- NVIDIA driver install করা নেই
- CPU version install হয়েছে

---

## SSL Error

```text
SSL CERTIFICATE_VERIFY_FAILED
```

Fix:

```bash
pip install --upgrade certifi
```

---

# Useful Hugging Face Libraries

| Library | Purpose |
|---|---|
| transformers | NLP models |
| datasets | Dataset loading |
| accelerate | Multi-GPU training |
| peft | LoRA / PEFT training |
| diffusers | Image generation |
| tokenizers | Fast tokenization |
| evaluate | Model evaluation |

---

# Install Additional Libraries

```bash
uv pip install datasets accelerate evaluate peft
```

---

# Hugging Face Login

CLI login:

```bash
huggingface-cli login
```

Python login:

```python
from huggingface_hub import login

login("your_token")
```

---

# Best Practices

- সবসময় virtual environment ব্যবহার করো
- GPU compatible PyTorch install করো
- বড় model-এর জন্য SSD storage ব্যবহার করো
- cache directory আলাদা drive-এ রাখো
- production-এ stable release ব্যবহার করো
- development-এ source install ব্যবহার করা যায়

---

# Useful Links

## Official Websites

- https://huggingface.co/
- https://github.com/huggingface/transformers
- https://pytorch.org/
- https://docs.astral.sh/uv/

---

# Conclusion

এই guide-এ আমরা শিখলাম:

- Transformers install
- PyTorch setup
- GPU configuration
- Source install
- Editable install
- Cache management
- Offline mode
- Basic usage
- Common troubleshooting

এখন তুমি সহজেই Hugging Face Transformers ব্যবহার করে NLP, LLM, Translation, Summarization, Text Generation সহ বিভিন্ন AI application build করতে পারবে।
