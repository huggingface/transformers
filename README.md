[![Secret Leaks](https://github.com/nodoubtz/2lit-formers/actions/workflows/trufflehog.yml/badge.svg?branch=Nodoubtz)](https://github.com/nodoubtz/2lit-formers/actions/workflows/trufflehog.yml)

[![Deploy static content to Pages](https://github.com/nodoubtz/2lit-formers/actions/workflows/static.yml/badge.svg)](https://github.com/nodoubtz/2lit-formers/actions/workflows/static.yml)

# 2lit-formers

[![Homepage](https://img.shields.io/badge/docs-website-blue)](https://nodoubtz.github.io/2lit-formers/)
[![GitHub](https://img.shields.io/badge/source-GitHub-green)](https://github.com/nodoubtz/2lit-formers)

## Overview

**2lit-formers** is a fork of the popular [Hugging Face Transformers](https://github.com/huggingface/transformers) library, providing state-of-the-art machine learning tools for PyTorch, TensorFlow, and JAX. This repository aims to advance and secure transformer models for research and production.

- **Language**: Python
- **License**: Other
- **Default branch**: `2lit`
- **Homepage**: [https://nodoubtz.github.io/2lit-formers/](https://nodoubtz.github.io/2lit-formers/)

## Features

- ⚡ Ready-to-use transformer architectures for NLP, vision, and more.
- 🔒 Focus on secure model management and vulnerability mitigation.
- 🛠️ Compatible with PyTorch, TensorFlow, and JAX.
- 🚀 Based on the Hugging Face Transformers ecosystem, with improvements and extensions.

## Getting Started

### Installation

codespace-organic-succotash-5rqgw4j5xqv376pr
```bash
pip install git+https://github.com/nodoubtz/2lit-formers.git
=======
Explore the [Hub](https://huggingface.com/) today to find a model and use Transformers to help you get started right away.

## Installation

Transformers works with Python 3.9+ [PyTorch](https://pytorch.org/get-started/locally/) 2.1+, [TensorFlow](https://www.tensorflow.org/install/pip) 2.6+, and [Flax](https://flax.readthedocs.io/en/latest/) 0.4.1+.

Create and activate a virtual environment with [venv](https://docs.python.org/3/library/venv.html) or [uv](https://docs.astral.sh/uv/), a fast Rust-based Python package and project manager.

```py
# venv
python -m venv .my-env
source .my-env/bin/activate
# uv
uv venv .my-env
source .my-env/bin/activate
```

Install Transformers in your virtual environment.

```py
# pip
pip install "transformers[torch]"

# uv
uv pip install "transformers[torch]"
```

Install Transformers from source if you want the latest changes in the library or are interested in contributing. However, the *latest* version may not be stable. Feel free to open an [issue](https://github.com/huggingface/transformers/issues) if you encounter an error.

```shell
git clone https://github.com/huggingface/transformers.git
cd transformers

# pip
pip install .[torch]

# uv
uv pip install .[torch]
```

## Quickstart

Get started with Transformers right away with the [Pipeline](https://huggingface.co/docs/transformers/pipeline_tutorial) API. The `Pipeline` is a high-level inference class that supports text, audio, vision, and multimodal tasks. It handles preprocessing the input and returns the appropriate output.

Instantiate a pipeline and specify model to use for text generation. The model is downloaded and cached so you can easily reuse it again. Finally, pass some text to prompt the model.

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
pipeline("the secret to baking a really good cake is ")
[{'generated_text': 'the secret to baking a really good cake is 1) to use the right ingredients and 2) to follow the recipe exactly. the recipe for the cake is as follows: 1 cup of sugar, 1 cup of flour, 1 cup of milk, 1 cup of butter, 1 cup of eggs, 1 cup of chocolate chips. if you want to make 2 cakes, how much sugar do you need? To make 2 cakes, you will need 2 cups of sugar.'}]
```

To chat with a model, the usage pattern is the same. The only difference is you need to construct a chat history (the input to `Pipeline`) between you and the system.

> [!TIP]
> You can also chat with a model directly from the command line.
> ```shell
> transformers chat Qwen/Qwen2.5-0.5B-Instruct
> ```

```py
import torch
from transformers import pipeline

chat = [
    {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
]

pipeline = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
response = pipeline(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
```

Expand the examples below to see how `Pipeline` works for different modalities and tasks.

<details>
<summary>Automatic speech recognition</summary>

```py
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
main
```

### Usage

Import and use pre-trained models as you would with Hugging Face Transformers:

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
```

> For more examples and API documentation, see the [project homepage](https://nodoubtz.github.io/2lit-formers/).

## Contributing

We welcome contributions to enhance features, security, and performance.

- Please check existing issues before opening new ones.
- Follow standard Python code conventions and best practices.
- Ensure new code does not introduce vulnerabilities.

## Issues

If you find a bug or security vulnerability, please [open an issue](https://github.com/nodoubtz/2lit-formers/issues) with detailed information.

## License

This project uses a custom license. Please see the `LICENSE` file for details.

## Acknowledgments

- Forked from [huggingface/transformers](https://github.com/huggingface/transformers).
