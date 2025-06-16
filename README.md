# 2lit-formers

[![Badge indicating the status of the Secret Leaks workflow on GitHub Actions, showing the branch Nodoubtz](https://github.com/nodoubtz/2lit-formers/actions/workflows/trufflehog.yml/badge.svg?branch=Nodoubtz)](https://github.com/nodoubtz/2lit-formers/actions/workflows/trufflehog.yml)

[![Deploy static content to Pages](https://github.com/nodoubtz/2lit-formers/actions/workflows/static.yml/badge.svg)](https://github.com/nodoubtz/2lit-formers/actions/workflows/static.yml)

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

```bash
pip install git+https://github.com/nodoubtz/2lit-formers.git
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
