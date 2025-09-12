<!--Copyright 2025 OpenMOSS and The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
http://www.apache.org/licenses/LICENSE-2.0
-->


# XY-Tokenizer

<div style="float: right;">
<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
</div>
</div>

## Overview

**`XY-Tokenizer`** is a speech codec that simultaneously models both semantic and acoustic aspects of speech, converting audio into discrete tokens and decoding them back to high-quality audio. It achieves efficient speech representation at only 1kbps with RVQ8 quantization at 12.5Hz frame rate.

-   **Paper:** https://huggingface.co/papers/2506.23325
-   **Source Code:** [GitHub Repo](https://github.com/OpenMOSS/MOSS-TTSD/tree/main/XY_Tokenizer)
| [Hugging Face Repo](https://huggingface.co/spaces/fnlp/MOSS-TTSD/tree/main/XY_Tokenizer)

## 📚 Related Project: **[MOSS-TTSD](https://huggingface.co/fnlp/MOSS-TTSD-v0.5)**

**`XY-Tokenizer`** serves as the underlying neural codec for **`MOSS-TTSD`**, our 1.7B Audio Language Model. \
Explore **`MOSS-TTSD`** for advanced text-to-speech and other audio generation tasks on [GitHub](https://github.com/OpenMOSS/MOSS-TTSD), [Blog](http://www.open-moss.com/en/moss-ttsd/), [博客](https://www.open-moss.com/cn/moss-ttsd/), and [Space Demo](https://huggingface.co/spaces/fnlp/MOSS-TTSD).

## Features

- **Dual-channel modeling**: Simultaneously captures semantic meaning and acoustic details
- **Efficient representation**: 1kbps bitrate with RVQ8 quantization at 12.5Hz
- **High-quality audio tokenization**: Convert speech to discrete tokens and back with minimal quality loss
- **Long audio support**: Process audio files longer than 30 seconds using chunking with overlap
- **Batch processing**: Efficiently process multiple audio files in batches
- **24kHz output**: Generate high-quality 24kHz audio output

## Configuration

[[autodoc]] XYTokenizerConfig

## Feature Extraction

[[autodoc]] XYTokenizerFeatureExtractor

## Model

[[autodoc]] XYTokenizer
