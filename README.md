<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg">
    <img alt="Hugging Face Transformers Library" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg" width="352" height="59" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>

<p align="center">
    <a href="https://huggingface.com/models"><img alt="Checkpoints on Hub" src="https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/models&color=brightgreen"></a>
    <a href="https://circleci.com/gh/huggingface/transformers"><img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/main"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue"></a>
    <a href="https://huggingface.co/docs/transformers/index"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online"></a>
    <a href="https://github.com/huggingface/transformers/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md"><img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg"></a>
    <a href="https://zenodo.org/badge/latestdoi/155220641"><img src="https://zenodo.org/badge/155220641.svg" alt="DOI"></a>
</p>

<h4 align="center">
    <p>
        <b>English</b> |
        <a href="i18n/README_zh-hans.md">简体中文</a> |
        <a href="i18n/README_zh-hant.md">繁體中文</a> |
        <a href="i18n/README_ko.md">한국어</a> |
        <a href="i18n/README_es.md">Español</a> |
        <a href="i18n/README_ja.md">日本語</a> |
        <a href="i18n/README_hd.md">हिन्दी</a> |
        <a href="i18n/README_mr.md">मराठी</a> |
        <a href="i18n/README_ru.md">Русский</a> |
        <a href="i18n/README_pt-br.md">Português</a> |
        <a href="i18n/README_te.md">తెలుగు</a> |
        <a href="i18n/README_fr.md">Français</a> |
        <a href="i18n/README_de.md">Deutsch</a> |
        <a href="i18n/README_it.md">Italiano</a> |
        <a href="i18n/README_vi.md">Tiếng Việt</a> |
        <a href="i18n/README_ar.md">العربية</a> |
        <a href="i18n/README_ur.md">اردو</a> |
        <a href="i18n/README_bn.md">বাংলা</a> |
        <a href="i18n/README_fa.md">فارسی</a> |
        <a href="i18n/README_ro.md">Română</a> |
        <a href="i18n/README_tr.md">Türkçe</a>
    </p>
</h4>

<h3 align="center">
    <p>State-of-the-art pretrained models for inference and training</p>
</h3>

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png"/>
</h3>

Transformers is the model-definition framework for machine learning models in text, computer vision, audio, video, and multimodal tasks. It defines each model once, in PyTorch, and you use that definition for both inference and training.

Training frameworks, inference engines, and local runtimes build on that definition instead of writing their own. Transformers adds new architectures as they're released and matches each one to its original results, so the tools that build on it get every new model without reimplementing it.

<table>
<tr>
<td>
<a href="https://huggingface.co/docs/transformers/community_integrations/unsloth"><img src="https://cdn.jsdelivr.net/gh/selfhst/icons/webp/unsloth.webp" alt="Unsloth" width="32" height="32"></a><br>
Fine-tune with less memory. Unsloth trains LLMs in 4-bit, 8-bit, and 16-bit precision with custom kernels.
</td>
<td>
<a href="https://huggingface.co/docs/transformers/community_integrations/vllm"><img src="https://cdn.jsdelivr.net/gh/homarr-labs/dashboard-icons/svg/vllm.svg" alt="vLLM" width="40" height="40"></a><br>
Serve LLMs at scale. vLLM batches requests continuously and keeps KV cache memory compact with PagedAttention.
</td>
<td>
<a href="https://huggingface.co/docs/transformers/community_integrations/sglang"><img src="https://www.sglang.io/images/logo.png" alt="SGLang" width="80" height="24"></a><br>
Serve LLMs with low latency. SGLang is a high-throughput inference engine with a frontend language for agentic workflows.
</td>
<td>
<a href="https://huggingface.co/docs/transformers/community_integrations/llama_cpp"><picture><source media="(prefers-color-scheme: dark)" srcset="https://cdn.jsdelivr.net/gh/selfhst/icons/svg/llama-cpp-light.svg"><img src="https://cdn.jsdelivr.net/gh/selfhst/icons/svg/llama-cpp.svg" alt="llama.cpp" width="28" height="28"></picture></a><br>
Run LLMs locally without Python. llama.cpp is a C/C++ inference engine that runs quantized models in the GGUF format.
</td>
</tr>
</table>

Transformers standardizes how every model is written, so hundreds of architectures share one structure. Each model is plain PyTorch in one file that reads from top to bottom, kept readable and simple without giving up performance. An improvement made once reaches every architecture, so if you find one, [contribute it](./CONTRIBUTING.md).

Transformers is designed to be read and changed by humans and coding agents alike. To change how a model computes, register your own attention or experts function, or load an optimized kernel from the Hub, and the model uses it without any changes to its code.

## Try a model

Install Transformers with PyTorch. See the [Installation](https://huggingface.co/docs/transformers/installation) guide for other setups.

```bash
uv pip install "transformers[torch]"
```

Create a [`Pipeline`](https://huggingface.co/docs/transformers/main_classes/pipelines) with a chat model from the [Hugging Face Hub](https://huggingface.co/models?library=transformers&sort=trending) and pass it a message.

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="Qwen/Qwen3-0.6B")
messages = [
    {"role": "user", "content": "What's the secret to baking a really good cake? Answer in one sentence."},
]
output = pipeline(messages, tokenizer_encode_kwargs={"enable_thinking": False})
print(output[0]["generated_text"][-1]["content"])
# The secret to baking a really good cake is combining the perfect balance of ingredients, technique, and a warm, loving atmosphere.
```

The [Quickstart](https://huggingface.co/docs/transformers/quicktour) covers loading models, running inference on other tasks, and training.

## Set up your coding agent

Copy the prompt below into your coding agent. It has the agent check your installed version, read the docs for that version, and follow the best and latest practices from Transformers v5.

```text
I'm building with Hugging Face Transformers. Before you write any Transformers code:

1. Check the installed version with `python -c "import transformers; print(transformers.__version__)"`. If Transformers isn't installed, install it with `pip install "transformers[torch]"`.
2. Read the docs index for that version at https://huggingface.co/docs/transformers/v<version>/llms.txt, for example v5.22.2. For a .dev version, use https://huggingface.co/docs/transformers/main/llms.txt. Add .md to any docs page URL to read it as Markdown. Look up each class and argument there instead of relying on memory.
3. If the version is 5.0 or later, read https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md Don't use APIs from older versions.
4. Read the model card of each model you use on the Hugging Face Hub. Many models need specific arguments, like a chat template option or a minimum Transformers version.
```

## Run inference

Load a pretrained model and run it on text, images, audio, video, or any combination of modalities.

- [Pipeline](https://huggingface.co/docs/transformers/pipeline_tutorial): Preprocess the input, run the model, and postprocess the output for a task with one class.
- [Text generation](https://huggingface.co/docs/transformers/llm_tutorial): Generate text with LLMs and VLMs, and control the decoding strategy and streaming.
- [Chat with models](https://huggingface.co/docs/transformers/conversations): Start a chat in your terminal with `transformers chat`, or pass a chat history to a pipeline.

## Serve models

Serve a model to other applications over HTTP. `transformers serve` is for evaluation, experimentation, and moderate load. For large production deployments, use an inference engine with a Transformers model as the backend.

- [Serve CLI](https://huggingface.co/docs/transformers/serve-cli/serving): Start a local server compatible with the OpenAI SDK with `transformers serve`.
- [Continuous batching](https://huggingface.co/docs/transformers/continuous_batching): Raise throughput by adding new requests to the batch as other requests finish.
- [Inference engines](https://huggingface.co/docs/transformers/community_integrations/transformers_as_backend): Use a Transformers model as the backend for vLLM or SGLang.

## Train models

Fine-tune a pretrained model on your data, on one GPU or many.

- [Trainer](https://huggingface.co/docs/transformers/training): Run the training loop with support for mixed precision and `torch.compile`.
- [PEFT](https://huggingface.co/docs/transformers/peft): Train a small adapter instead of all of a model's weights.
- [FSDP](https://huggingface.co/docs/transformers/fsdp): Shard a model's parameters across GPUs to train models that don't fit on one.
- [Tensor parallelism](https://huggingface.co/docs/transformers/tensor_parallelism): Split each weight matrix across GPUs so every GPU computes part of each layer.
- [Expert parallelism](https://huggingface.co/docs/transformers/expert_parallelism): Place the experts of a mixture-of-experts model on different GPUs and route tokens to them.

## Use less memory and run faster

Fit a model on your hardware and speed it up.

- [Quantization](https://huggingface.co/docs/transformers/quantization/overview): Store weights in fewer bits so a model needs less memory.
- [Kernels](https://huggingface.co/docs/transformers/kernel_doc/overview): Load precompiled operations for your hardware from the Hub.
- [Attention backends](https://huggingface.co/docs/transformers/attention_interface): Switch between attention implementations without changing the model.

## Add or change a model

Add a model to Transformers or edit an existing one.

- [Modular Transformers](https://huggingface.co/docs/transformers/modular_transformers): Add a model by inheriting from existing models. A converter generates the standalone file.
- [Model structure rules](https://huggingface.co/docs/transformers/modeling_rules): Follow the conventions a checker enforces on every modeling, modular, and configuration file.
- [Writing model tests](https://huggingface.co/docs/transformers/testing): Write a small amount of model-specific test code, and the test mixins generate more than 100 tests from it.

## Citation

We now have a [paper](https://aclanthology.org/2020.emnlp-demos.6/) you can cite for the 🤗 Transformers library:
```bibtex
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-demos.6/",
    pages = "38--45"
}
```
