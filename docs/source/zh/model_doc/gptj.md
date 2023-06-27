<!--版权所有 2021 年 HuggingFace 团队。保留所有权利。
根据 Apache License，Version 2.0（“许可证”）授权；除非符合许可证的规定，否则禁止使用本文件。您可以在
http://www.apache.org/licenses/LICENSE-2.0
获取许可证的副本。除非适用法律要求或书面同意，根据许可证分发的软件是基于 "AS IS" 的基础上提供的，无论是明示或暗示的保证或条件。请参阅许可证以了解
特定语言下的权限和限制。正确地渲染。
-->
# GPT-J

## 概述

GPT-J 模型是由 Ben Wang 和 Aran Komatsuzaki 在 [kingoflolz/mesh-transformer-jax](https://github.com/kingoflolz/mesh-transformer-jax) 代码库中发布的。它是一个类似于 GPT-2 的因果语言模型，使用了 [Pile](https://pile.eleuther.ai/) 数据集进行训练。

此模型由 [Stella Biderman](https://huggingface.co/stellaathena) 贡献。

提示：

- 要加载 [float32 格式的 GPT-J](https://huggingface.co/EleutherAI/gpt-j-6B)，至少需要 2 倍的模型大小的  RAM：1 倍用于初始化权重，另外 1 倍用于加载检查点。因此，对于 GPT-J 模型，至少需要 48GB 的  RAM 才能加载模型。为了减少 RAM 的使用量，有几个选项。`torch_dtype` 参数可以  用于仅在 CUDA 设备上以半精度初始化模型。还有一个 fp16 分支，用于存储 fp16 权重，  可以进一步减少 RAM 的使用量：
```python
>>> from transformers import GPTJForCausalLM
>>> import torch

>>> device = "cuda"
>>> model = GPTJForCausalLM.from_pretrained(
...     "EleutherAI/gpt-j-6B",
...     revision="float16",
...     torch_dtype=torch.float16,
... ).to(device)
```

- 模型在推理时应适合 16GB 的 GPU。对于训练/微调，需要更多的 GPU RAM。例如，Adam  优化器会创建模型的四个副本：模型、梯度、梯度的平均值和平方平均值。  因此，即使使用混合精度，梯度更新也是 fp32 格式，所以至少需要 4 倍模型大小的 GPU 内存。  这还不包括激活和数据批次，这些也需要额外的 GPU RAM。因此，应该探索  一些解决方案，例如使用 DeepSpeed 来训练/微调模型。

另一个选择是使用原始代码库  在 TPU 上训练/微调模型，然后将模型转换为 Transformers 格式进行推理。有关  此过程的说明可以在 [这里找到](https://github.com/kingoflolz/mesh-transformer-jax/blob/master/howto_finetune.md)
- 尽管嵌入矩阵的大小为 50400，但 GPT-2 分词器 (Tokenizer)仅使用 50257 个条目。为了在 TPU 上提高效率，  添加了这些额外的令牌。为了避免嵌入矩阵大小与词汇表大小不匹配，  [GPT-J 分词器 (Tokenizer)](https://huggingface.co/EleutherAI/gpt-j-6B) 包含了额外的 143 个令牌  `<|extratoken_1|>...<|extratoken_143|>`，因此分词器 (Tokenizer)的 `vocab_size` 也变为了 50400。

### 生成

可以使用 [`~generation.GenerationMixin.generate`] 方法使用 GPT-J 模型生成文本。
```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
>>> tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

>>> prompt = (
...     "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
...     "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
...     "researchers was the fact that the unicorns spoke perfect English."
... )

>>> input_ids = tokenizer(prompt, return_tensors="pt").input_ids

>>> gen_tokens = model.generate(
...     input_ids,
...     do_sample=True,
...     temperature=0.9,
...     max_length=100,
... )
>>> gen_text = tokenizer.batch_decode(gen_tokens)[0]
```

...或者使用 float16 精度：
```python
>>> from transformers import GPTJForCausalLM, AutoTokenizer
>>> import torch

>>> device = "cuda"
>>> model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16).to(device)
>>> tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

>>> prompt = (
...     "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
...     "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
...     "researchers was the fact that the unicorns spoke perfect English."
... )

>>> input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

>>> gen_tokens = model.generate(
...     input_ids,
...     do_sample=True,
...     temperature=0.9,
...     max_length=100,
... )
>>> gen_text = tokenizer.batch_decode(gen_tokens)[0]
```

## 资源

以下是官方 Hugging Face 和社区（由🌎表示）资源的列表，可帮助您开始使用 GPT-J。如果您有兴趣
提交资源以包含在此处，请随时提出 Pull Request，我们将对其进行审核！资源应该
展示一些新的东西，而不是重复现有资源。- [GPT-J 的描述](https://huggingface.co/EleutherAI/gpt-j-6B)- 有关如何 [使用 Hugging Face Transformers 和 Amazon SageMaker 部署 GPT-J 6B 进行推理](https://huggingface.co/blog/gptj-sagemaker) 的博客文章
- 介绍 [GPT-J-6B: 6B JAX-Based Transformer](https://arankomatsuzaki.wordpress.com/2021/06/04/gpt-j/) 的博客文章 🌎
- [GPT-J-6B 推理演示的 Notebook](https://colab.research.google.com/github/kingoflolz/mesh-transformer-jax/blob/master/colab_demo.ipynb) 🌎
- 另一个演示 [GPT-J-6B 推理的 Notebook](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/GPT-J-6B/Inference_with_GPT_J_6B.ipynb)
- 🤗 Hugging Face 课程中的 [因果语言建模](https://huggingface.co/course/en/chapter7/6?fw=pt#training-a-causal-language-model-from-scratch) 章节
- [`GPTJForCausalLM`] 支持使用此 [因果语言建模示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#gpt-2gpt-and-causal-language-modeling)、[文本生成示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-generation) 和 [Notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb)
- [`TFGPTJForCausalLM`] 支持使用此 [因果语言建模示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_clmpy) 和 [Notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb)
- [`FlaxGPTJForCausalLM`] 支持使用此 [因果语言建模示例脚本](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#causal-language-modeling) 和 [Notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/causal_language_modeling_flax.ipynb)

**文档资源**
- [文本分类任务指南](../tasks/sequence_classification)
- [问答任务指南](../tasks/question_answering)
- [因果语言建模任务指南](../tasks/language_modeling)

## GPTJConfig

[[autodoc]] GPTJConfig
    - all

## GPTJModel

[[autodoc]] GPTJModel
    - forward

## GPTJForCausalLM

[[autodoc]] GPTJForCausalLM
    - forward

## GPTJForSequenceClassification

[[autodoc]] GPTJForSequenceClassification
    - forward

## GPTJForQuestionAnswering

[[autodoc]] GPTJForQuestionAnswering
    - forward

## TFGPTJModel

[[autodoc]] TFGPTJModel
    - call

## TFGPTJForCausalLM

[[autodoc]] TFGPTJForCausalLM
    - call

## TFGPTJForSequenceClassification

[[autodoc]] TFGPTJForSequenceClassification
    - call

## TFGPTJForQuestionAnswering

[[autodoc]] TFGPTJForQuestionAnswering
    - call

## FlaxGPTJModel

[[autodoc]] FlaxGPTJModel
    - __call__

## FlaxGPTJForCausalLM

[[autodoc]] FlaxGPTJForCausalLM
    - __call__
