<!--版权所有2021年NVIDIA Corporation和HuggingFace团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证的规定，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何形式的担保或条件。请参阅许可证以获取特定语言下的权限和限制。⚠️ 请注意，此文件为 Markdown 格式，但包含我们的文档生成器（类似于 MDX）的特定语法，可能在 Markdown 查看器中无法正确渲染。
-->
# MegatronGPT2

## 概述

MegatronGPT2 模型是由 Mohammad Shoeybi，Mostofa Patwary，Raul Puri，Patrick LeGresley，Jared Casper 和 Bryan Catanzaro 在《Megatron-LM：使用模型并行性训练数十亿参数的语言模型》[Megatron-LM: Training Multi-Billion Parameter Language Models Using Model
Parallelism](https://arxiv.org/abs/1909.08053) 中提出的。
论文中的摘要如下：

*最近在语言建模方面的工作表明，训练大型 Transformer 模型可以推动自然语言处理应用的最新进展。然而，非常大的模型由于内存限制可能非常难以训练。在这项工作中，我们提出了训练非常大的 Transformer 模型的技术，并实现了一种简单高效的层内模型并行方法，使得可以训练具有数十亿参数的 Transformer 模型。我们的方法不需要新的编译器或库更改，是与管道模型并行性正交且互补的，可以通过在本机 PyTorch 中插入几个通信操作来完全实现。我们使用 512 个 GPU 将基于 Transformer 的模型收敛到 83 亿参数。与支持 39 TeraFLOPs 的强大单个 GPU 基准相比，我们在整个应用程序上保持了 15.1 PetaFLOPs 的性能扩展效率为 76%。为了证明大型语言模型可以进一步推动最新技术的发展，我们训练了一个类似于 GPT-2 的 83 亿参数的 Transformer 语言模型和一个类似于 BERT 的 39 亿参数的模型。我们表明，对于 BERT 样式的模型，对于层归一化的放置需要特别注意，以实现随着模型规模的增长而提高的性能。使用 GPT-2 模型，我们在 WikiText103（10.8，相对于最新技术的 15.8 困惑度）和 LAMBADA（66.5%，相对于最新技术的 63.2%准确率）数据集上取得了最新技术的结果。我们的 BERT 模型在 RACE 数据集上取得了最新技术的结果（90.9%，相对于最新技术的 89.4%准确率）.* 

提示：

我们提供了预训练的 [GPT2-345M](https://ngc.nvidia.com/catalog/models/nvidia:megatron_lm_345m) 检查点，用于评估或微调后续任务。

要访问这些检查点，首先 [注册](https://ngc.nvidia.com/signup) 并设置 NVIDIA GPU 云（NGC）注册表 CLI。有关下载模型的更多文档可以在 [NGC 文档](https://docs.nvidia.com/dgx/ngc-registry-cli-user-guide/index.html#topic_6_4_1) 中找到。Registry CLI. Further documentation for downloading models can be found in the [NGC documentation](https://docs.nvidia.com/dgx/ngc-registry-cli-user-guide/index.html#topic_6_4_1).

或者，您可以直接使用以下方式下载检查点：

```bash
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_lm_345m/versions/v0.0/zip -O
megatron_gpt2_345m_v0_0.zip
```

一旦您从 NVIDIA GPU 云（NGC）获取了检查点，您必须将其转换为 Hugging Face Transformers GPT2 实现轻松加载的格式。以下命令可用于执行此转换。我们假设文件夹 `models/megatron_gpt2` 包含 `megatron_gpt2_345m_v0_0.zip`，并且命令是从该文件夹运行的：



```bash
python3 $PATH_TO_TRANSFORMERS/models/megatron_gpt2/convert_megatron_gpt2_checkpoint.py megatron_gpt2_345m_v0_0.zip
```

此模型由 [jdemouth](https://huggingface.co/jdemouth) 贡献。原始代码可在 [此处](https://github.com/NVIDIA/Megatron-LM) 找到。该存储库包含了 Megatron 语言模型的多 GPU 和多节点实现。特别是，它包含使用“张量并行”和“流水线并行”技术的混合模型并行方法。

