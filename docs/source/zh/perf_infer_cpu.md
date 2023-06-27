<!--版权所有2022年The HuggingFace团队。保留所有权利。

根据 Apache 许可证第 2.0 版（“许可证”），除非符合许可证的规定，否则您不得使用此文件。您可以在以下位置获取许可证的副本：
http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何明示或暗示的保证或条件。有关许可证的详细信息.
⚠️ 请注意，此文件是 Markdown 格式的，但包含特定于我们的文档生成器（类似于 MDX）的语法，可能在您的 Markdown 查看器中无法正确渲染。

-->

# 在 CPU 上进行高效推断

本指南侧重于在 CPU 上高效推断大型模型。

## 使用 `BetterTransformer` 进行更快的推断

我们最近集成了 `BetterTransformer`，可在 CPU 上更快地进行文本、图像和音频模型的推断。有关此集成的更多详细信息，请查看 [此处](https://huggingface.co/docs/optimum/bettertransformer/overview) 的文档。

## PyTorch JIT 模式（TorchScript）

TorchScript 是一种从 PyTorch 代码创建可序列化和可优化模型的方法。任何 TorchScript 程序都可以从 Python 进程中保存并加载到没有 Python 依赖项的进程中。

与默认的即时模式（eager mode）相比，PyTorch 中的 jit 模式通常通过操作融合等优化方法为模型推断提供更好的性能。

有关 TorchScript 的简明介绍，请参阅 [PyTorch TorchScript 教程](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html#tracing-modules)。

### IPEX JIT 模式下的图优化

Intel ® Extension for PyTorch 为 Transformers 系列模型的 jit 模式提供了进一步的优化。强烈建议用户在 jit 模式下充分利用 Intel ® Extension for PyTorch。

Transformers 模型中的一些常用操作模式已在 Intel ® Extension for PyTorch 的 jit 模式融合中得到支持。这些融合模式包括 Multi-head-attention 融合、Concat Linear、Linear+Add、Linear+Gelu、Add+LayerNorm 融合等，并且效果良好。

融合的好处以透明的方式传递给用户。根据分析，对于问答、文本分类和标记分类等最常见的 NLP 任务，这些融合模式在 Float32 精度和 BFloat16 混合精度下都能带来性能提升。

详细信息请参阅 [IPEX 图优化](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/features/graph_optimization.html)。

#### IPEX 安装：

IPEX 的发布遵循 PyTorch 的发布。请查看 [IPEX 安装方法](https://intel.github.io/intel-extension-for-pytorch/)。

### JIT 模式的用法
要在 Trainer 中启用 JIT 模式进行评估或预测，用户应在 Trainer 命令参数中添加 `jit_mode_eval`。

<Tip warning={true}>

对于 PyTorch >= 1.14.0，JIT 模式可以使任何模型在预测和评估时受益，因为 jit.trace 支持字典输入。

对于 PyTorch < 1.14.0，JIT 模式可以使那些 forward 参数顺序与 jit.trace 中的元组输入顺序匹配的模型受益，比如问答模型。
但对于 forward 参数顺序与 jit.trace 中的元组输入顺序不匹配的情况，比如文本分类模型，jit.trace 将失败，我们在此处捕获该异常以进行回退。使用日志记录来通知用户。

</Tip>

以 [Transformers 问答](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) 为例，以下是用例：


- 在 CPU 上使用 jit 模式进行推断：
<pre> python run_qa.py \--model_name_or_path csarron/bert-base-uncased-squad-v1 \--dataset_name squad \--do_eval \--max_seq_length 384 \--doc_stride 128 \--output_dir /tmp/ \--no_cuda \<b>--jit_mode_eval </b> </pre> 
- 在 CPU 上使用 jit 模式和 IPEX 进行推断：
<pre> python run_qa.py \--model_name_or_path csarron/bert-base-uncased-squad-v1 \--dataset_name squad \--do_eval \--max_seq_length 384 \--doc_stride 128 \--output_dir /tmp/ \--no_cuda \<b>--use_ipex \</b> <b>--jit_mode_eval </b> </pre> 