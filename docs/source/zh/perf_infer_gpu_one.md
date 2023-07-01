<!--版权所有 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache License，Version 2.0（即“许可证”）获得许可；除非符合许可证的规定，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按 "按原样" 基础分发的，不附带任何明示或暗示的保证或条件。有关许可证的详细信息，请参见许可证。
⚠️ 请注意，此文件是 Markdown 格式，但包含特定于我们的 doc-builder（类似于 MDX）的语法，可能无法在您的 Markdown 查看器中正确呈现。渲染。
-->


# 在单个 GPU 上进行高效推理

除了本指南外，还可以在 [在单个 GPU 上进行训练的指南](perf_train_gpu_one) 和 [在 CPU 上进行推理的指南](perf_infer_cpu) 中找到相关信息。
## 更好的 Transformer：基于 PyTorch 的 Transformer 快速路径

基于 PyTorch 的 [`nn.MultiHeadAttention`](https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference/) 注意力快速路径，称为 BetterTransformer，可以通过 [🤗 Optimum 库](https://huggingface.co/docs/optimum/bettertransformer/overview) 中的集成与 Transformers 一起使用。

PyTorch 的注意力快速路径通过内核融合和使用 [嵌套张量](https://pytorch.org/docs/stable/nested.html) 来加速推理。详细的基准测试可以在 [此博客文章](https://medium.com/pytorch/bettertransformer-out-of-the-box-performance-for-huggingface-transformers-3fbe27d50ab2) 中找到。

在安装了 [`optimum`](https://github.com/huggingface/optimum) 包之后，在推理过程中使用 Better Transformer，可以通过调用 [`~PreTrainedModel.to_bettertransformer`] 来替换相关的内部模块：
```python
model = model.to_bettertransformer()
```

方法 [`~PreTrainedModel.reverse_bettertransformer`] 允许返回原始建模，在保存模型之前应使用该方法，以使用规范的 transformers 建模：
```python
model = model.reverse_bettertransformer()
model.save_pretrained("saved_model")
```

从 PyTorch 2.0 开始，注意力快速路径支持编码器和解码器。支持的架构列表可以在 [这里](https://huggingface.co/docs/optimum/bettertransformer/overview#supported-models) 找到。

## `bitsandbytes` 集成用于 FP4 混合精度推理
您可以安装 `bitsandbytes` 并从中受益，以便在 GPU 上轻松压缩模型。使用 FP4 量化，与其本机完全精度版本相比，您可以将模型大小减小高达 8 倍。请查看以下如何入门。
<Tip>

请注意，此功能也可以在多 GPU 设置中使用。
</Tip>

### 要求

- 最新的 `bitsandbytes` 库 `pip install bitsandbytes>=0.39.0`
- 从源代码安装最新的 `accelerate` `pip install git+https://github.com/huggingface/accelerate.git`
- 从源代码安装最新的 `transformers` `pip install git+https://github.com/huggingface/transformers.git`

### 运行 FP4 模型 - 单个 GPU 设置 - 快速入门

您可以通过运行以下代码快速在单个 GPU 上运行 FP4 模型：
```py
from transformers import AutoModelForCausalLM

model_name = "bigscience/bloom-2b5"
model_8bit = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)
```

请注意，`device_map` 是可选的，但在推理中设置 `device_map = 'auto'` 是首选，因为它将高效地将模型分配到可用的资源上。

### 运行 FP4 模型 - 多 GPU 设置

将混合 8 位模型加载到多个 GPU 中的方法如下（与单个 GPU 设置相同）：

```py
model_name = "bigscience/bloom-2b5"
model_8bit = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)
```
但是，您可以使用 `accelerate` 来控制要在每个 GPU 上分配的 GPU RAM。使用 `max_memory` 参数如下所示：
```py
max_memory_mapping = {0: "600MB", 1: "1GB"}
model_name = "bigscience/bloom-3b"
model_8bit = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", load_in_4bit=True, max_memory=max_memory_mapping
)
```

在此示例中，第一个 GPU 将使用 600MB 的内存，第二个 GPU 将使用 1GB 的内存。

### 高级用法

有关此方法的更高级用法，请参阅 [量化](main_classes/quantization) 文档页面。

## `bitsandbytes` 集成用于 Int8 混合精度矩阵分解

<Tip>
请注意，此功能也可以在多 GPU 设置中使用。
从论文 [`LLM.int8()：大规模Transformer的8位矩阵乘法`](https://arxiv.org/abs/2208.07339) 开始，我们支持 Hub 中所有模型的 Hugging Face 集成，只需几行代码。

该方法通过将 `nn.Linear` 大小减小 2（对于 `float16` 和 `bfloat16` 权重）和 4（对于 `float32` 权重）来操作半精度中的离群点，几乎不会对质量产生影响。
![HFxbitsandbytes.png](https://s3.amazonaws.com/moonup/production/uploads/1659861207959-62441d1d9fdefb55a0b7d12c.png)
Int8 混合精度矩阵分解通过将矩阵乘法分为两个流（1）在 fp16 中进行矩阵乘法的系统特征离群点流（0.01%），（2）进行 int8 矩阵乘法的常规流（99.9%）来工作。使用此方法，对于非常大的模型，可以进行 int8 推理而几乎不会产生预测性能下降。有关该方法的更多详细信息，请查看 [论文](https://arxiv.org/abs/2208.07339) 或我们的 [关于集成的博客文章](https://huggingface.co/blog/hf-bitsandbytes-integration)。
![MixedInt8.gif](https://s3.amazonaws.com/moonup/production/uploads/1660567469965-62441d1d9fdefb55a0b7d12c.gif)

请注意，要运行混合 8 位模型，您需要一个 GPU，因为内核仅针对 GPU 进行了编译。在使用此功能之前，请确保您有足够的 GPU 内存来存储模型的四分之一（或半精度的权重）。以下是一些帮助您使用此模块的注意事项，或者可以在 [Google Colab 的演示](#colab-demos) 中查看演示。

### 要求

- 如果您使用的是 `bitsandbytes<0.37.0`，请确保在支持 8 位张量核心（图灵，安培或更新的架构，例如 T4，RTX20s RTX30s，A40-A100）的 NVIDIA GPU 上运行。对于 `bitsandbytes>=0.37.0`，应支持所有 GPU。- 安装正确版本的 `bitsandbytes`，请运行：`pip install bitsandbytes>=0.31.5`- 安装 `accelerate` `pip install accelerate>=0.12.0`

### 运行混合 Int8 模型 - 单个 GPU 设置

在安装所需的库之后，加载混合 8 位模型的方法如下：

```py
from transformers import AutoModelForCausalLM

model_name = "bigscience/bloom-2b5"
model_8bit = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
```

对于文本生成，我们建议：
- 使用模型的 `generate()` 方法而不是 `pipeline()` 函数。尽管可以使用 `pipeline()` 函数进行推理，但它对于混合 8 位模型来说并不是最优化的，并且比使用 `generate()` 方法慢。此外，某些采样策略（如核心采样）不支持混合 8 位模型的 `pipeline()` 函数。- 将所有输入放置在与模型相同的设备上。

以下是一个简单的示例：

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "bigscience/bloom-2b5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_8bit = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)

prompt = "Hello, my llama is cute"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
generated_ids = model.generate(**inputs)
outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
```


### 运行混合 int8 模型 - 多 GPU 设置

将混合 8 位模型加载到多个 GPU 中的方法如下（与单个 GPU 设置相同）：
```py
model_name = "bigscience/bloom-2b5"
model_8bit = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
```

但是，您可以使用 `accelerate` 来控制要在每个 GPU 上分配的 GPU RAM。使用 `max_memory` 参数如下所示：
```py
max_memory_mapping = {0: "1GB", 1: "2GB"}
model_name = "bigscience/bloom-3b"
model_8bit = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", load_in_8bit=True, max_memory=max_memory_mapping
)
```

在此示例中，第一个 GPU 将使用 1GB 的内存，第二个 GPU 将使用 2GB 的内存。

### Colab 演示

使用此方法您可以对以前无法在 Google Colab 上进行推理的模型进行推理。查看在 Google Colab 上运行 T5-11b（42GB in fp32）的演示！使用 8 位量化：
[![在 Colab 中打开：T5-11b 演示](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YORPWx4okIHXnjW7MSAidXN29mPVNT7F?usp=sharing)
或者使用 BLOOM-3B 进行演示：

[![在 Colab 中打开：BLOOM-3b 演示](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qOjXfQIAULfKvZqwCen8-MoWKGdSatZ4?usp=sharing)