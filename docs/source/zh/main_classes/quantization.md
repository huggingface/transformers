<!--版权所有 2023 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）许可；除非符合许可证的规定，否则您不得使用此文件。您可以在以下位置获得许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件以“按原样”基础分发，不附带任何明示或暗示的担保或条件。请参阅许可证以获得特定语言下的权限和限制。
⚠️请注意，此文件是 Markdown 格式的，但包含了我们的文档构建器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确呈现。
-->

# 量化🤗 Transformers 模型

## `bitsandbytes` 集成

🤗 Transformers 与 `bitsandbytes` 上使用最频繁的模块紧密集成。您只需使用几行代码即可将模型加载到 8 位精度中。自 `bitsandbytes` 的 `0.37.0` 版本发布以来，大多数 GPU 硬件都支持此功能。

了解有关 [LLM.int8()](https://arxiv.org/abs/2208.07339) 论文中的量化方法的更多信息，或了解有关合作的 [博客文章](https://huggingface.co/blog/hf-bitsandbytes-integration) 的更多信息。

自 `0.39.0` 版本发布以来，您可以使用 4 位量化加载任何支持 `device_map` 的模型，利用 FP4 数据类型。

以下是使用 `bitsandbytes` 集成可以完成的操作

### FP4 量化

#### 要求

在运行以下任何代码片段之前，请确保已安装以下要求。

- 最新的 `bitsandbytes` 库 `pip install bitsandbytes>=0.39.0`
- 从源代码安装最新的 `accelerate` 库 `pip install git+https://github.com/huggingface/accelerate.git`
- 从源代码安装最新的 `transformers` 库 `pip install git+https://github.com/huggingface/transformers.git`

#### 使用 4 位加载大型模型

在调用 `.from_pretrained` 方法时使用 `load_in_4bit=True`，可以将内存使用减少 4 倍（大约）。
```python
# pip install transformers accelerate bitsandbytes
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "bigscience/bloom-1b7"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bit=True)
```

<Tip warning={true}>

请注意，一旦一个模型以 4 位加载，目前无法将量化的权重推送到 Hub。还请注意，目前不支持训练 4 位权重。但是，您可以使用 4 位模型来训练额外的参数，这将在下一节中介绍。
</Tip>

### 使用 8 位加载大型模型

您可以通过在调用 `.from_pretrained` 方法时使用 `load_in_8bit=True` 参数，将模型的内存需求大致减半。

```python
# pip install transformers accelerate bitsandbytes
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "bigscience/bloom-1b7"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_8bit=True)
```

然后，像通常使用 [`PreTrainedModel`] 一样使用您的模型。

您可以使用 `get_memory_footprint` 方法检查模型的内存占用。
```python
print(model.get_memory_footprint())
```

通过此集成，我们能够在较小的设备上加载大型模型并且运行正常。
<Tip warning={true}>

请注意，一旦一个模型以 8 位加载，目前无法将量化的权重推送到 Hub，除非您使用最新的 `transformers` 和 `bitsandbytes`。还请注意，目前不支持训练 8 位权重。但是，您可以使用 8 位模型来训练额外的参数，这将在下一节中介绍。还请注意，`device_map` 是可选的，但对于推理，建议将 `device_map = 'auto'` 设置为自动将模型高效地分派到可用资源上。
</Tip>

#### 高级用例

在这里，我们将介绍一些使用 FP4 量化可以执行的高级用例

##### 更改计算数据类型

计算数据类型用于更改计算过程中使用的数据类型。例如，隐藏状态可以是 `float32`，但计算可以设置为 bf16 以提高速度。默认情况下，计算数据类型设置为 `float32`。
```python
import torch
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
```

##### 使用 NF4（正常浮点 4 位）数据类型

您还可以使用 NF4 数据类型，这是一种针对使用正态分布初始化的权重的新的 4 位数据类型。运行以下命令：
```python
from transformers import BitsAndBytesConfig

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
)

model_nf4 = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)
```

##### 使用嵌套量化进行更高效的内存推理

我们还建议用户使用嵌套量化技术。这可以在不影响性能的情况下节省更多的内存-根据我们的经验观察，在 NVIDIA-T4 16GB 上，这可以使 llama-13b 模型在序列长度为 1024、批次大小为 1 和梯度累积步骤为 4 的情况下进行微调。

```python
from transformers import BitsAndBytesConfig

double_quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
)

model_double_quant = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=double_quant_config)
```


### 将量化模型推送到🤗 Hub

您可以通过简单地使用 `push_to_hub` 方法将量化模型推送到 Hub。这将首先推送量化配置文件，然后推送量化模型权重。请确保使用的是 `bitsandbytes>0.37.2`（在撰写本文时，我们在 `bitsandbytes==0.38.0.post1` 上进行了测试），以便使用此功能。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m", device_map="auto", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")

model.push_to_hub("bloom-560m-8bit")
```

<Tip warning={true}>

强烈建议将 8 位模型推送到 Hub 以便于大型模型。这将使社区受益于内存占用的减少，例如在 Google Colab 上加载大型模型。
</Tip>

### 从🤗 Hub 加载量化模型

您可以使用 `from_pretrained` 方法从 Hub 加载量化模型。通过检查模型配置对象中是否存在 `quantization_config` 属性，确保推送的权重已经量化。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{your_username}/bloom-560m-8bit", device_map="auto")
```

请注意，在这种情况下，您无需指定 `load_in_8bit=True` 参数，但您需要确保已安装 `bitsandbytes` 和 `accelerate`。还请注意，`device_map` 是可选的，但对于推理，建议将 `device_map = 'auto'` 设置为自动将模型高效地分派到可用资源上。

### 高级用例

本节面向高级用户，他们希望探索除加载和运行 8 位模型之外的其他可能性。

#### 在 `cpu` 和 `gpu` 之间进行卸载

其中一个高级用例是能够在 `CPU` 和 `GPU` 之间加载模型并分派权重。请注意，将在 CPU 上分派的权重 **不会** 转换为 8 位，而是保持为 `float32`。此功能适用于想要适应非常大的模型并在 GPU 和 CPU 之间分派模型的用户。

首先，从 `transformers` 中加载 `BitsAndBytesConfig` 并将属性 `llm_int8_enable_fp32_cpu_offload` 设置为 `True`：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
```

假设您要加载 `bigscience/bloom-1b7` 模型，并且您的 GPU RAM 刚好足够容纳整个模型，但不包括 `lm_head`。因此，编写自定义的 `device_map`，如下所示：
```python
device_map = {
    "transformer.word_embeddings": 0,
    "transformer.word_embeddings_layernorm": 0,
    "lm_head": "cpu",
    "transformer.h": 0,
    "transformer.ln_f": 0,
}
```

然后，按如下方式加载您的模型：

```python
model_8bit = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-1b7",
    device_map=device_map,
    quantization_config=quantization_config,
)
```

就是这样！享受您的模型吧！

#### 调整 `llm_int8_threshold`

您可以调整 `llm_int8_threshold` 参数以更改离群值的阈值。"离群值" 是大于某个特定阈值的隐藏状态值。

这对应于在 `LLM.int8()` 论文中描述的异常值检测的异常值阈值。任何超过此阈值的隐藏状态值都将被视为异常值，并且对这些值的操作将使用 fp16 进行。这些值通常服从正态分布，即大多数值在 [-3.5, 3.5] 范围内，但对于大型模型，有一些异常系统性异常值的分布方式非常不同。这些异常值通常在区间 [-60, -6] 或 [6, 60] 内。对于大小约为 5 的值，Int8 量化效果很好，但超过该范围会导致显著的性能损失。一个很好的默认阈值是 6，但对于不稳定的模型（小模型、微调），可能需要更低的阈值。

此参数可以影响模型的推断速度。建议调整此参数以找到最适合您的用例的值。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "bigscience/bloom-1b7"

quantization_config = BitsAndBytesConfig(
    llm_int8_threshold=10,
)

model_8bit = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device_map,
    quantization_config=quantization_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

#### 跳过某些模块的转换

某些模型具有多个模块，需要在 8 位中不进行转换以确保稳定性。例如，Jukebox 模型具有多个应跳过的 `lm_head` 模块。请使用 `llm_int8_skip_modules` 进行调整。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "bigscience/bloom-1b7"

quantization_config = BitsAndBytesConfig(
    llm_int8_skip_modules=["lm_head"],
)

model_8bit = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device_map,
    quantization_config=quantization_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

#### 对已加载的 8 位模型进行微调

在 Hugging Face 生态系统中正式支持适配器后，您可以对已加载的 8 位模型进行微调。这使得可以在单个 Google Colab 中微调大型模型，如 `flan-t5-large` 或 `facebook/opt-6.7b`。请参阅 [`peft`](https://github.com/huggingface/peft) 库了解更多详细信息。
请注意，在加载模型进行训练时，无需传递 `device_map`。它将自动将您的模型加载到 GPU 上。如果需要，您还可以将设备映射设置为特定设备（例如 `cuda:0`，`0`，`torch.device('cuda:0')`）。

请注意，`device_map=auto` 仅适用于推断。

### BitsAndBytesConfig
[[autodoc]] BitsAndBytesConfig

## 使用🤗 `optimum` 进行量化

请查看 [Optimum 文档](https://huggingface.co/docs/optimum/index) 以了解更多关于 `optimum` 支持的量化方法，并查看这些方法是否适用于您的用例。
