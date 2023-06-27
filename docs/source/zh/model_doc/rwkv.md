<!--版权所有 2023 年的 HuggingFace 团队。保留所有权利。
根据 Apache 许可证 2.0 版（“许可证”）许可；除非符合许可证，否则您不得使用此文件。您可以在
http://www.apache.org/licenses/LICENSE-2.0
适用法律要求或书面同意的情况下，根据许可证分发的软件是按照“按原样”基础分发的，没有任何明示或暗示的保证或条件。请参阅许可证以了解特定语言下的权限和限制。
⚠️ 请注意，此文件是 Markdown 格式，但包含了特定于我们 doc-builder 的语法（类似于 MDX），可能无法在您的 Markdown 查看器中正确呈现。
-->
# RWKV

## 概述

RWKV 模型是在 [此 repo](https://github.com/BlinkDL/RWKV-LM) 中提出的
它建议在传统的 Transformer 注意力机制中进行调整，使其变为线性。这样，该模型可以用作循环网络：将时间戳为 0 和时间戳为 1 的输入一起传递与在时间戳为 0 传递的输入，然后是时间戳为 1 的输入以及时间戳为 0 的状态（见下面的示例）。

这比常规 Transformer 更高效，并且可以处理任意长度的句子（即使模型在训练中使用的是固定的上下文长度）。
此模型由 [sgugger](https://huggingface.co/sgugger) 贡献。可以在 [此处](https://github.com/BlinkDL/RWKV-LM) 找到原始代码。

使用示例：
```py
import torch
from transformers import AutoTokenizer, RwkvConfig, RwkvModel

model = RwkvModel.from_pretrained("sgugger/rwkv-430M-pile")
tokenizer = AutoTokenizer.from_pretrained("sgugger/rwkv-430M-pile")

inputs = tokenizer("This is an example.", return_tensors="pt")
# Feed everything to the model
outputs = model(inputs["input_ids"])
output_whole = outputs.last_hidden_state

outputs = model(inputs["input_ids"][:, :2])
output_one = outputs.last_hidden_state

# Using the state computed on the first inputs, we will get the same output
outputs = model(inputs["input_ids"][:, 2:], state=outputs.state)
output_two = outputs.last_hidden_state

torch.allclose(torch.cat([output_one, output_two], dim=1), output_whole, atol=1e-5)
```

## RwkvConfig
[[autodoc]] RwkvConfig

## RwkvModel
[[autodoc]] RwkvModel
    - forward
## RwkvLMHeadModel
[[autodoc]] RwkvForCausalLM
    - forward

## Rwkv 注意力和循环公式

在传统的自回归 Transformer 中，注意力的表示为
$$O = \hbox{softmax}(QK^{T} / \sqrt{d}) V$$
其中\\(Q\\)、\\(K\\)和\\(V\\)是形状为 `seq_len x hidden_size` 的矩阵，分别称为查询（query）、键（key）和值（value）（实际上它们是具有批次维度和注意力头维度的更大矩阵，但我们只关心后两者，这是矩阵乘积进行的地方，为了简单起见，我们只考虑这两个）。矩阵乘积\\(QK^{T}\\)的形状为 `seq_len x seq_len`，我们可以与\\(V\\)进行矩阵乘积，得到与其他矩阵相同形状的输出\\(O\\)。
通过替换 softmax 的值，我们得到了
$$O_{i} = \frac{\sum_{j=1}^{i} e^{Q_{i} K_{j}^{T} / \sqrt{d}} V_{j}}{\sum_{j=1}^{i} e^{Q_{i} K_{j}^{T} / \sqrt{d}}}$$
请注意，对应于\\(j > i\\)的\\(QK^{T}\\)项被屏蔽了（求和在\\(j\\)处停止），因为注意力不能查看未来的令牌（只能看到过去的）。

相比之下，RWKV 注意力由以下表示
$$O_{i} = \sigma(R_{i}) \frac{\sum_{j=1}^{i} e^{W_{i-j} + K_{j}} V_{j}}{\sum_{j=1}^{i} e^{W_{i-j} + K_{j}}}$$

其中\\(R\\)是作者提出的称为 receptance 的新矩阵，\\(K\\)和\\(V\\)仍然是键和值（\\(\sigma\\)在这里是 sigmoid 函数）。\\(W\\)是表示令牌位置的新向量，由以下定义
$$W_{0} = u \hbox{  and  } W_{k} = (k-1)w \hbox{ for } k \geq 1$$
其中\\(u\\)和\\(w\\)是称为代码中的 `time_first` 和 `time_decay` 的可学习参数。分子和分母都可以递归地表示。我们将它们命名为\\(N_{i}\\)和\\(D_{i}\\)
$$N_{i} = e^{u + K_{i}} V_{i} + \hat{N}_{i} \hbox{  where  } \hat{N}_{i} = e^{K_{i-1}} V_{i-1} + e^{w + K_{i-2}} V_{i-2} \cdots + e^{(i-2)w + K_{1}} V_{1}$$
因此\\(\hat{N}_{i}\\)（代码中称为 `numerator_state`）满足
$$\hat{N}_{0} = 0 \hbox{  and  } \hat{N}_{j+1} = e^{K_{j}} V_{j} + e^{w} \hat{N}_{j}$$
以及
$$D_{i} = e^{u + K_{i}} + \hat{D}_{i} \hbox{  where  } \hat{D}_{i} = e^{K_{i-1}} + e^{w + K_{i-2}} \cdots + e^{(i-2)w + K_{1}}$$
因此\\(\hat{D}_{i}\\)（代码中称为 `denominator_state`）满足
$$\hat{D}_{0} = 0 \hbox{  and  } \hat{D}_{j+1} = e^{K_{j}} + e^{w} \hat{D}_{j}$$

实际使用的循环公式要稍微复杂一些，因为为了数值稳定性，我们不想计算指数的大数值。通常，softmax 不是按原样计算的，而是将最大项的指数除以分子和分母:
$$\frac{e^{x_{i}}}{\sum_{j=1}^{n} e^{x_{j}}} = \frac{e^{x_{i} - M}}{\sum_{j=1}^{n} e^{x_{j} - M}}$$

其中\\(M\\)是所有\\(x_{j}\\)的最大值。因此，除了保存分子状态（\\(\hat{N}\\)）和分母状态（\\(\hat{D}\\)）之外，我们还要跟踪所有指数中遇到的最大值。因此，我们实际上使用了
$$\tilde{N}_{i} = e^{-M_{i}} \hat{N}_{i} \hbox{  and  } \tilde{D}_{i} = e^{-M_{i}} \hat{D}_{i}$$

它们由以下循环公式定义:
$$\tilde{N}_{0} = 0 \hbox{  and  } \tilde{N}_{j+1} = e^{K_{j} - q} V_{j} + e^{w + M_{j} - q} \tilde{N}_{j} \hbox{  where  } q = \max(K_{j}, w + M_{j})$$

以及
$$\tilde{D}_{0} = 0 \hbox{  and  } \tilde{D}_{j+1} = e^{K_{j} - q} + e^{w + M_{j} - q} \tilde{D}_{j} \hbox{  where  } q = \max(K_{j}, w + M_{j})$$

以及\\(M_{j+1} = q\\)。通过这些，我们可以计算
$$N_{i} = e^{u + K_{i} - q} V_{i} + e^{M_{i}} \tilde{N}_{i} \hbox{  where  } q = \max(u + K_{i}, M_{i})$$

和
$$D_{i} = e^{u + K_{i} - q} + e^{M_{i}} \tilde{D}_{i} \hbox{  where  } q = \max(u + K_{i}, M_{i})$$

这最终给出了
$$O_{i} = \sigma(R_{i}) \frac{N_{i}}{D_{i}}$$