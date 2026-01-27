<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 调试

## 多GPU网络问题调试

当使用`DistributedDataParallel`和多个GPU进行训练或推理时，如果遇到进程和（或）节点之间的互联问题，您可以使用以下脚本来诊断网络问题。

```bash
wget https://raw.githubusercontent.com/huggingface/transformers/main/scripts/distributed/torch-distributed-gpu-test.py
```

例如，要测试两个GPU之间的互联，请执行以下操作：

```bash
python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
```

如果两个进程能够相互通信并分配GPU内存，它们各自将打印出 "OK" 状态。

对于更多的GPU或节点，可以根据脚本中的参数进行调整。

在诊断脚本内部，您将找到更多详细信息，甚至有关如何在SLURM环境中运行它的说明。

另一种级别的调试是添加 `NCCL_DEBUG=INFO` 环境变量，如下所示：


```bash
NCCL_DEBUG=INFO python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
```

这将产生大量与NCCL相关的调试信息，如果发现有问题报告，您可以在线搜索以获取相关信息。或者，如果您不确定如何解释输出，可以在`issue`中分享日志文件。


## 下溢和上溢检测

<Tip>

目前，此功能仅适用于PyTorch。

</Tip>

<Tip>

对于多GPU训练，它需要使用DDP（`torch.distributed.launch`）。

</Tip>

<Tip>

此功能可以与任何基于`nn.Module`的模型一起使用。

</Tip>

如果您开始发现`loss=NaN`或模型因激活值或权重中的`inf`或`nan`而出现一些异常行为，就需要发现第一个下溢或上溢发生的地方以及导致它的原因。幸运的是，您可以通过激活一个特殊模块来自动进行检测。

如果您正在使用[`Trainer`]，只需把以下内容：


```bash
--debug underflow_overflow
```

添加到常规命令行参数中，或在创建[`TrainingArguments`]对象时传递 `debug="underflow_overflow"`。

如果您正在使用自己的训练循环或其他Trainer，您可以通过以下方式实现相同的功能：

```python
from transformers.debug_utils import DebugUnderflowOverflow

debug_overflow = DebugUnderflowOverflow(model)
```

[`debug_utils.DebugUnderflowOverflow`] 将`hooks`插入模型，紧跟在每次前向调用之后，进而测试输入和输出变量，以及相应模块的权重。一旦在激活值或权重的至少一个元素中检测到`inf`或`nan`，程序将执行`assert`并打印报告，就像这样（这是在`google/mt5-small`下使用fp16混合精度捕获的）：

```
Detected inf/nan during batch_number=0
Last 21 forward frames:
abs min  abs max  metadata
                  encoder.block.1.layer.1.DenseReluDense.dropout Dropout
0.00e+00 2.57e+02 input[0]
0.00e+00 2.85e+02 output
[...]
                  encoder.block.2.layer.0 T5LayerSelfAttention
6.78e-04 3.15e+03 input[0]
2.65e-04 3.42e+03 output[0]
             None output[1]
2.25e-01 1.00e+04 output[2]
                  encoder.block.2.layer.1.layer_norm T5LayerNorm
8.69e-02 4.18e-01 weight
2.65e-04 3.42e+03 input[0]
1.79e-06 4.65e+00 output
                  encoder.block.2.layer.1.DenseReluDense.wi_0 Linear
2.17e-07 4.50e+00 weight
1.79e-06 4.65e+00 input[0]
2.68e-06 3.70e+01 output
                  encoder.block.2.layer.1.DenseReluDense.wi_1 Linear
8.08e-07 2.66e+01 weight
1.79e-06 4.65e+00 input[0]
1.27e-04 2.37e+02 output
                  encoder.block.2.layer.1.DenseReluDense.dropout Dropout
0.00e+00 8.76e+03 input[0]
0.00e+00 9.74e+03 output
                  encoder.block.2.layer.1.DenseReluDense.wo Linear
1.01e-06 6.44e+00 weight
0.00e+00 9.74e+03 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.DenseReluDense T5DenseGatedGeluDense
1.79e-06 4.65e+00 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.dropout Dropout
3.18e-04 6.27e+04 input[0]
0.00e+00      inf output
```

由于篇幅原因，示例输出中间的部分已经被缩减。

第二列显示了绝对最大元素的值，因此，如果您仔细查看最后`frame`，输入和输出都在`1e4`的范围内。因此，在使用fp16混合精度进行训练时，最后一步发生了溢出（因为在`fp16`下，在`inf`之前的最大数字是`64e3`）。为了避免在`fp16`下发生溢出，激活值必须保持低于`1e4`，因为`1e4 * 1e4 = 1e8`，因此任何具有大激活值的矩阵乘法都会导致数值溢出。

在跟踪的开始处，您可以发现问题发生在哪个批次（这里的`Detected inf/nan during batch_number=0`表示问题发生在第一个批次）。

每个报告的`frame`都以声明相应模块的层信息为开头，说明这一`frame`是为哪个模块报告的。如果只看这个`frame`：

```
                  encoder.block.2.layer.1.layer_norm T5LayerNorm
8.69e-02 4.18e-01 weight
2.65e-04 3.42e+03 input[0]
1.79e-06 4.65e+00 output
```

在这里，`encoder.block.2.layer.1.layer_norm` 表示它是编码器的第二个块中第一层的`layer norm`。而 `forward` 的具体调用是 `T5LayerNorm`。

让我们看看该报告的最后几个`frame`：

```
Detected inf/nan during batch_number=0
Last 21 forward frames:
abs min  abs max  metadata
[...]
                  encoder.block.2.layer.1.DenseReluDense.wi_0 Linear
2.17e-07 4.50e+00 weight
1.79e-06 4.65e+00 input[0]
2.68e-06 3.70e+01 output
                  encoder.block.2.layer.1.DenseReluDense.wi_1 Linear
8.08e-07 2.66e+01 weight
1.79e-06 4.65e+00 input[0]
1.27e-04 2.37e+02 output
                  encoder.block.2.layer.1.DenseReluDense.wo Linear
1.01e-06 6.44e+00 weight
0.00e+00 9.74e+03 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.DenseReluDense T5DenseGatedGeluDense
1.79e-06 4.65e+00 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.dropout Dropout
3.18e-04 6.27e+04 input[0]
0.00e+00      inf output
```

最后一个`frame`报告了`Dropout.forward`函数，第一个条目是唯一的输入，第二个条目是唯一的输出。您可以看到，它是从`DenseReluDense`类内的属性`dropout`中调用的。我们可以看到它发生在第2个块的第1层，也就是在第一个批次期间。最后，绝对最大的输入元素值为`6.27e+04`，输出也是`inf`。

您可以在这里看到，`T5DenseGatedGeluDense.forward`产生了输出激活值，其绝对最大值约为62.7K，非常接近fp16的上限64K。在下一个`frame`中，我们有`Dropout`对权重进行重新归一化，之后将某些元素归零，将绝对最大值推到了64K以上，导致溢出（`inf`）。

正如你所看到的，我们需要查看前面的`frame`, 从那里fp16数字开始变得非常大。

让我们将报告与`models/t5/modeling_t5.py`中的代码匹配：

```python
class T5DenseGatedGeluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gelu_act = ACT2FN["gelu_new"]

    def forward(self, hidden_states):
        hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states
```

现在很容易看到`dropout`调用，以及所有之前的调用。

由于检测是在前向`hook`中进行的，这些报告将立即在每个`forward`返回后打印出来。

回到完整的报告，要采取措施并解决问题，我们需要往回看几个`frame`，在那里数字开始上升，并且最有可能切换到fp32模式以便在乘法或求和时数字不会溢出。当然，可能还有其他解决方案。例如，如果启用了`amp`，我们可以在将原始`forward`移到`helper wrapper`中后，暂时关闭它，如下所示：

```python
def _forward(self, hidden_states):
    hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
    hidden_linear = self.wi_1(hidden_states)
    hidden_states = hidden_gelu * hidden_linear
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.wo(hidden_states)
    return hidden_states


import torch


def forward(self, hidden_states):
    if torch.is_autocast_enabled():
        with torch.cuda.amp.autocast(enabled=False):
            return self._forward(hidden_states)
    else:
        return self._forward(hidden_states)
```

由于自动检测器仅报告完整`frame`的输入和输出，一旦知道在哪里查找，您可能还希望分析特定`forward`函数的中间阶段。在这种情况下，您可以使用`detect_overflow`辅助函数将检测器放到希望的位置，例如：

```python
from debug_utils import detect_overflow


class T5LayerFF(nn.Module):
    [...]

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        detect_overflow(forwarded_states, "after layer_norm")
        forwarded_states = self.DenseReluDense(forwarded_states)
        detect_overflow(forwarded_states, "after DenseReluDense")
        return hidden_states + self.dropout(forwarded_states)
```

可以看到，我们添加了2个检测器，现在我们可以跟踪是否在`forwarded_states`中间的某个地方检测到了`inf`或`nan`。

实际上，检测器已经报告了这些，因为上面示例中的每个调用都是一个`nn.Module`，但假设如果您有一些本地的直接计算，这就是您将如何执行的方式。

此外，如果您在自己的代码中实例化调试器，您可以调整从其默认打印的`frame`数，例如：

```python
from transformers.debug_utils import DebugUnderflowOverflow

debug_overflow = DebugUnderflowOverflow(model, max_frames_to_save=100)
```

### 特定批次的绝对最小值和最大值跟踪

当关闭下溢/上溢检测功能, 同样的调试类可以用于批处理跟踪。

假设您想要监视给定批次的每个`forward`调用的所有成分的绝对最小值和最大值，并且仅对批次1和3执行此操作，您可以这样实例化这个类：

```python
debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1, 3])
```

现在，完整的批次1和3将以与下溢/上溢检测器相同的格式进行跟踪。

批次从0开始计数。

如果您知道程序在某个批次编号之后开始出现问题，那么您可以直接快进到该区域。以下是一个截取的配置示例输出：

```
                  *** Starting batch number=1 ***
abs min  abs max  metadata
                  shared Embedding
1.01e-06 7.92e+02 weight
0.00e+00 2.47e+04 input[0]
5.36e-05 7.92e+02 output
[...]
                  decoder.dropout Dropout
1.60e-07 2.27e+01 input[0]
0.00e+00 2.52e+01 output
                  decoder T5Stack
     not a tensor output
                  lm_head Linear
1.01e-06 7.92e+02 weight
0.00e+00 1.11e+00 input[0]
6.06e-02 8.39e+01 output
                   T5ForConditionalGeneration
     not a tensor output

                  *** Starting batch number=3 ***
abs min  abs max  metadata
                  shared Embedding
1.01e-06 7.92e+02 weight
0.00e+00 2.78e+04 input[0]
5.36e-05 7.92e+02 output
[...]
```

在这里，您将获得大量的`frame`被`dump` - 与您的模型中的前向调用一样多，它有可能符合也可能不符合您的要求，但有时对于调试目的来说，它可能比正常的调试器更容易使用。例如，如果问题开始发生在批次号150上，您可以`dump`批次149和150的跟踪，并比较数字开始发散的地方。

你还可以使用以下命令指定停止训练的批次号：

```python
debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1, 3], abort_after_batch_num=3)
```
