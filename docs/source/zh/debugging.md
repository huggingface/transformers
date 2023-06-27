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

## 多 GPU 网络问题调试

当使用 `DistributedDataParallel` 和多个 GPU 进行训练或推理时，如果遇到进程和/或节点之间的通信问题，您可以使用以下脚本诊断网络问题。

```bash
wget https://raw.githubusercontent.com/huggingface/transformers/main/scripts/distributed/torch-distributed-gpu-test.py
```

例如，要测试两个 GPU 之间的交互方式，请执行以下操作：

```bash
python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
```

如果两个进程都能相互通信并分配 GPU 内存，则每个进程都会打印出 OK 的状态。

对于更多的 GPU 或节点，请在脚本中调整参数。

在诊断脚本中，您会找到更多详细信息，甚至可以找到在 SLURM 环境中运行它的方法。

另一个调试级别是添加 `NCCL_DEBUG=INFO` 环境变量，如下所示：

```bash
NCCL_DEBUG=INFO python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
```

这将输出大量与 NCCL 相关的调试信息，如果发现某些问题已报告，您可以在网上搜索这些信息。如果您不确定如何解释输出，您可以在 Issue 中共享日志文件。



## 下溢和上溢检测

<Tip>

此功能目前仅适用于 PyTorch。

</Tip>

<Tip>

对于多 GPU 训练，它需要 DDP (`torch.distributed.launch`)。

</Tip>

<Tip>

此功能可以与任何基于 `nn.Module` 的模型一起使用。

</Tip>

如果开始出现 `loss=NaN` 或模型因激活或权重中的 `inf` 或 `nan` 而表现异常，您需要找出第一个下溢或上溢发生的位置以及导致其发生的原因。幸运的是，您可以通过激活一个特殊模块来自动进行检测来轻松完成这项任务。

如果您正在使用 [`Trainer`]，您只需要将以下内容添加到常规命令行参数中：

```bash
--debug underflow_overflow
```

或者在创建 [`TrainingArguments`] 对象时传递 `debug="underflow_overflow"`。

如果您正在使用自己的训练循环或其他 Trainer，您可以使用以下方式实现相同的效果：

```python
from transformers.debug_utils import DebugUnderflowOverflow

debug_overflow = DebugUnderflowOverflow(model)
```

[`~debug_utils.DebugUnderflowOverflow`] 在模型中插入钩子，在每次 forward 调用之后立即测试输入和输出变量以及相应模块的权重。一旦在激活或权重的至少一个元素中检测到 `inf` 或 `nan`，程序将断言并打印出报告（下面的报告是在 fp16 混合精度下使用 google/mt5-small 捕获的）：

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

该示例输出已经为简洁起见进行了修剪。

第二列显示了绝对最大元素的值，因此如果您仔细观察最后几个帧，输入和输出的范围在 `1e4` 的范围内。因此，在使用 `fp16` 混合精度下进行训练时，最后一个步骤发生了溢出（因为在 `fp16` 下，最大的数字是 `64e3`，所以 `1e4 * 1e4 = 1e8`，因此任何具有较大激活的矩阵乘法都将导致数值溢出条件。

在跟踪的开始处，您可以发现问题发生的批次号（这里的 `Detected inf/nan during batch_number=0` 表示问题发生在第一批次）。

每个报告的帧都以声明完全限定的条目开头，该帧报告的是相应模块的条目。如果我们只看这一帧：

```
                  encoder.block.2.layer.1.layer_norm T5LayerNorm
8.69e-02 4.18e-01 weight
2.65e-04 3.42e+03 input[0]
1.79e-06 4.65e+00 output
```

这里，`encoder.block.2.layer.1.layer_norm` 表示它是编码器第二个块的第一层的层归一化操作。而 `forward` 的具体调用是 `T5LayerNorm`。

让我们看一下报告的最后几个帧：

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

最后一帧报告了 `Dropout.forward` 函数，第一个条目是输入，第二个条目是输出。可以看到它是在 `DenseReluDense` 类内的 `dropout` 属性中调用的。我们可以看到这发生在第二个块的第一层，在第一个批次中。最后，输入中的绝对最大元素为 `6.27e+04`，输出中的绝对最大元素也为 `inf`。

可以看到，在 `T5DenseGatedGeluDense.forward` 中得到的输出激活值的绝对最大值约为 62.7K，非常接近 fp16 的上限 64K。在下一个帧中，我们有一个 `Dropout` 操作，在将一些元素置零后重新归一化权重，将绝对最大值推到了超过 64K，导致了溢出（`inf`）。

如你所见，我们需要查看前面的帧，当数字开始变得非常大时，才能找到问题所在。

让我们将报告与 `models/t5/modeling_t5.py` 中的代码进行匹配：
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

现在很容易看到 `dropout` 调用以及之前的所有调用。

由于检测是在前向钩子中进行的，这些报告会在每个 `forward` 返回后立即打印出来。

回到完整的报告，为了采取措施并解决问题，我们需要向上查看几帧，找到数字开始增大的地方，并且很可能在这里切换到 `fp32` 模式，以防止数字在相乘或相加时溢出。当然，可能还有其他解决方案。例如，我们可以暂时关闭 `amp`（如果启用），在将原始的 `forward` 移动到助手包装器中之后，如下所示：

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

由于自动检测器只报告完整帧的输入和输出，一旦你知道要查看的位置，你可能还想分析任何特定 `forward` 函数的中间阶段。在这种情况下，你可以使用 `detect_overflow` 助手函数将检测器注入到你希望的位置，例如：

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

你可以看到我们添加了其中的2个，并且现在跟踪是否在中间某个地方检测到了 `forwarded_states` 的 `inf` 或 `nan`。

实际上，检测器已经报告了这些问题，因为上面的每个调用都是一个 `nn.Module`，但是假设你有一些本地的直接计算，你可以按照下面的方式进行调试。

此外，如果你在自己的代码中实例化了调试器，你可以调整打印的帧数，默认为：

```python
from transformers.debug_utils import DebugUnderflowOverflow

debug_overflow = DebugUnderflowOverflow(model, max_frames_to_save=100)
```

同样的调试类可以用于对每个批次进行跟踪，关闭了下溢/上溢检测功能。

假设你想要观察给定批次的每个 `forward` 调用的所有参数的绝对最小值和最大值，只想在第1批和第3批进行观察。那么你可以按照以下方式实例化该类：

```python
debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1, 3])
```

And now full batches 1 and 3 will be traced using the same format as the underflow/overflow detector does.

Batches are 0-indexed.

如果您知道程序在某个批次号之后开始出现问题，您可以快进到该区域。以下是一个截断输出的示例配置: 在这里，您将获得大量的帧转储 - 与您的模型中的前向调用数量相同，因此它可能并不是您想要的，但有时它比正常的调试器更容易使用。例如，如果问题在第 150 个批次号开始发生。因此，您可以对第 149 和 150 批次号进行跟踪转储，并比较数字开始发散的位置。
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

您还可以使用以下方式指定训练停止的批次号之后: 如果问题在第 150 个批次号开始发生。因此，您可以对第 149 和 150 批次号进行跟踪转储，并比较数字开始发散的位置。如果问题在第 150 个批次号开始发生。

因此，您可以对第 149 和 150 批次号进行跟踪转储，并比较数字开始发散的位置。数字开始发散的位置。
使用以下方式指定训练停止的批次号之后:

```python
debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1, 3], abort_after_batch_num=3)
```
