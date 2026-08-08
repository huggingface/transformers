<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 加速器选择

在分布式训练时，你可以控制 PyTorch 可见的加速器（CUDA、XPU、MPS、HPU 等）及其顺序。你可以优先使用更快的设备，或者只使用部分可用硬件进行训练。此功能同时适用于 [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) 和 [DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)，并且不需要 Accelerate 或 [DeepSpeed 集成](./main_classes/deepspeed)。

## 加速器顺序

使用硬件对应的环境变量来选择加速器并设置它们的顺序。你可以在每次运行时在命令行中设置该环境变量，也可以将其添加到 `~/.bashrc` 或其他启动配置文件中。

> [!WARNING]
> 避免使用 export 导出环境变量，因为一旦忘记环境变量是如何设置的，你可能会在错误的加速器上训练而毫无察觉。请在启动训练的同一条命令行中设置环境变量。

例如，要在四个加速器中选择第 0 个和第 2 个：

<hfoptions id="accelerator-type">
<hfoption id="CUDA">

```cli
CUDA_VISIBLE_DEVICES=0,2 torchrun trainer-program.py ...
```

PyTorch 只能看到 GPU 0 和 GPU 2，它们分别被映射为 `cuda:0` 和 `cuda:1`。要反转顺序（将 GPU 2 用作 `cuda:0`、GPU 0 用作 `cuda:1`）：

```cli
CUDA_VISIBLE_DEVICES=2,0 torchrun trainer-program.py ...
```

要在不使用任何 GPU 的情况下运行：

```cli
CUDA_VISIBLE_DEVICES= python trainer-program.py ...
```

使用 `CUDA_DEVICE_ORDER` 控制 CUDA 设备的顺序。

- 按 PCIe 总线 ID 排序（与 `nvidia-smi` 显示的顺序一致）：

    ```cli
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    ```

- 按计算能力排序（最快的设备优先）：

    ```cli
    export CUDA_DEVICE_ORDER=FASTEST_FIRST
    ```

</hfoption>
<hfoption id="Intel XPU">

```cli
ZE_AFFINITY_MASK=0,2 torchrun trainer-program.py ...
```

PyTorch 只能看到 XPU 0 和 XPU 2，它们分别被映射为 `xpu:0` 和 `xpu:1`。要反转顺序（将 XPU 2 用作 `xpu:0`、XPU 0 用作 `xpu:1`）：

```cli
ZE_AFFINITY_MASK=2,0 torchrun trainer-program.py ...
```

使用以下命令控制 Intel XPU 的顺序：

```cli
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
```

有关 Intel XPU 上设备枚举和排序的更多信息，请参阅 [Level Zero](https://github.com/oneapi-src/level-zero/blob/master/README.md?plain=1#L87) 文档。

</hfoption>
</hfoptions>
