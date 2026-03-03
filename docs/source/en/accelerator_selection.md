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

# Accelerator selection

You can control which accelerators (CUDA, XPU, MPS, HPU, etc.) PyTorch sees and in what order during distributed training. Prioritize faster devices or limit training to a subset of available hardware. It works with both [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) and [DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html), and doesn't require Accelerate or the [DeepSpeed integration](./main_classes/deepspeed).

## Order of accelerators

Use the hardware-specific environment variable to select accelerators and set their order. Set it on the command line per run, or add it to `~/.bashrc` or another startup config file.

> [!WARNING]
> Avoid exporting environment variables because if you forget a previously exported value, you may silently train on the wrong accelerators. Set the environment variable on the same command line as the training run.

For example, to select accelerators 0 and 2 out of four:

<hfoptions id="accelerator-type">
<hfoption id="CUDA">

```bash
CUDA_VISIBLE_DEVICES=0,2 torchrun trainer-program.py ...
```

PyTorch sees only GPUs 0 and 2, which are mapped to `cuda:0` and `cuda:1`. To reverse the order (use GPU 2 as `cuda:0` and GPU 0 as `cuda:1`):

```bash
CUDA_VISIBLE_DEVICES=2,0 torchrun trainer-program.py ...
```

To run without any GPUs:

```bash
CUDA_VISIBLE_DEVICES= python trainer-program.py ...
```

Control the order of CUDA devices with `CUDA_DEVICE_ORDER`.

- Order by PCIe bus ID (matches `nvidia-smi`):

    ```bash
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    ```

- Order by compute capability (fastest first):

    ```bash
    export CUDA_DEVICE_ORDER=FASTEST_FIRST
    ```

</hfoption>
<hfoption id="Intel XPU">

```bash
ZE_AFFINITY_MASK=0,2 torchrun trainer-program.py ...
```

PyTorch sees only XPUs 0 and 2, which are mapped to `xpu:0` and `xpu:1`. To reverse the order (use XPU 2 as `xpu:0` and XPU 0 as `xpu:1`):

```bash
ZE_AFFINITY_MASK=2,0 torchrun trainer-program.py ...
```

Control the order of Intel XPUs with:

```bash
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
```

For more on device enumeration and sorting on Intel XPU, see the [Level Zero](https://github.com/oneapi-src/level-zero/blob/master/README.md?plain=1#L87) documentation.

</hfoption>
</hfoptions>
