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

During distributed training, you can specify the number and order of accelerators (CUDA, XPU, MPS, HPU, etc.) to use. This can be useful when you have accelerators with different computing power and you want to use the faster accelerator first. Or you could only use a subset of the available accelerators. The selection process works for both [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) and [DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html). You don't need Accelerate or [DeepSpeed integration](./main_classes/deepspeed).

This guide will show you how to select the number of accelerators to use and the order to use them in.

## Number of accelerators

For example, if there are 4 accelerators and you only want to use the first 2, run the command below.

<hfoptions id="select-accelerator">
<hfoption id="torchrun">

Use the `--nproc_per_node` to select how many accelerators to use.

```bash
torchrun --nproc_per_node=2  trainer-program.py ...
```

</hfoption>
<hfoption id="Accelerate">

Use `--num_processes` to select how many accelerators to use.

```bash
accelerate launch --num_processes 2 trainer-program.py ...
```

</hfoption>
<hfoption id="DeepSpeed">

Use `--num_gpus` to select how many GPUs to use.

```bash
deepspeed --num_gpus 2 trainer-program.py ...
```

</hfoption>
</hfoptions>

## Order of accelerators
To select specific accelerators to use and their order, use the environment variable appropriate for your hardware. This is often set on the command line for each run, but can also be added to your `~/.bashrc` or other startup config file.

For example, if there are 4 accelerators (0, 1, 2, 3) and you only want to run accelerators 0 and 2:

<hfoptions id="accelerator-type">
<hfoption id="CUDA">

```bash
CUDA_VISIBLE_DEVICES=0,2 torchrun trainer-program.py ...
```

Only GPUs 0 and 2 are "visible" to PyTorch and are mapped to `cuda:0` and `cuda:1` respectively.  
To reverse the order (use GPU 2 as `cuda:0` and GPU 0 as `cuda:1`):


```bash
CUDA_VISIBLE_DEVICES=2,0 torchrun trainer-program.py ...
```

To run without any GPUs:

```bash
CUDA_VISIBLE_DEVICES= python trainer-program.py ...
```

You can also control the order of CUDA devices using `CUDA_DEVICE_ORDER`:

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

Only XPUs 0 and 2 are "visible" to PyTorch and are mapped to `xpu:0` and `xpu:1` respectively.  
To reverse the order (use XPU 2 as `xpu:0` and XPU 0 as `xpu:1`):

```bash
ZE_AFFINITY_MASK=2,0 torchrun trainer-program.py ...
```


You can also control the order of Intel XPUs with:

```bash
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
```

For more information about device enumeration and sorting on Intel XPU, please refer to the [Level Zero](https://github.com/oneapi-src/level-zero/blob/master/README.md?plain=1#L87) documentation.

</hfoption>
</hfoptions>



> [!WARNING]
> Environment variables can be exported instead of being added to the command line. This is not recommended because it can be confusing if you forget how the environment variable was set up and you end up using the wrong accelerators. Instead, it is common practice to set the environment variable for a specific training run on the same command line.
