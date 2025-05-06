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

# GPU selection

During distributed training, you can specify the number of GPUs to use and in what order. This can be useful when you have GPUs with different computing power and you want to use the faster GPU first. Or you could only use a subset of the available GPUs. The selection process works for both [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) and [DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html). You don't need Accelerate or [DeepSpeed integration](./main_classes/deepspeed).

This guide will show you how to select the number of GPUs to use and the order to use them in.

## Number of GPUs

For example, if there are 4 GPUs and you only want to use the first 2, run the command below.

<hfoptions id="select-gpu">
<hfoption id="torchrun">

Use the `--nproc_per_node` to select how many GPUs to use.

```bash
torchrun --nproc_per_node=2  trainer-program.py ...
```

</hfoption>
<hfoption id="Accelerate">

Use `--num_processes` to select how many GPUs to use.

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

### Order of GPUs

To select specific GPUs to use and their order, configure the `CUDA_VISIBLE_DEVICES` environment variable. It is easiest to set the environment variable in `~/bashrc` or another startup config file. `CUDA_VISIBLE_DEVICES` is used to map which GPUs are used. For example, if there are 4 GPUs (0, 1, 2, 3) and you only want to run GPUs 0 and 2:

```bash
CUDA_VISIBLE_DEVICES=0,2 torchrun trainer-program.py ...
```

Only the 2 physical GPUs (0 and 2) are "visible" to PyTorch and these are mapped to `cuda:0` and `cuda:1` respectively. You can also reverse the order of the GPUs to use 2 first. The mapping becomes `cuda:1` for GPU 0 and `cuda:0` for GPU 2.

```bash
CUDA_VISIBLE_DEVICES=2,0 torchrun trainer-program.py ...
```

You can also set the `CUDA_VISIBLE_DEVICES` environment variable to an empty value to create an environment without GPUs.

```bash
CUDA_VISIBLE_DEVICES= python trainer-program.py ...
```

> [!WARNING]
> As with any environment variable, they can be exported instead of being added to the command line. However, this is not recommended because it can be confusing if you forget how the environment variable was set up and you end up using the wrong GPUs. Instead, it is common practice to set the environment variable for a specific training run on the same command line.

`CUDA_DEVICE_ORDER` is an alternative environment variable you can use to control how the GPUs are ordered. You can order according to the following.

1. PCIe bus IDs that matches the order of [`nvidia-smi`](https://developer.nvidia.com/nvidia-system-management-interface) and [`rocm-smi`](https://rocm.docs.amd.com/projects/rocm_smi_lib/en/latest/.doxygen/docBin/html/index.html) for NVIDIA and AMD GPUs respectively.

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

2. GPU compute ability.

```bash
export CUDA_DEVICE_ORDER=FASTEST_FIRST
```

The `CUDA_DEVICE_ORDER` is especially useful if your training setup consists of an older and newer GPU, where the older GPU appears first, but you cannot physically swap the cards to make the newer GPU appear first. In this case, set `CUDA_DEVICE_ORDER=FASTEST_FIRST` to always use the newer and faster GPU first (`nvidia-smi` or `rocm-smi` still reports the GPUs in their PCIe order). Or you could also set `export CUDA_VISIBLE_DEVICES=1,0`.