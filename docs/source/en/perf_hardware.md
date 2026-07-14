<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Building a GPU workstation

The GPU is one of the most important choices when building a deep learning machine. Tensor cores handle matrix multiplication efficiently, and high memory bandwidth keeps data flowing. Training large models requires a more powerful GPU, multiple GPUs, or offloading techniques that move work to the CPU or NVMe.

The tips below cover practical GPU setup for deep learning.

## Power

High-end consumer GPUs may have two or three PCIe 8-pin power sockets. Connect a separate 12V PCIe 8-pin cable to each socket. Don't use a *pigtail cable* (a single cable with two splits at one end) to connect two sockets, otherwise, you won't get full performance from the GPU.

Connect each PCIe 8-pin power cable to a 12V rail on the power supply unit (PSU). Each cable delivers up to 150W. Some GPUs use a PCIe 12-pin connector that delivers up to 500-600W. Lower-end GPUs may use a PCIe 6-pin connector that supplies up to 75W.

A PSU must maintain stable voltage because unstable voltage can starve the GPU of power during peak usage.

## Cooling

An overheated GPU throttles performance and shuts down to prevent damage. Keep temperatures between 158–167°F (70–75 Celsius) for full performance and a longer lifespan. Above 183–194°F (84–90 Celsius), the GPU usually starts throttling.

## Multi-GPU connectivity

How your GPUs connect matters for multi-GPU setups. [NVLink](https://www.nvidia.com/en-us/design-visualization/nvlink-bridges/) connections are faster than PCIe bridges, but the impact depends on your parallelism strategy. DDP has less GPU-to-GPU communication than ZeRO, so connection speed matters less.

Run the command below to check how your GPUs are connected.

```bash
nvidia-smi topo -m
```

<hfoptions id="nvlink">
<hfoption id="NVLink">

[NVLink](https://www.nvidia.com/en-us/design-visualization/nvlink-bridges/) is NVIDIA's high-speed communication system for connecting multiple GPUs.

```bash
        GPU0    GPU1    CPU Affinity    NUMA Affinity
GPU0     X      NV2     0-23            N/A
GPU1    NV2      X      0-23            N/A
```

`NV2` indicates `GPU0` and `GPU1` are connected by 2 NVLinks.

</hfoption>
<hfoption id="PCIe bridge">

```bash
        GPU0    GPU1    CPU Affinity    NUMA Affinity
GPU0     X      PHB     0-11            N/A
GPU1    PHB      X      0-11            N/A
```

`PHB` indicates `GPU0` and `GPU1` are connected by a PCIe bridge.

</hfoption>
</hfoptions>

## Next steps

-  See the [Which GPU(s) to Get for Deep Learning](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/) blog post for a deeper comparison of GPUs.