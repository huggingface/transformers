<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# CPU

CPUs run Transformers models with the device string `cpu`, without an accelerator. They work for inference and training when GPUs aren't available, or when you want a cost-effective setup.

## Run inference

Models load on CPU by default when you don't pass `device_map`, and [`pipeline`] uses the CPU when no accelerator is available. Pass `device_map="cpu"` or `device="cpu"` to use the CPU on a machine that also has an accelerator.

```py
from transformers import pipeline

generator = pipeline("text-generation", model="Qwen/Qwen3-0.6B", device="cpu")
print(generator("CPUs are", max_new_tokens=50, do_sample=False)[0]["generated_text"])
```

## Train with Trainer

CPUs with bf16 support, such as recent Intel and AMD processors, can use bf16 mixed precision training with [PyTorch's AMP](https://docs.pytorch.org/docs/stable/amp), which reduces memory usage and speeds up training.

The examples below cover three setups, from a single CPU to several machines when one CPU is too slow:

- a single CPU
- multiple processes on one machine (one per CPU socket)
- multiple processes across several machines

The single-node and multi-node examples launch processes with `mpirun` from [Intel MPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html), part of the [Intel oneAPI HPC Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/hpc-toolkit.html), and [`Trainer`] runs DDP over the `gloo` backend.

<hfoptions id="distrib-cpu">
<hfoption id="single CPU">

Set `bf16=True` and `use_cpu=True` in [`TrainingArguments`], or pass `--bf16` and `--use_cpu` to a training script. Prefer bf16 over fp16 on CPU because it's more numerically stable.

```py
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./outputs",
    bf16=True,
    use_cpu=True,
)
```

</hfoption>
<hfoption id="distributed CPU (single node)">

On a dual-socket CPU, run one process per socket. Keeping memory accesses local to each socket improves throughput. The example below launches two processes on a single machine with one process per socket, where `train.py` is your `Trainer` training script.

> [!TIP]
> Set `OMP_NUM_THREADS` to the number of physical cores on one socket, minus one core reserved for the OS. For example, on a 24-core socket set `OMP_NUM_THREADS=23`.

```bash
export MASTER_ADDR=127.0.0.1
mpirun -n 2 -genv OMP_NUM_THREADS=23 \
python3 train.py \
 --output_dir ./outputs \
 --use_cpu \
 --bf16
```

</hfoption>
<hfoption id="distributed CPU (multiple nodes)">

Scale training to four processes across two machines (`node0` and `node1`) using a `hostfile` that lists each node's IP address. The `-n 4` flag sets the total number of processes. `-ppn 2` sets two processes per node (one per socket).

Run this script from `node0`, which acts as the main process.

> [!TIP]
> Set `OMP_NUM_THREADS` to the number of physical cores on one socket, minus one core reserved for the OS. For example, on a 24-core socket set `OMP_NUM_THREADS=23`.

Create a hostfile with the IP address of each node.

```bash
cat hostfile
xxx.xxx.xxx.xxx #node0 ip
xxx.xxx.xxx.xxx #node1 ip
```

Then run the training script.

```bash
export MASTER_ADDR=xxx.xxx.xxx.xxx #node0 ip
mpirun -f hostfile -n 4 -ppn 2 \
 -genv OMP_NUM_THREADS=23 \
python3 train.py \
 --output_dir ./outputs \
 --use_cpu \
 --bf16
```

</hfoption>
</hfoptions>

## Tune CPU performance

Preload a faster memory allocator and Intel's OpenMP runtime to reduce overhead in CPU training. Install `google-perftools` (for `libtcmalloc`) and the `intel-openmp` Python package (for `libiomp5`), then add both libraries to the `LD_PRELOAD` environment variable before you launch training.

Set `OMP_NUM_THREADS` to the number of cores each process can use. In a container or a cluster job with CPU limits, that's the number of allocated CPU units, so leave some cores unallocated for system processes.

## Next steps

- Read the [Accelerating PyTorch Transformers with Intel Sapphire Rapids](https://huggingface.co/blog/intel-sapphire-rapids) blog post for a deeper look at BF16 performance on modern Intel hardware.
- Read [DDP](./ddp) for more on the distributed strategy the multi-process examples use.
- Read the PyTorch [CPU threading](https://docs.pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html) notes to learn how `OMP_NUM_THREADS` affects performance.
- Read the Kubeflow [PyTorchJob guide](https://www.kubeflow.org/docs/components/training/user-guides/pytorch/) to run distributed CPU training on Kubernetes.
