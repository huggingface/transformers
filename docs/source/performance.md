<!---
Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Performance and Scalability: How To Fit a Bigger Model and Train It Faster

For now the software sections of this document are mainly Pytorch-specific, but the guide can be extended to other frameworks in the future.

## Quick notes

This section gives brief ideas on how to make training faster and support bigger models. Later sections will expand, demonstrate and elucidate each of these.

### Faster Training

Hardware:

- fast connectivity between GPUs
    * intra-node: NVLink
    * inter-node: Infiniband / Intel OPA

Software:

- Data Parallel / Distributed Data Parallel
- fp16 (autocast caching)


### Bigger Models

Hardware:

- bigger GPUs
- more GPUs
- more CPU and NVMe (offloaded to by DeepSpeed)

Software:

- Deepspeed ZeRO
- Deepspeed ZeRO-Offload
- Megatron-LM 3D Parallelism
- Pipeline Parallelism
- Tensor Parallelism
- Low-memory Optimizers
- fp16/bf16 (smaller data)



## Hardware

### Multi-GPU Connectivity

If you use multiple GPUs the way cards are inter-connected can have a huge impact on the total training time.

If the GPUs are on the same physical node, you can run:

```
nvidia-smi topo -m
```

and it will tell you how the GPUs are inter-connected.

On a machine with dual-GPU and which are connected with NVLink, you will most likely see something like:

```
        GPU0    GPU1    CPU Affinity    NUMA Affinity
GPU0     X      NV2     0-23            N/A
GPU1    NV2      X      0-23            N/A
```

on a different machine w/o NVLink we may see:
```
        GPU0    GPU1    CPU Affinity    NUMA Affinity
GPU0     X      PHB     0-11            N/A
GPU1    PHB      X      0-11            N/A
```

The report includes this legend:

```
  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```

So the first report `NV2` tells us the GPUs are interconnected with 2 NVLinks, and the second report `PHB` we have a typical consumer-level PCIe+Bridge setup.

Check what type of connectivity you have on your setup. Some of these will make the communication between cards faster (e.g. NVLink), others slower (e.g. PHB).

Depending on the type of scalability solution used, the connectivity speed could have a major or a minor impact. If the GPUs need to sync rarely, as in DDP, the impact of a slower connection will be less significant. If the GPUs need to send messages to each other often, as in ZeRO-DP, then faster connectivity becomes super important to achieve faster training.

### NVlink

[NVLink](https://en.wikipedia.org/wiki/NVLink) is a wire-based serial multi-lane near-range communications link developed by Nvidia.

Each new generation provides a faster bandwidth, e.g. here is a quote from [Nvidia Ampere GA102 GPU Architecture](https://www.nvidia.com/content/dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf):

> Third-Generation NVLink®
> GA102 GPUs utilize NVIDIA’s third-generation NVLink interface, which includes four x4 links,
> with each link providing 14.0625 GB/sec bandwidth in each direction between two GPUs. Four
> links provide 56.25 GB/sec bandwidth in each direction, and 112.5 GB/sec total bandwidth
> between two GPUs. Two RTX 3090 GPUs can be connected together for SLI using NVLink.
> (Note that 3-Way and 4-Way SLI configurations are not supported.)

So the higher `X` you get in the report of `NVX` in the output of `nvidia-smi topo -m` the better. The generation will depend on your GPU architecture.

Let's compare the execution of a gpt2 language model training over a small sample of wikitext.

The results are:


| NVlink | Time |
| -----  | ---: |
| Y      | 101s |
| N      | 131s |


You can see that NVLink completes the training ~23% faster.

In the second benchmark we use `NCCL_P2P_DISABLE=1` to tell the GPUs not to use NVLink.

Here is the full benchmark code and outputs:

```
# DDP w/ NVLink

rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py --model_name_or_path gpt2 \
--dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train \
--output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 101.9003, 'train_samples_per_second': 1.963, 'epoch': 0.69}

# DDP w/o NVLink

rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 python -m torch.distributed.launch \
--nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py --model_name_or_path gpt2 \
--dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train
--output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 131.4367, 'train_samples_per_second': 1.522, 'epoch': 0.69}
```

Hardware: 2x TITAN RTX 24GB each + NVlink with 2 NVLinks (`NV2` in `nvidia-smi topo -m`)
Software: `pytorch-1.8-to-be` + `cuda-11.0` / `transformers==4.3.0.dev0`

## Software

### Anatomy of Model's Memory

The components on GPU memory are the following:
- the model weights
- the forward activations saved for gradient computation
- the gradients
- the optimizer state

### `forward` vs `backward` Execution Speed

For convolutions and linear layers there are 2x flops in the backward compared to the forward, which generally translates into ~2x slower (sometimes more, because sizes in the backward tend to be more awkward). Activations are usually bandwidth-limited, and it’s typical for an activation to have to read more data in the backward than in the forward (e.g. activation forward reads once, writes once, activation backward reads twice, gradOutput and output of the forward, and writes once, gradInput).

### fp16

AMP = Automatic Mixed Precision

If we look at what's happening with FP16 training (mixed precision) we have:
- the model has two copies in memory: one in half-precision for the forward/backward computations and one in full precision - no memory saved here
- the forward activations saved for gradient computation are in half-precision - memory is saved here
- the gradients are computed in half-precision *but* converted to full-precision for the update, no saving there
- the optimizer states are in full precision as all the updates are done in full-precision

So the savings only happen for the forward activations saved for the backward computation, and there is a slight overhead because the model weights are stored both in half- and full-precision.

Now let's look at a simple text-classification fine-tuning on 2 GPUs (I'm giving the command for reference):
```
export BS=16
python -m torch.distributed.launch \
    --nproc_per_node 2 examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path bert-base-cased \
    --task_name mrpc \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size $BS \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir /tmp/mrpc \
    --overwrite_output_dir \
    --fp16
```
Since the only savings we get are in the model activations saved for the backward passed, it's logical that the bigger those activations are, the bigger the saving will be. If we try different batch sizes, I indeed get (this is with `nvidia-smi` so not completely reliable as said above but it will be a fair comparison):

| batch size | w/o --fp16 | w/ --fp16 | savings |
| ---------: | ---------: | --------: | ------: |
|          8 |       4247 |      4163 |      84 |
|         16 |       4971 |      4793 |     178 |
|         32 |       6827 |      6207 |     620 |
|         64 |      10037 |      8061 |    1976 |

So there is only a real memory saving if we train at a high batch size (and it's not half) and at batch sizes lower than 8, you actually get a bigger memory footprint (because of the overhead mentioned above). The gain for FP16 training is that in each of those cases, the training with the flag `--fp16` is twice as fast, which does require every tensor to have every dimension be a multiple of 8 (examples pad the tensors to a sequence length that is a multiple of 8).

Summary: FP16 with apex or AMP will only give you some memory savings with a reasonably high batch size.

Additionally, under mixed precision when possible, it's important that the batch size is a multiple of 8 to efficiently use tensor cores.

Some amazing tutorials to read on mixed precision:
- @sgugger wrote a great explanation of mixed precision [here](https://docs.fast.ai/callback.fp16.html#A-little-bit-of-theory)
- Aleksey Bilogur's [A developer-friendly guide to mixed precision training with PyTorch](https://spell.ml/blog/mixed-precision-training-with-pytorch-Xuk7YBEAACAASJam)

### fp16 caching

pytorch `autocast` which performs AMP include a caching feature, which speed things up by caching fp16-converted values. Here is the full description from this [comment](https://discuss.pytorch.org/t/autocast-and-torch-no-grad-unexpected-behaviour/93475/3):

Autocast maintains a cache of the FP16 casts of model params (leaves). This helps streamline parameter reuse: if the same FP32 param is used in several different FP16list ops, like several matmuls, instead of re-casting the param to FP16 on entering each matmul, the cast will occur on the first matmul, the casted FP16 copy will be cached, and for all later matmuls the FP16 copy will be reused. The cache is maintained only within a particular outermost autocast context. When you exit the autocast context the cache is dropped. For recommended usage, in which autocast wraps the forward pass, and then you exit the context before calling backward(), this means the cache only lasts the duration of the forward pass each iteration, and will be rebuilt next iteration. (The cache of FP16-casted copies MUST be rebuilt each iteration. The FP32 params get updated by the optimizer, so the FP16 copies must be recreated, otherwise the FP16 values will be stale.)

### DP vs DDP

`DistributedDataParallel` (DDP) is typically faster than `DataParallel` (DP), but it is not always the case:
* while DP is python threads-based, DDP is multiprocess-based - and as such it has no python threads limitations, such as GIL
* on the other hand a slow inter-connectivity between the GPU cards could lead to an actual slower outcome with DDP

Here are the main differences in the inter-GPU communication overhead between the two modes:

[DDP](https://pytorch.org/docs/master/notes/ddp.html):

- At the start time the main process replicates the model once from gpu 0 to the rest of gpus
- Then for each batch:
   1. each gpu consumes each own mini-batch of data directly
   2. during `backward`, once the local gradients are ready, they are then averaged across all processes

[DP](https://pytorch.org/docs/master/generated/torch.nn.DataParallel.html):

For each batch:
   1. gpu 0 reads the batch of data and then sends a mini-batch to each gpu
   2. replicates the up-to-date model from gpu 0 to each gpu
   3. runs `forward` and sends output from each gpu to gpu 0, computes loss
   4. scatters loss from gpu 0 to all gpus, runs `backward`
   5. sends gradients from each gpu to gpu 0 and averages those

The only communication DDP performs per batch is sending gradients, whereas DP does 5 different data exchanges per batch.

DP copies data within the process via python threads, whereas DDP copies data via [torch.distributed](https://pytorch.org/docs/master/distributed.html).

Under DP gpu 0 performs a lot more work than the rest of the gpus, thus resulting in under-utilization of gpus.

You can use DDP across multiple machines, but this is not the case with DP.

There are other differences between DP and DDP but they aren't relevant to this discussion.

If you want to go really deep into understanding these 2 modes, this [article](https://www.telesens.co/2019/04/04/distributed-data-parallel-training-using-pytorch-on-aws/) is highly recommended, as it has great diagrams, includes multiple benchmarks and profiler outputs on various hardware, explains all the nuances that you may need to know.

Let's look at an actual benchmark:

| Type   | NVlink | Time |
| :----- | -----  | ---: |
| 2:DP   | Y      | 110s |
| 2:DDP  | Y      | 101s |
| 2:DDP  | N      | 131s |


Analysis:

Here DP is ~10% slower than DDP w/ NVlink, but ~15% faster than DDP w/o NVlink

The real difference will depend on how much data each GPU needs to sync with the others - the more there is to sync, the more a slow link will slow down the total runtime.

Here is the full benchmark code and outputs:

`NCCL_P2P_DISABLE=1` was used to disable the NVLink feature on the corresponding benchmark.

```

# DP
rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 \
python examples/pytorch/language-modeling/run_clm.py \
--model_name_or_path gpt2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
--do_train --output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 110.5948, 'train_samples_per_second': 1.808, 'epoch': 0.69}

# DDP w/ NVlink
rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py \
--model_name_or_path gpt2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
--do_train --output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 101.9003, 'train_samples_per_second': 1.963, 'epoch': 0.69}

# DDP w/o NVlink
rm -r /tmp/test-clm; NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py \
--model_name_or_path gpt2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
--do_train --output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 131.4367, 'train_samples_per_second': 1.522, 'epoch': 0.69}
```

Hardware: 2x TITAN RTX 24GB each + NVlink with 2 NVLinks (`NV2` in `nvidia-smi topo -m`)
Software: `pytorch-1.8-to-be` + `cuda-11.0` / `transformers==4.3.0.dev0`


### DataLoader

One of the important requirements to reach great training speed is the ability to feed the GPU at the maximum speed it can handle. By default everything happens in the main process and it might not be able to read the data from disk fast enough, and thus create a bottleneck, leading to GPU under-utilization.

- `DataLoader(pin_memory=True, ...)` which ensures that the data gets preloaded into the pinned memory on CPU and typically leads to much faster transfers from CPU to GPU memory.
-  `DataLoader(num_workers=4, ...)` - spawn several workers to pre-load data faster - during training watch the GPU utilization stats and if it's far from 100% experiment with raising the number of workers. Of course, the problem could be elsewhere so a very big number of workers won't necessarily lead to a better performance.

### Faster optimizer

pytorch-nightly introduced `torch.optim._multi_tensor` which should significantly speed up the optimizers for situations with lots of small feature tensors. It should eventually become the default, but if you want to experiment with it sooner and don't mind using the bleed-edge, see: https://github.com/huggingface/transformers/issues/9965


## Contribute

This document is far from being complete and a lot more needs to be added, so if you have additions or corrections to make please don't hesitate to open a PR or if you aren't sure start an Issue and we can discuss the details there.

When making contributions that A is better than B, please try to include a reproducible benchmark and/or a link to the source of that information (unless it comes directly from you).
