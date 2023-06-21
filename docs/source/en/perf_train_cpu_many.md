<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Efficient Training on Multiple CPUs

When training on a single CPU is too slow, we can use multiple CPUs. This guide focuses on PyTorch-based DDP enabling distributed CPU training efficiently.

## Intel® oneCCL Bindings for PyTorch

[Intel® oneCCL](https://github.com/oneapi-src/oneCCL) (collective communications library) is a library for efficient distributed deep learning training implementing such collectives like allreduce, allgather, alltoall. For more information on oneCCL, please refer to the [oneCCL documentation](https://spec.oneapi.com/versions/latest/elements/oneCCL/source/index.html) and [oneCCL specification](https://spec.oneapi.com/versions/latest/elements/oneCCL/source/index.html).

Module `oneccl_bindings_for_pytorch` (`torch_ccl` before version 1.12)  implements PyTorch C10D ProcessGroup API and can be dynamically loaded as external ProcessGroup and only works on Linux platform now

Check more detailed information for [oneccl_bind_pt](https://github.com/intel/torch-ccl).

### Intel® oneCCL Bindings for PyTorch installation:

Wheel files are available for the following Python versions:

| Extension Version | Python 3.6 | Python 3.7 | Python 3.8 | Python 3.9 | Python 3.10 |
| :---------------: | :--------: | :--------: | :--------: | :--------: | :---------: |
| 1.13.0            |            | √          | √          | √          | √           |
| 1.12.100          |            | √          | √          | √          | √           |
| 1.12.0            |            | √          | √          | √          | √           |
| 1.11.0            |            | √          | √          | √          | √           |
| 1.10.0            | √          | √          | √          | √          |             |

```
pip install oneccl_bind_pt=={pytorch_version} -f https://developer.intel.com/ipex-whl-stable-cpu
```
where `{pytorch_version}` should be your PyTorch version, for instance 1.13.0.
Check more approaches for [oneccl_bind_pt installation](https://github.com/intel/torch-ccl).
Versions of oneCCL and PyTorch must match.

<Tip warning={true}>

oneccl_bindings_for_pytorch 1.12.0 prebuilt wheel does not work with PyTorch 1.12.1 (it is for PyTorch 1.12.0)
PyTorch 1.12.1 should work with oneccl_bindings_for_pytorch 1.12.100

</Tip>

## Intel® MPI library
Use this standards-based MPI implementation to deliver flexible, efficient, scalable cluster messaging on Intel® architecture. This component is part of the Intel® oneAPI HPC Toolkit.

oneccl_bindings_for_pytorch is installed along with the MPI tool set. Need to source the environment before using it.

for Intel® oneCCL >= 1.12.0
```
oneccl_bindings_for_pytorch_path=$(python -c "from oneccl_bindings_for_pytorch import cwd; print(cwd)")
source $oneccl_bindings_for_pytorch_path/env/setvars.sh
```

for Intel® oneCCL whose version < 1.12.0
```
torch_ccl_path=$(python -c "import torch; import torch_ccl; import os;  print(os.path.abspath(os.path.dirname(torch_ccl.__file__)))")
source $torch_ccl_path/env/setvars.sh
```

#### IPEX installation:

IPEX provides performance optimizations for CPU training with both Float32 and BFloat16, you could refer [single CPU section](./perf_train_cpu).


The following "Usage in Trainer" takes mpirun in Intel® MPI library as an example.


## Usage in Trainer
To enable multi CPU distributed training in the Trainer with the ccl backend, users should add **`--ddp_backend ccl`** in the command arguments.

Let's see an example with the [question-answering example](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering)


The following command enables training with 2 processes on one Xeon node, with one process running per one socket. The variables OMP_NUM_THREADS/CCL_WORKER_COUNT can be tuned for optimal performance.
```shell script
 export CCL_WORKER_COUNT=1
 export MASTER_ADDR=127.0.0.1
 mpirun -n 2 -genv OMP_NUM_THREADS=23 \
 python3 run_qa.py \
 --model_name_or_path bert-large-uncased \
 --dataset_name squad \
 --do_train \
 --do_eval \
 --per_device_train_batch_size 12  \
 --learning_rate 3e-5  \
 --num_train_epochs 2  \
 --max_seq_length 384 \
 --doc_stride 128  \
 --output_dir /tmp/debug_squad/ \
 --no_cuda \
 --ddp_backend ccl \
 --use_ipex
```
The following command enables training with a total of four processes on two Xeons (node0 and node1, taking node0 as the main process), ppn (processes per node) is set to 2, with one process running per one socket. The variables OMP_NUM_THREADS/CCL_WORKER_COUNT can be tuned for optimal performance.

In node0, you need to create a configuration file which contains the IP addresses of each node (for example hostfile) and pass that configuration file path as an argument.
```shell script
 cat hostfile
 xxx.xxx.xxx.xxx #node0 ip
 xxx.xxx.xxx.xxx #node1 ip
```
Now, run the following command in node0 and **4DDP** will be enabled in node0 and node1 with BF16 auto mixed precision:
```shell script
 export CCL_WORKER_COUNT=1
 export MASTER_ADDR=xxx.xxx.xxx.xxx #node0 ip
 mpirun -f hostfile -n 4 -ppn 2 \
 -genv OMP_NUM_THREADS=23 \
 python3 run_qa.py \
 --model_name_or_path bert-large-uncased \
 --dataset_name squad \
 --do_train \
 --do_eval \
 --per_device_train_batch_size 12  \
 --learning_rate 3e-5  \
 --num_train_epochs 2  \
 --max_seq_length 384 \
 --doc_stride 128  \
 --output_dir /tmp/debug_squad/ \
 --no_cuda \
 --ddp_backend ccl \
 --use_ipex \
 --bf16
```
