<!--版权所有 2022 HuggingFace 团队保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）许可；除非符合许可证，否则您不得使用此文件。您可以在
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，否则根据许可证分发的软件基于“按原样”分发，不附带任何明示或暗示的保证或条件。有关许可证的详细信息，请参阅
⚠️ 请注意，此文件是 Markdown 格式，但包含我们的文档生成器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确呈现。
-->
# 多 CPU 上的高效训练

当单个 CPU 上的训练速度过慢时，我们可以使用多个 CPU。本指南重点介绍基于 PyTorch 的 DDP，以实现分布式 CPU 训练的高效性。

## PyTorch 的 Intel ® oneCCL 绑定

[Intel ® oneCCL](https://github.com/oneapi-src/oneCCL)（集体通信库）是用于高效分布式深度学习训练的库，实现了诸如 allreduce、allgather、alltoall 等集体通信操作。有关 oneCCL 的更多信息，请参阅 [oneCCL 文档](https://spec.oneapi.com/versions/latest/elements/oneCCL/source/index.html) 和 [oneCCL 规范](https://spec.oneapi.com/versions/latest/elements/oneCCL/source/index.html)。

模块 `oneccl_bindings_for_pytorch`（1.12 版本之前为 `torch_ccl`）实现了 PyTorch C10D ProcessGroup API，并可以作为外部 ProcessGroup 进行动态加载，目前仅适用于 Linux 平台。

详细信息请查看 [oneccl_bind_pt](https://github.com/intel/torch-ccl)。

### 安装 PyTorch 的 Intel ® oneCCL 绑定:

Wheel 文件适用于以下 Python 版本:

| 扩展版本 | Python 3.6 | Python 3.7 | Python 3.8 | Python 3.9 | Python 3.10 || :------: | :--------: | :--------: | :--------: | :--------: | :---------: || 1.13.0  |            |     √      |     √      |     √      |      √      || 1.12.100|            |     √      |     √      |     √      |      √      || 1.12.0  |            |     √      |     √      |     √      |      √      || 1.11.0  |            |     √      |     √      |     √      |      √      || 1.10.0  |     √      |     √      |     √      |     √      |             |
```
pip install oneccl_bind_pt=={pytorch_version} -f https://developer.intel.com/ipex-whl-stable-cpu
```
其中 `{pytorch_version}` 应为您的 PyTorch 版本，例如 1.13.0。有关 [oneccl_bind_pt 安装的更多方法](https://github.com/intel/torch-ccl)。oneCCL 和 PyTorch 的版本必须匹配。
<Tip warning={true}>
oneccl_bindings_for_pytorch 1.12.0 的预构建 Wheel 文件不适用于 PyTorch 1.12.1（适用于 PyTorch 1.12.0）PyTorch 1.12.1 应与 oneccl_bindings_for_pytorch 1.12.100 配合使用
</Tip>

## Intel ® MPI 库

使用这个基于标准的 MPI 实现，在 Intel ®架构上提供灵活、高效、可扩展的集群通信。该组件是 Intel ® oneAPI HPC Toolkit 的一部分。

oneccl_bindings_for_pytorch 与 MPI 工具集一起安装。在使用之前需要设置环境变量。
适用于 Intel ® oneCCL >= 1.12.0

```
oneccl_bindings_for_pytorch_path=$(python -c "from oneccl_bindings_for_pytorch import cwd; print(cwd)")
source $oneccl_bindings_for_pytorch_path/env/setvars.sh
```

适用于 Intel ® oneCCL 版本 < 1.12.0
```
torch_ccl_path=$(python -c "import torch; import torch_ccl; import os;  print(os.path.abspath(os.path.dirname(torch_ccl.__file__)))")
source $torch_ccl_path/env/setvars.sh
```

#### 安装 IPEX:
IPEX 为使用 Float32 和 BFloat16 进行 CPU 训练提供了性能优化，您可以参考 [单 CPU 部分](./perf_train_cpu)。
以下 "Trainer 中的用法" 以 Intel ® MPI 库中的 mpirun 为例。

## Trainer 中的用法
要在 Trainer 中启用多 CPU 分布式训练并使用 ccl 后端，用户应在命令参数中添加 **`--ddp_backend ccl`**。
让我们以 [问答示例](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) 为例。

以下命令在一台 Xeon 节点上启用使用 2 个进程进行训练，每个进程在一个 socket 上运行。可以调整 OMP_NUM_THREADS/CCL_WORKER_COUNT 变量以获得最佳性能。
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
以下命令在两个 Xeon 上总共启用 4 个进程进行训练（node0 和 node1，以 node0 为主进程），ppn（每个节点的进程数）设置为 2，每个进程在一个 socket 上运行。可以调整 OMP_NUM_THREADS/CCL_WORKER_COUNT 变量以获得最佳性能。

在 node0 中，您需要创建一个包含每个节点的 IP 地址（例如 hostfile）的配置文件，并将该配置文件路径作为参数传递。
```shell script
 cat hostfile
 xxx.xxx.xxx.xxx #node0 ip
 xxx.xxx.xxx.xxx #node1 ip
```
现在，在 node0 中运行以下命令，将启用 **4DDP** 在 node0 和 node1 中使用 BF16 混合精度：```shell script
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
