<!--版权所有2022 HuggingFace团队保留所有权利。
根据Apache许可证第2.0版（“许可证”）许可；除非符合许可证，否则您不得使用此文件。您可以在
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，否则根据许可证分发的软件基于“按原样”分发，不附带任何明示或暗示的保证或条件。有关许可证的详细信息，请参阅
⚠️ 请注意，此文件是Markdown格式，但包含我们的文档生成器（类似于MDX）的特定语法，可能无法在您的Markdown查看器中正确呈现。
-->
# 多CPU上的高效训练

当单个CPU上的训练速度过慢时，我们可以使用多个CPU。本指南重点介绍基于PyTorch的DDP，以实现分布式CPU训练的高效性。

## PyTorch的Intel® oneCCL绑定

[Intel® oneCCL](https://github.com/oneapi-src/oneCCL)（集体通信库）是用于高效分布式深度学习训练的库，实现了诸如allreduce、allgather、alltoall等集体通信操作。有关oneCCL的更多信息，请参阅[oneCCL文档](https://spec.oneapi.com/versions/latest/elements/oneCCL/source/index.html)和[oneCCL规范](https://spec.oneapi.com/versions/latest/elements/oneCCL/source/index.html)。

模块`oneccl_bindings_for_pytorch`（1.12版本之前为`torch_ccl`）实现了PyTorch C10D ProcessGroup API，并可以作为外部ProcessGroup进行动态加载，目前仅适用于Linux平台。

详细信息请查看[oneccl_bind_pt](https://github.com/intel/torch-ccl)。

### 安装PyTorch的Intel® oneCCL绑定:

Wheel文件适用于以下Python版本:

| 扩展版本 | Python 3.6 | Python 3.7 | Python 3.8 | Python 3.9 | Python 3.10 || :------: | :--------: | :--------: | :--------: | :--------: | :---------: || 1.13.0  |            |     √      |     √      |     √      |      √      || 1.12.100|            |     √      |     √      |     √      |      √      || 1.12.0  |            |     √      |     √      |     √      |      √      || 1.11.0  |            |     √      |     √      |     √      |      √      || 1.10.0  |     √      |     √      |     √      |     √      |             |
```
pip install oneccl_bind_pt=={pytorch_version} -f https://developer.intel.com/ipex-whl-stable-cpu
```
其中`{pytorch_version}`应为您的PyTorch版本，例如1.13.0。有关[oneccl_bind_pt安装的更多方法](https://github.com/intel/torch-ccl)。oneCCL和PyTorch的版本必须匹配。
<Tip warning={true}>
oneccl_bindings_for_pytorch 1.12.0的预构建Wheel文件不适用于PyTorch 1.12.1（适用于PyTorch 1.12.0）PyTorch 1.12.1应与oneccl_bindings_for_pytorch 1.12.100配合使用
</Tip>

## Intel® MPI库

使用这个基于标准的MPI实现，在Intel®架构上提供灵活、高效、可扩展的集群通信。该组件是Intel® oneAPI HPC Toolkit的一部分。

oneccl_bindings_for_pytorch与MPI工具集一起安装。在使用之前需要设置环境变量。
适用于Intel® oneCCL >= 1.12.0

```
oneccl_bindings_for_pytorch_path=$(python -c "from oneccl_bindings_for_pytorch import cwd; print(cwd)")
source $oneccl_bindings_for_pytorch_path/env/setvars.sh
```

适用于Intel® oneCCL版本 < 1.12.0
```
torch_ccl_path=$(python -c "import torch; import torch_ccl; import os;  print(os.path.abspath(os.path.dirname(torch_ccl.__file__)))")
source $torch_ccl_path/env/setvars.sh
```

#### 安装IPEX:
IPEX为使用Float32和BFloat16进行CPU训练提供了性能优化，您可以参考[单CPU部分](./perf_train_cpu)。
以下"Trainer中的用法"以Intel® MPI库中的mpirun为例。

## Trainer中的用法
要在Trainer中启用多CPU分布式训练并使用ccl后端，用户应在命令参数中添加 **`--ddp_backend ccl`**。
让我们以[问答示例](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering)为例。

以下命令在一台Xeon节点上启用使用2个进程进行训练，每个进程在一个socket上运行。可以调整OMP_NUM_THREADS/CCL_WORKER_COUNT变量以获得最佳性能。
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
以下命令在两个Xeon上总共启用4个进程进行训练（node0和node1，以node0为主进程），ppn（每个节点的进程数）设置为2，每个进程在一个socket上运行。可以调整OMP_NUM_THREADS/CCL_WORKER_COUNT变量以获得最佳性能。

在node0中，您需要创建一个包含每个节点的IP地址（例如hostfile）的配置文件，并将该配置文件路径作为参数传递。
```shell script
 cat hostfile
 xxx.xxx.xxx.xxx #node0 ip
 xxx.xxx.xxx.xxx #node1 ip
```
现在，在node0中运行以下命令，将启用**4DDP**在node0和node1中使用BF16混合精度：```shell script
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
