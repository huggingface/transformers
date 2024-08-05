<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 다중 CPU에서 효율적으로 훈련하기 [[efficient-training-on-multiple-cpus]]

하나의 CPU에서 훈련하는 것이 너무 느릴 때는 다중 CPU를 사용할 수 있습니다. 이 가이드는 PyTorch 기반의 DDP를 사용하여 분산 CPU 훈련을 효율적으로 수행하는 방법에 대해 설명합니다.

## PyTorch용 Intel® oneCCL 바인딩 [[intel-oneccl-bindings-for-pytorch]]

[Intel® oneCCL](https://github.com/oneapi-src/oneCCL) (collective communications library)은 allreduce, allgather, alltoall과 같은 집합 통신(collective communications)을 구현한 효율적인 분산 딥러닝 훈련을 위한 라이브러리입니다. oneCCL에 대한 자세한 정보는 [oneCCL 문서](https://spec.oneapi.com/versions/latest/elements/oneCCL/source/index.html)와 [oneCCL 사양](https://spec.oneapi.com/versions/latest/elements/oneCCL/source/index.html)을 참조하세요.

`oneccl_bindings_for_pytorch` 모듈 (`torch_ccl`은 버전 1.12 이전에 사용)은 PyTorch C10D ProcessGroup API를 구현하며, 외부 ProcessGroup로 동적으로 가져올 수 있으며 현재 Linux 플랫폼에서만 작동합니다.

[oneccl_bind_pt](https://github.com/intel/torch-ccl)에서 더 자세한 정보를 확인하세요.

### PyTorch용 Intel® oneCCL 바인딩 설치: [[intel-oneccl-bindings-for-pytorch-installation]]

다음 Python 버전에 대한 Wheel 파일을 사용할 수 있습니다.

| Extension Version | Python 3.6 | Python 3.7 | Python 3.8 | Python 3.9 | Python 3.10 |
| :---------------: | :--------: | :--------: | :--------: | :--------: | :---------: |
| 1.13.0            |            | √          | √          | √          | √           |
| 1.12.100          |            | √          | √          | √          | √           |
| 1.12.0            |            | √          | √          | √          | √           |
| 1.11.0            |            | √          | √          | √          | √           |
| 1.10.0            | √          | √          | √          | √          |             |

```bash
pip install oneccl_bind_pt=={pytorch_version} -f https://developer.intel.com/ipex-whl-stable-cpu
```
`{pytorch_version}`은 1.13.0과 같이 PyTorch 버전을 나타냅니다.
[oneccl_bind_pt 설치](https://github.com/intel/torch-ccl)에 대한 더 많은 접근 방법을 확인해 보세요.
oneCCL과 PyTorch의 버전은 일치해야 합니다.

<Tip warning={true}>

oneccl_bindings_for_pytorch 1.12.0 버전의 미리 빌드된 Wheel 파일은 PyTorch 1.12.1과 호환되지 않습니다(PyTorch 1.12.0용입니다).
PyTorch 1.12.1은 oneccl_bindings_for_pytorch 1.12.10 버전과 함께 사용해야 합니다.

</Tip>

## Intel® MPI 라이브러리 [[intel-mpi-library]]
이 표준 기반 MPI 구현을 사용하여 Intel® 아키텍처에서 유연하고 효율적이며 확장 가능한 클러스터 메시징을 제공하세요. 이 구성 요소는 Intel® oneAPI HPC Toolkit의 일부입니다.

oneccl_bindings_for_pytorch는 MPI 도구 세트와 함께 설치됩니다. 사용하기 전에 환경을 소스로 지정해야 합니다.

Intel® oneCCL 버전 1.12.0 이상인 경우
```bash
oneccl_bindings_for_pytorch_path=$(python -c "from oneccl_bindings_for_pytorch import cwd; print(cwd)")
source $oneccl_bindings_for_pytorch_path/env/setvars.sh
```

Intel® oneCCL 버전이 1.12.0 미만인 경우
```bash
torch_ccl_path=$(python -c "import torch; import torch_ccl; import os;  print(os.path.abspath(os.path.dirname(torch_ccl.__file__)))")
source $torch_ccl_path/env/setvars.sh
```

#### IPEX 설치: [[ipex-installation]]

IPEX는 Float32와 BFloat16을 모두 사용하는 CPU 훈련을 위한 성능 최적화를 제공합니다. [single CPU section](./perf_train_cpu)을 참조하세요.


이어서 나오는 "Trainer에서의 사용"은 Intel® MPI 라이브러리의 mpirun을 예로 들었습니다.


## Trainer에서의 사용 [[usage-in-trainer]]
Trainer에서 ccl 백엔드를 사용하여 멀티 CPU 분산 훈련을 활성화하려면 명령 인수에 **`--ddp_backend ccl`**을 추가해야 합니다.

[질의 응답 예제](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering)를 사용한 예를 살펴보겠습니다.


다음 명령은 한 Xeon 노드에서 2개의 프로세스로 훈련을 활성화하며, 각 소켓당 하나의 프로세스가 실행됩니다. OMP_NUM_THREADS/CCL_WORKER_COUNT 변수는 최적의 성능을 위해 조정할 수 있습니다.
```shell script
 export CCL_WORKER_COUNT=1
 export MASTER_ADDR=127.0.0.1
 mpirun -n 2 -genv OMP_NUM_THREADS=23 \
 python3 run_qa.py \
 --model_name_or_path google-bert/bert-large-uncased \
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
다음 명령은 두 개의 Xeon(노드0 및 노드1, 주 프로세스로 노드0을 사용)에서 총 4개의 프로세스로 훈련을 활성화하며, 각 소켓당 하나의 프로세스가 실행됩니다. OMP_NUM_THREADS/CCL_WORKER_COUNT 변수는 최적의 성능을 위해 조정할 수 있습니다.

노드0에서는 각 노드의 IP 주소를 포함하는 구성 파일(예: hostfile)을 생성하고 해당 구성 파일 경로를 인수로 전달해야 합니다.
```shell script
 cat hostfile
 xxx.xxx.xxx.xxx #node0 ip
 xxx.xxx.xxx.xxx #node1 ip
```
이제 노드0에서 다음 명령을 실행하면 **4DDP**가 노드0 및 노드1에서 BF16 자동 혼합 정밀도로 활성화됩니다.
```shell script
 export CCL_WORKER_COUNT=1
 export MASTER_ADDR=xxx.xxx.xxx.xxx #node0 ip
 mpirun -f hostfile -n 4 -ppn 2 \
 -genv OMP_NUM_THREADS=23 \
 python3 run_qa.py \
 --model_name_or_path google-bert/bert-large-uncased \
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
