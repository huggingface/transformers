<!---
Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->


# 훈련용 사용자 맞춤형 하드웨어 [[custom-hardware-for-training]]

모델 훈련과 추론에 사용하는 하드웨어는 성능에 큰 영향을 미칠 수 있습니다. GPU에 대해 자세히 알아보려면, Tim Dettmer의 훌륭한 블로그 포스트를 확인해보세요. [블로그 포스트 링크](https://timdettmers.com/2020/09/07/which-gpu-for-deep-learning/) (영어로 작성됨).

GPU 설정에 대한 실용적인 조언을 살펴보겠습니다.

## GPU [[gpu]]
더 큰 모델을 훈련시킬 때는 기본적으로 세 가지 옵션이 있습니다:

- 더 큰 GPU
- 더 많은 GPU
- 더 많은 CPU 및 NVMe ([DeepSpeed-Infinity](../en/main_classes/deepspeed#nvme-support)를 통한 오프로드(offload))

우선, 하나의 GPU만 사용하는 경우부터 시작해봅시다.

### 전원 공급과 냉각 [[power-and-cooling]]

비싼 고성능 GPU를 구매한 경우, 올바른 전원 공급과 충분한 냉각을 제공해야 합니다.

**전원 공급**:

일부 고성능 소비자용 GPU는 2개 혹은 가끔가다 3개의 PCI-E 8핀 전원 소켓이 있습니다. 카드에 있는 소켓 수만큼 독립적인 12V PCI-E 8핀 케이블이 연결되어 있는지 확인하세요. 같은 케이블의 한쪽 끝에 있는 2개의 스플릿(또는 피그테일(pigtail) 케이블)을 사용하지 마세요. 즉, GPU에 2개의 소켓이 있다면, PSU(전원 공급 장치)에서 카드로 연결되는 2개의 PCI-E 8핀 케이블이 필요하며, 끝에 2개의 PCI-E 8핀 커넥터가 있는 케이블이 필요하지 않습니다! 그렇지 않으면 카드의 전체 성능을 제대로 발휘하지 못할 수 있습니다.

각각의 PCI-E 8핀 전원 케이블은 PSU 쪽의 12V 레일에 연결되어야 하며 최대 150W의 전력을 공급할 수 있습니다.

일부 다른 GPU는 PCI-E 12핀 커넥터를 사용하며, 이러한 커넥터는 최대 500W-600W의 전력을 공급할 수 있습니다.

저가형 GPU는 6핀 커넥터를 사용하며, 최대 75W의 전력을 공급합니다.

또한 GPU가 안정적인 전압을 받을 수 있도록 고급 PSU를 선택해야 합니다. 일부 저품질의 PSU는 GPU가 최고 성능으로 동작하기 위해 필요한 전압을 안정적으로 공급하지 못할 수 있습니다.

물론, PSU는 GPU에 전원을 공급하기에 충분한 여분의 전력 용량을 가져야 합니다.

**냉각**:

GPU가 과열되면 성능이 저하되고 최대 성능을 발휘하지 못할 수 있으며, 너무 뜨거워지면 중지될 수 있습니다.

GPU가 과열될 때 정확한 적정 온도를 알기 어려우나, 아마도 +80℃ 미만이면 좋지만 더 낮을수록 좋습니다. 70℃-75℃ 정도가 훌륭한 온도 범위입니다. 성능 저하가 발생하기 시작하는 온도는 대략 84℃-90℃ 정도일 것입니다. 하지만 성능 저하 이외에도 지속적으로 매우 높은 온도는 GPU 수명을 단축시킬 수 있습니다.

이어서, 여러 개의 GPU를 사용할 때 가장 중요한 측면 중 하나인 GPU 간 연결 방식을 살펴보겠습니다.

### 다중 GPU 연결 방식 [[multigpu-connectivity]]

다중 GPU를 사용하는 경우 GPU 간의 연결 방식은 전체 훈련 시간에 큰 영향을 미칠 수 있습니다. 만약 GPU가 동일한 물리적 노드에 있을 경우, 다음과 같이 확인할 수 있습니다:

```bash
nvidia-smi topo -m
```

만약 NVLink로 연결된 듀얼 GPU 환경이라면, 다음과 같은 결과를 확인할 수 있습니다:

```
        GPU0    GPU1    CPU Affinity    NUMA Affinity
GPU0     X      NV2     0-23            N/A
GPU1    NV2      X      0-23            N/A
```

NVLink를 지원하지 않는 다른 환경의 경우에는 다음과 같은 결과를 확인할 수 있습니다:
```
        GPU0    GPU1    CPU Affinity    NUMA Affinity
GPU0     X      PHB     0-11            N/A
GPU1    PHB      X      0-11            N/A
```

이 결과에는 다음과 같은 범례가 포함되어 있습니다:

```
  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```

따라서 첫 번째 결과의 `NV2`는 GPU가 2개의 NVLink로 연결되어 있다는 것을 나타내고, 두 번째 결과의 `PHB`는 일반적인 소비자용 PCIe+브릿지 설정을 가지고 있다는 것을 나타냅니다.

설정에서 어떤 유형의 연결 방식을 가지고 있는지 확인하세요. 일부 연결 방식은 GPU 간 통신을 더 빠르게 만들 수 있으며(NVLink와 같이), 어떤 연결 방식은 더 느리게 만들 수 있습니다(PHB와 같이).

사용하는 확장성 솔루션의 종류에 따라 연결 속도가 주요한 영향을 미칠 수도 있고 미미한 영향을 미칠 수도 있습니다. DDP와 같이 GPU가 거의 동기화하지 않아도 되는 경우, 연결 속도가 느려도 큰 영향을 받지 않습니다. 반면 ZeRO-DP와 같이 GPU간 통신이 많이 필요한 경우, 더 빠른 훈련을 위해서는 더 빠른 연결 속도가 중요합니다.

#### NVLink [[nvlink]]

[NVLink](https://en.wikipedia.org/wiki/NVLink)는 Nvidia에서 개발한 유선 기반의 직렬 다중 레인 근거리 통신 링크입니다.

새로운 세대의 NVLink는 더 빠른 대역폭을 제공합니다. [Nvidia Ampere GA102 GPU Architecture](https://www.nvidia.com/content/dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf)에서 아래와 같은 정보를 확인하실 수 있습니다:

> 3세대 NVLink®
> GA102 GPU는 4개의 x4 링크를 포함하는 NVIDIA의 3세대 NVLink 인터페이스를 활용하며,
> 각 링크는 두 개의 GPU 간에 각 방향으로 초당 14.0625GB의 대역폭을 제공합니다.
> 4개의 링크는 각 방향에 초당 56.25GB의 대역폭을 제공하며, 두 개의 GPU 간에는 초당 112.5GB의 총 대역폭을 제공합니다.
> 두 개의 RTX 3090 GPU를 NVLink를 사용해 SLI로 연결할 수 있습니다.
> (3-Way 및 4-Way SLI 구성은 지원되지 않음에 유의하세요.)


따라서 `nvidia-smi topo -m`의 결과에서 `NVX`의 값이 높을수록 더 좋습니다. 세대는 GPU 아키텍처에 따라 다를 수 있습니다.

그렇다면, openai-community/gpt2를 작은 wikitext 샘플로 학습시키는 예제를 통해, NVLink가 훈련에 어떤 영향을 미치는지 살펴보겠습니다.

결과는 다음과 같습니다:


| NVlink | Time |
| -----  | ---: |
| Y      | 101s |
| N      | 131s |


NVLink 사용 시 훈련이 약 23% 더 빠르게 완료됨을 확인할 수 있습니다. 두 번째 벤치마크에서는 `NCCL_P2P_DISABLE=1`을 사용하여 NVLink를 사용하지 않도록 설정했습니다.

전체 벤치마크 코드와 결과는 다음과 같습니다:

```bash
# DDP w/ NVLink

rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 torchrun \
--nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py --model_name_or_path openai-community/gpt2 \
--dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train \
--output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 101.9003, 'train_samples_per_second': 1.963, 'epoch': 0.69}

# DDP w/o NVLink

rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 torchrun \
--nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py --model_name_or_path openai-community/gpt2 \
--dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train
--output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 131.4367, 'train_samples_per_second': 1.522, 'epoch': 0.69}
```

하드웨어: 각각 2개의 TITAN RTX 24GB + 2개의 NVLink (`NV2` in `nvidia-smi topo -m`)
소프트웨어: `pytorch-1.8-to-be` + `cuda-11.0` / `transformers==4.3.0.dev0`
