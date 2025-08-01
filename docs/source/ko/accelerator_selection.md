<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

<!-- TODO: this was copied from the korean version of `gpu_selection.md`, and is not up to date with the english version of `accelerator_selection.md` -->

# GPU 선택하기 [[gpu-selection]]

분산 학습 과정에서 사용할 GPU의 개수와 순서를 정할 수 있습니다. 이 방법은 서로 다른 연산 성능을 가진 GPU가 있을 때 더 빠른 GPU를 우선적으로 사용하거나, 사용 가능한 GPU 중 일부만 선택하여 활용하고자 할 때 유용합니다. 이 선택 과정은 [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)과 [DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)에서 모두 작동합니다. Accelerate나 [DeepSpeed 통합](./main_classes/deepspeed)은 필요하지 않습니다.

이 가이드는 사용할 GPU의 개수를 선택하는 방법과 사용 순서를 설정하는 방법을 설명합니다.

## GPU 개수 지정 [[number-of-gpus]]

예를 들어, GPU가 4개 있고 그중 처음 2개만 사용하려는 경우, 아래 명령어를 실행하세요.

<hfoptions id="select-gpu">
<hfoption id="torchrun">

사용할 GPU 개수를 정하기 위해 `--nproc_per_node` 옵션을 사용하세요.

```bash
torchrun --nproc_per_node=2  trainer-program.py ...
```

</hfoption>
<hfoption id="Accelerate">

사용할 GPU 개수를 정하기 위해 `--num_processes` 옵션을 사용하세요.

```bash
accelerate launch --num_processes 2 trainer-program.py ...
```

</hfoption>
<hfoption id="DeepSpeed">

사용할 GPU 개수를 정하기 위해 `--num_gpus` 옵션을 사용하세요.

```bash
deepspeed --num_gpus 2 trainer-program.py ...
```

</hfoption>
</hfoptions>

### GPU 순서 [[order-of-gpus]]

사용할 GPU와 그 순서를 지정하려면 `CUDA_VISIBLE_DEVICES` 환경 변수를 설정하세요. 가장 쉬운 방법은 `~/bashrc` 또는 다른 시작 설정 파일에서 해당 변수를 설정하는 것입니다. `CUDA_VISIBLE_DEVICES`는 사용할 GPU를 매핑하는 데 사용됩니다. 예를 들어, GPU가 4개 (0, 1, 2, 3) 있고 그중에서 0번과 2번 GPU만 사용하고 싶을 경우, 다음과 같이 설정할 수 있습니다:

```bash
CUDA_VISIBLE_DEVICES=0,2 torchrun trainer-program.py ...
```

오직 두 개의 물리적 GPU(0, 2)만 PyTorch에서 "보이는" 상태가 되며, 각각 `cuda:0`과 `cuda:1`로 매핑됩니다. 또한, GPU 사용 순서를 반대로 설정할 수도 있습니다. 이 경우, GPU 0이 `cuda:1`, GPU 2가 `cuda:0`으로 매핑됩니다."

```bash
CUDA_VISIBLE_DEVICES=2,0 torchrun trainer-program.py ...
```

`CUDA_VISIBLE_DEVICES` 환경 변수를 빈 값으로 설정하여 GPU가 없는 환경을 만들 수도 있습니다.

```bash
CUDA_VISIBLE_DEVICES= python trainer-program.py ...
```

> [!WARNING]
> 다른 환경 변수와 마찬가지로, CUDA_VISIBLE_DEVICES를 커맨드 라인에 추가하는 대신 export하여 설정할 수도 있습니다. 그러나 이 방식은 환경 변수가 어떻게 설정되었는지를 잊어버릴 경우, 잘못된 GPU를 사용할 위험이 있기 때문에 권장하지 않습니다. 특정 학습 실행에 대해 동일한 커맨드 라인에서 환경 변수를 설정하는 것이 일반적인 방법입니다.

`CUDA_DEVICE_ORDER`는 GPU의 순서를 제어하는 데 사용할 수 있는 대체 환경 변수입니다. 이 변수를 사용하면 다음과 같은 방식으로 GPU 순서를 지정할 수 있습니다:

1. NVIDIA 및 AMD GPU의 PCIe 버스 ID는 각각 [nvidia-smi](https://developer.nvidia.com/nvidia-system-management-interface)와 [rocm-smi](https://rocm.docs.amd.com/projects/rocm_smi_lib/en/latest/.doxygen/docBin/html/index.html)의 순서와 일치합니다.

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

2. GPU 연산 능력

```bash
export CUDA_DEVICE_ORDER=FASTEST_FIRST
```

The `CUDA_DEVICE_ORDER` is especially useful if your training setup consists of an older and newer GPU, where the older GPU appears first, but you cannot physically swap the cards to make the newer GPU appear first. In this case, set `CUDA_DEVICE_ORDER=FASTEST_FIRST` to always use the newer and faster GPU first (`nvidia-smi` or `rocm-smi` still reports the GPUs in their PCIe order). Or you could also set `export CUDA_VISIBLE_DEVICES=1,0`.

`CUDA_DEVICE_ORDER`는 구형 GPU와 신형 GPU가 혼합된 환경에서 특히 유용합니다. 예를 들어, 구형 GPU가 먼저 표시되지만 물리적으로 교체할 수 없는 경우, `CUDA_DEVICE_ORDER=FASTEST_FIRST`를 설정하면 항상 신형 및 더 빠른 GPU를 우선적으로 사용(nvidia-smi 또는 rocm-smi는 PCIe 순서대로 GPU를 표시함)할 수 있습니다. 또는, `export CUDA_VISIBLE_DEVICES=1,0`을 설정하여 GPU 사용 순서를 직접 지정할 수도 있습니다.
