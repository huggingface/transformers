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

# 가속기 선택

분산 학습 중에 PyTorch가 어떤 가속기(CUDA, XPU, MPS, HPU 등)를 어떤 순서로 인식할지 제어할 수 있습니다. 더 빠른 장치를 우선적으로 사용하거나, 사용 가능한 하드웨어 중 일부만 학습에 사용하도록 제한할 수 있습니다. 이 방법은 [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)과 [DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) 모두에서 작동하며, Accelerate나 [DeepSpeed 통합](./main_classes/deepspeed)이 필요하지 않습니다.

## 가속기 순서

하드웨어별 환경 변수를 사용해 가속기를 선택하고 순서를 설정하세요. 실행할 때마다 명령줄에서 설정하거나 `~/.bashrc` 또는 다른 시작 설정 파일에 추가할 수 있습니다.

> [!WARNING]
> `export`로 환경 변수를 설정하는 방식은 피하세요. 환경 변수가 어떻게 설정되었는지 잊어버리면 잘못된 가속기에서 별도 알림 없이 학습하게 될 수 있습니다. 학습 실행 명령과 같은 명령줄에서 환경 변수를 설정하세요.

예를 들어, 네 개의 가속기 중 0번과 2번을 선택하려면 다음과 같이 실행하세요.

<hfoptions id="accelerator-type">
<hfoption id="CUDA">

```cli
CUDA_VISIBLE_DEVICES=0,2 torchrun trainer-program.py ...
```

PyTorch는 0번과 2번 GPU만 인식하며, 이를 각각 `cuda:0`과 `cuda:1`로 매핑합니다. 순서를 반대로 바꾸려면(2번 GPU를 `cuda:0`으로, 0번 GPU를 `cuda:1`로 사용하려면) 다음과 같이 실행하세요.

```cli
CUDA_VISIBLE_DEVICES=2,0 torchrun trainer-program.py ...
```

GPU 없이 실행하려면 다음과 같이 실행하세요.

```cli
CUDA_VISIBLE_DEVICES= python trainer-program.py ...
```

`CUDA_DEVICE_ORDER`로 CUDA 장치의 순서를 제어할 수 있습니다.

- PCIe 버스 ID 순서로 정렬(`nvidia-smi`와 일치):

    ```cli
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    ```

- 연산 능력(compute capability) 순서로 정렬(가장 빠른 장치부터):

    ```cli
    export CUDA_DEVICE_ORDER=FASTEST_FIRST
    ```

</hfoption>
<hfoption id="Intel XPU">

```cli
ZE_AFFINITY_MASK=0,2 torchrun trainer-program.py ...
```

PyTorch는 0번과 2번 XPU만 인식하며, 이를 각각 `xpu:0`과 `xpu:1`로 매핑합니다. 순서를 반대로 바꾸려면(2번 XPU를 `xpu:0`으로, 0번 XPU를 `xpu:1`로 사용하려면) 다음과 같이 실행하세요.

```cli
ZE_AFFINITY_MASK=2,0 torchrun trainer-program.py ...
```

Intel XPU의 순서는 다음과 같이 제어할 수 있습니다.

```cli
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
```

Intel XPU의 장치 열거와 정렬에 대한 자세한 내용은 [Level Zero](https://github.com/oneapi-src/level-zero/blob/master/README.md?plain=1#L87) 문서를 참고하세요.

</hfoption>
</hfoptions>
