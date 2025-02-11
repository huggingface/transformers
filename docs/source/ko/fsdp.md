<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 완전 분할 데이터 병렬 처리(FSDP) [[fully-sharded-data-parallel]]

[Fully Sharded Data Parallel (FSDP)](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)은 모델의 매개변수, 그레이디언트 및 옵티마이저 상태를 사용 가능한 GPU(작업자 또는 *랭크*라고도 함) 수에 따라 분할하는 데이터 병렬 처리 방식입니다. [DistributedDataParallel (DDP)](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)와 달리, FSDP는 각 GPU에 모델을 복제하기 때문에 메모리 사용량을 줄입니다. 이는 GPU 메모리 효율성을 향상시키며 적은 수의 GPU로 훨씬 더 큰 모델을 훈련할 수 있게 합니다. FSDP는 분산 환경에서의 훈련을 쉽게 관리할 수 있는 라이브러리인 Accelerate와 통합되어 있으며, 따라서 [`Trainer`] 클래스에서 사용할 수 있습니다.

시작하기 전에 Accelerate가 설치되어 있고 최소 PyTorch 2.1.0 이상의 버전이 설치되어 있는지 확인하세요.

```bash
pip install accelerate
```

## FSDP 구성 [[fsdp-configuration]]

시작하려면 [`accelerate config`](https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-config) 명령을 실행하여 훈련 환경에 대한 구성 파일을 생성하세요. Accelerate는 이 구성 파일을 사용하여 `accelerate config`에서 선택한 훈련 옵션에 따라 자동으로 올바른 훈련 환경을 설정합니다.

```bash
accelerate config
```

`accelerate config`를 실행하면 훈련 환경을 구성하기 위한 일련의 옵션들이 나타납니다. 이 섹션에서는 가장 중요한 FSDP 옵션 중 일부를 다룹니다. 다른 사용 가능한 FSDP 옵션에 대해 더 알아보고 싶다면 [fsdp_config](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.fsdp_config) 매개변수를 참조하세요.

### 분할 전략 [[sharding-strategy]]

FSDP는 여러 가지 분할 전략을 제공합니다:

* `FULL_SHARD` - 모델 매개변수, 그레이디언트 및 옵티마이저 상태를 작업자 간에 분할; 이 옵션을 선택하려면 `1`을 선택하세요
* `SHARD_GRAD_OP` - 그레이디언트 및 옵티마이저 상태를 작업자 간에 분할; 이 옵션을 선택하려면 `2`를 선택하세요
* `NO_SHARD` - 아무 것도 분할하지 않음 (DDP와 동일); 이 옵션을 선택하려면 `3`을 선택하세요
* `HYBRID_SHARD` - 각 작업자가 전체 복사본을 가지고 있는 상태에서 모델 매개변수, 그레이디언트 및 옵티마이저 상태를 작업자 내에서 분할; 이 옵션을 선택하려면 `4`를 선택하세요
* `HYBRID_SHARD_ZERO2` - 각 작업자가 전체 복사본을 가지고 있는 상태에서 그레이디언트 및 옵티마이저 상태를 작업자 내에서 분할; 이 옵션을 선택하려면 `5`를 선택하세요

이것은 `fsdp_sharding_strategy` 플래그로 활성화됩니다.

### CPU 오프로드 [[cpu-offload]]

사용하지 않는 매개변수와 그레이디언트를 CPU로 오프로드하여 더 많은 GPU 메모리를 절약하고 FSDP로도 충분하지 않은 큰 모델을 GPU에 적재할 수 있도록 할 수 있습니다. 이는 `accelerate config`를 실행할 때 `fsdp_offload_params: true`로 설정하여 활성화됩니다.

### 래핑 정책 [[wrapping-policy]]

FSDP는 네트워크의 각 레이어를 래핑하여 적용됩니다. 래핑은 일반적으로 중첩 방식으로 적용되며 각각 순방향으로 지나간 후 전체 가중치를 삭제하여 다음 레이어에서 사용할 메모리를 절약합니다. *자동 래핑* 정책은 이를 구현하는 가장 간단한 방법이며 코드를 변경할 필요가 없습니다. Transformer 레이어를 래핑하려면 `fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP`를 선택하고 래핑할 레이어를 지정하려면 `fsdp_transformer_layer_cls_to_wrap`를 선택하세요 (예: `BertLayer`).

또는 특정 매개변수 수를 초과할 경우 FSDP가 레이어에 적용되는 크기 기반 래핑 정책을 선택할 수 있습니다. 이는 `fsdp_wrap_policy: SIZE_BASED_WRAP` 및 `min_num_param`을 원하는 크기의 임계값으로 설정하여 활성화됩니다.

### 체크포인트 [[checkpointing]]

중간 체크포인트는 `fsdp_state_dict_type: SHARDED_STATE_DICT`로 저장해야 합니다. CPU 오프로드가 활성화된 랭크 0에서 전체 상태 딕셔너리를 저장하는 데 시간이 많이 걸리고, 브로드캐스팅 중 무기한 대기하여 `NCCL Timeout` 오류가 발생할 수 있기 때문입니다. [`~accelerate.Accelerator.load_state`] 메서드를 사용하여 분할된 상태 딕셔너리로 훈련을 재개할 수 있습니다.

```py
# 경로가 내재된 체크포인트
accelerator.load_state("ckpt")
```

그러나 훈련이 끝나면 전체 상태 딕셔너리를 저장해야 합니다. 분할된 상태 딕셔너리는 FSDP와만 호환되기 때문입니다.

```py
if trainer.is_fsdp_enabled:
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

trainer.save_model(script_args.output_dir)
```

### TPU [[tpu]]

[PyTorch XLA](https://pytorch.org/xla/release/2.1/index.html)는 TPU에 대한 FSDP 훈련을 지원하며 `accelerate config`로 생성된 FSDP 구성 파일을 수정하여 활성화할 수 있습니다. 위에서 지정한 분할 전략 및 래핑 옵션 외에도 아래에 표시된 매개변수를 파일에 추가할 수 있습니다.

```yaml
xla: True # PyTorch/XLA를 활성화하려면 True로 설정해야 합니다
xla_fsdp_settings: # XLA 특정 FSDP 매개변수
xla_fsdp_grad_ckpt: True # gradient checkpointing을 사용합니다
```

[`xla_fsdp_settings`](https://github.com/pytorch/xla/blob/2e6e183e0724818f137c8135b34ef273dea33318/torch_xla/distributed/fsdp/xla_fully_sharded_data_parallel.py#L128)는 FSDP에 대한 추가적인 XLA 특정 매개변수를 구성할 수 있게 합니다.

## 훈련 시작 [[launch-training]]

예시 FSDP 구성 파일은 다음과 같을 수 있습니다:

```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: true
  fsdp_sharding_strategy: 1
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_transformer_layer_cls_to_wrap: BertLayer
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

훈련을 시작하려면 [`accelerate launch`](https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-launch) 명령을 실행하세요. 이 때 전에 `accelerate config`로 생성한 구성 파일을 자동으로 사용합니다.

```bash
accelerate launch my-trainer-script.py
```

```bash
accelerate launch --fsdp="full shard" --fsdp_config="path/to/fsdp_config/ my-trainer-script.py
```

## 다음 단계 [[next-steps]]

FSDP는 매우 큰 모델을 훈련할 때 강력한 도구가 될 수 있으며, 여러 개의 GPU나 TPU를 사용할 수 있습니다. 모델 매개변수, 옵티마이저 및 그레이디언트 상태를 분할하고 비활성 상태일 때, CPU로 오프로드하면 FSDP는 대규모 훈련의 높은 연산 비용을 줄일 수 있습니다. 더 알아보고 싶다면 다음 자료가 도움이 될 수 있습니다:

* [FSDP](https://huggingface.co/docs/accelerate/usage_guides/fsdp)에 대한 더 깊이 있는 Accelerate 가이드를 따라가 보세요.
* [PyTorch의 완전 분할 데이터 병렬 처리 (FSDP) API를 소개합니다](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) 블로그 글을 읽어보세요.
* [FSDP를 사용하여 클라우드 TPU에서 PyTorch 모델 크기 조절하기](https://pytorch.org/blog/scaling-pytorch-models-on-cloud-tpus-with-fsdp/) 블로그 글을 읽어보세요.
