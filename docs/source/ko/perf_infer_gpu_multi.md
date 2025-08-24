<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 분산 추론[[distributed-inference]]

모델이 단일 GPU에 올라가지 않는 경우, [텐서 병렬 처리](./perf_train_gpu_many#tensor-parallelism)를 사용한 분산 추론이 도움이 될 수 있습니다. 텐서 병렬화는 모델을 여러 가속기(CUDA GPU, Intel XPU 등)에 분할하여 행렬 곱셈과 같은 계산을 병렬화합니다. 이를 통해 더 큰 모델을 메모리에 올릴 수 있으며, 각 가속기가 텐서의 일부를 처리하므로 추론 속도가 향상됩니다.

그러나 텐서 병렬화는 통신 오버헤드를 발생시키므로, 빠른 노드 내 통신을 활용할 수 있는 다중 가속기 환경에서 사용하는 것이 가장 효과적입니다. 다중 노드 학습 환경에서는 사용 사례에 따라 파이프라인 병렬화나 데이터 병렬화를 사용하는 것이 더 효율적일 수 있습니다.

> [!TIP]
> 텐서 병렬화에 대해 더 자세히 알아보려면 [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism)의 텐서 병렬화 섹션을 참조하세요.

아래 목록에서 텐서 병렬 처리를 기본적으로 지원하는 모델을 확인할 수 있습니다. 새로운 모델에 대한 지원을 추가하려면 GitHub 이슈나 풀 리퀘스트를 열어주세요.

<details>
<summary>지원되는 모델 보기</summary>

* [Cohere](./model_doc/cohere) 및 [Cohere 2](./model_doc/cohere2)
* [Gemma](./model_doc/gemma) 및 [Gemma 2](./model_doc/gemma2)
* [GLM](./model_doc/glm)
* [Granite](./model_doc/granite)
* [Llama](./model_doc/llama)
* [Mistral](./model_doc/mistral)
* [Mixtral](./model_doc/mixtral)
* [OLMo](./model_doc/olmo) 및 [OLMo2](./model_doc/olmo2)
* [Phi](./model_doc/phi) 및 [Phi-3](./model_doc/phi3)
* [Qwen2](./model_doc/qwen2), [Qwen2Moe](./model_doc/qwen2_moe), 및 [Qwen2-VL](./model_doc/qwen2_5_vl)
* [Starcoder2](./model_doc/starcoder2)

</details>

이 가이드는 Transformers에서 다양한 분할 전략을 사용하여 텐서 병렬화를 활성화하는 방법을 설명합니다.

## 모델 분할[[partitioning-a-model]]

Transformers는 `tp_plan`매개변수를 활용할 수 있는 모델에 대해 텐서 병렬 처리를 지원합니다. 모델 분할 방식은 두 가지가 있습니다.

- `auto` 텐서 병렬화 계획은 사전 정의된 구성을 기반으로 모델(위에 언급된 지원 모델)을 자동으로 분할합니다.
- 사용자 지정 분할 계획을 직접 정의하여 [~PreTrainedModel.from_pretrained] 메소드의 `tp_plan` 매개변수로 전달할 수 있습니다.

<hfoptions id="sharding">
<hfoption id="auto plan">

```py
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct" # 모든 가능한 전략을 시각화하기에 더 좋음
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"  # 적은 수의 GPU에 더 좋음

model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, tp_plan="auto")
print(model._tp_plan)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
prompt = "Can I help"
inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

# 분산 실행
outputs = model(inputs)
```

위의 추론 스크립트를 GPU당 4개 프로세스로 [torchrun](https://pytorch.org/docs/stable/elastic/run.html)에서 실행하세요.

```bash
torchrun --nproc-per-node 4 demo.py
```

</hfoption>
<hfoption id="manual plan">

각 레이어에 대한 텐서 병렬 계획을 `tp_plan`에 정의한 후 [`~PreTrainedModel.from_pretrained`]에 전달하세요. 아래 예시는 열 및 행 분할을 조합하여 사용합니다. 지원되는 다른 분할 전략은 [분할 전략](#partitioning-strategies) 섹션을 참고하세요.

> [!WARNING]
> 사용자 지정 분할 계획을 수동으로 지정하려면 모델 아키텍처와 분할 전략이 함께 상호 작용하는 방식에 대한 충분한 이해가 필요합니다. 분할 전략을 잘못 설정하면 모델이 매우 느려지거나, 오류가 발생하거나, 부정확한 결과를 낼 수 있습니다. 자세히 알아보려면 [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism)을 참고하세요.

```py
from transformers import AutoModelForCausalLM

tp_plan = {
    "model.layers.*.self_attn.q_proj": "colwise",
    "model.layers.*.self_attn.k_proj": "colwise",
    "model.layers.*.self_attn.v_proj": "colwise",
    "model.layers.*.self_attn.o_proj": "rowwise",
    ...
}

model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, tp_plan=tp_plan)
print(model._tp_plan)
```

</hfoption>
</hfoptions>

## 분할 전략[[partitioning-strategies]]

모든 분할 전략은 문자열을 전략 구현에 매핑하는 [`ParallelInterface`] 클래스에서 정의됩니다. 모든 전략은 [`~PreTrainedModel.from_pretrained`]의 `tp_plan`을 통해 설정되므로 이 클래스와 직접 상호 작용할 필요는 없지만, 어떤 전략을 사용할 수 있는지 확인할 때 유용합니다.

```py
class ParallelInterface(MutableMapping):
    """
    허용된 어텐션 함수를 추적하는 딕셔너리 같은 객체입니다. `register()` 호출로 새로운 어텐션 함수를 쉽게 추가할 수 있습니다. 
    모델이 기존 어텐션 함수(예: `sdpa`)를 로컬에서 덮어쓰려면 `modeling_<model>.py` 내부에서 이 클래스의 새 인스턴스를 선언하고 
    해당 인스턴스에서 선언해야 합니다.
    """
    _global_mapping = {
        "colwise": ColwiseParallel(),
        "rowwise": RowwiseParallel(),
        "colwise_rep": ColwiseParallel(output_layouts=Replicate()),
        "rowwise_rep": RowwiseParallel(input_layouts=Replicate()),
        "local_colwise": ColwiseParallel(use_dtensor=False),
        "local_rowwise": RowwiseParallel(use_dtensor=False),
        "local": IsolatedParallel(),
        "gather": GatherParallel(),
        "local_packed_rowwise": PackedRowwiseParallel(use_dtensor=False),
        "sequence_parallel": SequenceParallel(),
        "replicate": ReplicateParallel(),
    }
```

각 전략에 대해 자세히 알아보려면 아래 표를 참고하세요.

| 전략 | 설명 |
|---|---|
| `ColwiseParallel` | 가중치와 편향의 열 방향 분할. |
| `RowwiseParallel` | 가중치와 편향의 행 방향 분할. `nn.Embedding` 모듈 분할도 지원. |
| `SequenceParallel` | `LayerNorm`과 `Dropout` 레이어를 지원하는 시퀀스 병렬 구현. [RMSNorm](https://github.com/facebookresearch/llama/blob/main/llama/model.py#L34)의 Python 구현도 지원. |
| `PackedColwiseParallel` | 패킹된 가중치를 지원하는 `ColwiseParallel`의 변형(예: `up_proj`와 `gate_proj`를 함께 패킹). 자세한 내용은 [코드](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/tensor_parallel.py#L79-#L108)를 참조하세요. |
| `PackedRowwiseParallel` | 패킹된 가중치를 지원하는 `RowwiseParallel`의 변형([코드](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/tensor_parallel.py#L79-#L108) 참조). |
| `GatherParallel` | 기기 간 모듈의 출력을 수집. |
| `IsolatedParallel` | Mixture-of-Experts(MoE) 레이어의 전문가에 사용되어 다른 기기로부터 모듈을 격리. |
| `ReplicateParallel` | 부분적으로 분할된 모델로 인해 `torch.distributed` API가 중단되는 것을 방지하기 위해 모든 기기에 모듈을 복제. |

### 패킹된 전략[[packed-strategies]]

가중치 패킹은 여러 선형 레이어를 하나의 더 큰 레이어로 합치는 기법입니다. 패킹된 전략인 `PackedColwiseParallel`과 `PackedRowwiseParallel`은 패킹된 가중치를 분할하는 데 사용됩니다. 기본적인 `ColwiseParallel`이나 `RowwiseParallel`은 패킹된 가중치를 올바르게 분할하지 못합니다.

아래 예시는 `up_proj`와 `gate_proj`를 단일 `gate_up_proj` 모듈로 패킹하고 `gate_up_proj`를 분할하기 위해 `PackedRowwiseParallel` 전략이 필요합니다.

```python
class Llama4TextExperts(nn.Module):
    ...
    self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim))
```

배치 행렬 곱셈을 `forward` 패스에서 사용하여 `gate_up_proj` 모듈의 출력을 계산할 수 있습니다.

```python
def forward(self, hidden_states):
    ...
    gate_up = torch.bmm(hidden_states, self.gate_up_proj) # gate_up_proj 모듈의 출력 계산
    gate, up = gate_up.chunk(2, dim=-1) # 출력을 gate와 up으로 분할
```

> [!TIP]
> `Packed*`를 사용해야 하는 이유에 대한 시각적 표현은 [이 주석](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/tensor_parallel.py#L79-#L108)을 참고하세요.

### 로컬 전략[[local-strategies]]

로컬 전략(`local_colwise`, `local_rowwise`, `local_packed_rowwise`)은 [torch.chunk](https://docs.pytorch.org/docs/stable/generated/torch.chunk.html)와 같은 일부 연산에서 지원되지 않기 때문에 [DTensor](https://docs.pytorch.org/docs/stable/distributed.tensor.html)를 사용하지 않습니다. 대신 로컬 전략은 기본 [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html)를 사용하고 일부 분산 로직을 수동으로 수행합니다.

<!--
Readd this when I get the exact error message
> [!TIP]
> 사용자 정의 분할 전략을 사용하는데 `... is not supported` 오류로 작동하지 않는 경우, `local*` 전략을 사용해서 더 잘 작동하는지 시도해보세요.
-->

## 사용자 정의 분할 전략[[custom-partitioning-strategies]]

사용자 정의 분할 전략은 [`TensorParallelLayer`](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/tensor_parallel.py)를 상속하고 `partition_tensor`, `_prepare_input_fn`, `_prepare_output_fn`을 구현해야 합니다.

그런 다음 `tp_plan`에서 해당 전략을 지정했을 때 디스패칭 로직이 찾을 수 있도록 `ParallelInterface` 매핑에 등록해야 합니다.

아래 예시는 이 워크플로우로 `ColwiseParallel`을 구현하는 방법을 보여줍니다.

1. `TensorParallelLayer`를 상속합니다. `__init__` 메소드에서 입력 및 출력 텐서가 기기에 어떻게 배치되어야 하는지 설명하는 `input_layouts`과 `output_layouts`을 정의합니다. `desired_input_layouts` 속성은 입력이 기기에 어떻게 배치*되어야만* 하는지를 명시하는 데 사용됩니다.

    ```python
    class ColwiseParallel(TensorParallelLayer):
        def __init__(
            self,
            *,
            input_layouts: Optional[Placement] = None, # 이전 레이어에서 오는 입력 레이아웃
            output_layouts: Optional[Placement] = None, # 달성하고자 하는 출력 레이아웃
            use_local_output: bool = True, # 로컬 출력 사용 여부
            use_dtensor=True, # DTensor 사용 여부
        ):
            self.input_layouts = (input_layouts or Replicate(),) # 이전 레이어에서 오는 입력 분할
            self.output_layouts = (output_layouts or Shard(-1),) # 원하는 출력 분할
            self.desired_input_layouts = (Replicate(),) # 원하는 입력 분할, 입력은 GPU 간에 복제되어야 함
            self.use_local_output = use_local_output
            self.use_dtensor = use_dtensor
    ```

2. `partition_tensor`, `_prepare_input_fn`, `_prepare_output_fn` 메서드를 구현합니다.

    `partition_tensor` 메소드는 텐서를 분할하고 분할된 텐서로 `empty_param`을 채웁니다. 유틸리티 함수 `get_tensor_shard`를 사용하여 주어진 랭크에 대한 원본 매개변수의 올바른 분할을 얻고, 패킹된 가중치에 대해서는 `get_packed_weights`를 사용하세요.

    ```python
    def partition_tensor(
        self,
        param, # 매개변수의 전체 텐서
        empty_param, # 매개변수의 빈 텐서, 분할된 텐서로 채워짐
        param_type, # 매개변수 유형, `bias` 또는 `weight`
        param_casting_dtype, # 매개변수를 캐스팅할 유형
        to_contiguous, # 텐서를 연속적인 메모리 레이아웃으로 변환할지 여부
        rank, # 현재 기기의 랭크
        device_mesh, # 기기 메시
    ) -> nn.Parameter: # 분할된 매개변수 반환
        ...
    ```

    `_prepare_input_fn`과 `_prepare_output_fn` 메소드는 [사전 포워드](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_pre_hook.html) 및 [포워드](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html) 훅에서 사용됩니다. `__init__`에서 지정된 대로 입력과 출력을 원하는 레이아웃으로 재분배합니다.

    ```python
    def _prepare_input_fn(input_layouts, desired_input_layouts, mod, inputs, device_mesh):
        ...
        # 사용자 정의 로직 수행, DTensor로 캐스팅 등.
        ...
        return inputs.redistribute(placements=desired_input_layouts, device_mesh=device_mesh)
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        ...
        # 사용자 정의 로직 수행, DTensor로 캐스팅 등.
        ...
        return outputs.redistribute(placements=output_layouts, device_mesh=device_mesh)
    ```

3. `tp_plan`과 함께 사용할 수 있도록 전략을 [`ParallelInterface`]에 등록합니다.

    ```python
    from transformers.integrations.tensor_parallel import ParallelInterface

    ParallelInterface.register_strategy("colwise_custom", ColwiseParallel)
    tp_plan = {
        "model.layers.*.self_attn.q_proj": "colwise_custom",
        ...
    }
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, tp_plan=tp_plan)
    ```

## 벤치마크[[benchmarks]]

텐서 병렬화는 특히 큰 배치 크기나 긴 시퀀스를 가진 입력에 대한 추론 속도를 크게 향상시킬 수 있습니다.

시퀀스 길이가 512인 [Llama](./model_doc/llama)에서 단일 포워드 패스에 대한 예상 속도 향상 수치는 아래 차트를 참조하세요.

<div style="text-align: center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Meta-Llama-3-8B-Instruct%2C%20seqlen%20%3D%20512%2C%20python%2C%20w_%20compile.png">
</div>

## 설계 구현[[design-implementation]]

Transformers 텐서 병렬화 구현은 프레임워크에 구애받지 않지만, 구체적인 구현을 위해서는 [DeviceMesh](https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html)와 [torch.distributed](https://docs.pytorch.org/tutorials/beginner/dist_overview.html)의 [DTensor](https://docs.pytorch.org/docs/stable/distributed.tensor.html)에 의존하여 간단하고 확장 가능한 인터페이스를 제공합니다.

### DeviceMesh[[devicemesh]]

`DeviceMesh`를 함께 통신하는 기기들의 다차원 그리드로 상상해보세요. 병렬 처리 전략마다 각기 다른 통신 패턴이 필요하므로, 여러 하위 메시를 가진 `DeviceMesh`를 만들 수 있습니다.

```python
from torch.distributed.device_mesh import init_device_mesh

# 4개 GPU의 1D 메시 생성
device_mesh = init_device_mesh("cuda", (4,), mesh_dim_names=["tp"])
```

`torch.distributed`에서 정의된 대부분의 병렬화 전략은 메시 자체나 하위 메시에 적용할 수 있으며, 자동으로 통신 패턴을 처리합니다.

### DTensor[[dtensor]]

`DTensor`(분산 텐서)는 일반적인 텐서 연산 위에 분산 로직을 처리하는 텐서 하위 클래스입니다. 텐서 병렬화의 대부분의 모델 가중치는 `DTensor` 형태로 저장됩니다.

DTensor의 가장 중요한 부분은 `placement` 속성입니다. 이는 PyTorch에게 텐서가 `DeviceMesh`의 기기에 어떻게 배치되는지 알려주기 때문입니다. `placement` 속성은 다음 값을 가질 수 있습니다.

- `Shard(dimension)` - `DTensor`가 구성된 `DeviceMesh`에서 주어진 차원에 걸쳐 어떻게 분할되는지 나타냅니다. 아래 예시는 열 방향 분할을 위해 다양한 차원에 걸쳐 가중치를 분할하는 방법을 보여줍니다.

    ```python
    weight = ...
    weight = DTensor.from_local(weight, device_mesh["tp"], placements=[Shard(0)]) # 첫 번째(열 방향) 차원에 걸쳐 분할
    bias = ...
    bias = DTensor.from_local(bias, device_mesh["tp"], placements=[Shard(-1)]) # 유일한 차원에 걸쳐 분할
    ```

    이 예시는 행 방향 분할을 위해 여러 차원에 걸쳐 가중치를 분할하는 방법을 보여줍니다.

    ```python
    weight = ...
    weight = DTensor.from_local(weight, device_mesh["tp"], placements=[Shard(1)]) # 두 번째(행 방향) 차원에 걸쳐 분할
    bias = ...
    bias = DTensor.from_local(bias, device_mesh["tp"], placements=[Replicate()]) # 모든 GPU에 편향 복제
    ```

- `Replicate()` - `DTensor`가 `DeviceMesh`에 걸쳐 복제됨을 나타냅니다. 각 기기에 텐서의 전체 사본만 생성합니다.

    ```py
    bias = ...
    bias = DTensor.from_local(bias, device_mesh["tp"], placements=[Replicate()]) # 모든 GPU에 편향 복제
    ```

- `Partial()` - 텐서가 감소 연산을 기다리고 있는 상태임을 나타냅니다 (일반적으로 Transformers에서의 사용 사례와는 직접적인 관련이 적습니다).
