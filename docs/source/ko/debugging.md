<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 디버깅 [[debugging]]

## Multi-GPU 네트워크 문제 디버그 [[multigpu-network-issues-debug]]

`DistributedDataParallel` 및 다중 GPU를 사용하여 훈련하거나 추론할 때, 프로세스 및/또는 노드 간의 상호 통신 문제가 발생하는 경우, 다음 스크립트를 사용하여 네트워크 문제를 진단할 수 있습니다.

```bash
wget https://raw.githubusercontent.com/huggingface/transformers/main/scripts/distributed/torch-distributed-gpu-test.py
```

예를 들어, 2개의 GPU가 상호 작용하는 방식을 테스트하려면 다음을 실행하세요:

```bash
python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
```
두 프로세스가 서로 통신하고 GPU 메모리를 할당하는 경우, 각각 "OK" 상태를 출력합니다.

더 많은 GPU 또는 노드의 경우 스크립트의 인수를 조정하면 됩니다.

진단 스크립트 내에서 더 많은 세부 정보와 SLURM 환경에서 실행하는 방법에 대한 레시피를 찾을 수 있습니다.

추가적인 디버그 수준은 다음과 같이 `NCCL_DEBUG=INFO` 환경 변수를 추가하는 것입니다:

```bash
NCCL_DEBUG=INFO python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
```

이렇게 하면 NCCL 관련 디버그 정보가 많이 출력되며, 문제가 보고된 경우에는 인터넷에서 검색할 수 있습니다. 또는 출력을 해석하는 방법을 잘 모르는 경우 로그 파일을 이슈에 공유할 수 있습니다.



## 언더플로 및 오버플로 감지 [[underflow-and-overflow-detection]]


<Tip>

이 기능은 현재 PyTorch에서만 사용할 수 있습니다.

</Tip>

<Tip>

다중 GPU 훈련을 위해서는 DDP (`torch.distributed.launch`)가 필요합니다.

</Tip>

<Tip>

이 기능은 `nn.Module`을 기반으로 하는 모델과 함께 사용할 수 있습니다.

</Tip>

`loss=NaN`이 나타나거나 모델이 `inf` 또는 `nan`으로 인해 다른 이상한 동작을 하는 경우, 언더플로 또는 오버플로의 첫 번째 발생 위치와 그 원인을 파악해야 합니다. 다행히도 이를 자동으로 감지하는 특수 모듈을 활성화하여 쉽게 알아낼 수 있습니다.

[`Trainer`]를 사용하는 경우, 다음을 기존의 명령줄 인수에 추가하면 됩니다.

```bash
--debug underflow_overflow
```
또는 [`TrainingArguments`] 객체를 생성할 때 `debug="underflow_overflow"`를 전달합니다.

자체 훈련 루프나 다른 Trainer를 사용하는 경우, 다음과 같이 수행할 수 있습니다. 

```python
from transformers.debug_utils import DebugUnderflowOverflow

debug_overflow = DebugUnderflowOverflow(model)
```

[`~debug_utils.DebugUnderflowOverflow`]는 모델에 후크를 삽입하여 각 forward 호출 직후에 입력 및 출력 변수 및 해당 모듈의 가중치를 테스트합니다. 활성화나 가중치의 최소한 하나의 요소에서 `inf` 또는 `nan`이 감지되면 프로그램이 어설트되고 다음과 같은 보고서가 출력됩니다. (이 예제는 fp16 혼합 정밀도에서 `google/mt5-small`에서 캡처된 것입니다):

```
Detected inf/nan during batch_number=0
Last 21 forward frames:
abs min  abs max  metadata
                  encoder.block.1.layer.1.DenseReluDense.dropout Dropout
0.00e+00 2.57e+02 input[0]
0.00e+00 2.85e+02 output
[...]
                  encoder.block.2.layer.0 T5LayerSelfAttention
6.78e-04 3.15e+03 input[0]
2.65e-04 3.42e+03 output[0]
             None output[1]
2.25e-01 1.00e+04 output[2]
                  encoder.block.2.layer.1.layer_norm T5LayerNorm
8.69e-02 4.18e-01 weight
2.65e-04 3.42e+03 input[0]
1.79e-06 4.65e+00 output
                  encoder.block.2.layer.1.DenseReluDense.wi_0 Linear
2.17e-07 4.50e+00 weight
1.79e-06 4.65e+00 input[0]
2.68e-06 3.70e+01 output
                  encoder.block.2.layer.1.DenseReluDense.wi_1 Linear
8.08e-07 2.66e+01 weight
1.79e-06 4.65e+00 input[0]
1.27e-04 2.37e+02 output
                  encoder.block.2.layer.1.DenseReluDense.dropout Dropout
0.00e+00 8.76e+03 input[0]
0.00e+00 9.74e+03 output
                  encoder.block.2.layer.1.DenseReluDense.wo Linear
1.01e-06 6.44e+00 weight
0.00e+00 9.74e+03 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.DenseReluDense T5DenseGatedGeluDense
1.79e-06 4.65e+00 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.dropout Dropout
3.18e-04 6.27e+04 input[0]
0.00e+00      inf output
```

예제 출력은 간략성을 위해 중간 부분이 잘려 있습니다.

두 번째 열은 절대적으로 가장 큰 요소의 값이며, 따라서 마지막 몇 개의 프레임을 자세히 살펴보면 입력과 출력이 `1e4` 범위에 있음을 알 수 있습니다. 따라서 이 훈련은 `fp16` 혼합 정밀도로 수행될 때 가장 마지막 단계에서 오버플로우가 발생했습니다 (`fp16`에서 `inf` 이전의 가장 큰 숫자는 `64e3`입니다). `fp16` 아래에서 오버플로우를 피하기 위해서는 활성화는 `1e4`보다 훨씬 작아야 합니다. 왜냐하면 `1e4 * 1e4 = 1e8`이기 때문에 큰 활성화와의 행렬 곱은 수치적인 오버플로우 조건으로 이어질 것입니다.

추적의 맨 처음에서 어느 배치 번호에서 문제가 발생했는지 알 수 있습니다 (여기서 `Detected inf/nan during batch_number=0`은 문제가 첫 번째 배치에서 발생했음을 의미합니다).

각 보고된 프레임은 해당 프레임이 보고하는 해당 모듈에 대한 완전한 항목을 선언하며, 이 프레임만 살펴보면 다음과 같습니다.

```
                  encoder.block.2.layer.1.layer_norm T5LayerNorm
8.69e-02 4.18e-01 weight
2.65e-04 3.42e+03 input[0]
1.79e-06 4.65e+00 output
```

여기서 `encoder.block.2.layer.1.layer_norm`은 인코더의 두 번째 블록의 첫 번째 레이어에 대한 레이어 정규화를 의미하며, `forward`의 특정 호출은 `T5LayerNorm`입니다.

이 보고서의 마지막 몇 개 프레임을 살펴보겠습니다:

```
Detected inf/nan during batch_number=0
Last 21 forward frames:
abs min  abs max  metadata
[...]
                  encoder.block.2.layer.1.DenseReluDense.wi_0 Linear
2.17e-07 4.50e+00 weight
1.79e-06 4.65e+00 input[0]
2.68e-06 3.70e+01 output
                  encoder.block.2.layer.1.DenseReluDense.wi_1 Linear
8.08e-07 2.66e+01 weight
1.79e-06 4.65e+00 input[0]
1.27e-04 2.37e+02 output
                  encoder.block.2.layer.1.DenseReluDense.wo Linear
1.01e-06 6.44e+00 weight
0.00e+00 9.74e+03 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.DenseReluDense T5DenseGatedGeluDense
1.79e-06 4.65e+00 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.dropout Dropout
3.18e-04 6.27e+04 input[0]
0.00e+00      inf output
```

마지막 프레임은 `Dropout.forward` 함수에 대한 보고입니다. 첫 번째 항목은 유일한 입력을 나타내고 두 번째 항목은 유일한 출력을 나타냅니다. 이 함수가 `DenseReluDense` 클래스 내부의 `dropout` 속성에서 호출된 것을 볼 수 있습니다. 이는 첫 번째 레이어의 두 번째 블록에서 첫 번째 배치 중에 발생했다는 것을 알 수 있습니다. 마지막으로, 절대적으로 가장 큰 입력 요소는 `6.27e+04`이고 출력도 마찬가지로 `inf`입니다.

여기에서는 `T5DenseGatedGeluDense.forward`가 출력 활성화를 생성하는데, 절대적으로 가장 큰 값이 약 62.7K인 것을 볼 수 있습니다. 이 값은 fp16의 최대 제한인 64K에 매우 근접합니다. 다음 프레임에서는 일부 요소를 0으로 만든 후 가중치를 재정규화하는 `Dropout`이 있습니다. 이로 인해 절대 최대값이 64K를 초과하고 오버플로우(`inf`)가 발생합니다.

보시다시피, fp16 숫자의 경우 숫자가 매우 커질 때 이전 프레임을 살펴보아야 합니다.

보고서를 `models/t5/modeling_t5.py`의 코드와 일치시켜 보겠습니다.

```python
class T5DenseGatedGeluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gelu_act = ACT2FN["gelu_new"]

    def forward(self, hidden_states):
        hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states
```

이제 `dropout` 호출과 이전의 모든 호출을 쉽게 확인할 수 있습니다.

감지는 `forward` 후크에서 발생하므로, 이러한 보고서는 각 `forward`가 반환된 직후에 즉시 출력됩니다.

전체 보고서로 돌아가서 문제에 대한 조치 및 수정을 하려면, 숫자가 증가하기 시작한 몇 개의 프레임 위로 이동해서 여기서 `fp32` 모드로 전환해야 합니다. 이렇게 해야 숫자가 곱해지거나 합쳐질 때 오버플로우되지 않을 가능성이 높습니다. 물론 다른 해결책도 있을 수 있습니다. 예를 들어, `amp`가 활성화된 경우 일시적으로 끄고 원래의 `forward`를 도우미 래퍼로 이동한 후 다음과 같이 할 수 있습니다:

```python
def _forward(self, hidden_states):
    hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
    hidden_linear = self.wi_1(hidden_states)
    hidden_states = hidden_gelu * hidden_linear
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.wo(hidden_states)
    return hidden_states


import torch


def forward(self, hidden_states):
    if torch.is_autocast_enabled():
        with torch.cuda.amp.autocast(enabled=False):
            return self._forward(hidden_states)
    else:
        return self._forward(hidden_states)
```

자동 감지기는 전체 프레임의 입력과 출력에 대해서만 보고하므로, 어디를 살펴봐야 하는지 알면 특정 `forward` 함수의 중간 단계도 분석할 수 있습니다. 이 경우에는 `detect_overflow` 도우미 함수를 사용하여 원하는 위치에 감지기를 삽입할 수 있습니다. 예를 들어:

```python
from debug_utils import detect_overflow


class T5LayerFF(nn.Module):
    [...]

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        detect_overflow(forwarded_states, "after layer_norm")
        forwarded_states = self.DenseReluDense(forwarded_states)
        detect_overflow(forwarded_states, "after DenseReluDense")
        return hidden_states + self.dropout(forwarded_states)
```

여기서는 이를 추가하여 2개의 것을 추적하고 이제 `forwarded_states`의 `inf` 또는 `nan`이 중간에 감지되었는지를 추적합니다.

실제로 위의 예제에서 각 호출이 `nn.Module`이기 때문에 탐지기가 이미 이를 보고합니다. 로컬에서 직접 계산하는 경우 이렇게 수행한다고 가정해 봅시다.

또한, 자체 코드에서 디버거를 인스턴스화하는 경우 기본값에서 출력되는 프레임 수를 조정할 수 있습니다. 예를 들어:

```python
from transformers.debug_utils import DebugUnderflowOverflow

debug_overflow = DebugUnderflowOverflow(model, max_frames_to_save=100)
```

### 특정 배치의 절댓값 최소 및 최대 값 추적 [[specific-batch-absolute-min-and-max-value-tracing]]

동일한 디버깅 클래스는 언더플로우/오버플로우 감지 기능이 꺼진 상태에서 배치별 추적에도 사용할 수 있습니다.

예를 들어, 특정 배치의 각 `forward` 호출의 모든 구성 성분에 대한 절대 최솟값과 최댓값을 확인하고, 이를 배치 1과 3에 대해서만 수행하려면 다음과 같이 이 클래스를 인스턴스화합니다:

```python
debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1, 3])
```

그러면 이제 배치 1과 3 전체가 언더플로우/오버플로우 감지기와 동일한 형식으로 추적됩니다.

배치는 0부터 시작합니다.

이는 프로그램이 특정 배치 번호 이후에 오작동하기 시작하는 것을 알고 있는 경우에 유용합니다. 그렇기 때문에 해당 영역으로 바로 이동할 수 있습니다. 이런 구성에 대한 샘플 축소된 출력은 다음과 같습니다.

```
                  *** Starting batch number=1 ***
abs min  abs max  metadata
                  shared Embedding
1.01e-06 7.92e+02 weight
0.00e+00 2.47e+04 input[0]
5.36e-05 7.92e+02 output
[...]
                  decoder.dropout Dropout
1.60e-07 2.27e+01 input[0]
0.00e+00 2.52e+01 output
                  decoder T5Stack
     not a tensor output
                  lm_head Linear
1.01e-06 7.92e+02 weight
0.00e+00 1.11e+00 input[0]
6.06e-02 8.39e+01 output
                   T5ForConditionalGeneration
     not a tensor output

                  *** Starting batch number=3 ***
abs min  abs max  metadata
                  shared Embedding
1.01e-06 7.92e+02 weight
0.00e+00 2.78e+04 input[0]
5.36e-05 7.92e+02 output
[...]
```

여기에서는 모델의 forward 호출 수와 동일한 수의 프레임이 덤프되므로 많은 수의 프레임이 생성됩니다. 따라서 원하는 것일 수도 있고 아닐 수도 있습니다. 그러나 때로는 일반 디버거보다 디버깅 목적으로 더 쉽게 사용할 수 있습니다. 예를 들어, 문제가 배치 번호 150에서 시작하는 경우 149와 150의 추적을 덤프하고 숫자가 어디서부터 다르게 되었는지 비교할 수 있습니다.

또한, 훈련을 중지할 배치 번호를 지정할 수도 있습니다. 다음과 같이 지정할 수 있습니다.

```python
debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1, 3], abort_after_batch_num=3)
```
