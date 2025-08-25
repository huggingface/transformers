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

# 옵티마이저[[optimizers]]

Transformers는 AdamW 및 AdaFactor와 같은 두 가지 기본 옵티마이저를 제공합니다. 또한, 보다 특화된 옵티마이저와의 통합도 지원합니다. 원하는 옵티마이저를 제공하는 라이브러리를 설치한 후, [`TrainingArguments`]의 `optim` 파라미터에 해당 옵티마이저명을 지정하시면 됩니다. 

이 가이드에서는 아래에 제시된 [`TrainingArguments`]와 함께 [`Trainer`]에서 이러한 옵티마이저를 사용하는 방법을 안내합니다. 

```py
import torch
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM, Trainer

args = TrainingArguments(
    output_dir="./test-optimizer",
    max_steps=1000,
    per_device_train_batch_size=4,
    logging_strategy="steps",
    logging_steps=1,
    learning_rate=2e-5,
    save_strategy="no",
    run_name="optimizer-name",
)
```

## APOLLO[[apollo]]

```bash
pip install apollo-torch
```

[Approximated Gradient Scaling for Memory Efficient LLM Optimization (APOLLO)](https://github.com/zhuhanqing/APOLLO) 는 사전 학습과 미세 조정 모두에 대해 전체 파라미터 학습을 지원하는, 메모리 효율적인 옵티마이저입니다. 이 옵티마이저는 SGD와 유사한 메모리 효율성으로 AdamW 수준의 성능을 유지합니다. 극한의 메모리 효율성이 필요하다면 APOLLO의 rank 1 변형인 APOLLO-Mini를 사용할 수 있습니다. APOLLO 옵티마이저는 다음과 같은 특징을 지원합니다. 

* 초저랭크(rank) 효율성. [GaLoRE](./trainer#galore)보다 훨씬 낮은 랭크를 사용할 수 있으며, 랭크 1로도 충분합니다.
* 고비용 SVD 연산 회피. APOLLO는 학습 중단(training stalls)을 피하기 위해 무작위 투영(random projections)을 활용합니다.

학습할 레이어를 지정하려면 `optim_target_modules` 파라미터를 사용하세요. 

```diff
import torch
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./test-apollo",
    max_steps=100,
    per_device_train_batch_size=2,
+   optim="apollo_adamw",
+   optim_target_modules=[r".*.attn.*", r".*.mlp.*"],
    logging_strategy="steps",
    logging_steps=1,
    learning_rate=2e-5,
    save_strategy="no",
    run_name="apollo_adamw",
)
```

추가적인 학습 옵션이 필요하다면, `optim_args`를 사용하여 `rank`, `scale` 등과 같은 하이퍼파라미터를 설정할 수 있습니다. 사용 가능한 하이퍼파라미터 목록은 아래 표를 참고하세요.

> [!TIP]
> `scale` 파라미터는 `n/r`으로 설정할 수 있습니다. 이때, `n`은 원본 공간 차원이고 `r`은 저랭크(low-rank) 공간 차원입니다. `scale`을 기본값으로 유지하면서 학습률만 조정해도 비슷한 효과를 얻을 수 있습니다.

| 매개 변수 | 설명 | APOLLO | APOLLO-Mini |
|---|---|---|---|
| rank | 그래디언트 스케일링을 위한 보조 부분 공간(sub-space)의 랭크 | 256 | 1 |
| scale_type | 스케일링 인자(factor)를 적용하는 방법 | `channel` (채널별 스케일링) | `tensor` (텐서별 스케일링) |
| scale | 그래디언트 업데이트를 조정하여 학습을 안정화 | 1.0 | 128 |
| update_proj_gap | 투영 행렬(projection matrices)을 업데이트하기 전 단계(step) 수 | 200 | 200 |
| proj | 투영(projection) 유형 | `random` | `random` |

아래 예시는 APOLLO-Mini 옵티마이저를 활성화하는 방법입니다.

```py
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./test-apollo_mini",
    max_steps=100,
    per_device_train_batch_size=2,
    optim="apollo_adamw",
    optim_target_modules=[r".*.attn.*", r".*.mlp.*"],
    optim_args="proj=random,rank=1,scale=128.0,scale_type=tensor,update_proj_gap=200",
)
```

## GrokAdamW[[grokadamw]]

```bash
pip install grokadamw
```

[GrokAdamW](https://github.com/cognitivecomputations/grokadamw)는 *grokking* 현상(기울기가 천천히 변화해 일반화가 지연되는 현상)에서 성능이 향상되는 모델들에게 적합하도록 설계된 옵티마이저입니다. GrokAdamW는 더 뛰어난 성능과 안정성을 위해 고급 최적화 기술이 필요한 모델에 특히 유용합니다. 

```diff
import torch
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./test-grokadamw",
    max_steps=1000,
    per_device_train_batch_size=4,
+   optim="grokadamw",
    logging_strategy="steps",
    logging_steps=1,
    learning_rate=2e-5,
    save_strategy="no",
    run_name="grokadamw",
)
```

## LOMO[[lomo]]

```bash
pip install lomo-optim
```

[Low-Memory Optimization (LOMO)](https://github.com/OpenLMLab/LOMO)는 LLM의 전체 파라미터를 메모리 효율적으로 미세 조정하기 위해 설계된 옵티마이저 제품군이며, [LOMO](https://huggingface.co/papers/2306.09782)와 [AdaLomo](https://hf.co/papers/2310.10195) 두 가지 버전이 있습니다. 두 LOMO 옵티마이저는 모두 메모리 사용량을 줄이기 위해 그래디언트 계산과 매개변수 업데이트를 한 단계로 통합합니다. AdaLomo는 LOMO를 기반으로, Adam 옵티마이저처럼 각 매개변수에 대해 적응형 학습률을 적용하는 기능이 추가되었습니다.

> [!TIP]
> 더 나은 성능과 높은 처리량을 위해서는 `grad_norm` 없이 AdaLomo를 사용하는 것을 권장합니다.

```diff
args = TrainingArguments(
    output_dir="./test-lomo",
    max_steps=1000,
    per_device_train_batch_size=4,
+   optim="adalomo",
    gradient_checkpointing=True,
    logging_strategy="steps",
    logging_steps=1,
    learning_rate=2e-6,
    save_strategy="no",
    run_name="adalomo",
)
```

## Schedule Free[[schedule-free]]

```bash
pip install schedulefree
```

[Schedule Free optimizer (SFO)](https://hf.co/papers/2405.15682)는 기본 옵티마이저의 모멘텀 대신 평균화(averaging)와 보간(interpolation)을 조합하여 사용합니다. 덕분에 기존의 학습률 스케줄러와 달리, SFO는 학습률을 점진적으로 낮추는 절차가 아예 필요 없습니다.

SFO는 RAdam(`schedule_free_radam`), AdamW(`schedule_free_adamw`), SGD(`schedule_free_sgd`) 옵티마이저를 지원합니다. RAdam 스케줄러는 `warmup_steps`나 `warmup_ratio` 설정이 필요하지 않습니다. 

기본적으로 `lr_scheduler_type="constant"`로 설정하는 것을 권장합니다. 다른 `lr_scheduler_type` 값도 동작할 순 있으나, SFO 옵티마이저와 다른 학습률 스케줄을 함께 사용하면 SFO의 의도된 동작과 성능에 영향을 줄 수 있습니다. 

```diff
args = TrainingArguments(
    output_dir="./test-schedulefree",
    max_steps=1000,
    per_device_train_batch_size=4,
+   optim="schedule_free_radamw",
+   lr_scheduler_type="constant",
    gradient_checkpointing=True,
    logging_strategy="steps",
    logging_steps=1,
    learning_rate=2e-6,
    save_strategy="no",
    run_name="sfo",
)
```

## StableAdamW[[stableadamw]]

```bash
pip install torch-optimi
```

[StableAdamW](https://huggingface.co/papers/2304.13013)는 AdamW와 AdaFactor를 결합한 하이브리드 옵티마이저입니다. AdaFactor의 업데이트 클리핑(update clipping)이 AdamW에 도입되어 별도의 그래디언트 클리핑(gradient clipping)이 필요 없습니다. 그 외의 동작에서는 AdamW와 완벽히 호환되는 대체제로 사용할 수 있습니다.

> [!TIP]
> 배치(batch) 크기가 크거나 훈련 손실(training loss)이 계속해서 급격하게 변동한다면, beta_2 값을 [0.95, 0.99] 사이로 줄여보세요.

```diff
args = TrainingArguments(
    output_dir="./test-stable-adamw",
    max_steps=1000,
    per_device_train_batch_size=4,
+   optim="stable_adamw",
    gradient_checkpointing=True,
    logging_strategy="steps",
    logging_steps=1,
    learning_rate=2e-6,
    save_strategy="no",
    run_name="stable-adamw",
)
```