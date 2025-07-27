<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 최적화[[optimization]]

`.optimization` 모듈은 다음을 제공합니다:

- 미세 조정된 모델에 사용할 수 있는 가중치 감쇠가 적용된 옵티마이저
- `_LRSchedule`을 상속받는 스케줄 객체 형태의 여러 스케줄
- 여러 배치의 그래디언트를 누적하는 그래디언트 누적 클래스


## AdaFactor (PyTorch)[[transformers.Adafactor]]

[[autodoc]] Adafactor

## AdamWeightDecay (TensorFlow)[[transformers.AdamWeightDecay]]

[[autodoc]] AdamWeightDecay

[[autodoc]] create_optimizer

## 스케줄[[schedules]]

### 학습률 스케줄 (PyTorch)[[transformers.SchedulerType]]

[[autodoc]] SchedulerType

[[autodoc]] get_scheduler

[[autodoc]] get_constant_schedule

[[autodoc]] get_constant_schedule_with_warmup

<img alt="" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/warmup_constant_schedule.png"/>

[[autodoc]] get_cosine_schedule_with_warmup

<img alt="" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/warmup_cosine_schedule.png"/>

[[autodoc]] get_cosine_with_hard_restarts_schedule_with_warmup

<img alt="" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/warmup_cosine_hard_restarts_schedule.png"/>

[[autodoc]] get_linear_schedule_with_warmup

<img alt="" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/warmup_linear_schedule.png"/>

[[autodoc]] get_polynomial_decay_schedule_with_warmup

[[autodoc]] get_inverse_sqrt_schedule

[[autodoc]] get_wsd_schedule

### 웜업 (TensorFlow)[[transformers.WarmUp]]

[[autodoc]] WarmUp

## 그래디언트 전략[[gradient-strategies]]

### GradientAccumulator (TensorFlow)[[transformers.GradientAccumulator]]

[[autodoc]] GradientAccumulator
