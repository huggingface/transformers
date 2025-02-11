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

# Optimization

`.optimization` モジュールは以下を提供します。

- モデルの微調整に使用できる重み減衰が修正されたオプティマイザー、および
- `_LRSchedule` から継承するスケジュール オブジェクトの形式のいくつかのスケジュール:
- 複数のバッチの勾配を累積するための勾配累積クラス

## AdamW (PyTorch)

[[autodoc]] AdamW

## AdaFactor (PyTorch)

[[autodoc]] Adafactor

## AdamWeightDecay (TensorFlow)

[[autodoc]] AdamWeightDecay

[[autodoc]] create_optimizer

## Schedules

### Learning Rate Schedules (Pytorch)

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

### Warmup (TensorFlow)

[[autodoc]] WarmUp

## Gradient Strategies

### GradientAccumulator (TensorFlow)

[[autodoc]] GradientAccumulator
