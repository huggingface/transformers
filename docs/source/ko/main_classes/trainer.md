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

# Trainer

[`Trainer`] 클래스는 PyTorch에서 완전한 기능의 훈련을 위한 API를 제공하며, 다중 GPU/TPU에서의 분산 훈련, [NVIDIA GPU](https://nvidia.github.io/apex/), [AMD GPU](https://rocm.docs.amd.com/en/latest/rocm.html)를 위한 혼합 정밀도, 그리고 PyTorch의 [`torch.amp`](https://pytorch.org/docs/stable/amp.html)를 지원합니다. [`Trainer`]는 모델의 훈련 방식을 커스터마이즈할 수 있는 다양한 옵션을 제공하는 [`TrainingArguments`] 클래스와 함께 사용됩니다. 이 두 클래스는 함께 완전한 훈련 API를 제공합니다.

[`Seq2SeqTrainer`]와 [`Seq2SeqTrainingArguments`]는 [`Trainer`]와 [`TrainingArguments`] 클래스를 상속하며, 요약이나 번역과 같은 시퀀스-투-시퀀스 작업을 위한 모델 훈련에 적합하게 조정되어 있습니다.

<Tip warning={true}>

[`Trainer`] 클래스는 🤗 Transformers 모델에 최적화되어 있으며, 다른 모델과 함께 사용될 때 예상치 못한 동작을 할 수 있습니다. 자신만의 모델을 사용할 때는 다음을 확인하세요:

- 모델이 항상 튜플이나 [`~utils.ModelOutput`]의 서브클래스를 반환해야 합니다.
- `labels` 인자가 제공되면 손실을 계산할 수 있고, 그 손실이 튜플의 첫 번째 요소로 반환되어야 합니다 (모델이 튜플을 반환하는 경우).
- 모델이 여러 개의 레이블 인자를 수락할 수 있어야 하며, [`Trainer`]에게 이름을 알리기 위해 [`TrainingArguments`]에서 `label_names`를 사용하지만, 그 중 어느 것도 `"label"`로 명명되어서는 안 됩니다.

</Tip>

## Trainer[[api-reference]]

[[autodoc]] Trainer
    - all

## Seq2SeqTrainer

[[autodoc]] Seq2SeqTrainer
    - evaluate
    - predict

## TrainingArguments

[[autodoc]] TrainingArguments
    - all

## Seq2SeqTrainingArguments

[[autodoc]] Seq2SeqTrainingArguments
    - all
    