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

# 콜백 [[callbacks]]

콜백은 PyTorch [`Trainer`]의 반복 학습 동작을 사용자 정의할 수 있는 객체입니다
(이 기능은 TensorFlow에서는 아직 구현되지 않았습니다). 콜백은 반복 학습의 상태를
검사하여 (진행 상황 보고, TensorBoard 또는 기타 머신 러닝 플랫폼에 로그 남기기 등) 
결정(예: 조기 종료)을 내릴 수 있습니다.

콜백은 [`TrainerControl`] 객체를 반환하는 것 외에는 반복 학습에서 어떤 것도 변경할 수 없는
"읽기 전용" 코드 조각입니다. 반복 학습에 변경이 필요한 사용자 정의 작업이 필요한 경우, 
[`Trainer`]를 서브클래스로 만들어 필요한 메소드들을 오버라이드해야 합니다 (예시는 [trainer](trainer)를 참조하세요).

기본적으로 `TrainingArguments.report_to`는 `"all"`로 설정되어 있으므로, [`Trainer`]는 다음 콜백을 사용합니다.

- [`DefaultFlowCallback`]는 로그, 저장, 평가에 대한 기본 동작을 처리합니다.
- [`PrinterCallback`] 또는 [`ProgressCallback`]는 진행 상황을 표시하고 로그를 출력합니다 
  ([`TrainingArguments`]를 통해 tqdm을 비활성화하면 첫 번째 콜백이 사용되고, 그렇지 않으면 두 번째가 사용됩니다).
- [`~integrations.TensorBoardCallback`]는 TensorBoard가 (PyTorch >= 1.4
 또는 tensorboardX를 통해) 접근 가능하면 사용됩니다.
- [`~integrations.WandbCallback`]는 [wandb](https://www.wandb.com/)가 설치되어 있으면
 사용됩니다.
- [`~integrations.CometCallback`]는 [comet_ml](https://www.comet.com/site/)이 설치되어 있으면 사용됩니다.
- [`~integrations.MLflowCallback`]는 [mlflow](https://www.mlflow.org/)가 설치되어 있으면 사용됩니다.
- [`~integrations.NeptuneCallback`]는 [neptune](https://neptune.ai/)이 설치되어 있으면 사용됩니다.
- [`~integrations.AzureMLCallback`]는 [azureml-sdk](https://pypi.org/project/azureml-sdk/)가 설치되어
 있으면 사용됩니다.
- [`~integrations.CodeCarbonCallback`]는 [codecarbon](https://pypi.org/project/codecarbon/)이 설치되어
 있으면 사용됩니다.
- [`~integrations.ClearMLCallback`]는 [clearml](https://github.com/allegroai/clearml)이 설치되어 있으면 사용됩니다.
- [`~integrations.DagsHubCallback`]는 [dagshub](https://dagshub.com/)이 설치되어 있으면 사용됩니다.
- [`~integrations.FlyteCallback`]는 [flyte](https://flyte.org/)가 설치되어 있으면 사용됩니다.
- [`~integrations.DVCLiveCallback`]는 [dvclive](https://dvc.org/doc/dvclive)가 설치되어 있으면 사용됩니다.

패키지가 설치되어 있지만 해당 통합 기능을 사용하고 싶지 않다면, `TrainingArguments.report_to`를 사용하고자 하는 통합 기능 목록으로 변경할 수 있습니다 (예: `["azure_ml", "wandb"]`).

콜백을 구현하는 주요 클래스는 [`TrainerCallback`]입니다. 이 클래스는 [`Trainer`]를 
인스턴스화하는 데 사용된 [`TrainingArguments`]를 가져오고, 해당 Trainer의 내부 상태를 
[`TrainerState`]를 통해 접근할 수 있으며, [`TrainerControl`]을 통해 반복 학습에서 일부 
작업을 수행할 수 있습니다.


## 사용 가능한 콜백 [[available-callbacks]]

라이브러리에서 사용 가능한 [`TrainerCallback`] 목록은 다음과 같습니다:

[[autodoc]] integrations.CometCallback
    - setup

[[autodoc]] DefaultFlowCallback

[[autodoc]] PrinterCallback

[[autodoc]] ProgressCallback

[[autodoc]] EarlyStoppingCallback

[[autodoc]] integrations.TensorBoardCallback

[[autodoc]] integrations.WandbCallback
    - setup

[[autodoc]] integrations.MLflowCallback
    - setup

[[autodoc]] integrations.AzureMLCallback

[[autodoc]] integrations.CodeCarbonCallback

[[autodoc]] integrations.NeptuneCallback

[[autodoc]] integrations.ClearMLCallback

[[autodoc]] integrations.DagsHubCallback

[[autodoc]] integrations.FlyteCallback

[[autodoc]] integrations.DVCLiveCallback
    - setup

## TrainerCallback [[trainercallback]]

[[autodoc]] TrainerCallback

여기 PyTorch [`Trainer`]와 함께 사용자 정의 콜백을 등록하는 예시가 있습니다:

```python
class MyCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"

    def on_train_begin(self, args, state, control, **kwargs):
        print("Starting training")


trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[MyCallback],  # 우리는 콜백 클래스를 이 방식으로 전달하거나 그것의 인스턴스(MyCallback())를 전달할 수 있습니다
)
```

또 다른 콜백을 등록하는 방법은 `trainer.add_callback()`을 호출하는 것입니다:

```python
trainer = Trainer(...)
trainer.add_callback(MyCallback)
# 다른 방법으로는 콜백 클래스의 인스턴스를 전달할 수 있습니다
trainer.add_callback(MyCallback())
```

## TrainerState [[trainerstate]]

[[autodoc]] TrainerState

## TrainerControl [[trainercontrol]]

[[autodoc]] TrainerControl
