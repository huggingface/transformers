<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# コールバック

コールバックは、PyTorchの[`Trainer`]内でトレーニングループの動作をカスタマイズできるオブジェクトです（この機能はTensorFlowではまだ実装されていません）。これらのコールバックはトレーニングループの状態を調査でき（進捗報告、TensorBoardや他のMLプラットフォームへのログ記録など）、早期停止などの決定を行うことができます。

コールバックは "読み取り専用" のコードの断片であり、[`TrainerControl`]オブジェクトを返す以外の場合、トレーニングループ内で何も変更できません。トレーニングループ内で変更を必要とするカスタマイズについては、[`Trainer`]をサブクラス化して必要なメソッドをオーバーライドする必要があります（例については[trainer](trainer)を参照）。

デフォルトでは、[`Trainer`]は次のコールバックを使用します：

- [`DefaultFlowCallback`]：ログのデフォルトの動作、保存、評価を処理します。
- [`PrinterCallback`]または[`ProgressCallback`]：進行状況の表示とログの印刷（`tqdm`を[`TrainingArguments`]を介して無効にしない限り、最初のものが使用されます）。
- [`~integrations.TensorBoardCallback`]：tensorboardがアクセス可能な場合（PyTorch >= 1.4またはtensorboardXを介して）。
- [`~integrations.WandbCallback`]：[wandb](https://www.wandb.com/)がインストールされている場合。
- [`~integrations.CometCallback`]：[comet_ml](https://www.comet.ml/site/)がインストールされている場合。
- [`~integrations.MLflowCallback`]：[mlflow](https://www.mlflow.org/)がインストールされている場合。
- [`~integrations.NeptuneCallback`]：[neptune](https://neptune.ai/)がインストールされている場合。
- [`~integrations.AzureMLCallback`]：[azureml-sdk](https://pypi.org/project/azureml-sdk/)がインストールされている場合。
- [`~integrations.CodeCarbonCallback`]：[codecarbon](https://pypi.org/project/codecarbon/)がインストールされている場合。
- [`~integrations.ClearMLCallback`]：[clearml](https://github.com/allegroai/clearml)がインストールされている場合。
- [`~integrations.DagsHubCallback`]：[dagshub](https://dagshub.com/)がインストールされている場合。
- [`~integrations.FlyteCallback`]：[flyte](https://flyte.org/)がインストールされている場合。

コールバックを実装する主要なクラスは[`TrainerCallback`]です。これは[`Trainer`]をインスタンス化するために使用された[`TrainingArguments`]にアクセスでき、そのTrainerの内部状態に[`TrainerState`]を介してアクセスでき、[`TrainerControl`]を介してトレーニングループ上でいくつかのアクションを実行できます。

## Available Callbacks


ライブラリで利用可能な [`TrainerCallback`] のリストは次のとおりです。

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

## TrainerCallback

[[autodoc]] TrainerCallback


以下は、カスタム コールバックを PyTorch [`Trainer`] に登録する方法の例です。

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
    callbacks=[MyCallback],  # We can either pass the callback class this way or an instance of it (MyCallback())
)
```


コールバックを登録する別の方法は、次のように `trainer.add_callback()` を呼び出すことです。

```python
trainer = Trainer(...)
trainer.add_callback(MyCallback)
# Alternatively, we can pass an instance of the callback class
trainer.add_callback(MyCallback())
```

## TrainerState

[[autodoc]] TrainerState

## TrainerControl

[[autodoc]] TrainerControl
