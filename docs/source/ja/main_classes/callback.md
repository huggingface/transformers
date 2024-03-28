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


# コールバック数

コールバックは、PyTorch のトレーニング ループの動作をカスタマイズできるオブジェクトです。
トレーニング ループを検査できる [`Trainer`] (この機能は TensorFlow にはまだ実装されていません)
状態を確認し (進捗レポート、TensorBoard または他の ML プラットフォームへのログ記録など)、決定を下します (初期段階など)。
停止中）。

コールバックは、返される [`TrainerControl`] オブジェクトを除けば、「読み取り専用」のコード部分です。
トレーニング ループ内では何も変更できません。トレーニング ループの変更が必要なカスタマイズの場合は、次のことを行う必要があります。
[`Trainer`] をサブクラス化し、必要なメソッドをオーバーライドします (例については、[trainer](trainer) を参照してください)。

デフォルトでは、`TrainingArguments.report_to` は `"all"` に設定されているため、[`Trainer`] は次のコールバックを使用します。

- [`DefaultFlowCallback`] は、ログ記録、保存、評価のデフォルトの動作を処理します。
- [`PrinterCallback`] または [`ProgressCallback`] で進行状況を表示し、
  ログ (最初のログは、[`TrainingArguments`] を通じて tqdm を非アクティブ化する場合に使用され、そうでない場合に使用されます)
  2番目です)。
- [`~integrations.TensorBoardCallback`] (PyTorch >= 1.4 を介して) tensorboard にアクセスできる場合
  またはテンソルボードX）。
- [`~integrations.WandbCallback`] [wandb](https://www.wandb.com/) がインストールされている場合。
- [`~integrations.CometCallback`] [comet_ml](https://www.comet.ml/site/) がインストールされている場合。
- [mlflow](https://www.mlflow.org/) がインストールされている場合は [`~integrations.MLflowCallback`]。
- [`~integrations.NeptuneCallback`] [neptune](https://neptune.ai/) がインストールされている場合。
- [`~integrations.AzureMLCallback`] [azureml-sdk](https://pypi.org/project/azureml-sdk/) の場合
  インストールされています。
- [`~integrations.CodeCarbonCallback`] [codecarbon](https://pypi.org/project/codecarbon/) の場合
  インストールされています。
- [`~integrations.ClearMLCallback`] [clearml](https://github.com/allegroai/clearml) がインストールされている場合。
- [`~integrations.DagsHubCallback`] [dagshub](https://dagshub.com/) がインストールされている場合。
- [`~integrations.FlyteCallback`] [flyte](https://flyte.org/) がインストールされている場合。
- [`~integrations.DVCLiveCallback`] [dvclive](https://www.dvc.org/doc/dvclive) がインストールされている場合。

パッケージがインストールされているが、付随する統合を使用したくない場合は、`TrainingArguments.report_to` を、使用したい統合のみのリストに変更できます (例: `["azure_ml", "wandb"]`) 。

コールバックを実装するメインクラスは [`TrainerCallback`] です。それは、
[`TrainingArguments`] は [`Trainer`] をインスタンス化するために使用され、それにアクセスできます。
[`TrainerState`] を介してトレーナーの内部状態を取得し、トレーニング ループ上でいくつかのアクションを実行できます。
[`TrainerControl`]。

## 利用可能なコールバック

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

[[autodoc]] integrations.DVCLiveCallback
    - setup

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


