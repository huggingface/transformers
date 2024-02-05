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

# Callbacks


Callbacks可以用来自定义PyTorch [Trainer]中训练循环行为的对象（此功能尚未在TensorFlow中实现），该对象可以检查训练循环状态（用于进度报告、在TensorBoard或其他ML平台上记录日志等），并做出决策（例如提前停止）。

Callbacks是“只读”的代码片段，除了它们返回的[TrainerControl]对象外，它们不能更改训练循环中的任何内容。对于需要更改训练循环的自定义，您应该继承[Trainer]并重载您需要的方法（有关示例，请参见[trainer](trainer)）。

默认情况下，`TrainingArguments.report_to` 设置为"all"，然后[Trainer]将使用以下callbacks。


- [`DefaultFlowCallback`]，它处理默认的日志记录、保存和评估行为
- [`PrinterCallback`] 或 [`ProgressCallback`]，用于显示进度和打印日志（如果通过[`TrainingArguments`]停用tqdm，则使用第一个函数；否则使用第二个）。
- [`~integrations.TensorBoardCallback`]，如果TensorBoard可访问（通过PyTorch版本 >= 1.4 或者 tensorboardX）。
- [`~integrations.WandbCallback`]，如果安装了[wandb](https://www.wandb.com/)。
- [`~integrations.CometCallback`]，如果安装了[comet_ml](https://www.comet.ml/site/)。
- [`~integrations.MLflowCallback`]，如果安装了[mlflow](https://www.mlflow.org/)。
- [`~integrations.NeptuneCallback`]，如果安装了[neptune](https://neptune.ai/)。
- [`~integrations.AzureMLCallback`]，如果安装了[azureml-sdk](https://pypi.org/project/azureml-sdk/)。
- [`~integrations.CodeCarbonCallback`]，如果安装了[codecarbon](https://pypi.org/project/codecarbon/)。
- [`~integrations.ClearMLCallback`]，如果安装了[clearml](https://github.com/allegroai/clearml)。
- [`~integrations.DagsHubCallback`]，如果安装了[dagshub](https://dagshub.com/)。
- [`~integrations.FlyteCallback`]，如果安装了[flyte](https://flyte.org/)。
- [`~integrations.DVCLiveCallback`]，如果安装了[dvclive](https://dvc.org/doc/dvclive)。

如果安装了一个软件包，但您不希望使用相关的集成，您可以将 `TrainingArguments.report_to` 更改为仅包含您想要使用的集成的列表（例如 `["azure_ml", "wandb"]`）。

实现callbacks的主要类是[`TrainerCallback`]。它获取用于实例化[`Trainer`]的[`TrainingArguments`]，可以通过[`TrainerState`]访问该Trainer的内部状态，并可以通过[`TrainerControl`]对训练循环执行一些操作。


## 可用的Callbacks

这里是库里可用[`TrainerCallback`]的列表：

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

以下是如何使用PyTorch注册自定义callback的示例：

[`Trainer`]:

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

注册callback的另一种方式是调用 `trainer.add_callback()`，如下所示：


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
