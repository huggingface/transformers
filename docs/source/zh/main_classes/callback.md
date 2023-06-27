<!--版权所有 2020 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证，否则您不得使用此文件。您可以在以下位置获取许可证副本：
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“原样”提供的，没有任何形式的保证或条件。请参阅许可证以了解特定语言下的权限和限制。
⚠️ 请注意，此文件是 Markdown 格式，但包含了针对我们的文档构建工具（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确渲染。
-->

# 回调函数

回调函数是可以自定义 PyTorch 的训练循环行为的对象 [`Trainer`]（此功能尚未在 TensorFlow 中实现），它可以检查训练循环状态（用于进度报告、TensorBoard 或其他 ML 平台上的日志记录...）并做出决策（例如提前停止）。

回调函数是“只读”的代码片段，除了它们返回的 [`TrainerControl`] 对象，它们不能更改训练循环中的任何内容。对于需要更改训练循环的自定义操作，您应该子类化 [`Trainer`] 并覆盖所需的方法（请参阅 [trainer](trainer) 中的示例）。

默认情况下，[`Trainer`] 将使用以下回调函数：
- [`DefaultFlowCallback`] 用于处理日志记录、保存和评估的默认行为。
- [`PrinterCallback`] 或 [`ProgressCallback`] 用于显示进度和打印  日志（如果通过 [`TrainingArguments`] 停用 tqdm，则使用第一个，否则  使用第二个）
- 如果 tensorboard 可访问（通过 PyTorch >= 1.4 或 tensorboardX）：[`~integrations.TensorBoardCallback`]
- 如果已安装 [wandb](https://www.wandb.com/)：[`~integrations.WandbCallback`]
- 如果已安装 [comet_ml](https://www.comet.ml/site/)：[`~integrations.CometCallback`]
- 如果已安装 [mlflow](https://www.mlflow.org/)：[`~integrations.MLflowCallback`]
- 如果已安装 [neptune](https://neptune.ai/)：[`~integrations.NeptuneCallback`]
- 如果已安装 [azureml-sdk](https://pypi.org/project/azureml-sdk/)：[`~integrations.AzureMLCallback`]
- 如果已安装 [codecarbon](https://pypi.org/project/codecarbon/)：[`~integrations.CodeCarbonCallback`]
- 如果已安装 [clearml](https://github.com/allegroai/clearml)：[`~integrations.ClearMLCallback`]
- 如果已安装 [dagshub](https://dagshub.com/)：[`~integrations.DagsHubCallback`]
- 如果已安装 [flyte](https://flyte.org/)：[`~integrations.FlyteCallback`]
- [`~integrations.ClearMLCallback`] if [clearml](https://github.com/allegroai/clearml) is installed.
- [`~integrations.DagsHubCallback`] 如果已安装 [dagshub](https://dagshub.com/)
- [`~integrations.FlyteCallback`] 如果已安装 [flyte](https://flyte.org/)。

实现回调函数的主要类是 [`TrainerCallback`]。它获取用于实例化 [`Trainer`] 的 [`TrainingArguments`]，可以通过它访问 Trainer 的内部状态，并可以通过 [`TrainerControl`] 对训练循环执行一些操作。[`TrainerControl`].


## 可用的回调函数

以下是库中可用的 [`TrainerCallback`] 列表：

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

以下是如何在 PyTorch [`Trainer`] 中注册自定义回调的示例：
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

另一种注册回调的方式是调用 `trainer.add_callback()`，如下所示：
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
