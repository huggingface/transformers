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

Callbacks are objects that can customize the behavior of the training loop in the PyTorch
[`Trainer`] that can inspect the training loop state (for progress reporting, logging on TensorBoard or other ML
platforms...) and take decisions (like early stopping).

Callbacks are "read only" pieces of code, apart from the [`TrainerControl`] object they return, they
cannot change anything in the training loop. For customizations that require changes in the training loop, you should
subclass [`Trainer`] and override the methods you need (see [trainer](trainer) for examples).

By default, `TrainingArguments.report_to` is set to `"all"`, so a [`Trainer`] will use the following callbacks.

- [`DefaultFlowCallback`] which handles the default behavior for logging, saving and evaluation.
- [`PrinterCallback`] or [`ProgressCallback`] to display progress and print the
  logs (the first one is used if you deactivate tqdm through the [`TrainingArguments`], otherwise
  it's the second one).
- [`~integrations.TensorBoardCallback`] if tensorboard is accessible (either through PyTorch >= 1.4
  or tensorboardX).
- [`~integrations.TrackioCallback`] if [trackio](https://github.com/gradio-app/trackio) is installed.
- [`~integrations.WandbCallback`] if [wandb](https://www.wandb.com/) is installed.
- [`~integrations.CometCallback`] if [comet_ml](https://www.comet.com/site/) is installed.
- [`~integrations.MLflowCallback`] if [mlflow](https://www.mlflow.org/) is installed.
- [`~integrations.NeptuneCallback`] if [neptune](https://neptune.ai/) is installed.
- [`~integrations.AzureMLCallback`] if [azureml-sdk](https://pypi.org/project/azureml-sdk/) is
  installed.
- [`~integrations.CodeCarbonCallback`] if [codecarbon](https://pypi.org/project/codecarbon/) is
  installed.
- [`~integrations.ClearMLCallback`] if [clearml](https://github.com/allegroai/clearml) is installed.
- [`~integrations.DagsHubCallback`] if [dagshub](https://dagshub.com/) is installed.
- [`~integrations.FlyteCallback`] if [flyte](https://flyte.org/) is installed.
- [`~integrations.DVCLiveCallback`] if [dvclive](https://dvc.org/doc/dvclive) is installed.
- [`~integrations.SwanLabCallback`] if [swanlab](http://swanlab.cn/) is installed.

If a package is installed but you don't wish to use the accompanying integration, you can change `TrainingArguments.report_to` to a list of just those integrations you want to use (e.g. `["azure_ml", "wandb"]`).

The main class that implements callbacks is [`TrainerCallback`]. It gets the
[`TrainingArguments`] used to instantiate the [`Trainer`], can access that
Trainer's internal state via [`TrainerState`], and can take some actions on the training loop via
[`TrainerControl`].


## Available Callbacks

Here is the list of the available [`TrainerCallback`] in the library:

[[autodoc]] integrations.CometCallback
    - setup

[[autodoc]] DefaultFlowCallback

[[autodoc]] PrinterCallback

[[autodoc]] ProgressCallback

[[autodoc]] EarlyStoppingCallback

[[autodoc]] integrations.TensorBoardCallback

[[autodoc]] integrations.TrackioCallback
    - setup

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

[[autodoc]] integrations.SwanLabCallback
    - setup

## TrainerCallback

[[autodoc]] TrainerCallback

Here is an example of how to register a custom callback with the PyTorch [`Trainer`]:

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

Another way to register a callback is to call `trainer.add_callback()` as follows:

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
