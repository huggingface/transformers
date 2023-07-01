<!--版权所有 2022 年的 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证的规定，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件以“按原样”分发，不附带任何形式的保证或条件。有关许可证的详细信息请参阅许可证。
⚠️ 请注意，此文件使用 Markdown 编写，但包含我们文档生成器（类似于 MDX）的特定语法，可能无法在 Markdown 查看器中正确渲染。rendered properly in your Markdown viewer.
-->

# 使用 Trainer API 进行超参数搜索

🤗 Transformers 提供了一个经过优化的 [`Trainer`] 类，用于训练🤗 Transformers 模型，使得开始训练而无需手动编写自己的训练循环更加容易。[`Trainer`] 提供了超参数搜索的 API。本文档展示了如何在示例中启用超参数搜索。

## 超参数搜索后端

[`Trainer`] 当前支持四个超参数搜索后端：
[optuna](https://optuna.org/)，[sigopt](https://sigopt.com/)，[raytune](https://docs.ray.io/en/latest/tune/index.html) 和 [wandb](https://wandb.ai/site/sweeps)。

在使用超参数搜索后端之前，您应该先安装它们。
```bash
pip install optuna/sigopt/wandb/ray[tune] 
```

## 如何在示例中启用超参数搜索

定义超参数搜索空间，不同的后端需要不同的格式。
对于 sigopt，请参阅 sigopt [object_parameter](https://docs.sigopt.com/ai-module-api-references/api_reference/objects/object_parameter)，格式如下：
```py
>>> def sigopt_hp_space(trial):
...     return [
...         {"bounds": {"min": 1e-6, "max": 1e-4}, "name": "learning_rate", "type": "double"},
...         {
...             "categorical_values": ["16", "32", "64", "128"],
...             "name": "per_device_train_batch_size",
...             "type": "categorical",
...         },
...     ]
```

对于 optuna，请参阅 optuna [object_parameter](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html#sphx-glr-tutorial-10-key-features-002-configurations-py)，格式如下：
```py
>>> def optuna_hp_space(trial):
...     return {
...         "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
...         "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128]),
...     }
```

对于 raytune，请参阅 raytune [object_parameter](https://docs.ray.io/en/latest/tune/api/search_space.html)，格式如下：
```py
>>> def ray_hp_space(trial):
...     return {
...         "learning_rate": tune.loguniform(1e-6, 1e-4),
...         "per_device_train_batch_size": tune.choice([16, 32, 64, 128]),
...     }
```

对于 wandb，请参阅 wandb [object_parameter](https://docs.wandb.ai/guides/sweeps/configuration)，格式如下：
```py
>>> def wandb_hp_space(trial):
...     return {
...         "method": "random",
...         "metric": {"name": "objective", "goal": "minimize"},
...         "parameters": {
...             "learning_rate": {"distribution": "uniform", "min": 1e-6, "max": 1e-4},
...             "per_device_train_batch_size": {"values": [16, 32, 64, 128]},
...         },
...     }
```

定义一个 `model_init` 函数，并将其作为示例传递给 [`Trainer`]：```py
>>> def model_init(trial):
...     return AutoModelForSequenceClassification.from_pretrained(
...         model_args.model_name_or_path,
...         from_tf=bool(".ckpt" in model_args.model_name_or_path),
...         config=config,
...         cache_dir=model_args.cache_dir,
...         revision=model_args.model_revision,
...         use_auth_token=True if model_args.use_auth_token else None,
...     )
```

使用您的 `model_init` 函数、训练参数、训练和测试数据集以及评估函数创建一个 [`Trainer`]：
```py
>>> trainer = Trainer(
...     model=None,
...     args=training_args,
...     train_dataset=small_train_dataset,
...     eval_dataset=small_eval_dataset,
...     compute_metrics=compute_metrics,
...     tokenizer=tokenizer,
...     model_init=model_init,
...     data_collator=data_collator,
... )
```

调用超参数搜索，获取最佳的试验参数，后端可以是 `"optuna"`/`"sigopt"`/`"wandb"`/`"ray"`。direction 可以是 `"minimize"` 或 `"maximize"`，表示优化更大或更小的目标。
如果未定义，您可以定义自己的 compute_objective 函数，将调用默认的 compute_objective 函数，并将评估指标（如 f1）的总和作为目标值返回。
```py
>>> best_trial = trainer.hyperparameter_search(
...     direction="maximize",
...     backend="optuna",
...     hp_space=optuna_hp_space,
...     n_trials=20,
...     compute_objective=compute_objective,
... )
```

## DDP 微调的超参数搜索

目前，DDP 的超参数搜索仅支持 optuna 和 sigopt。仅 rank-zero 进程将生成搜索试验并将参数传递给其他 rank。