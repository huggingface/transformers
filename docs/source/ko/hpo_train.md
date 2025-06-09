<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 使用Trainer API进行超参数搜索

🤗 Transformers库提供了一个优化过的[`Trainer`]类，用于训练🤗 Transformers模型，相比于手动编写自己的训练循环，这更容易开始训练。[`Trainer`]提供了超参数搜索的API。本文档展示了如何在示例中启用它。

## 超参数搜索后端

[`Trainer`] 目前支持三种超参数搜索后端：[optuna](https://optuna.org/)，[raytune](https://docs.ray.io/en/latest/tune/index.html)，[wandb](https://wandb.ai/site/sweeps)

在使用它们之前，您应该先安装它们作为超参数搜索后端。

```bash
pip install optuna/wandb/ray[tune]
```

## 在示例中启用超参数搜索

定义超参数搜索空间。超参数搜索后端需要不同的格式。

optuna的例子，参考[object_parameter](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html#sphx-glr-tutorial-10-key-features-002-configurations-py)文档，如下：

```py
>>> def optuna_hp_space(trial):
...     return {
...         "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
...         "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128]),
...     }
```

raytune的例子，参考[object_parameter](https://docs.ray.io/en/latest/tune/api/search_space.html)文档，如下：

```py
>>> def ray_hp_space(trial):
...     return {
...         "learning_rate": tune.loguniform(1e-6, 1e-4),
...         "per_device_train_batch_size": tune.choice([16, 32, 64, 128]),
...     }
```

wandb的例子，参考[object_parameter](https://docs.wandb.ai/guides/sweeps/configuration)文档，如下：

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

定义`model_init`函数并将其传递给[`Trainer`]。以下是示例：
```py
>>> def model_init(trial):
...     return AutoModelForSequenceClassification.from_pretrained(
...         model_args.model_name_or_path,
...         from_tf=bool(".ckpt" in model_args.model_name_or_path),
...         config=config,
...         cache_dir=model_args.cache_dir,
...         revision=model_args.model_revision,
...         token=True if model_args.use_auth_token else None,
...     )
```

使用以下方法创建[`Trainer`]：

```py
>>> trainer = Trainer(
...     model=None,
...     args=training_args,
...     train_dataset=small_train_dataset,
...     eval_dataset=small_eval_dataset,
...     compute_metrics=compute_metrics,
...     processing_class=tokenizer,
...     model_init=model_init,
...     data_collator=data_collator,
... )
```

调用超参数搜索并获取最佳试验参数。后端可以是`"optuna"`/`"wandb"`/`"ray"`中的一个，方向可以是`"minimize"`或`"maximize"`，取决于您是要最小化还是最大化目标。

您可以定义自己的compute_objective函数。如果未定义，则将调用默认的compute_objective，并返回f1等评估指标的和作为目标值。

```py
>>> best_trial = trainer.hyperparameter_search(
...     direction="maximize",
...     backend="optuna",
...     hp_space=optuna_hp_space,
...     n_trials=20,
...     compute_objective=compute_objective,
... )
```

## DDP 미세 조정을 위한 하이퍼파라미터 탐색
현재 DDP(분산 데이터 병렬)를 위한 하이퍼파라미터 탐색은 Optuna에만 활성화되어 있습니다. 랭크-0 프로세스만이 탐색 시험을 생성하고 다른 랭크에 매개변수를 전달합니다.
