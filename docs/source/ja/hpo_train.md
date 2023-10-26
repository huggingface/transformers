<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Hyperparameter Search using Trainer API

🤗 Transformersは、🤗 Transformersモデルのトレーニングを最適化する[`Trainer`]クラスを提供し、独自のトレーニングループを手動で記述せずにトレーニングを開始するのが簡単になります。[`Trainer`]はハイパーパラメーター検索のAPIも提供しています。このドキュメントでは、それを例示します。

## Hyperparameter Search backend

[`Trainer`]は現在、4つのハイパーパラメーター検索バックエンドをサポートしています：
[optuna](https://optuna.org/)、[sigopt](https://sigopt.com/)、[raytune](https://docs.ray.io/en/latest/tune/index.html)、および[wandb](https://wandb.ai/site/sweeps)。

これらを使用する前に、ハイパーパラメーター検索バックエンドをインストールする必要があります。
```bash
pip install optuna/sigopt/wandb/ray[tune] 
```

## How to enable Hyperparameter search in example

ハイパーパラメータの検索スペースを定義し、異なるバックエンドには異なるフォーマットが必要です。

Sigoptの場合、sigopt [object_parameter](https://docs.sigopt.com/ai-module-api-references/api_reference/objects/object_parameter) を参照してください。それは以下のようなものです：
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


Optunaに関しては、[object_parameter](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html#sphx-glr-tutorial-10-key-features-002-configurations-py)をご覧ください。以下のようになります：


```py
>>> def optuna_hp_space(trial):
...     return {
...         "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
...         "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128]),
...     }
```

Optunaは、多目的のハイパーパラメータ最適化（HPO）を提供しています。 `hyperparameter_search` で `direction` を渡し、複数の目的関数値を返すための独自の `compute_objective` を定義することができます。 Pareto Front（`List[BestRun]`）は `hyperparameter_search` で返され、[test_trainer](https://github.com/huggingface/transformers/blob/main/tests/trainer/test_trainer.py) のテストケース `TrainerHyperParameterMultiObjectOptunaIntegrationTest` を参照する必要があります。これは以下のようになります。


```py
>>> best_trials = trainer.hyperparameter_search(
...     direction=["minimize", "maximize"],
...     backend="optuna",
...     hp_space=optuna_hp_space,
...     n_trials=20,
...     compute_objective=compute_objective,
... )
```

Ray Tuneに関して、[object_parameter](https://docs.ray.io/en/latest/tune/api/search_space.html)を参照してください。以下のようになります：


```py
>>> def ray_hp_space(trial):
...     return {
...         "learning_rate": tune.loguniform(1e-6, 1e-4),
...         "per_device_train_batch_size": tune.choice([16, 32, 64, 128]),
...     }
```

Wandbについては、[object_parameter](https://docs.wandb.ai/guides/sweeps/configuration)をご覧ください。これは以下のようになります：

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

`model_init` 関数を定義し、それを [`Trainer`] に渡す例を示します：


```py
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

[`Trainer`] を `model_init` 関数、トレーニング引数、トレーニングデータセット、テストデータセット、および評価関数と共に作成してください:


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

ハイパーパラメーターの探索を呼び出し、最良のトライアル パラメーターを取得します。バックエンドは `"optuna"` / `"sigopt"` / `"wandb"` / `"ray"` である可能性があります。方向は `"minimize"` または `"maximize"` であり、目標をより大きくするか小さくするかを示します。

`compute_objective` 関数を独自に定義することもできます。定義されていない場合、デフォルトの `compute_objective` が呼び出され、F1などの評価メトリックの合計が目標値として返されます。


```py
>>> best_trial = trainer.hyperparameter_search(
...     direction="maximize",
...     backend="optuna",
...     hp_space=optuna_hp_space,
...     n_trials=20,
...     compute_objective=compute_objective,
... )
```

## Hyperparameter search For DDP finetune
現在、DDP（Distributed Data Parallel）のためのハイパーパラメーター検索は、Optuna と SigOpt に対して有効になっています。ランクゼロプロセスのみが検索トライアルを生成し、他のランクに引数を渡します。






