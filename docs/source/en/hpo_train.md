<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Hyperparameter Search using Trainer API

🤗 Transformers provides a [`Trainer`] class optimized for training 🤗 Transformers models, making it easier to start training without manually writing your own training loop. The [`Trainer`] provides API for hyperparameter search. This doc shows how to enable it in example. 

## Hyperparameter Search backend

[`Trainer`] supports four hyperparameter search backends currently:
[optuna](https://optuna.org/), [sigopt](https://sigopt.com/), [raytune](https://docs.ray.io/en/latest/tune/index.html) and [wandb](https://wandb.ai/site/sweeps).

you should install them before using them as the hyperparameter search backend
```bash
pip install optuna/sigopt/wandb/ray[tune] 
```

## How to enable Hyperparameter search in example

Define the hyperparameter search space, different backends need different format.

For sigopt, see sigopt [object_parameter](https://docs.sigopt.com/ai-module-api-references/api_reference/objects/object_parameter), it's like following:
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

For optuna, see optuna [object_parameter](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html#sphx-glr-tutorial-10-key-features-002-configurations-py), it's like following:

```py
>>> def optuna_hp_space(trial):
...     return {
...         "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
...         "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128]),
...     }
```

For raytune, see raytune [object_parameter](https://docs.ray.io/en/latest/tune/api/search_space.html), it's like following:

```py
>>> def ray_hp_space(trial):
...     return {
...         "learning_rate": tune.loguniform(1e-6, 1e-4),
...         "per_device_train_batch_size": tune.choice([16, 32, 64, 128]),
...     }
```

For wandb, see wandb [object_parameter](https://docs.wandb.ai/guides/sweeps/configuration), it's like following:

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

Define a `model_init` function and pass it to the [`Trainer`], as an example:
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

Create a [`Trainer`] with your `model_init` function, training arguments, training and test datasets, and evaluation function:

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

Call hyperparameter search, get the best trial parameters, backend could be `"optuna"`/`"sigopt"`/`"wandb"`/`"ray"`. direction can be`"minimize"` or `"maximize"`, which indicates whether to optimize greater or lower objective.

You could define your own compute_objective function, if not defined, the default compute_objective will be called, and the sum of eval metric like f1 is returned as objective value.

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
Currently, Hyperparameter search for DDP is enabled for optuna and sigopt. Only the rank-zero process will generate the search trial and pass the argument to other ranks.
