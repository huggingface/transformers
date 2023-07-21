<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Trainer API를 사용한 하이퍼파라미터 탐색 [[hyperparameter-search-using-trainer-api]]

🤗 Transformers는 트레이닝을 더욱 간편하게 시작할 수 있도록 Transformers 모델을 최적화한 [`Trainer`] 클래스를 제공합니다. 🤗 이 클래스를 사용하면 사용자는 직접 훈련 루프를 작성할 필요 없이 학습을 진행할 수 있습니다. 또한, [`Trainer`]는 하이퍼파라미터 탐색을 위한 API를 제공합니다. 이 문서에서는 이 API들을 활용하는 방법을 예시와 함께 설명합니다.

## 하이퍼파라미터 탐색 백엔드 [[hyperparameter-search-backend]]

현재, [`Trainer`]는 아래와 같은 네 가지 하이퍼파라미터 탐색 백엔드를 지원합니다.
[optuna](https://optuna.org/), [sigopt](https://sigopt.com/), [raytune](https://docs.ray.io/en/latest/tune/index.html) 그리고 [wandb](https://wandb.ai/site/sweeps) 입니다.

하이퍼파라미터 탐색 백엔드를 사용하기 전에 아래의 명령어를 사용하여 라이브러리들을 설치합니다.
```bash
pip install optuna/sigopt/wandb/ray[tune] 
```

## 예제에서 하이퍼파라미터 탐색을 활성화하는 방법 [[how-to-enable-hyperparameter-search-in-example]]

하이퍼파라미터 탐색 공간을 정의하세요. 서로 다른 하이퍼파라미터 탐색 백엔드들은 다른 형식으로 작성되어야 합니다.

sigopt는 sigopt의 [object_parameter](https://docs.sigopt.com/ai-module-api-references/api_reference/objects/object_parameter)를 참고하세요. 아래와 같이 작성할 수 있습니다.
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

optuna는 optuna의 [object_parameter](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html#sphx-glr-tutorial-10-key-features-002-configurations-py)를 참고하세요. 아래와 같이 작성할 수 있습니다.

```py
>>> def optuna_hp_space(trial):
...     return {
...         "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
...         "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128]),
...     }
```

raytune는 raytune의 [object_parameter](https://docs.ray.io/en/latest/tune/api/search_space.html)를 참고하세요. 아래와 같이 작성할 수 있습니다.

```py
>>> def ray_hp_space(trial):
...     return {
...         "learning_rate": tune.loguniform(1e-6, 1e-4),
...         "per_device_train_batch_size": tune.choice([16, 32, 64, 128]),
...     }
```

wandb는 wandb의 [object_parameter](https://docs.wandb.ai/guides/sweeps/configuration)를 참고하세요. 아래와 같이 작성할 수 있습니다.

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

`model_init` 함수를 정의하고 이를 [`Trainer`]에 전달하세요. 아래는 그 예시입니다.
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

아래와 같이 `model_init` 함수, 훈련 인수, 훈련 및 테스트 데이터셋, 그리고 평가 함수를 사용하여 [`Trainer`]를 생성합니다.

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

하이퍼파라미터 탐색을 호출하고, 최적의 시험 매개변수를 가져옵니다. backend는 `"optuna"`/`"sigopt"`/`"wandb"`/`"ray"` 중에서 선택할 수 있습니다. direction은 `"minimize"` 또는 `"maximize"`가 될 수 있으며, 목표를 최대화할 것인지, 최소화할 것인지를 결정합니다.

자신만의 compute_objective 함수를 정의할 수 있습니다. 만약 이 함수를 정의하지 않으면, 기본 compute_objective가 호출되고, f1과 같은 평가 지표의 합이 목푯값으로 반환됩니다.

```py
>>> best_trial = trainer.hyperparameter_search(
...     direction="maximize",
...     backend="optuna",
...     hp_space=optuna_hp_space,
...     n_trials=20,
...     compute_objective=compute_objective,
... )
```

## DDP 미세 조정을 위한 하이퍼파라미터 탐색 [[hyperparameter-search-for-ddp-finetune]]
현재, DDP(Distributed Data Parallelism)를 위한 하이퍼파라미터 탐색은 optuna와 sigopt에서 가능합니다. 최상위 프로세스가 하이퍼파라미터 탐색 과정을 시작하고 그 결과를 다른 프로세스에 전달합니다.
