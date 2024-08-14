# البحث عن أفضل المعلمات باستخدام واجهة برمجة تطبيقات المدرب

يوفر 🤗 Transformers فئة [`Trainer`] تم تحسينها لتدريب نماذج 🤗 Transformers، مما يسهل بدء التدريب دون الحاجة إلى كتابة حلقة التدريب الخاصة بك يدويًا. توفر واجهة برمجة التطبيقات [`Trainer`] واجهة برمجة تطبيقات للبحث عن أفضل المعلمات. توضح هذه الوثيقة كيفية تمكينها في المثال.

## backend البحث عن أفضل المعلمات

تدعم [`Trainer`] حاليًا أربع واجهات خلفية للبحث عن أفضل المعلمات: [optuna](https://optuna.org/)، [sigopt](https://sigopt.com/)، [raytune](https://docs.ray.io/en/latest/tune/index.html) و [wandb](https://wandb.ai/site/sweeps).

يجب تثبيتها قبل استخدامها كخلفية للبحث عن أفضل المعلمات
```bash
pip install optuna/sigopt/wandb/ray[tune] 
```

## كيفية تمكين البحث عن أفضل المعلمات في المثال

قم بتعريف مساحة البحث عن أفضل المعلمات، حيث تحتاج الخلفيات المختلفة إلى تنسيق مختلف.

بالنسبة إلى sigopt، راجع sigopt [object_parameter](https://docs.sigopt.com/ai-module-api-references/api_reference/objects/object_parameter)، فهو يشبه ما يلي:
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

بالنسبة إلى optuna، راجع optuna [object_parameter](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html#sphx-glr-tutorial-10-key-features-002-configurations-py)، فهو يشبه ما يلي:

```py
>>> def optuna_hp_space(trial):
...     return {
...         "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
...         "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128]),
...     }
```

يوفر Optuna HPO متعدد الأهداف. يمكنك تمرير `direction` في `hyperparameter_search` وتعريف compute_objective الخاص بك لإرجاع قيم الهدف المتعددة. سيتم إرجاع Pareto Front (`List[BestRun]`) في hyperparameter_search، ويجب أن ترجع إلى حالة الاختبار `TrainerHyperParameterMultiObjectOptunaIntegrationTest` في [test_trainer](https://github.com/huggingface/transformers/blob/main/tests/trainer/test_trainer.py). إنه يشبه ما يلي

```py
>>> best_trials = trainer.hyperparameter_search(
...     direction=["minimize", "maximize"],
...     backend="optuna",
...     hp_space=optuna_hp_space,
...     n_trials=20,
...     compute_objective=compute_objective,
... )
```

بالنسبة إلى raytune، راجع raytune [object_parameter](https://docs.ray.io/en/latest/tune/api/search_space.html)، فهو يشبه ما يلي:

```py
>>> def ray_hp_space(trial):
...     return {
...         "learning_rate": tune.loguniform(1e-6, 1e-4),
...         "per_device_train_batch_size": tune.choice([16, 32, 64, 128]),
...     }
```

بالنسبة إلى wandb، راجع wandb [object_parameter](https://docs.wandb.ai/guides/sweeps/configuration)، فهو يشبه ما يلي:
بالنسبة إلى wandb، راجع wandb [object_parameter](https://docs.wandb.ai/guides/sweeps/configuration)، فهو يشبه ما يلي:

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

قم بتعريف دالة `model_init` ومررها إلى [`Trainer`]، كمثال:
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

قم بإنشاء [`Trainer`] باستخدام دالة `model_init` الخاصة بك، وحجج التدريب، ومجموعات البيانات التدريبية والاختبارية، ودالة التقييم:

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

قم بالبحث عن أفضل المعلمات، واحصل على أفضل معلمات التجربة، ويمكن أن تكون الخلفية `"optuna"`/`"sigopt"`/`"wandb"`/`"ray"`. يمكن أن يكون الاتجاه `"minimize"` أو `"maximize"`، مما يشير إلى ما إذا كان سيتم تحسين الهدف الأكبر أو الأصغر.

يمكنك تعريف دالة compute_objective الخاصة بك، وإذا لم يتم تعريفها، فسيتم استدعاء دالة compute_objective الافتراضية، وسيتم إرجاع مجموع مقياس التقييم مثل f1 كقيمة للهدف.

```py
>>> best_trial = trainer.hyperparameter_search(
...     direction="maximize",
...     backend="optuna",
...     hp_space=optuna_hp_space,
...     n_trials=20,
...     compute_objective=compute_objective,
... )
```

## البحث عن أفضل المعلمات لضبط دقيق DDP
حاليًا، يتم تمكين البحث عن أفضل المعلمات لضبط دقيق DDP لـ optuna و sigopt. ستولد العملية ذات الترتيب الصفري فقط تجربة البحث وستمرر الحجة إلى الرتب الأخرى.