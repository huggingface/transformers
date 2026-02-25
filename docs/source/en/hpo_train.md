<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Hyperparameter search

Hyperparameters - learning rate, batch size, number of epochs - are choices that affect training and can produce very different results depending on your choice. [`Trainer.hyperparameter_search`] finds the best combination of hyperparameters by running multiple training trials, each with a different set of values, and returning the best one.

For each trial, [`Trainer.hyperparameter_search`] initializes a model from scratch with `model_init`, samples a new set of hyperparameters and runs a full training loop with them, evaluates and computes an objective, and reports the objective to the search backend which is used to inform the next trial. After a number of trials, the best set of hyperparameters are returned in [`~trainer.utils.BestRun`].

## Initializing a model

Start each trial with a fresh model to avoid the previous runs training state. `model_init` is called at the start of each trial and returns a new model instance, so every trial starts from the same initial weights.

```py
from transformers import AutoModelForCausalLM

def model_init(trial):
    return AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

trainer = Trainer(
    model_init=model_init,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
```

Don't pass `model=` with `model_init=`, otherwise it'll return an error.

## Define the search space

Create a function that defines the search space. The format depends on the backend. If you don't define a `hp_space` function, the default searches over `learning_rate`, `num_train_epochs`, and `per_device_train_batch_size`.

```bash
# install one of these hyperparam search backends
pip install optuna
pip install wandb
pip install ray[tune]
```

<hfoptions id="backends">
<hfoption id="Optuna">

[Optuna](https://optuna.readthedocs.io/en/stable/index.html) is a lightweight framework for hyperparameter optimization.

```py
def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128]),
    }
```

</hfoption>
<hfoption id="Ray Tune">

[Ray Tune](https://docs.ray.io/en/latest/tune/index.html) is a scalable hyperparameter tuning library that can also distribute trials across multiple machines.

```py
from ray import tune

def hp_space(trial):
    return {
        "learning_rate": tune.loguniform(1e-6, 1e-4),
        "per_device_train_batch_size": tune.choice([16, 32, 64, 128]),
    }
```

</hfoption>
<hfoption id="Weights & Biases">

[Weights & Biases](https://docs.wandb.ai/) is an experiment tracking platform with built-in hyperparameter search. It supports Bayesian, random, and grid search strategies.

```py
def hp_space(trial):
    return {
        "method": "random",
        "metric": {"name": "objective", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"distribution": "uniform", "min": 1e-6, "max": 1e-4},
            "per_device_train_batch_size": {"values": [16, 32, 64, 128]},
        },
    }
```

</hfoption>
</hfoptions>

## Run the search

Optionally provide a `compute_objective` function that describes what to optimize for. It defaults to `eval_loss`, or the sum of all eval metrics if no loss is reported. The search `backend` optimizes for this value over `n_trials` search runs in your desired `direction`.

```py
def compute_objective(metrics):
    return metrics["eval_loss"]

best_run = trainer.hyperparameter_search(
    hp_space=hp_space,
    compute_objective=compute_objective,
    n_trials=30,               # how many trials to run
    direction="minimize",      # or "maximize" for metrics like accuracy/F1
    backend="optuna",          # "optuna", "ray", or "wandb"
)
```

This returns a [`~trainer.utils.BestRun`] object that contains the objective and best combination of hyperparameters.

```py
best_run = trainer.hyperparameter_search(...)

best_run.objective        # 0.38  (best eval loss)
best_run.hyperparameters  # {"learning_rate": 5e-5, "num_train_epochs": 4, ...}
```

Apply the best hyperparameters to [`TrainingArguments`] and retrain on the full dataset.
