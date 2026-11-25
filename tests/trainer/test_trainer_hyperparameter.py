# Copyright 2018 the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Trainer hyperparameter search tests: Optuna (single/multi-objective, full eval),
Ray Tune (with client), W&B sweeps, and backend availability detection.
"""

import tempfile
import unittest

from transformers import TrainingArguments
from transformers.hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS, HPSearchBackend
from transformers.testing_utils import require_optuna, require_ray, require_torch, require_wandb, torch_device
from transformers.trainer_utils import IntervalStrategy
from transformers.utils.hp_naming import TrialShortNamer

from .trainer_test_utils import (
    AlmostAccuracy,
    RegressionModelConfig,
    RegressionPreTrainedModel,
    get_regression_trainer,
)


@require_torch
@require_optuna
class TrainerHyperParameterOptunaIntegrationTest(unittest.TestCase):
    def setUp(self):
        args = TrainingArguments("..")
        self.n_epochs = args.num_train_epochs
        self.batch_size = args.train_batch_size

    def test_hyperparameter_search(self):
        class MyTrialShortNamer(TrialShortNamer):
            DEFAULTS = {"a": 0, "b": 0}

        def hp_space(trial):
            return {}

        def model_init(trial):
            if trial is not None:
                a = trial.suggest_int("a", -4, 4)
                b = trial.suggest_int("b", -4, 4)
            else:
                a = 0
                b = 0
            config = RegressionModelConfig(a=a, b=b, double_output=False)

            return RegressionPreTrainedModel(config).to(torch_device)

        def hp_name(trial):
            return MyTrialShortNamer.shortname(trial.params)

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                output_dir=tmp_dir,
                learning_rate=0.1,
                logging_steps=1,
                eval_strategy=IntervalStrategy.EPOCH,
                save_strategy=IntervalStrategy.EPOCH,
                num_train_epochs=4,
                disable_tqdm=True,
                load_best_model_at_end=True,
                run_name="test",
                model_init=model_init,
            )
            trainer.hyperparameter_search(direction="minimize", hp_space=hp_space, hp_name=hp_name, n_trials=4)


@require_torch
@require_optuna
class TrainerHyperParameterMultiObjectOptunaIntegrationTest(unittest.TestCase):
    def setUp(self):
        args = TrainingArguments("..")
        self.n_epochs = args.num_train_epochs
        self.batch_size = args.train_batch_size

    def test_hyperparameter_search(self):
        class MyTrialShortNamer(TrialShortNamer):
            DEFAULTS = {"a": 0, "b": 0}

        def hp_space(trial):
            return {}

        def model_init(trial):
            if trial is not None:
                a = trial.suggest_int("a", -4, 4)
                b = trial.suggest_int("b", -4, 4)
            else:
                a = 0
                b = 0
            config = RegressionModelConfig(a=a, b=b, double_output=False)

            return RegressionPreTrainedModel(config).to(torch_device)

        def hp_name(trial):
            return MyTrialShortNamer.shortname(trial.params)

        def compute_objective(metrics: dict[str, float]) -> list[float]:
            return metrics["eval_loss"], metrics["eval_accuracy"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                output_dir=tmp_dir,
                learning_rate=0.1,
                logging_steps=1,
                eval_strategy=IntervalStrategy.EPOCH,
                save_strategy=IntervalStrategy.EPOCH,
                num_train_epochs=10,
                disable_tqdm=True,
                load_best_model_at_end=True,
                run_name="test",
                model_init=model_init,
                compute_metrics=AlmostAccuracy(),
            )
            trainer.hyperparameter_search(
                direction=["minimize", "maximize"],
                hp_space=hp_space,
                hp_name=hp_name,
                n_trials=4,
                compute_objective=compute_objective,
            )


@require_torch
@require_optuna
class TrainerHyperParameterOptunaIntegrationTestWithFullEval(unittest.TestCase):
    def test_hyperparameter_search(self):
        def hp_space(trial):
            return {}

        def model_init(trial):
            if trial is not None:
                a = trial.suggest_int("a", -4, 4)
                b = trial.suggest_int("b", -4, 4)
            else:
                a = 0
                b = 0
            config = RegressionModelConfig(a=a, b=b, double_output=False)

            return RegressionPreTrainedModel(config).to(torch_device)

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                output_dir=tmp_dir,
                disable_tqdm=True,
                model_init=model_init,
                fp16_full_eval=True,
            )
            trainer.hyperparameter_search(
                direction="minimize",
                hp_space=hp_space,
                n_trials=2,
            )


@require_torch
@require_ray
@unittest.skip("don't work because of a serialization issue")
class TrainerHyperParameterRayIntegrationTest(unittest.TestCase):
    def setUp(self):
        args = TrainingArguments("..")
        self.n_epochs = args.num_train_epochs
        self.batch_size = args.train_batch_size

    def ray_hyperparameter_search(self):
        class MyTrialShortNamer(TrialShortNamer):
            DEFAULTS = {"a": 0, "b": 0}

        def hp_space(trial):
            from ray import tune

            return {
                "a": tune.randint(-4, 4),
                "b": tune.randint(-4, 4),
            }

        def model_init(config):
            if config is None:
                a = 0
                b = 0
            else:
                a = config["a"]
                b = config["b"]
            model_config = RegressionModelConfig(a=a, b=b, double_output=False)

            return RegressionPreTrainedModel(model_config).to(torch_device)

        def hp_name(params):
            return MyTrialShortNamer.shortname(params)

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                output_dir=tmp_dir,
                learning_rate=0.1,
                logging_steps=1,
                eval_strategy=IntervalStrategy.EPOCH,
                save_strategy=IntervalStrategy.EPOCH,
                num_train_epochs=4,
                disable_tqdm=True,
                load_best_model_at_end=True,
                run_name="test",
                model_init=model_init,
            )
            trainer.hyperparameter_search(
                direction="minimize", hp_space=hp_space, hp_name=hp_name, backend="ray", n_trials=4
            )

    def test_hyperparameter_search(self):
        self.ray_hyperparameter_search()

    def test_hyperparameter_search_ray_client(self):
        import ray
        from ray.util.client.ray_client_helpers import ray_start_client_server

        with ray_start_client_server():
            assert ray.util.client.ray.is_connected()
            self.ray_hyperparameter_search()


@require_torch
@require_wandb
class TrainerHyperParameterWandbIntegrationTest(unittest.TestCase):
    def setUp(self):
        args = TrainingArguments("..")
        self.n_epochs = args.num_train_epochs
        self.batch_size = args.train_batch_size

    def test_hyperparameter_search(self):
        def hp_space(trial):
            return {
                "method": "random",
                "metric": {},
                "parameters": {
                    "a": {"distribution": "uniform", "min": 1e-6, "max": 1e-4},
                    "b": {"distribution": "int_uniform", "min": 1, "max": 6},
                },
            }

        def model_init(config):
            if config is None:
                a = 0
                b = 0
            else:
                a = config["a"]
                b = config["b"]
            model_config = RegressionModelConfig(a=a, b=b, double_output=False)

            return RegressionPreTrainedModel(model_config).to(torch_device)

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                output_dir=tmp_dir,
                learning_rate=0.1,
                logging_steps=1,
                eval_strategy=IntervalStrategy.EPOCH,
                save_strategy=IntervalStrategy.EPOCH,
                num_train_epochs=4,
                disable_tqdm=True,
                load_best_model_at_end=True,
                run_name="test",
                model_init=model_init,
            )
            sweep_kwargs = {
                "direction": "minimize",
                "hp_space": hp_space,
                "backend": "wandb",
                "n_trials": 4,
            }
            best_run = trainer.hyperparameter_search(**sweep_kwargs)

            self.assertIsNotNone(best_run.run_id)
            self.assertIsNotNone(best_run.run_summary)
            hp_keys = set(best_run.hyperparameters.keys())
            self.assertSetEqual(hp_keys, {"a", "b", "assignments", "metric"})

            # pretend restarting the process purged the environ
            import os

            del os.environ["WANDB_ENTITY"]
            del os.environ["WANDB_PROJECT"]
            sweep_kwargs["sweep_id"] = best_run.run_summary
            updated_best_run = trainer.hyperparameter_search(**sweep_kwargs)

            self.assertIsNotNone(updated_best_run.run_id)
            self.assertEqual(updated_best_run.run_summary, best_run.run_summary)
            updated_hp_keys = set(updated_best_run.hyperparameters.keys())
            self.assertSetEqual(updated_hp_keys, {"a", "b", "assignments", "metric"})


class HyperParameterSearchBackendsTest(unittest.TestCase):
    def test_hyperparameter_search_backends(self):
        self.assertEqual(
            list(ALL_HYPERPARAMETER_SEARCH_BACKENDS.keys()),
            list(HPSearchBackend),
        )
