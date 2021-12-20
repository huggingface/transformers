# coding=utf-8
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

import dataclasses
import gc
import math
import os
import random
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from huggingface_hub import Repository, delete_repo, login
from requests.exceptions import HTTPError
from transformers import (
    AutoTokenizer,
    IntervalStrategy,
    PretrainedConfig,
    TrainingArguments,
    is_torch_available,
    logging,
)
from transformers.file_utils import WEIGHTS_NAME
from transformers.testing_utils import (
    ENDPOINT_STAGING,
    PASS,
    USER,
    CaptureLogger,
    TestCasePlus,
    get_gpu_count,
    get_tests_dir,
    is_staging_test,
    require_optuna,
    require_ray,
    require_sentencepiece,
    require_sigopt,
    require_tokenizers,
    require_torch,
    require_torch_bf16,
    require_torch_gpu,
    require_torch_multi_gpu,
    require_torch_non_multi_gpu,
    require_torch_tf32,
    require_torch_up_to_2_gpus,
    slow,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils.hp_naming import TrialShortNamer


if is_torch_available():
    import torch
    from torch import nn
    from torch.utils.data import IterableDataset

    from transformers import (
        AutoModelForSequenceClassification,
        EarlyStoppingCallback,
        GlueDataset,
        GlueDataTrainingArguments,
        GPT2Config,
        GPT2LMHeadModel,
        LineByLineTextDataset,
        PreTrainedModel,
        Trainer,
        TrainerState,
    )
    from transformers.modeling_utils import unwrap_model


PATH_SAMPLE_TEXT = f"{get_tests_dir()}/fixtures/sample_text.txt"


class RegressionDataset:
    def __init__(self, a=2, b=3, length=64, seed=42, label_names=None):
        np.random.seed(seed)
        self.label_names = ["labels"] if label_names is None else label_names
        self.length = length
        self.x = np.random.normal(size=(length,)).astype(np.float32)
        self.ys = [a * self.x + b + np.random.normal(scale=0.1, size=(length,)) for _ in self.label_names]
        self.ys = [y.astype(np.float32) for y in self.ys]

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        result = {name: y[i] for name, y in zip(self.label_names, self.ys)}
        result["input_x"] = self.x[i]
        return result


@dataclasses.dataclass
class RegressionTrainingArguments(TrainingArguments):
    a: float = 0.0
    b: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        # save resources not dealing with reporting (also avoids the warning when it's not set)
        self.report_to = []


class RepeatDataset:
    def __init__(self, x, length=64):
        self.x = x
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return {"input_ids": self.x, "labels": self.x}


class DynamicShapesDataset:
    def __init__(self, length=64, seed=42, batch_size=8):
        self.length = length
        np.random.seed(seed)
        sizes = np.random.randint(1, 20, (length // batch_size,))
        # For easy batching, we make every batch_size consecutive samples the same size.
        self.xs = [np.random.normal(size=(s,)) for s in sizes.repeat(batch_size)]
        self.ys = [np.random.normal(size=(s,)) for s in sizes.repeat(batch_size)]

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return {"input_x": self.xs[i], "labels": self.ys[i]}


class AlmostAccuracy:
    def __init__(self, thresh=0.25):
        self.thresh = thresh

    def __call__(self, eval_pred):
        predictions, labels = eval_pred
        true = np.abs(predictions - labels) <= self.thresh
        return {"accuracy": true.astype(np.float32).mean().item()}


class RegressionModelConfig(PretrainedConfig):
    def __init__(self, a=0, b=0, double_output=False, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b
        self.double_output = double_output
        self.hidden_size = 1


if is_torch_available():

    class SampleIterableDataset(IterableDataset):
        def __init__(self, a=2, b=3, length=64, seed=42, label_names=None):
            self.dataset = RegressionDataset(a=a, b=b, length=length, seed=seed, label_names=label_names)

        def __iter__(self):
            for i in range(len(self.dataset)):
                yield self.dataset[i]

    class FiniteIterableDataset(SampleIterableDataset):
        def __init__(self, a=2, b=3, length=64, seed=42, label_names=None):
            super().__init__(a, b, length, seed, label_names)
            self.current_sample = 0

        def __iter__(self):
            while self.current_sample < len(self.dataset):
                yield self.dataset[self.current_sample]
                self.current_sample += 1

    class RegressionModel(nn.Module):
        def __init__(self, a=0, b=0, double_output=False):
            super().__init__()
            self.a = nn.Parameter(torch.tensor(a).float())
            self.b = nn.Parameter(torch.tensor(b).float())
            self.double_output = double_output
            self.config = None

        def forward(self, input_x, labels=None, **kwargs):
            y = input_x * self.a + self.b
            if labels is None:
                return (y, y) if self.double_output else (y,)
            loss = nn.functional.mse_loss(y, labels)
            return (loss, y, y) if self.double_output else (loss, y)

    class RegressionDictModel(nn.Module):
        def __init__(self, a=0, b=0):
            super().__init__()
            self.a = nn.Parameter(torch.tensor(a).float())
            self.b = nn.Parameter(torch.tensor(b).float())
            self.config = None

        def forward(self, input_x, labels=None, **kwargs):
            y = input_x * self.a + self.b
            result = {"output": y}
            if labels is not None:
                result["loss"] = nn.functional.mse_loss(y, labels)
            return result

    class RegressionPreTrainedModel(PreTrainedModel):
        config_class = RegressionModelConfig
        base_model_prefix = "regression"

        def __init__(self, config):
            super().__init__(config)
            self.a = nn.Parameter(torch.tensor(config.a).float())
            self.b = nn.Parameter(torch.tensor(config.b).float())
            self.double_output = config.double_output

        def forward(self, input_x, labels=None, **kwargs):
            y = input_x * self.a + self.b
            if labels is None:
                return (y, y) if self.double_output else (y,)
            loss = nn.functional.mse_loss(y, labels)
            return (loss, y, y) if self.double_output else (loss, y)

    class RegressionRandomPreTrainedModel(PreTrainedModel):
        config_class = RegressionModelConfig
        base_model_prefix = "regression"

        def __init__(self, config):
            super().__init__(config)
            self.a = nn.Parameter(torch.tensor(config.a).float())
            self.b = nn.Parameter(torch.tensor(config.b).float())

        def forward(self, input_x, labels=None, **kwargs):
            y = input_x * self.a + self.b
            torch_rand = torch.randn(1).squeeze()
            np_rand = np.random.rand()
            rand_rand = random.random()

            y += 0.05 * torch_rand + 0.05 * torch.tensor(np_rand + rand_rand)

            if labels is None:
                return (y,)
            loss = nn.functional.mse_loss(y, labels)
            return (loss, y)

    class TstLayer(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.linear1 = nn.Linear(hidden_size, hidden_size)
            self.ln1 = nn.LayerNorm(hidden_size)
            self.linear2 = nn.Linear(hidden_size, hidden_size)
            self.ln2 = nn.LayerNorm(hidden_size)
            self.bias = nn.Parameter(torch.zeros(hidden_size))

        def forward(self, x):
            h = self.ln1(nn.functional.relu(self.linear1(x)))
            h = nn.functional.relu(self.linear2(x))
            return self.ln2(x + h + self.bias)

    def get_regression_trainer(a=0, b=0, double_output=False, train_len=64, eval_len=64, pretrained=True, **kwargs):
        label_names = kwargs.get("label_names", None)
        train_dataset = RegressionDataset(length=train_len, label_names=label_names)
        eval_dataset = RegressionDataset(length=eval_len, label_names=label_names)

        model_init = kwargs.pop("model_init", None)
        if model_init is not None:
            model = None
        else:
            if pretrained:
                config = RegressionModelConfig(a=a, b=b, double_output=double_output)
                model = RegressionPreTrainedModel(config)
            else:
                model = RegressionModel(a=a, b=b, double_output=double_output)

        compute_metrics = kwargs.pop("compute_metrics", None)
        data_collator = kwargs.pop("data_collator", None)
        optimizers = kwargs.pop("optimizers", (None, None))
        output_dir = kwargs.pop("output_dir", "./regression")

        args = RegressionTrainingArguments(output_dir, a=a, b=b, **kwargs)
        return Trainer(
            model,
            args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            optimizers=optimizers,
            model_init=model_init,
        )


class TrainerIntegrationCommon:
    def check_saved_checkpoints(self, output_dir, freq, total, is_pretrained=True):
        file_list = [WEIGHTS_NAME, "training_args.bin", "optimizer.pt", "scheduler.pt", "trainer_state.json"]
        if is_pretrained:
            file_list.append("config.json")
        for step in range(freq, total, freq):
            checkpoint = os.path.join(output_dir, f"checkpoint-{step}")
            self.assertTrue(os.path.isdir(checkpoint))
            for filename in file_list:
                self.assertTrue(os.path.isfile(os.path.join(checkpoint, filename)))

    def check_best_model_has_been_loaded(
        self, output_dir, freq, total, trainer, metric, greater_is_better=False, is_pretrained=True
    ):
        checkpoint = os.path.join(output_dir, f"checkpoint-{(total // freq) * freq}")
        log_history = TrainerState.load_from_json(os.path.join(checkpoint, "trainer_state.json")).log_history

        values = [d[metric] for d in log_history]
        best_value = max(values) if greater_is_better else min(values)
        best_checkpoint = (values.index(best_value) + 1) * freq
        checkpoint = os.path.join(output_dir, f"checkpoint-{best_checkpoint}")
        if is_pretrained:
            best_model = RegressionPreTrainedModel.from_pretrained(checkpoint)
            best_model.to(trainer.args.device)
        else:
            best_model = RegressionModel()
            state_dict = torch.load(os.path.join(checkpoint, WEIGHTS_NAME))
            best_model.load_state_dict(state_dict)
            best_model.to(trainer.args.device)
        self.assertTrue(torch.allclose(best_model.a, trainer.model.a))
        self.assertTrue(torch.allclose(best_model.b, trainer.model.b))

        metrics = trainer.evaluate()
        self.assertEqual(metrics[metric], best_value)

    def check_trainer_state_are_the_same(self, trainer_state, trainer_state1):
        # We'll pop things so operate on copies.
        state = trainer_state.copy()
        state1 = trainer_state1.copy()
        # Log history main contain different logs for the time metrics (after resuming a training).
        log_history = state.pop("log_history", None)
        log_history1 = state1.pop("log_history", None)
        self.assertEqual(state, state1)
        skip_log_keys = ["train_runtime", "train_samples_per_second", "train_steps_per_second", "train_loss"]
        for log, log1 in zip(log_history, log_history1):
            for key in skip_log_keys:
                _ = log.pop(key, None)
                _ = log1.pop(key, None)
            self.assertEqual(log, log1)


@require_torch
@require_sentencepiece
@require_tokenizers
class TrainerIntegrationPrerunTest(TestCasePlus, TrainerIntegrationCommon):
    """
    Only tests that want to tap into the auto-pre-run 2 trainings:
    - self.default_trained_model
    - self.alternate_trained_model
    directly, or via check_trained_model
    """

    def setUp(self):
        super().setUp()
        args = TrainingArguments(".")
        self.n_epochs = args.num_train_epochs
        self.batch_size = args.train_batch_size
        trainer = get_regression_trainer(learning_rate=0.1)
        trainer.train()
        self.default_trained_model = (trainer.model.a, trainer.model.b)

        trainer = get_regression_trainer(learning_rate=0.1, seed=314)
        trainer.train()
        self.alternate_trained_model = (trainer.model.a, trainer.model.b)

    def check_trained_model(self, model, alternate_seed=False):
        # Checks a training seeded with learning_rate = 0.1
        (a, b) = self.alternate_trained_model if alternate_seed else self.default_trained_model
        self.assertTrue(torch.allclose(model.a, a))
        self.assertTrue(torch.allclose(model.b, b))

    def test_reproducible_training(self):
        # Checks that training worked, model trained and seed made a reproducible training.
        trainer = get_regression_trainer(learning_rate=0.1)
        trainer.train()
        self.check_trained_model(trainer.model)

        # Checks that a different seed gets different (reproducible) results.
        trainer = get_regression_trainer(learning_rate=0.1, seed=314)
        trainer.train()
        self.check_trained_model(trainer.model, alternate_seed=True)

    def test_trainer_with_datasets(self):
        import datasets

        np.random.seed(42)
        x = np.random.normal(size=(64,)).astype(np.float32)
        y = 2.0 * x + 3.0 + np.random.normal(scale=0.1, size=(64,))
        train_dataset = datasets.Dataset.from_dict({"input_x": x, "label": y})

        # Base training. Should have the same results as test_reproducible_training
        model = RegressionModel()
        args = TrainingArguments("./regression", learning_rate=0.1)
        trainer = Trainer(model, args, train_dataset=train_dataset)
        trainer.train()
        self.check_trained_model(trainer.model)

        # Can return tensors.
        train_dataset.set_format(type="torch", dtype=torch.float32)
        model = RegressionModel()
        trainer = Trainer(model, args, train_dataset=train_dataset)
        trainer.train()
        self.check_trained_model(trainer.model)

        # Adding one column not used by the model should have no impact
        z = np.random.normal(size=(64,)).astype(np.float32)
        train_dataset = datasets.Dataset.from_dict({"input_x": x, "label": y, "extra": z})
        model = RegressionModel()
        trainer = Trainer(model, args, train_dataset=train_dataset)
        trainer.train()
        self.check_trained_model(trainer.model)

    def test_model_init(self):
        train_dataset = RegressionDataset()
        args = TrainingArguments("./regression", learning_rate=0.1)
        trainer = Trainer(args=args, train_dataset=train_dataset, model_init=lambda: RegressionModel())
        trainer.train()
        self.check_trained_model(trainer.model)

        # Re-training should restart from scratch, thus lead the same results.
        trainer.train()
        self.check_trained_model(trainer.model)

        # Re-training should restart from scratch, thus lead the same results and new seed should be used.
        trainer.args.seed = 314
        trainer.train()
        self.check_trained_model(trainer.model, alternate_seed=True)

    def test_gradient_accumulation(self):
        # Training with half the batch size but accumulation steps as 2 should give the same results.
        trainer = get_regression_trainer(
            gradient_accumulation_steps=2, per_device_train_batch_size=4, learning_rate=0.1
        )
        trainer.train()
        self.check_trained_model(trainer.model)

    def test_custom_optimizer(self):
        train_dataset = RegressionDataset()
        args = TrainingArguments("./regression")
        model = RegressionModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1.0)
        trainer = Trainer(model, args, train_dataset=train_dataset, optimizers=(optimizer, lr_scheduler))
        trainer.train()

        (a, b) = self.default_trained_model
        self.assertFalse(torch.allclose(trainer.model.a, a))
        self.assertFalse(torch.allclose(trainer.model.b, b))
        self.assertEqual(trainer.optimizer.state_dict()["param_groups"][0]["lr"], 1.0)

    def test_adafactor_lr_none(self):
        # test the special case where lr=None, since Trainer can't not have lr_scheduler

        from transformers.optimization import Adafactor, AdafactorSchedule

        train_dataset = RegressionDataset()
        args = TrainingArguments("./regression")
        model = RegressionModel()
        optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
        lr_scheduler = AdafactorSchedule(optimizer)
        trainer = Trainer(model, args, train_dataset=train_dataset, optimizers=(optimizer, lr_scheduler))
        trainer.train()

        (a, b) = self.default_trained_model
        self.assertFalse(torch.allclose(trainer.model.a, a))
        self.assertFalse(torch.allclose(trainer.model.b, b))
        self.assertGreater(trainer.optimizer.state_dict()["param_groups"][0]["lr"], 0)

    @require_torch_gpu
    @require_torch_bf16
    def test_mixed_bf16(self):

        # very basic test
        trainer = get_regression_trainer(learning_rate=0.1, bf16=True)
        trainer.train()
        self.check_trained_model(trainer.model)

        # --bf16 --half_precision_backend apex can't be used together
        with self.assertRaises(ValueError):
            trainer = get_regression_trainer(learning_rate=0.1, bf16=True, half_precision_backend="apex")

        # will add more specific tests once there are some bugs to fix

    @require_torch_gpu
    @require_torch_tf32
    def test_tf32(self):

        # very basic test
        trainer = get_regression_trainer(learning_rate=0.1, tf32=True)
        trainer.train()
        self.check_trained_model(trainer.model)


@require_torch
@require_sentencepiece
@require_tokenizers
class TrainerIntegrationTest(TestCasePlus, TrainerIntegrationCommon):
    def setUp(self):
        super().setUp()
        args = TrainingArguments(".")
        self.n_epochs = args.num_train_epochs
        self.batch_size = args.train_batch_size

    def test_trainer_works_with_dict(self):
        # Edge case because Apex with mode O2 will change our models to return dicts. This test checks it doesn't break
        # anything.
        train_dataset = RegressionDataset()
        eval_dataset = RegressionDataset()
        model = RegressionDictModel()
        args = TrainingArguments("./regression")
        trainer = Trainer(model, args, train_dataset=train_dataset, eval_dataset=eval_dataset)
        trainer.train()
        _ = trainer.evaluate()
        _ = trainer.predict(eval_dataset)

    def test_evaluation_with_keys_to_drop(self):
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=32, n_layer=3, n_head=4)
        tiny_gpt2 = GPT2LMHeadModel(config)
        x = torch.randint(0, 100, (128,))
        eval_dataset = RepeatDataset(x)
        args = TrainingArguments("./test")
        trainer = Trainer(tiny_gpt2, args, eval_dataset=eval_dataset)
        # By default the past_key_values are removed
        result = trainer.predict(eval_dataset)
        self.assertTrue(isinstance(result.predictions, np.ndarray))
        # We can still get them by setting ignore_keys to []
        result = trainer.predict(eval_dataset, ignore_keys=[])
        self.assertTrue(isinstance(result.predictions, tuple))
        self.assertEqual(len(result.predictions), 2)

    def test_training_arguments_are_left_untouched(self):
        trainer = get_regression_trainer()
        trainer.train()
        args = TrainingArguments("./regression", report_to=[])
        dict1, dict2 = args.to_dict(), trainer.args.to_dict()
        for key in dict1.keys():
            # Logging dir can be slightly different as they default to something with the time.
            if key != "logging_dir":
                self.assertEqual(dict1[key], dict2[key])

    def test_number_of_steps_in_training(self):
        # Regular training has n_epochs * len(train_dl) steps
        trainer = get_regression_trainer(learning_rate=0.1)
        train_output = trainer.train()
        self.assertEqual(train_output.global_step, self.n_epochs * 64 / self.batch_size)

        # Check passing num_train_epochs works (and a float version too):
        trainer = get_regression_trainer(learning_rate=0.1, num_train_epochs=1.5)
        train_output = trainer.train()
        self.assertEqual(train_output.global_step, int(1.5 * 64 / self.batch_size))

        # If we pass a max_steps, num_train_epochs is ignored
        trainer = get_regression_trainer(learning_rate=0.1, max_steps=10)
        train_output = trainer.train()
        self.assertEqual(train_output.global_step, 10)

    def test_logging_inf_nan_filter(self):
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=32, n_layer=3, n_head=4)
        tiny_gpt2 = GPT2LMHeadModel(config)
        x = torch.randint(0, 100, (128,))
        train_dataset = RepeatDataset(x)

        # Trainer without inf/nan filter
        args = TrainingArguments("./test", learning_rate=1e9, logging_steps=5, logging_nan_inf_filter=False)
        trainer = Trainer(tiny_gpt2, args, train_dataset=train_dataset)
        trainer.train()
        log_history_no_filter = trainer.state.log_history

        # Trainer with inf/nan filter
        args = TrainingArguments("./test", learning_rate=1e9, logging_steps=5, logging_nan_inf_filter=True)
        trainer = Trainer(tiny_gpt2, args, train_dataset=train_dataset)
        trainer.train()
        log_history_filter = trainer.state.log_history

        def is_any_loss_nan_or_inf(log_history):
            losses = [l["loss"] for l in log_history[:-1]]
            return any(math.isnan(x) for x in losses) or any(math.isinf(x) for x in losses)

        self.assertTrue(is_any_loss_nan_or_inf(log_history_no_filter))
        self.assertFalse(is_any_loss_nan_or_inf(log_history_filter))

    def test_train_and_eval_dataloaders(self):
        n_gpu = max(1, torch.cuda.device_count())
        trainer = get_regression_trainer(learning_rate=0.1, per_device_train_batch_size=16)
        self.assertEqual(trainer.get_train_dataloader().batch_size, 16 * n_gpu)
        trainer = get_regression_trainer(learning_rate=0.1, per_device_eval_batch_size=16)
        self.assertEqual(trainer.get_eval_dataloader().batch_size, 16 * n_gpu)

        # Check drop_last works
        trainer = get_regression_trainer(
            train_len=66, eval_len=74, learning_rate=0.1, per_device_train_batch_size=16, per_device_eval_batch_size=32
        )
        self.assertEqual(len(trainer.get_train_dataloader()), 66 // (16 * n_gpu) + 1)
        self.assertEqual(len(trainer.get_eval_dataloader()), 74 // (32 * n_gpu) + 1)

        trainer = get_regression_trainer(
            train_len=66,
            eval_len=74,
            learning_rate=0.1,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            dataloader_drop_last=True,
        )
        self.assertEqual(len(trainer.get_train_dataloader()), 66 // (16 * n_gpu))
        self.assertEqual(len(trainer.get_eval_dataloader()), 74 // (32 * n_gpu))

        # Check passing a new dataset for evaluation works
        new_eval_dataset = RegressionDataset(length=128)
        self.assertEqual(len(trainer.get_eval_dataloader(new_eval_dataset)), 128 // (32 * n_gpu))

    @require_torch_multi_gpu
    def test_data_is_not_parallelized_when_model_is_parallel(self):
        model = RegressionModel()
        # Make the Trainer believe it's a parallelized model
        model.is_parallelizable = True
        model.model_parallel = True
        args = TrainingArguments("./regression", per_device_train_batch_size=16, per_device_eval_batch_size=16)
        trainer = Trainer(model, args, train_dataset=RegressionDataset(), eval_dataset=RegressionDataset())
        # Check the Trainer was fooled
        self.assertTrue(trainer.is_model_parallel)
        self.assertEqual(trainer.args.n_gpu, 1)

        # The batch size of the training and evaluation dataloaders should be 16, not 16 * n_gpu
        self.assertEqual(trainer.get_train_dataloader().batch_size, 16)
        self.assertEqual(len(trainer.get_train_dataloader()), 64 // 16)
        self.assertEqual(trainer.get_eval_dataloader().batch_size, 16)
        self.assertEqual(len(trainer.get_eval_dataloader()), 64 // 16)

    def test_evaluate(self):
        trainer = get_regression_trainer(a=1.5, b=2.5, compute_metrics=AlmostAccuracy())
        results = trainer.evaluate()

        x, y = trainer.eval_dataset.x, trainer.eval_dataset.ys[0]
        pred = 1.5 * x + 2.5
        expected_loss = ((pred - y) ** 2).mean()
        self.assertAlmostEqual(results["eval_loss"], expected_loss)
        expected_acc = AlmostAccuracy()((pred, y))["accuracy"]
        self.assertAlmostEqual(results["eval_accuracy"], expected_acc)

        # With a number of elements not a round multiple of the batch size
        trainer = get_regression_trainer(a=1.5, b=2.5, eval_len=66, compute_metrics=AlmostAccuracy())
        results = trainer.evaluate()

        x, y = trainer.eval_dataset.x, trainer.eval_dataset.ys[0]
        pred = 1.5 * x + 2.5
        expected_loss = ((pred - y) ** 2).mean()
        self.assertAlmostEqual(results["eval_loss"], expected_loss)
        expected_acc = AlmostAccuracy()((pred, y))["accuracy"]
        self.assertAlmostEqual(results["eval_accuracy"], expected_acc)

    def test_predict(self):
        trainer = get_regression_trainer(a=1.5, b=2.5)
        preds = trainer.predict(trainer.eval_dataset).predictions
        x = trainer.eval_dataset.x
        self.assertTrue(np.allclose(preds, 1.5 * x + 2.5))

        # With a number of elements not a round multiple of the batch size
        trainer = get_regression_trainer(a=1.5, b=2.5, eval_len=66)
        preds = trainer.predict(trainer.eval_dataset).predictions
        x = trainer.eval_dataset.x
        self.assertTrue(np.allclose(preds, 1.5 * x + 2.5))

        # With more than one output of the model
        trainer = get_regression_trainer(a=1.5, b=2.5, double_output=True)
        preds = trainer.predict(trainer.eval_dataset).predictions
        x = trainer.eval_dataset.x
        self.assertTrue(len(preds), 2)
        self.assertTrue(np.allclose(preds[0], 1.5 * x + 2.5))
        self.assertTrue(np.allclose(preds[1], 1.5 * x + 2.5))

        # With more than one output/label of the model
        trainer = get_regression_trainer(a=1.5, b=2.5, double_output=True, label_names=["labels", "labels_2"])
        outputs = trainer.predict(trainer.eval_dataset)
        preds = outputs.predictions
        labels = outputs.label_ids
        x = trainer.eval_dataset.x
        self.assertTrue(len(preds), 2)
        self.assertTrue(np.allclose(preds[0], 1.5 * x + 2.5))
        self.assertTrue(np.allclose(preds[1], 1.5 * x + 2.5))
        self.assertTrue(np.array_equal(labels[0], trainer.eval_dataset.ys[0]))
        self.assertTrue(np.array_equal(labels[1], trainer.eval_dataset.ys[1]))

    def test_dynamic_shapes(self):
        eval_dataset = DynamicShapesDataset(batch_size=self.batch_size)
        model = RegressionModel(a=2, b=1)
        args = TrainingArguments("./regression")
        trainer = Trainer(model, args, eval_dataset=eval_dataset)

        # Check evaluation can run to completion
        _ = trainer.evaluate()

        # Check predictions
        preds = trainer.predict(eval_dataset)
        for expected, seen in zip(eval_dataset.ys, preds.label_ids):
            self.assertTrue(np.array_equal(expected, seen[: expected.shape[0]]))
            self.assertTrue(np.all(seen[expected.shape[0] :] == -100))

        for expected, seen in zip(eval_dataset.xs, preds.predictions):
            self.assertTrue(np.array_equal(2 * expected + 1, seen[: expected.shape[0]]))
            self.assertTrue(np.all(seen[expected.shape[0] :] == -100))

        # Same tests with eval accumulation
        args = TrainingArguments("./regression", eval_accumulation_steps=2)
        trainer = Trainer(model, args, eval_dataset=eval_dataset)

        # Check evaluation can run to completion
        _ = trainer.evaluate()

        # Check predictions
        preds = trainer.predict(eval_dataset)
        for expected, seen in zip(eval_dataset.ys, preds.label_ids):
            self.assertTrue(np.array_equal(expected, seen[: expected.shape[0]]))
            self.assertTrue(np.all(seen[expected.shape[0] :] == -100))

        for expected, seen in zip(eval_dataset.xs, preds.predictions):
            self.assertTrue(np.array_equal(2 * expected + 1, seen[: expected.shape[0]]))
            self.assertTrue(np.all(seen[expected.shape[0] :] == -100))

    def test_log_level(self):
        # testing only --log_level (--log_level_replica requires multiple gpus and DDP and is tested elsewhere)
        logger = logging.get_logger()
        log_info_string = "Running training"

        # test with the default log_level - should be info and thus log on the main process
        with CaptureLogger(logger) as cl:
            trainer = get_regression_trainer()
            trainer.train()
        self.assertIn(log_info_string, cl.out)

        # test with low log_level - lower than info
        with CaptureLogger(logger) as cl:
            trainer = get_regression_trainer(log_level="debug")
            trainer.train()
        self.assertIn(log_info_string, cl.out)

        # test with high log_level - should be quiet
        with CaptureLogger(logger) as cl:
            trainer = get_regression_trainer(log_level="error")
            trainer.train()
        self.assertNotIn(log_info_string, cl.out)

    def test_save_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(output_dir=tmpdir, save_steps=5)
            trainer.train()
            self.check_saved_checkpoints(tmpdir, 5, int(self.n_epochs * 64 / self.batch_size))

        # With a regular model that is not a PreTrainedModel
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(output_dir=tmpdir, save_steps=5, pretrained=False)
            trainer.train()
            self.check_saved_checkpoints(tmpdir, 5, int(self.n_epochs * 64 / self.batch_size), False)

    @require_torch_multi_gpu
    def test_run_seq2seq_double_train_wrap_once(self):
        # test that we don't wrap the model more than once
        # since wrapping primarily happens on multi-gpu setup we want multiple gpus to test for
        # example DataParallel(DataParallel(model))

        trainer = get_regression_trainer()
        trainer.train()
        model_wrapped_before = trainer.model_wrapped
        trainer.train()
        model_wrapped_after = trainer.model_wrapped
        self.assertIs(model_wrapped_before, model_wrapped_after, "should be not wrapped twice")

    @require_torch_up_to_2_gpus
    def test_can_resume_training(self):
        # This test will fail for more than 2 GPUs since the batch size will get bigger and with the number of
        # save_steps, the checkpoint will resume training at epoch 2 or more (so the data seen by the model
        # won't be the same since the training dataloader is shuffled).

        with tempfile.TemporaryDirectory() as tmpdir:
            kwargs = dict(output_dir=tmpdir, train_len=128, save_steps=5, learning_rate=0.1)
            trainer = get_regression_trainer(**kwargs)
            trainer.train()
            (a, b) = trainer.model.a.item(), trainer.model.b.item()
            state = dataclasses.asdict(trainer.state)

            checkpoint = os.path.join(tmpdir, "checkpoint-5")

            # Reinitialize trainer
            trainer = get_regression_trainer(**kwargs)

            trainer.train(resume_from_checkpoint=checkpoint)
            (a1, b1) = trainer.model.a.item(), trainer.model.b.item()
            state1 = dataclasses.asdict(trainer.state)
            self.assertEqual(a, a1)
            self.assertEqual(b, b1)
            self.check_trainer_state_are_the_same(state, state1)

            # Now check with a later checkpoint that it also works when we span over one epoch
            checkpoint = os.path.join(tmpdir, "checkpoint-15")

            # Reinitialize trainer and load model
            trainer = get_regression_trainer(**kwargs)

            trainer.train(resume_from_checkpoint=checkpoint)
            (a1, b1) = trainer.model.a.item(), trainer.model.b.item()
            state1 = dataclasses.asdict(trainer.state)
            self.assertEqual(a, a1)
            self.assertEqual(b, b1)
            self.check_trainer_state_are_the_same(state, state1)

        # With a regular model that is not a PreTrainedModel
        with tempfile.TemporaryDirectory() as tmpdir:
            kwargs = dict(output_dir=tmpdir, train_len=128, save_steps=5, learning_rate=0.1, pretrained=False)

            trainer = get_regression_trainer(**kwargs)
            trainer.train()
            (a, b) = trainer.model.a.item(), trainer.model.b.item()
            state = dataclasses.asdict(trainer.state)

            checkpoint = os.path.join(tmpdir, "checkpoint-5")

            # Reinitialize trainer and load model
            trainer = get_regression_trainer(**kwargs)

            trainer.train(resume_from_checkpoint=checkpoint)
            (a1, b1) = trainer.model.a.item(), trainer.model.b.item()
            state1 = dataclasses.asdict(trainer.state)
            self.assertEqual(a, a1)
            self.assertEqual(b, b1)
            self.check_trainer_state_are_the_same(state, state1)

            # Now check with a later checkpoint that it also works when we span over one epoch
            checkpoint = os.path.join(tmpdir, "checkpoint-15")

            # Reinitialize trainer and load model
            trainer = get_regression_trainer(**kwargs)

            trainer.train(resume_from_checkpoint=checkpoint)
            (a1, b1) = trainer.model.a.item(), trainer.model.b.item()
            state1 = dataclasses.asdict(trainer.state)
            self.assertEqual(a, a1)
            self.assertEqual(b, b1)
            self.check_trainer_state_are_the_same(state, state1)

        # Now check failures

        # 1. fail to find a bogus checkpoint
        trainer = get_regression_trainer()
        with self.assertRaises(Exception) as context:
            trainer.train(resume_from_checkpoint=f"{checkpoint}-bogus")
        self.assertTrue("Can't find a valid checkpoint at" in str(context.exception))

        # 2. fail to find any checkpoint - due a fresh output_dir
        output_dir2 = self.get_auto_remove_tmp_dir()
        trainer = get_regression_trainer(output_dir=output_dir2)
        with self.assertRaises(Exception) as context:
            trainer.train(resume_from_checkpoint=True)
        self.assertTrue("No valid checkpoint found in output directory" in str(context.exception))

    @require_torch_non_multi_gpu
    def test_resume_training_with_randomness(self):
        # This test will fail flakily for more than 1 GPUs since the result will be slightly more different
        # TODO: investigate why it fails for 2 GPUs?

        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
        train_dataset = RegressionDataset(length=128)
        eval_dataset = RegressionDataset()

        config = RegressionModelConfig(a=0, b=2)
        model = RegressionRandomPreTrainedModel(config)

        tmp_dir = self.get_auto_remove_tmp_dir()
        args = RegressionTrainingArguments(tmp_dir, save_steps=5, learning_rate=0.1)
        trainer = Trainer(model, args, train_dataset=train_dataset, eval_dataset=eval_dataset)

        trainer.train()
        (a, b) = trainer.model.a.item(), trainer.model.b.item()

        model = RegressionRandomPreTrainedModel(config)
        trainer = Trainer(model, args, train_dataset=train_dataset, eval_dataset=eval_dataset)
        trainer.train(resume_from_checkpoint=os.path.join(tmp_dir, "checkpoint-15"))
        (a1, b1) = trainer.model.a.item(), trainer.model.b.item()

        self.assertAlmostEqual(a, a1, delta=1e-8)
        self.assertAlmostEqual(b, b1, delta=1e-8)

    # regression for this issue: https://github.com/huggingface/transformers/issues/12970
    def test_training_with_resume_from_checkpoint_false(self):
        train_dataset = RegressionDataset(length=128)
        eval_dataset = RegressionDataset()

        config = RegressionModelConfig(a=0, b=2)
        model = RegressionRandomPreTrainedModel(config)

        tmp_dir = self.get_auto_remove_tmp_dir()
        args = RegressionTrainingArguments(tmp_dir, save_steps=5, learning_rate=0.1)
        trainer = Trainer(model, args, train_dataset=train_dataset, eval_dataset=eval_dataset)

        trainer.train(resume_from_checkpoint=False)

    @require_torch_up_to_2_gpus
    def test_resume_training_with_gradient_accumulation(self):
        # This test will fail for more than 2 GPUs since the batch size will get bigger and with the number of
        # save_steps, the checkpoint will resume training at epoch 2 or more (so the data seen by the model
        # won't be the same since the training dataloader is shuffled).

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                output_dir=tmpdir,
                train_len=128,
                gradient_accumulation_steps=2,
                per_device_train_batch_size=4,
                save_steps=5,
                learning_rate=0.1,
            )
            trainer.train()
            (a, b) = trainer.model.a.item(), trainer.model.b.item()
            state = dataclasses.asdict(trainer.state)

            checkpoint = os.path.join(tmpdir, "checkpoint-5")

            # Reinitialize trainer
            trainer = get_regression_trainer(
                output_dir=tmpdir,
                train_len=128,
                gradient_accumulation_steps=2,
                per_device_train_batch_size=4,
                save_steps=5,
                learning_rate=0.1,
            )

            trainer.train(resume_from_checkpoint=checkpoint)
            (a1, b1) = trainer.model.a.item(), trainer.model.b.item()
            state1 = dataclasses.asdict(trainer.state)
            self.assertEqual(a, a1)
            self.assertEqual(b, b1)
            self.check_trainer_state_are_the_same(state, state1)

    @require_torch_up_to_2_gpus
    def test_resume_training_with_frozen_params(self):
        # This test will fail for more than 2 GPUs since the batch size will get bigger and with the number of
        # save_steps, the checkpoint will resume training at epoch 2 or more (so the data seen by the model
        # won't be the same since the training dataloader is shuffled).

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                output_dir=tmpdir,
                train_len=128,
                per_device_train_batch_size=4,
                save_steps=5,
                learning_rate=0.1,
            )
            trainer.model.a.requires_grad_(False)
            trainer.train()
            (a, b) = trainer.model.a.item(), trainer.model.b.item()
            state = dataclasses.asdict(trainer.state)

            checkpoint = os.path.join(tmpdir, "checkpoint-5")

            # Reinitialize trainer
            trainer = get_regression_trainer(
                output_dir=tmpdir,
                train_len=128,
                per_device_train_batch_size=4,
                save_steps=5,
                learning_rate=0.1,
            )
            trainer.model.a.requires_grad_(False)

            trainer.train(resume_from_checkpoint=checkpoint)

            self.assertFalse(trainer.model.a.requires_grad)
            (a1, b1) = trainer.model.a.item(), trainer.model.b.item()
            state1 = dataclasses.asdict(trainer.state)
            self.assertEqual(a, a1)
            self.assertEqual(b, b1)
            self.check_trainer_state_are_the_same(state, state1)

    def test_load_best_model_at_end(self):
        total = int(self.n_epochs * 64 / self.batch_size)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                a=1.5,
                b=2.5,
                output_dir=tmpdir,
                learning_rate=0.1,
                eval_steps=5,
                evaluation_strategy="steps",
                save_steps=5,
                load_best_model_at_end=True,
            )
            self.assertFalse(trainer.args.greater_is_better)
            trainer.train()
            self.check_saved_checkpoints(tmpdir, 5, total)
            self.check_best_model_has_been_loaded(tmpdir, 5, total, trainer, "eval_loss")

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                a=1.5,
                b=2.5,
                output_dir=tmpdir,
                learning_rate=0.1,
                eval_steps=5,
                evaluation_strategy="steps",
                save_steps=5,
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                compute_metrics=AlmostAccuracy(),
            )
            self.assertTrue(trainer.args.greater_is_better)
            trainer.train()
            self.check_saved_checkpoints(tmpdir, 5, total)
            self.check_best_model_has_been_loaded(tmpdir, 5, total, trainer, "eval_accuracy", greater_is_better=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                a=1.5,
                b=2.5,
                output_dir=tmpdir,
                learning_rate=0.1,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                compute_metrics=AlmostAccuracy(),
            )
            self.assertTrue(trainer.args.greater_is_better)
            trainer.train()
            self.check_saved_checkpoints(tmpdir, 64 // self.batch_size, total)
            self.check_best_model_has_been_loaded(
                tmpdir, 64 // self.batch_size, total, trainer, "eval_accuracy", greater_is_better=True
            )

        # Test this works with a non PreTrainedModel
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                output_dir=tmpdir,
                learning_rate=0.1,
                eval_steps=5,
                evaluation_strategy="steps",
                save_steps=5,
                load_best_model_at_end=True,
                pretrained=False,
            )
            self.assertFalse(trainer.args.greater_is_better)
            trainer.train()
            self.check_saved_checkpoints(tmpdir, 5, total, is_pretrained=False)
            self.check_best_model_has_been_loaded(tmpdir, 5, total, trainer, "eval_loss", is_pretrained=False)

    @slow
    def test_trainer_eval_mrpc(self):
        MODEL_ID = "bert-base-cased-finetuned-mrpc"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        data_args = GlueDataTrainingArguments(
            task_name="mrpc", data_dir=f"{get_tests_dir()}/fixtures/tests_samples/MRPC", overwrite_cache=True
        )
        eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="dev")

        training_args = TrainingArguments(output_dir="./examples", no_cuda=True)
        trainer = Trainer(model=model, args=training_args, eval_dataset=eval_dataset)
        result = trainer.evaluate()
        self.assertLess(result["eval_loss"], 0.2)

    @slow
    def test_trainer_eval_lm(self):
        MODEL_ID = "distilroberta-base"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        dataset = LineByLineTextDataset(
            tokenizer=tokenizer,
            file_path=PATH_SAMPLE_TEXT,
            block_size=tokenizer.max_len_single_sentence,
        )
        self.assertEqual(len(dataset), 31)

    def test_training_iterable_dataset(self):
        config = RegressionModelConfig()
        model = RegressionPreTrainedModel(config)
        train_dataset = SampleIterableDataset()

        args = RegressionTrainingArguments(output_dir="./examples", max_steps=4)
        trainer = Trainer(model=model, args=args, train_dataset=train_dataset)
        trainer.train()
        self.assertEqual(trainer.state.global_step, 4)

        loader = trainer.get_train_dataloader()
        self.assertIsInstance(loader, torch.utils.data.DataLoader)
        self.assertIsInstance(loader.sampler, torch.utils.data.dataloader._InfiniteConstantSampler)

    def test_training_finite_iterable_dataset(self):
        num_gpus = max(1, get_gpu_count())
        if num_gpus > 2:
            return

        config = RegressionModelConfig()
        model = RegressionPreTrainedModel(config)

        batch_size = 1
        num_samples = 10

        available_steps = num_samples // (batch_size * num_gpus)

        data = FiniteIterableDataset(length=num_samples)
        train_args = TrainingArguments(
            ".",
            max_steps=available_steps + 1,  # set a higher number than actually available
            per_device_train_batch_size=batch_size,
        )
        trainer = Trainer(model, train_dataset=data, args=train_args)
        with self.assertLogs("transformers.trainer", level="WARNING") as logs:
            trainer.train()
        self.assertIn(f"stopping training at step {available_steps}!", logs.output[0])

    def test_evaluation_iterable_dataset(self):
        config = RegressionModelConfig(a=1.5, b=2.5)
        model = RegressionPreTrainedModel(config)
        eval_dataset = SampleIterableDataset()

        args = RegressionTrainingArguments(output_dir="./examples")
        trainer = Trainer(model=model, args=args, eval_dataset=eval_dataset, compute_metrics=AlmostAccuracy())
        results = trainer.evaluate()

        x, y = trainer.eval_dataset.dataset.x, trainer.eval_dataset.dataset.ys[0]
        pred = 1.5 * x + 2.5
        expected_loss = ((pred - y) ** 2).mean()
        self.assertAlmostEqual(results["eval_loss"], expected_loss)
        expected_acc = AlmostAccuracy()((pred, y))["accuracy"]
        self.assertAlmostEqual(results["eval_accuracy"], expected_acc)

        # With a number of elements not a round multiple of the batch size
        eval_dataset = SampleIterableDataset(length=66)
        results = trainer.evaluate(eval_dataset)

        x, y = eval_dataset.dataset.x, eval_dataset.dataset.ys[0]
        pred = 1.5 * x + 2.5
        expected_loss = ((pred - y) ** 2).mean()
        self.assertAlmostEqual(results["eval_loss"], expected_loss)
        expected_acc = AlmostAccuracy()((pred, y))["accuracy"]
        self.assertAlmostEqual(results["eval_accuracy"], expected_acc)

    def test_predict_iterable_dataset(self):
        config = RegressionModelConfig(a=1.5, b=2.5)
        model = RegressionPreTrainedModel(config)
        eval_dataset = SampleIterableDataset()

        args = RegressionTrainingArguments(output_dir="./examples")
        trainer = Trainer(model=model, args=args, eval_dataset=eval_dataset, compute_metrics=AlmostAccuracy())

        preds = trainer.predict(trainer.eval_dataset).predictions
        x = eval_dataset.dataset.x
        self.assertTrue(np.allclose(preds, 1.5 * x + 2.5))

        # With a number of elements not a round multiple of the batch size
        test_dataset = SampleIterableDataset(length=66)
        preds = trainer.predict(test_dataset).predictions
        x = test_dataset.dataset.x
        self.assertTrue(np.allclose(preds, 1.5 * x + 2.5))

    def test_num_train_epochs_in_training(self):
        # len(train_dl) < gradient_accumulation_steps shouldn't give ``ZeroDivisionError`` when ``max_steps`` is given.
        # It should give 1 update step for each epoch.
        trainer = get_regression_trainer(
            max_steps=3, train_len=64, per_device_train_batch_size=16, gradient_accumulation_steps=5
        )
        train_output = trainer.train()
        self.assertEqual(train_output.global_step, 3)

        # Even ``max_steps`` is not specified, we still expect 1 update step for each epoch if
        # len(train_dl) < gradient_accumulation_steps.
        trainer = get_regression_trainer(train_len=64, per_device_train_batch_size=16, gradient_accumulation_steps=5)
        train_output = trainer.train()
        self.assertEqual(train_output.global_step, int(self.n_epochs))

    def test_early_stopping_callback(self):
        # early stopping stops training before num_training_epochs
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                output_dir=tmp_dir,
                num_train_epochs=20,
                gradient_accumulation_steps=1,
                per_device_train_batch_size=16,
                load_best_model_at_end=True,
                evaluation_strategy=IntervalStrategy.EPOCH,
                save_strategy=IntervalStrategy.EPOCH,
                compute_metrics=AlmostAccuracy(),
                metric_for_best_model="accuracy",
            )
            trainer.add_callback(EarlyStoppingCallback(1, 0.0001))
            train_output = trainer.train()
            self.assertLess(train_output.global_step, 20 * 64 / 16)

        # Invalid inputs to trainer with early stopping callback result in assertion error
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                output_dir=tmp_dir,
                num_train_epochs=20,
                gradient_accumulation_steps=1,
                per_device_train_batch_size=16,
                evaluation_strategy=IntervalStrategy.EPOCH,
                compute_metrics=AlmostAccuracy(),
                metric_for_best_model="accuracy",
            )
            trainer.add_callback(EarlyStoppingCallback(1))
            self.assertEqual(trainer.state.global_step, 0)
            try:
                trainer.train()
            except AssertionError:
                self.assertEqual(trainer.state.global_step, 0)

    def test_flos_extraction(self):
        trainer = get_regression_trainer(learning_rate=0.1)

        def assert_flos_extraction(trainer, wrapped_model_to_check):
            self.assertEqual(trainer.model, unwrap_model(wrapped_model_to_check))
            self.assertGreaterEqual(getattr(unwrap_model(wrapped_model_to_check).config, "total_flos", 0), 0)

        # with plain model
        assert_flos_extraction(trainer, trainer.model)

        # with enforced DataParallel
        assert_flos_extraction(trainer, nn.DataParallel(trainer.model))

        trainer.train()
        self.assertTrue(isinstance(trainer.state.total_flos, float))

    def check_checkpoint_deletion(self, trainer, output_dir, expected):
        # Make fake checkpoints
        for n in [5, 10, 15, 20, 25]:
            os.makedirs(os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}-{n}"), exist_ok=True)
        trainer._rotate_checkpoints(output_dir=output_dir)
        glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{PREFIX_CHECKPOINT_DIR}-*")]
        values = [int(re.match(f".*{PREFIX_CHECKPOINT_DIR}-([0-9]+)", d).groups()[0]) for d in glob_checkpoints]
        self.assertSetEqual(set(values), set(expected))

    def test_checkpoint_rotation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Without best model at end
            trainer = get_regression_trainer(output_dir=tmp_dir, save_total_limit=2)
            self.check_checkpoint_deletion(trainer, tmp_dir, [20, 25])

            # With best model at end
            trainer = get_regression_trainer(
                output_dir=tmp_dir, evaluation_strategy="steps", load_best_model_at_end=True, save_total_limit=2
            )
            trainer.state.best_model_checkpoint = os.path.join(tmp_dir, "checkpoint-5")
            self.check_checkpoint_deletion(trainer, tmp_dir, [5, 25])

            # Edge case: we don't always honor save_total_limit=1 if load_best_model_at_end=True to be able to resume
            # from checkpoint
            trainer = get_regression_trainer(
                output_dir=tmp_dir, evaluation_strategy="steps", load_best_model_at_end=True, save_total_limit=1
            )
            trainer.state.best_model_checkpoint = os.path.join(tmp_dir, "checkpoint-25")
            self.check_checkpoint_deletion(trainer, tmp_dir, [25])

            trainer.state.best_model_checkpoint = os.path.join(tmp_dir, "checkpoint-5")
            self.check_checkpoint_deletion(trainer, tmp_dir, [5, 25])

    def check_mem_metrics(self, trainer, check_func):
        metrics = trainer.train().metrics
        check_func("init_mem_cpu_alloc_delta", metrics)
        check_func("train_mem_cpu_alloc_delta", metrics)
        if torch.cuda.device_count() > 0:
            check_func("init_mem_gpu_alloc_delta", metrics)
            check_func("train_mem_gpu_alloc_delta", metrics)

        metrics = trainer.evaluate()
        check_func("eval_mem_cpu_alloc_delta", metrics)
        if torch.cuda.device_count() > 0:
            check_func("eval_mem_gpu_alloc_delta", metrics)

        metrics = trainer.predict(RegressionDataset()).metrics
        check_func("test_mem_cpu_alloc_delta", metrics)
        if torch.cuda.device_count() > 0:
            check_func("test_mem_gpu_alloc_delta", metrics)

    def test_mem_metrics(self):

        # with mem metrics enabled
        trainer = get_regression_trainer(skip_memory_metrics=False)
        self.check_mem_metrics(trainer, self.assertIn)

        # with mem metrics disabled
        trainer = get_regression_trainer(skip_memory_metrics=True)
        self.check_mem_metrics(trainer, self.assertNotIn)

    @require_torch_gpu
    def test_fp16_full_eval(self):

        # this is a sensitive test so let's keep debugging printouts in place for quick diagnosis.
        # it's using pretty large safety margins, but small enough to detect broken functionality.
        debug = 0
        n_gpus = get_gpu_count()

        bs = 8
        eval_len = 16 * n_gpus
        # make the params somewhat big so that there will be enough RAM consumed to be able to
        # measure things. We should get about 64KB for a+b in fp32
        a = torch.ones(1000, bs) + 0.001
        b = torch.ones(1000, bs) - 0.001

        # 1. with mem metrics enabled
        trainer = get_regression_trainer(a=a, b=b, eval_len=eval_len, skip_memory_metrics=False)
        metrics = trainer.evaluate()
        del trainer
        gc.collect()

        fp32_init = metrics["init_mem_gpu_alloc_delta"]
        fp32_eval = metrics["eval_mem_gpu_alloc_delta"]

        if debug:
            print(f"fp32_init {fp32_init}")
            print(f"fp32_eval {fp32_eval}")

        # here we expect the model to be preloaded in trainer.__init__ and consume around 64K gpu ram.
        # perfect world: fp32_init == 64<<10
        self.assertGreater(fp32_init, 59_000)
        # after eval should be no extra memory allocated - with a small margin (other than the peak
        # memory consumption for the forward calculation that gets recovered)
        # perfect world: fp32_eval == close to zero
        self.assertLess(fp32_eval, 5_000)

        # 2. with mem metrics disabled
        trainer = get_regression_trainer(a=a, b=b, eval_len=eval_len, fp16_full_eval=True, skip_memory_metrics=False)
        metrics = trainer.evaluate()
        fp16_init = metrics["init_mem_gpu_alloc_delta"]
        fp16_eval = metrics["eval_mem_gpu_alloc_delta"]

        if debug:
            print(f"fp16_init {fp16_init}")
            print(f"fp16_eval {fp16_eval}")

        # here we expect the model to not be preloaded in trainer.__init__, so with a small margin it should be close to 0
        # perfect world: fp16_init == close to zero
        self.assertLess(fp16_init, 5_000)
        # here we put the model on device in eval and only `half()` of it, i.e. about 32K,(again we ignore the peak margin which gets returned back)
        # perfect world: fp32_init == 32<<10
        self.assertGreater(fp16_eval, 27_000)

        # 3. relative comparison fp32 vs full fp16
        # should be about half of fp16_init
        # perfect world: fp32_init/2 == fp16_eval
        self.assertAlmostEqual(fp16_eval, fp32_init / 2, delta=5_000)

    @require_torch_gpu
    @require_torch_bf16
    def test_bf16_full_eval(self):
        # note: most of the logic is the same as test_fp16_full_eval

        # this is a sensitive test so let's keep debugging printouts in place for quick diagnosis.
        # it's using pretty large safety margins, but small enough to detect broken functionality.
        debug = 0
        n_gpus = get_gpu_count()

        bs = 8
        eval_len = 16 * n_gpus
        # make the params somewhat big so that there will be enough RAM consumed to be able to
        # measure things. We should get about 64KB for a+b in fp32
        a = torch.ones(1000, bs) + 0.001
        b = torch.ones(1000, bs) - 0.001

        # 1. with mem metrics enabled
        trainer = get_regression_trainer(a=a, b=b, eval_len=eval_len, skip_memory_metrics=False)
        metrics = trainer.evaluate()
        del trainer
        gc.collect()

        fp32_init = metrics["init_mem_gpu_alloc_delta"]
        fp32_eval = metrics["eval_mem_gpu_alloc_delta"]

        if debug:
            print(f"fp32_init {fp32_init}")
            print(f"fp32_eval {fp32_eval}")

        # here we expect the model to be preloaded in trainer.__init__ and consume around 64K gpu ram.
        # perfect world: fp32_init == 64<<10
        self.assertGreater(fp32_init, 59_000)
        # after eval should be no extra memory allocated - with a small margin (other than the peak
        # memory consumption for the forward calculation that gets recovered)
        # perfect world: fp32_eval == close to zero
        self.assertLess(fp32_eval, 5_000)

        # 2. with mem metrics disabled
        trainer = get_regression_trainer(a=a, b=b, eval_len=eval_len, bf16_full_eval=True, skip_memory_metrics=False)
        metrics = trainer.evaluate()
        bf16_init = metrics["init_mem_gpu_alloc_delta"]
        bf16_eval = metrics["eval_mem_gpu_alloc_delta"]

        if debug:
            print(f"bf16_init {bf16_init}")
            print(f"bf16_eval {bf16_eval}")

        # here we expect the model to not be preloaded in trainer.__init__, so with a small margin it should be close to 0
        # perfect world: bf16_init == close to zero
        self.assertLess(bf16_init, 5_000)
        # here we put the model on device in eval and only `half()` of it, i.e. about 32K,(again we ignore the peak margin which gets returned back)
        # perfect world: fp32_init == 32<<10
        self.assertGreater(bf16_eval, 27_000)

        # 3. relative comparison fp32 vs full bf16
        # should be about half of bf16_init
        # perfect world: fp32_init/2 == bf16_eval
        self.assertAlmostEqual(bf16_eval, fp32_init / 2, delta=5_000)

    def test_no_wd_param_group(self):
        model = nn.Sequential(TstLayer(128), nn.ModuleList([TstLayer(128), TstLayer(128)]))
        trainer = Trainer(model=model)
        trainer.create_optimizer_and_scheduler(10)
        # fmt: off
        wd_names = ['0.linear1.weight', '0.linear2.weight', '1.0.linear1.weight', '1.0.linear2.weight', '1.1.linear1.weight', '1.1.linear2.weight']
        # fmt: on
        wd_params = [p for n, p in model.named_parameters() if n in wd_names]
        no_wd_params = [p for n, p in model.named_parameters() if n not in wd_names]
        self.assertListEqual(trainer.optimizer.param_groups[0]["params"], wd_params)
        self.assertListEqual(trainer.optimizer.param_groups[1]["params"], no_wd_params)


@require_torch
@is_staging_test
class TrainerIntegrationWithHubTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._token = login(username=USER, password=PASS)

    @classmethod
    def tearDownClass(cls):
        for model in ["test-trainer", "test-trainer-epoch", "test-trainer-step"]:
            try:
                delete_repo(token=cls._token, name=model)
            except HTTPError:
                pass

        try:
            delete_repo(token=cls._token, name="test-trainer-org", organization="valid_org")
        except HTTPError:
            pass

    def test_push_to_hub(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                output_dir=os.path.join(tmp_dir, "test-trainer"),
                push_to_hub=True,
                hub_token=self._token,
            )
            url = trainer.push_to_hub()

            # Extract repo_name from the url
            re_search = re.search(ENDPOINT_STAGING + r"/([^/]+/[^/]+)/", url)
            self.assertTrue(re_search is not None)
            repo_name = re_search.groups()[0]

            self.assertEqual(repo_name, f"{USER}/test-trainer")

            model = RegressionPreTrainedModel.from_pretrained(repo_name)
            self.assertEqual(model.a.item(), trainer.model.a.item())
            self.assertEqual(model.b.item(), trainer.model.b.item())

    def test_push_to_hub_in_organization(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(output_dir=tmp_dir)
            trainer.save_model()
            trainer = get_regression_trainer(
                output_dir=os.path.join(tmp_dir, "test-trainer-org"),
                push_to_hub=True,
                hub_model_id="valid_org/test-trainer-org",
                hub_token=self._token,
            )
            url = trainer.push_to_hub()

            # Extract repo_name from the url
            re_search = re.search(ENDPOINT_STAGING + r"/([^/]+/[^/]+)/", url)
            self.assertTrue(re_search is not None)
            repo_name = re_search.groups()[0]
            self.assertEqual(repo_name, "valid_org/test-trainer-org")

            model = RegressionPreTrainedModel.from_pretrained("valid_org/test-trainer-org")
            self.assertEqual(model.a.item(), trainer.model.a.item())
            self.assertEqual(model.b.item(), trainer.model.b.item())

    def get_commit_history(self, repo):
        commit_logs = subprocess.run(
            "git log".split(),
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            check=True,
            encoding="utf-8",
            cwd=repo,
        ).stdout
        commits = commit_logs.split("\n\n")[1::2]
        return [commit.strip() for commit in commits]

    def test_push_to_hub_with_saves_each_epoch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                output_dir=os.path.join(tmp_dir, "test-trainer-epoch"),
                push_to_hub=True,
                hub_token=self._token,
                save_strategy="epoch",
            )
            trainer.train()

        with tempfile.TemporaryDirectory() as tmp_dir:
            _ = Repository(tmp_dir, clone_from=f"{USER}/test-trainer-epoch", use_auth_token=self._token)
            commits = self.get_commit_history(tmp_dir)
            expected_commits = [f"Training in progress, epoch {i}" for i in range(3, 0, -1)]
            expected_commits.append("initial commit")
            self.assertListEqual(commits, expected_commits)
            print(commits, len(commits))

    def test_push_to_hub_with_saves_each_n_steps(self):
        num_gpus = max(1, get_gpu_count())
        if num_gpus > 2:
            return

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                output_dir=os.path.join(tmp_dir, "test-trainer-step"),
                push_to_hub=True,
                hub_token=self._token,
                save_strategy="steps",
                save_steps=5,
            )
            trainer.train()

        with tempfile.TemporaryDirectory() as tmp_dir:
            _ = Repository(tmp_dir, clone_from=f"{USER}/test-trainer-step", use_auth_token=self._token)
            commits = self.get_commit_history(tmp_dir)
            total_steps = 20 // num_gpus
            expected_commits = [f"Training in progress, step {i}" for i in range(total_steps, 0, -5)]
            expected_commits.append("initial commit")
            self.assertListEqual(commits, expected_commits)
            print(commits, len(commits))


@require_torch
@require_optuna
class TrainerHyperParameterOptunaIntegrationTest(unittest.TestCase):
    def setUp(self):
        args = TrainingArguments(".")
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

            return RegressionPreTrainedModel(config)

        def hp_name(trial):
            return MyTrialShortNamer.shortname(trial.params)

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                output_dir=tmp_dir,
                learning_rate=0.1,
                logging_steps=1,
                evaluation_strategy=IntervalStrategy.EPOCH,
                save_strategy=IntervalStrategy.EPOCH,
                num_train_epochs=4,
                disable_tqdm=True,
                load_best_model_at_end=True,
                logging_dir="runs",
                run_name="test",
                model_init=model_init,
            )
            trainer.hyperparameter_search(direction="minimize", hp_space=hp_space, hp_name=hp_name, n_trials=4)


@require_torch
@require_ray
class TrainerHyperParameterRayIntegrationTest(unittest.TestCase):
    def setUp(self):
        args = TrainingArguments(".")
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

            return RegressionPreTrainedModel(model_config)

        def hp_name(params):
            return MyTrialShortNamer.shortname(params)

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                output_dir=tmp_dir,
                learning_rate=0.1,
                logging_steps=1,
                evaluation_strategy=IntervalStrategy.EPOCH,
                save_strategy=IntervalStrategy.EPOCH,
                num_train_epochs=4,
                disable_tqdm=True,
                load_best_model_at_end=True,
                logging_dir="runs",
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
@require_sigopt
class TrainerHyperParameterSigOptIntegrationTest(unittest.TestCase):
    def setUp(self):
        args = TrainingArguments(".")
        self.n_epochs = args.num_train_epochs
        self.batch_size = args.train_batch_size

    def test_hyperparameter_search(self):
        class MyTrialShortNamer(TrialShortNamer):
            DEFAULTS = {"a": 0, "b": 0}

        def hp_space(trial):
            return [
                {"bounds": {"min": -4, "max": 4}, "name": "a", "type": "int"},
                {"bounds": {"min": -4, "max": 4}, "name": "b", "type": "int"},
            ]

        def model_init(trial):
            if trial is not None:
                a = trial.assignments["a"]
                b = trial.assignments["b"]
            else:
                a = 0
                b = 0
            config = RegressionModelConfig(a=a, b=b, double_output=False)

            return RegressionPreTrainedModel(config)

        def hp_name(trial):
            return MyTrialShortNamer.shortname(trial.assignments)

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                output_dir=tmp_dir,
                learning_rate=0.1,
                logging_steps=1,
                evaluation_strategy=IntervalStrategy.EPOCH,
                save_strategy=IntervalStrategy.EPOCH,
                num_train_epochs=4,
                disable_tqdm=True,
                load_best_model_at_end=True,
                logging_dir="runs",
                run_name="test",
                model_init=model_init,
            )
            trainer.hyperparameter_search(
                direction="minimize", hp_space=hp_space, hp_name=hp_name, backend="sigopt", n_trials=4
            )
