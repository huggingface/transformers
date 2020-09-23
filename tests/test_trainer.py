import unittest

import datasets
import numpy as np

from transformers import AutoTokenizer, TrainingArguments, is_torch_available
from transformers.testing_utils import get_tests_dir, require_torch, slow


if is_torch_available():
    import torch
    from torch.utils.data import IterableDataset

    from transformers import (
        AutoModelForSequenceClassification,
        GlueDataset,
        GlueDataTrainingArguments,
        LineByLineTextDataset,
        Trainer,
    )


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


class AlmostAccuracy:
    def __init__(self, thresh=0.25):
        self.thresh = thresh

    def __call__(self, eval_pred):
        predictions, labels = eval_pred
        true = np.abs(predictions - labels) <= self.thresh
        return {"accuracy": true.astype(np.float32).mean().item()}


if is_torch_available():

    class SampleIterableDataset(IterableDataset):
        def __init__(self, file_path):
            self.file_path = file_path

        def parse_file(self):
            f = open(self.file_path, "r")
            return f.readlines()

        def __iter__(self):
            return iter(self.parse_file())

    class RegressionModel(torch.nn.Module):
        def __init__(self, a=0, b=0, double_output=False):
            super().__init__()
            self.a = torch.nn.Parameter(torch.tensor(a).float())
            self.b = torch.nn.Parameter(torch.tensor(b).float())
            self.double_output = double_output
            self.config = None

        def forward(self, input_x=None, labels=None, **kwargs):
            y = input_x * self.a + self.b
            if labels is None:
                return (y, y) if self.double_output else (y,)
            loss = torch.nn.functional.mse_loss(y, labels)
            return (loss, y, y) if self.double_output else (loss, y)

    def get_regression_trainer(a=0, b=0, double_output=False, train_len=64, eval_len=64, **kwargs):
        label_names = kwargs.get("label_names", None)
        train_dataset = RegressionDataset(length=train_len, label_names=label_names)
        eval_dataset = RegressionDataset(length=eval_len, label_names=label_names)
        model = RegressionModel(a, b, double_output)
        compute_metrics = kwargs.pop("compute_metrics", None)
        data_collator = kwargs.pop("data_collator", None)
        optimizers = kwargs.pop("optimizers", (None, None))
        args = TrainingArguments("./regression", **kwargs)
        return Trainer(
            model,
            args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            optimizers=optimizers,
        )


@require_torch
class TrainerIntegrationTest(unittest.TestCase):
    def setUp(self):
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

        # Check passing a new dataset for evaluation wors
        new_eval_dataset = RegressionDataset(length=128)
        self.assertEqual(len(trainer.get_eval_dataloader(new_eval_dataset)), 128 // (32 * n_gpu))

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

    def test_trainer_with_datasets(self):
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
        train_dataset.set_format(type="torch")
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

    def test_trainer_iterable_dataset(self):
        MODEL_ID = "sshleifer/tiny-distilbert-base-cased"
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        train_dataset = SampleIterableDataset(PATH_SAMPLE_TEXT)
        training_args = TrainingArguments(output_dir="./examples", no_cuda=True)
        trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
        loader = trainer.get_train_dataloader()
        self.assertIsInstance(loader, torch.utils.data.DataLoader)

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
