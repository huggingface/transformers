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
Trainer evaluation and prediction tests: evaluate, predict, batched metrics, dynamic shapes,
iterable datasets, early stopping, FP16/BF16 full eval memory, torch.compile, and MRPC/LM eval.
"""

import gc
import tempfile

import numpy as np

from transformers import (
    AutoTokenizer,
    TrainingArguments,
    is_torch_available,
)
from transformers.testing_utils import (
    TestCasePlus,
    backend_device_count,
    get_tests_dir,
    require_torch,
    require_torch_accelerator,
    require_torch_bf16,
    require_torch_fp16,
    slow,
    torch_device,
)

from .trainer_test_utils import (
    PATH_SAMPLE_TEXT,
    AlmostAccuracy,
    AlmostAccuracyBatched,
    RegressionDataset,
    RegressionDictModel,
    TrainerIntegrationCommon,
    get_dataset,
    get_regression_trainer,
)


if is_torch_available():
    import torch

    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        GlueDataset,
        GlueDataTrainingArguments,
        Trainer,
    )


# ---------------------------------------------------------------------------
# Core evaluate / predict tests
# ---------------------------------------------------------------------------


@require_torch
class TrainerEvaluationTest(TestCasePlus, TrainerIntegrationCommon):
    def setUp(self):
        super().setUp()
        args = TrainingArguments("..")
        self.n_epochs = args.num_train_epochs
        self.batch_size = args.train_batch_size

    def test_evaluate(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(a=1.5, b=2.5, compute_metrics=AlmostAccuracy(), output_dir=tmp_dir)
            results = trainer.evaluate()

            x, y = trainer.eval_dataset.x, trainer.eval_dataset.ys[0]
            pred = 1.5 * x + 2.5
            expected_loss = ((pred - y) ** 2).mean()
            self.assertAlmostEqual(results["eval_loss"], expected_loss)
            expected_acc = AlmostAccuracy()((pred, y))["accuracy"]
            self.assertAlmostEqual(results["eval_accuracy"], expected_acc)

            # With a number of elements not a round multiple of the batch size
            trainer = get_regression_trainer(
                a=1.5, b=2.5, eval_len=66, compute_metrics=AlmostAccuracy(), output_dir=tmp_dir
            )
            results = trainer.evaluate()

            x, y = trainer.eval_dataset.x, trainer.eval_dataset.ys[0]
            pred = 1.5 * x + 2.5
            expected_loss = ((pred - y) ** 2).mean()
            self.assertAlmostEqual(results["eval_loss"], expected_loss)
            expected_acc = AlmostAccuracy()((pred, y))["accuracy"]
            self.assertAlmostEqual(results["eval_accuracy"], expected_acc)

            # With logits preprocess
            trainer = get_regression_trainer(
                a=1.5,
                b=2.5,
                compute_metrics=AlmostAccuracy(),
                preprocess_logits_for_metrics=lambda logits, labels: logits + 1,
                output_dir=tmp_dir,
            )
            results = trainer.evaluate()

            x, y = trainer.eval_dataset.x, trainer.eval_dataset.ys[0]
            pred = 1.5 * x + 2.5
            expected_loss = ((pred - y) ** 2).mean()
            self.assertAlmostEqual(results["eval_loss"], expected_loss)
            expected_acc = AlmostAccuracy()((pred + 1, y))["accuracy"]
            self.assertAlmostEqual(results["eval_accuracy"], expected_acc)

    def test_predict(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(a=1.5, b=2.5, output_dir=tmp_dir)
            preds = trainer.predict(trainer.eval_dataset).predictions
            x = trainer.eval_dataset.x
            self.assertTrue(np.allclose(preds, 1.5 * x + 2.5))

            # With a number of elements not a round multiple of the batch size
            trainer = get_regression_trainer(a=1.5, b=2.5, eval_len=66, output_dir=tmp_dir)
            preds = trainer.predict(trainer.eval_dataset).predictions
            x = trainer.eval_dataset.x
            self.assertTrue(np.allclose(preds, 1.5 * x + 2.5))

            # With more than one output of the model
            trainer = get_regression_trainer(a=1.5, b=2.5, double_output=True, output_dir=tmp_dir)
            preds = trainer.predict(trainer.eval_dataset).predictions
            x = trainer.eval_dataset.x
            self.assertEqual(len(preds), 2)
            self.assertTrue(np.allclose(preds[0], 1.5 * x + 2.5))
            self.assertTrue(np.allclose(preds[1], 1.5 * x + 2.5))

            # With more than one output/label of the model
            trainer = get_regression_trainer(
                a=1.5, b=2.5, double_output=True, label_names=["labels", "labels_2"], output_dir=tmp_dir
            )
            outputs = trainer.predict(trainer.eval_dataset)
            preds = outputs.predictions
            labels = outputs.label_ids
            x = trainer.eval_dataset.x
            self.assertEqual(len(preds), 2)
            self.assertTrue(np.allclose(preds[0], 1.5 * x + 2.5))
            self.assertTrue(np.allclose(preds[1], 1.5 * x + 2.5))
            self.assertTrue(np.array_equal(labels[0], trainer.eval_dataset.ys[0]))
            self.assertTrue(np.array_equal(labels[1], trainer.eval_dataset.ys[1]))

    def test_train_and_predict_loss_parity(self):
        """
        Tests that the loss computed during a training_step is the same as the one computed during prediction_step.
        for the same inputs
        """
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
        # Create a dummy batch of inputs
        inputs = {}
        inputs["input_ids"] = []
        for row_ind in range(4):
            seq_len = torch.randint(32, 64, (1,)).item()
            x = torch.randint(1, 100, (seq_len,))
            inputs["input_ids"].append(x)
        inputs["input_ids"] = torch.nn.utils.rnn.pad_sequence(inputs["input_ids"], batch_first=True, padding_value=0)
        inputs["labels"] = inputs["input_ids"].clone()
        inputs["labels"][inputs["input_ids"] == 0] = -100
        num_items_in_batch = inputs["labels"].ne(-100).sum().item()

        def custom_loss_func(outputs, labels, num_items_in_batch=None):
            logits = outputs["logits"]
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            if num_items_in_batch is not None:
                return loss / num_items_in_batch  # multiply by number of items to get the sum
            return loss

        trainer = Trainer(model, train_dataset=None, compute_loss_func=custom_loss_func)

        # creating log history of trainer, results don't matter
        train_loss = trainer.training_step(model, inputs, num_items_in_batch)
        predict_loss = trainer.prediction_step(model, inputs, prediction_loss_only=True)[0]

        torch.testing.assert_close(train_loss, predict_loss, atol=1e-6, rtol=0)

    def test_eval_use_gather_object(self):
        train_dataset = RegressionDataset()
        eval_dataset = RegressionDataset()
        model = RegressionDictModel()
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(tmp_dir, eval_use_gather_object=True)
            trainer = Trainer(model, args, train_dataset=train_dataset, eval_dataset=eval_dataset)
            trainer.train()
            _ = trainer.evaluate()
            _ = trainer.predict(eval_dataset)


# ---------------------------------------------------------------------------
# Batch eval metrics tests
# ---------------------------------------------------------------------------


@require_torch
class TrainerBatchEvalMetricsTest(TestCasePlus, TrainerIntegrationCommon):
    def setUp(self):
        super().setUp()
        args = TrainingArguments("..")
        self.n_epochs = args.num_train_epochs
        self.batch_size = args.train_batch_size

    def test_evaluate_with_batch_eval_metrics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                a=1.5, b=2.5, compute_metrics=AlmostAccuracyBatched(), batch_eval_metrics=True, output_dir=tmp_dir
            )
            results = trainer.evaluate()

            x, y = trainer.eval_dataset.x, trainer.eval_dataset.ys[0]
            pred = 1.5 * x + 2.5
            expected_loss = ((pred - y) ** 2).mean()
            self.assertAlmostEqual(results["eval_loss"], expected_loss)
            expected_acc = AlmostAccuracy()((pred, y))["accuracy"]
            self.assertAlmostEqual(results["eval_accuracy"], expected_acc)

            # With a number of elements not a round multiple of the batch size
            trainer = get_regression_trainer(
                a=1.5,
                b=2.5,
                eval_len=66,
                compute_metrics=AlmostAccuracyBatched(),
                batch_eval_metrics=True,
                output_dir=tmp_dir,
            )
            results = trainer.evaluate()

            x, y = trainer.eval_dataset.x, trainer.eval_dataset.ys[0]
            pred = 1.5 * x + 2.5
            expected_loss = ((pred - y) ** 2).mean()
            self.assertAlmostEqual(results["eval_loss"], expected_loss)
            expected_acc = AlmostAccuracy()((pred, y))["accuracy"]
            self.assertAlmostEqual(results["eval_accuracy"], expected_acc)

            # With logits preprocess
            trainer = get_regression_trainer(
                a=1.5,
                b=2.5,
                compute_metrics=AlmostAccuracyBatched(),
                batch_eval_metrics=True,
                preprocess_logits_for_metrics=lambda logits, labels: logits + 1,
                output_dir=tmp_dir,
            )
            results = trainer.evaluate()

            x, y = trainer.eval_dataset.x, trainer.eval_dataset.ys[0]
            pred = 1.5 * x + 2.5
            expected_loss = ((pred - y) ** 2).mean()
            self.assertAlmostEqual(results["eval_loss"], expected_loss)
            expected_acc = AlmostAccuracy()((pred + 1, y))["accuracy"]
            self.assertAlmostEqual(results["eval_accuracy"], expected_acc)

    def test_predict_with_batch_eval_metrics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                a=1.5, b=2.5, compute_metrics=AlmostAccuracyBatched(), batch_eval_metrics=True, output_dir=tmp_dir
            )
            results = trainer.predict(trainer.eval_dataset)
            preds = results.predictions
            x, y = trainer.eval_dataset.x, trainer.eval_dataset.ys[0]
            gt = 1.5 * x + 2.5
            self.assertTrue(np.allclose(preds, gt))
            expected_acc = AlmostAccuracy()((preds, y))["accuracy"]
            self.assertAlmostEqual(results.metrics["test_accuracy"], expected_acc)

            # With a number of elements not a round multiple of the batch size
            trainer = get_regression_trainer(
                a=1.5,
                b=2.5,
                eval_len=66,
                compute_metrics=AlmostAccuracyBatched(),
                batch_eval_metrics=True,
                output_dir=tmp_dir,
            )
            results = trainer.predict(trainer.eval_dataset)
            preds = results.predictions
            x, y = trainer.eval_dataset.x, trainer.eval_dataset.ys[0]
            self.assertTrue(np.allclose(preds, 1.5 * x + 2.5))
            expected_acc = AlmostAccuracy()((preds, y))["accuracy"]
            self.assertAlmostEqual(results.metrics["test_accuracy"], expected_acc)

            # With more than one output of the model
            trainer = get_regression_trainer(
                a=1.5,
                b=2.5,
                double_output=True,
                compute_metrics=AlmostAccuracyBatched(),
                batch_eval_metrics=True,
                output_dir=tmp_dir,
            )
            preds = trainer.predict(trainer.eval_dataset).predictions
            x = trainer.eval_dataset.x
            self.assertEqual(len(preds), 2)
            self.assertTrue(np.allclose(preds[0], 1.5 * x + 2.5))
            self.assertTrue(np.allclose(preds[1], 1.5 * x + 2.5))

            # With more than one output/label of the model
            trainer = get_regression_trainer(
                a=1.5,
                b=2.5,
                double_output=True,
                label_names=["labels", "labels_2"],
                compute_metrics=AlmostAccuracyBatched(),
                batch_eval_metrics=True,
                output_dir=tmp_dir,
            )
            outputs = trainer.predict(trainer.eval_dataset)
            preds = outputs.predictions
            labels = outputs.label_ids
            x = trainer.eval_dataset.x
            self.assertEqual(len(preds), 2)
            self.assertTrue(np.allclose(preds[0], 1.5 * x + 2.5))
            self.assertTrue(np.allclose(preds[1], 1.5 * x + 2.5))
            self.assertTrue(np.array_equal(labels[0], trainer.eval_dataset.ys[0]))
            self.assertTrue(np.array_equal(labels[1], trainer.eval_dataset.ys[1]))


# ---------------------------------------------------------------------------
# FP16 / BF16 full eval memory tests
# ---------------------------------------------------------------------------


@require_torch
class TrainerFullEvalMemoryTest(TestCasePlus):
    @require_torch_fp16
    @require_torch_accelerator
    def test_fp16_full_eval(self):
        # this is a sensitive test so let's keep debugging printouts in place for quick diagnosis.
        # it's using pretty large safety margins, but small enough to detect broken functionality.
        debug = 0
        n_gpus = backend_device_count(torch_device)

        with tempfile.TemporaryDirectory() as tmp_dir:
            bs = 8
            eval_len = 16 * n_gpus
            # make the params somewhat big so that there will be enough RAM consumed to be able to
            # measure things. We should get about 64KB for a+b in fp32
            a = torch.ones(1000, bs) + 0.001
            b = torch.ones(1000, bs) - 0.001

            # 1. with fp16_full_eval disabled
            trainer = get_regression_trainer(
                a=a, b=b, eval_len=eval_len, skip_memory_metrics=False, output_dir=tmp_dir
            )
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

            # 2. with fp16_full_eval enabled
            trainer = get_regression_trainer(
                a=a, b=b, eval_len=eval_len, fp16_full_eval=True, skip_memory_metrics=False, output_dir=tmp_dir
            )
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

    @require_torch_accelerator
    @require_torch_bf16
    def test_bf16_full_eval(self):
        # note: most of the logic is the same as test_fp16_full_eval

        # this is a sensitive test so let's keep debugging printouts in place for quick diagnosis.
        # it's using pretty large safety margins, but small enough to detect broken functionality.
        debug = 0
        n_gpus = backend_device_count(torch_device)

        bs = 8
        eval_len = 16 * n_gpus
        # make the params somewhat big so that there will be enough RAM consumed to be able to
        # measure things. We should get about 64KB for a+b in fp32
        a = torch.ones(1000, bs) + 0.001
        b = torch.ones(1000, bs) - 0.001

        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. with bf16_full_eval disabled
            trainer = get_regression_trainer(
                a=a, b=b, eval_len=eval_len, skip_memory_metrics=False, output_dir=tmp_dir
            )
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

            # 2. with bf16_full_eval enabled
            trainer = get_regression_trainer(
                a=a, b=b, eval_len=eval_len, bf16_full_eval=True, skip_memory_metrics=False, output_dir=tmp_dir
            )
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


# ---------------------------------------------------------------------------
# Slow external model eval tests
# ---------------------------------------------------------------------------


@require_torch
class TrainerSlowEvalTest(TestCasePlus):
    @slow
    def test_trainer_eval_mrpc(self):
        MODEL_ID = "google-bert/bert-base-cased-finetuned-mrpc"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        data_args = GlueDataTrainingArguments(
            task_name="mrpc", data_dir=f"{get_tests_dir()}/fixtures/tests_samples/MRPC", overwrite_cache=True
        )
        eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="dev")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(output_dir=tmp_dir, use_cpu=True)
            trainer = Trainer(model=model, args=training_args, eval_dataset=eval_dataset)
            result = trainer.evaluate()
            self.assertLess(result["eval_loss"], 0.2)

    @slow
    def test_trainer_eval_multiple(self):
        MODEL_ID = "openai-community/gpt2"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

        dataset = get_dataset(PATH_SAMPLE_TEXT, tokenizer, 100)
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                use_cpu=True,
                per_device_eval_batch_size=1,
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                eval_dataset={
                    "data1": dataset,
                    "data2": dataset,
                },
            )
            result = trainer.evaluate()
            self.assertIn("eval_data1_loss", result)
            self.assertIn("eval_data2_loss", result)

    @slow
    def test_trainer_eval_lm(self):
        MODEL_ID = "distilbert/distilroberta-base"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        dataset = get_dataset(PATH_SAMPLE_TEXT, tokenizer, 100)
        self.assertEqual(len(dataset), 31)
