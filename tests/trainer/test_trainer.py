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
Core Trainer integration tests: reproducibility, gradient accumulation, gradient checkpointing,
mixed precision, logging, NEFTune, memory metrics, and end-to-end training.
"""

import math
import os
import tempfile
from functools import partial
from unittest.mock import patch

import datasets
import numpy as np
import pytest
import torch
from torch import nn

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    GPT2Config,
    GPT2LMHeadModel,
    IntervalStrategy,
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
    logging,
    set_seed,
)
from transformers.integrations import activate_neftune
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers.testing_utils import (
    CaptureLogger,
    LoggingLevel,
    TestCasePlus,
    backend_device_count,
    execute_subprocess_async,
    require_bitsandbytes,
    require_liger_kernel,
    require_non_hpu,
    require_peft,
    require_torch,
    require_torch_accelerator,
    require_torch_bf16,
    require_torch_fp16,
    require_torch_gpu,
    require_torch_multi_accelerator,
    require_torch_non_multi_accelerator,
    require_torch_tf32,
    run_first,
    slow,
    torch_device,
)

from .trainer_test_utils import (
    ATOL,
    PATH_SAMPLE_TEXT,
    RTOL,
    AlmostAccuracy,
    BasicTextGenerationModel,
    RegressionDataset,
    RegressionModel,
    RepeatDataset,
    StoreLossCallback,
    TrainerIntegrationCommon,
    get_dataset,
    get_regression_trainer,
)


# ---------------------------------------------------------------------------
# Mixed precision tests
# ---------------------------------------------------------------------------


@require_torch
class TrainerMixedPrecisionTest(TestCasePlus, TrainerIntegrationCommon):
    """Tests for FP16, BF16, and TF32 mixed precision training."""

    def setUp(self):
        super().setUp()

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(learning_rate=0.1, output_dir=tmp_dir)
            trainer.train()
            self.default_trained_model = (trainer.model.a, trainer.model.b)

    def check_trained_model(self, model, **kwargs):
        (a, b) = self.default_trained_model
        torch.testing.assert_close(model.a, a, **kwargs)
        torch.testing.assert_close(model.b, b, **kwargs)

    @require_torch_fp16
    @require_torch_accelerator
    def test_mixed_fp16(self):
        # very basic test
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(learning_rate=0.1, fp16=True, logging_steps=1, output_dir=tmp_dir)
            trainer.train()
            self.check_trained_model(trainer.model, atol=ATOL, rtol=RTOL)
            log_0 = trainer.state.log_history[:-1][0]
            # check that the grads were properly clipped due to the grad scaler. Otherwise, we get huge values
            self.assertEqual(log_0["grad_norm"] < 100, True)

    @require_torch_bf16
    @require_torch_accelerator
    def test_mixed_bf16(self):
        # very basic test
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(learning_rate=0.1, bf16=True, output_dir=tmp_dir)
            trainer.train()
            self.check_trained_model(trainer.model, atol=ATOL, rtol=RTOL)

    @require_torch_gpu
    @require_torch_tf32
    def test_tf32(self):
        # very basic test
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(learning_rate=0.1, tf32=True, output_dir=tmp_dir)
            trainer.train()
            self.check_trained_model(trainer.model)


# ---------------------------------------------------------------------------
# Gradient accumulation tests
# ---------------------------------------------------------------------------


@require_torch
class TrainerGradientAccumulationTest(TestCasePlus, TrainerIntegrationCommon):
    """Tests for gradient accumulation loss alignment and batch counting."""

    def setUp(self):
        super().setUp()

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(learning_rate=0.1, output_dir=tmp_dir)
            trainer.train()
            self.default_trained_model = (trainer.model.a, trainer.model.b)

    def check_trained_model(self, model, **kwargs):
        (a, b) = self.default_trained_model
        torch.testing.assert_close(model.a, a, **kwargs)
        torch.testing.assert_close(model.b, b, **kwargs)

    def test_gradient_accumulation(self):
        # Training with half the batch size but accumulation steps as 2 should give the same training losses.
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                gradient_accumulation_steps=2, per_device_train_batch_size=4, learning_rate=0.1, output_dir=tmp_dir
            )
            trainer.train()
            self.check_trained_model(trainer.model)

    @slow
    def test_gradient_accumulation_loss_alignment_with_model_loss(self):
        set_seed(42)

        model_name = "nickypro/tinyllama-15M"
        dataset_name = "wikitext"
        dataset_config = "wikitext-2-raw-v1"
        dataset = datasets.load_dataset(dataset_name, dataset_config, split="train[:40]")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        tokenizer.pad_token = tokenizer.eos_token

        def tokenize_function(examples):
            return tokenizer(examples["text"], max_length=16, padding="max_length", truncation=True)

        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        args_kwargs = {
            "logging_steps": 1,
            "max_steps": 5,
            "learning_rate": 3e-4,
            "disable_tqdm": True,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(
                tmp_dir,
                **args_kwargs,
            )
            # train with base loss
            set_seed(42)
            model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
            base_loss_callback = StoreLossCallback()
            trainer = Trainer(
                model,
                args,
                train_dataset=tokenized_dataset,
                callbacks=[base_loss_callback],
                data_collator=data_collator,
            )
            assert trainer.model_accepts_loss_kwargs
            trainer.train()

            args = TrainingArguments(
                tmp_dir,
                **args_kwargs,
                gradient_accumulation_steps=2,
                per_device_train_batch_size=4,
            )

            # train with gradient accumulation
            set_seed(42)
            model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
            grad_accum_loss_callback = StoreLossCallback()
            trainer = Trainer(
                model,
                args,
                train_dataset=tokenized_dataset,
                callbacks=[grad_accum_loss_callback],
                data_collator=data_collator,
            )
            assert trainer.model_accepts_loss_kwargs
            trainer.train()

            # train with broken loss
            set_seed(42)
            model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
            broken_loss_callback = StoreLossCallback()
            trainer = Trainer(
                model,
                args,
                train_dataset=tokenized_dataset,
                callbacks=[broken_loss_callback],
                data_collator=data_collator,
            )
            # disable model_accepts_loss_kwargs so that "num_items_in_batch" is not passed to the model
            trainer.model_accepts_loss_kwargs = False
            trainer.train()

        # Calculate the difference between the base loss and the grad_accum loss
        diff_truth = [
            abs(base - grad) for base, grad in zip(base_loss_callback.losses, grad_accum_loss_callback.losses)
        ]
        diff_broken = [abs(base - grad) for base, grad in zip(base_loss_callback.losses, broken_loss_callback.losses)]

        # all diff truth should be quite close
        self.assertLess(max(diff_truth), 0.01, f"Difference {max(diff_truth)} is not within 0.01")
        # max diff broken should be very off ("very off" is arbitrary, but as long as it's bigger than 0.1, it's fine)
        self.assertGreater(max(diff_broken), 0.7, f"Difference {max(diff_broken)} is not greater than 0.7")

        loss_base = sum(base_loss_callback.losses)
        loss_broken = sum(broken_loss_callback.losses)

        # mean/sum loss should not vary too much.
        relative_diff = abs(loss_base - loss_broken) / max(loss_base, loss_broken)
        self.assertLess(relative_diff, 0.2, f"Relative difference {relative_diff} is not within 0.2")

    def test_gradient_accumulation_loss_alignment_with_loss_func(self):
        set_seed(42)

        model_name = "roneneldan/TinyStories-33M"
        dataset_name = "Salesforce/wikitext"
        dataset_config = "wikitext-2-raw-v1"
        dataset = datasets.load_dataset(dataset_name, dataset_config, split="train[:40]")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        tokenizer.pad_token = tokenizer.eos_token

        def tokenize_function(examples):
            return tokenizer(examples["text"], max_length=16, padding="max_length", truncation=True)

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        tokenizer.pad_token = tokenizer.eos_token
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        model = AutoModelForCausalLM.from_pretrained(model_name)

        def compute_loss(logits, labels, vocab_size, num_items_in_batch, disable_num_items_in_batch=False):
            if disable_num_items_in_batch:
                num_items_in_batch = None
            return ForCausalLMLoss(logits["logits"], labels, vocab_size, num_items_in_batch)

        loss_fn = partial(compute_loss, vocab_size=model.config.vocab_size)

        base_loss_callback = StoreLossCallback()

        args_kwargs = {
            "logging_steps": 1,
            "max_steps": 5,
            "learning_rate": 3e-4,
            "disable_tqdm": True,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(
                tmp_dir,
                **args_kwargs,
            )
            trainer = Trainer(
                model,
                args,
                train_dataset=tokenized_dataset,
                callbacks=[base_loss_callback],
                compute_loss_func=loss_fn,
                data_collator=data_collator,
            )
            trainer.train()

        grad_accum_loss_callback = StoreLossCallback()
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(
                tmp_dir,
                **args_kwargs,
                gradient_accumulation_steps=2,
                per_device_train_batch_size=4,
            )
            set_seed(42)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            trainer = Trainer(
                model,
                args,
                train_dataset=tokenized_dataset,
                callbacks=[grad_accum_loss_callback],
                compute_loss_func=loss_fn,
                data_collator=data_collator,
            )
            trainer.train()

            set_seed(42)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            broken_loss_callback = StoreLossCallback()
            # we need to disable num_items_in_batch because since we are passing a custom loss,
            # we make the assumption that num_items_in_batch is handled correctly
            loss_fn = partial(compute_loss, vocab_size=model.config.vocab_size, disable_num_items_in_batch=True)
            trainer = Trainer(
                model,
                args,
                train_dataset=tokenized_dataset,
                callbacks=[broken_loss_callback],
                compute_loss_func=loss_fn,
                data_collator=data_collator,
            )
            trainer.train()

            # Calculate the difference between the base loss and the grad_accum loss
            diff_truth = [
                abs(base - grad) for base, grad in zip(base_loss_callback.losses, grad_accum_loss_callback.losses)
            ]
            diff_broken = [
                abs(base - grad) for base, grad in zip(base_loss_callback.losses, broken_loss_callback.losses)
            ]

            # all diff truth should be quite close
            self.assertLess(max(diff_truth), 0.01, f"Difference {max(diff_truth)} is not within 0.01")

            # max diff broken should be very off
            self.assertGreater(max(diff_broken), 3, f"Difference {max(diff_broken)} is not greater than 3")

    @require_torch_multi_accelerator
    def test_num_batches_in_training_with_gradient_accumulation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            for num_train_epochs in [1, 2]:
                for train_len in [123, 120]:
                    trainer = get_regression_trainer(
                        train_len=train_len,
                        per_device_train_batch_size=4,
                        gradient_accumulation_steps=5,
                        num_train_epochs=num_train_epochs,
                        output_dir=tmp_dir,
                    )

                    total_batch_samples = []

                    def wrap_get_batch_samples(fn):
                        def wrapped_fn(epoch_iterator, num_batches, device):
                            self.assertGreater(num_batches, 0)
                            batch_samples, num_items_in_batch = fn(epoch_iterator, num_batches, device)
                            self.assertEqual(len(batch_samples), num_batches)
                            total_batch_samples.append(num_batches)
                            return batch_samples, num_items_in_batch

                        return wrapped_fn

                    trainer.get_batch_samples = wrap_get_batch_samples(trainer.get_batch_samples)

                    trainer.train()

                    self.assertEqual(len(trainer.get_train_dataloader()) * num_train_epochs, sum(total_batch_samples))


# ---------------------------------------------------------------------------
# Gradient checkpointing tests
# ---------------------------------------------------------------------------


@require_torch
class TrainerGradientCheckpointingTest(TestCasePlus):
    """Tests for gradient checkpointing during training."""

    def test_gradient_checkpointing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                per_device_train_batch_size=1,
                learning_rate=0.1,
                gradient_checkpointing=True,
                output_dir=tmp_dir,
            )
            previous_params = {k: v.detach().clone() for k, v in trainer.model.named_parameters()}

            trainer.train()

            # Check if model weights have been updated
            for k, v in trainer.model.named_parameters():
                self.assertFalse(
                    torch.allclose(previous_params[k], v, rtol=1e-4, atol=1e-4),
                    f"Model weights for {k} have not been updated",
                )


# ---------------------------------------------------------------------------
# NEFTune tests
# ---------------------------------------------------------------------------


@require_torch
class TrainerNEFTuneTest(TestCasePlus):
    """Tests for NEFTune noise injection during training."""

    def test_neftune(self):
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=32, n_layer=3, n_head=4)
        tiny_gpt2 = GPT2LMHeadModel(config)
        x = torch.randint(0, 100, (128,))
        train_dataset = RepeatDataset(x)

        # Trainer without inf/nan filter
        args = TrainingArguments(
            self.get_auto_remove_tmp_dir(),
            learning_rate=1e-9,
            logging_steps=5,
            logging_nan_inf_filter=False,
            neftune_noise_alpha=0.4,
        )
        trainer = Trainer(tiny_gpt2, args, train_dataset=train_dataset)

        activate_neftune(trainer.model, trainer.args.neftune_noise_alpha)

        dummy_input = torch.LongTensor([[1, 0, 1]]).to(torch_device)

        emb1 = trainer.model.get_input_embeddings()(dummy_input)
        emb2 = trainer.model.get_input_embeddings()(dummy_input)

        self.assertFalse(torch.allclose(emb1, emb2), "Neftune noise is not applied!")

        # redefine the model
        tiny_gpt2 = GPT2LMHeadModel(config)
        # Trainer without inf/nan filter
        args = TrainingArguments(
            self.get_auto_remove_tmp_dir(),
            learning_rate=1e-9,
            logging_steps=5,
            logging_nan_inf_filter=False,
            neftune_noise_alpha=0.4,
        )
        trainer = Trainer(tiny_gpt2, args, train_dataset=train_dataset)

        # Check that it trains without errors
        trainer.train()

        # Make sure forward pass works fine
        _ = trainer.model(dummy_input)
        self.assertTrue(len(trainer.model.get_input_embeddings()._forward_hooks) == 0)

        trainer.model.eval()

        # Check that we get identical embeddings just in case
        emb1 = trainer.model.get_input_embeddings()(dummy_input)
        emb2 = trainer.model.get_input_embeddings()(dummy_input)
        torch.testing.assert_close(emb1, emb2)


# ---------------------------------------------------------------------------
# Logging tests
# ---------------------------------------------------------------------------


@require_torch
class TrainerLoggingTest(TestCasePlus):
    """Tests for logging behavior: inf/nan filter and log levels."""

    def test_logging_inf_nan_filter(self):
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=32, n_layer=3, n_head=4)
        tiny_gpt2 = GPT2LMHeadModel(config)
        x = torch.randint(0, 100, (128,))
        train_dataset = RepeatDataset(x)

        # Trainer without inf/nan filter
        args = TrainingArguments(
            self.get_auto_remove_tmp_dir(),
            learning_rate=1e9,
            logging_steps=5,
            logging_nan_inf_filter=False,
        )
        trainer = Trainer(tiny_gpt2, args, train_dataset=train_dataset)
        trainer.train()
        log_history_no_filter = trainer.state.log_history

        # Trainer with inf/nan filter
        args = TrainingArguments(
            self.get_auto_remove_tmp_dir(),
            learning_rate=1e9,
            logging_steps=5,
            logging_nan_inf_filter=True,
        )
        trainer = Trainer(tiny_gpt2, args, train_dataset=train_dataset)
        trainer.train()
        log_history_filter = trainer.state.log_history

        def is_any_loss_nan_or_inf(log_history):
            losses = [l["loss"] for l in log_history[:-1]]
            return any(math.isnan(x) for x in losses) or any(math.isinf(x) for x in losses)

        self.assertTrue(is_any_loss_nan_or_inf(log_history_no_filter))
        self.assertFalse(is_any_loss_nan_or_inf(log_history_filter))

    def test_log_level(self):
        # testing only --log_level (--log_level_replica requires multiple gpus and DDP and is tested elsewhere)
        logger = logging.get_logger()
        log_info_string = "Running training"

        # test with the default log_level - should be the same as before and thus we test depending on is_info
        is_info = logging.get_verbosity() <= 20

        with tempfile.TemporaryDirectory() as tmp_dir:
            with CaptureLogger(logger) as cl:
                trainer = get_regression_trainer(output_dir=tmp_dir)
                trainer.train()
            if is_info:
                self.assertIn(log_info_string, cl.out)
            else:
                self.assertNotIn(log_info_string, cl.out)

            with LoggingLevel(logging.INFO):
                # test with low log_level - lower than info
                with CaptureLogger(logger) as cl:
                    trainer = get_regression_trainer(log_level="debug", output_dir=tmp_dir)
                    trainer.train()
                self.assertIn(log_info_string, cl.out)

            with LoggingLevel(logging.INFO):
                # test with high log_level - should be quiet
                with CaptureLogger(logger) as cl:
                    trainer = get_regression_trainer(log_level="error", output_dir=tmp_dir)
                    trainer.train()
                self.assertNotIn(log_info_string, cl.out)


# ---------------------------------------------------------------------------
# Metrics tests (FLOS, memory, input tokens)
# ---------------------------------------------------------------------------


@require_torch
class TrainerMetricsTest(TestCasePlus):
    """Tests for FLOS extraction, memory metrics, and input token counting."""

    def test_flos_extraction(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(learning_rate=0.1, output_dir=tmp_dir)

            def assert_flos_extraction(trainer, wrapped_model_to_check):
                self.assertEqual(trainer.model, trainer.accelerator.unwrap_model(wrapped_model_to_check))
                self.assertGreaterEqual(
                    getattr(trainer.accelerator.unwrap_model(wrapped_model_to_check).config, "total_flos", 0), 0
                )

            # with plain model
            assert_flos_extraction(trainer, trainer.model)

            # with enforced DataParallel
            assert_flos_extraction(trainer, nn.DataParallel(trainer.model))

            trainer.train()
            self.assertTrue(isinstance(trainer.state.total_flos, float))

    def check_mem_metrics(self, trainer, check_func):
        metrics = trainer.train().metrics
        check_func("init_mem_cpu_alloc_delta", metrics)
        check_func("train_mem_cpu_alloc_delta", metrics)
        if backend_device_count(torch_device) > 0:
            check_func("init_mem_gpu_alloc_delta", metrics)
            check_func("train_mem_gpu_alloc_delta", metrics)

        metrics = trainer.evaluate()
        check_func("eval_mem_cpu_alloc_delta", metrics)
        if backend_device_count(torch_device) > 0:
            check_func("eval_mem_gpu_alloc_delta", metrics)

        metrics = trainer.predict(RegressionDataset()).metrics
        check_func("test_mem_cpu_alloc_delta", metrics)
        if backend_device_count(torch_device) > 0:
            check_func("test_mem_gpu_alloc_delta", metrics)

    def test_mem_metrics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # with mem metrics enabled
            trainer = get_regression_trainer(skip_memory_metrics=False, output_dir=tmp_dir)
            self.check_mem_metrics(trainer, self.assertIn)

            # with mem metrics disabled
            trainer = get_regression_trainer(skip_memory_metrics=True, output_dir=tmp_dir)
            self.check_mem_metrics(trainer, self.assertNotIn)

    def test_include_num_input_tokens_seen(self):
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        tokenizer.pad_token = "[PAD]"
        model.config.pad_token_id = tokenizer.pad_token_id

        sentences = ["This is a short sentence.", "This is a much longer sentence that will require padding."]
        labels = torch.tensor([0, 1])

        # 1. Test with attention_mask
        tokenized_dataset_with_mask = tokenizer(sentences, truncation=True, padding="longest", return_tensors="pt")
        tokenized_dataset_with_mask["labels"] = labels
        dataset_with_mask = datasets.Dataset.from_dict(tokenized_dataset_with_mask)

        # 2. Test without attention_mask
        tokenized_dataset_no_mask = {k: v for k, v in tokenized_dataset_with_mask.items() if k != "attention_mask"}
        dataset_no_mask = datasets.Dataset.from_dict(tokenized_dataset_no_mask)

        # 3. Test with no padding information
        tokenizer_no_pad = AutoTokenizer.from_pretrained("bert-base-cased")
        tokenizer_no_pad.pad_token = None

        data_collator = default_data_collator

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Test case 1: "non_padding" with attention_mask
            args = TrainingArguments(
                output_dir=tmp_dir,
                include_num_input_tokens_seen="non_padding",
                per_device_train_batch_size=2,
                max_steps=1,
            )
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=dataset_with_mask,
                data_collator=data_collator,
                processing_class=tokenizer,
            )
            trainer.train()
            attention_mask = tokenized_dataset_with_mask["attention_mask"]
            non_padded_tokens_with_mask = attention_mask.sum().item()
            self.assertEqual(trainer.state.num_input_tokens_seen, non_padded_tokens_with_mask)

            # Test case 2: "non_padding" without attention_mask (fallback to pad_token_id)
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=dataset_no_mask,
                data_collator=data_collator,
                processing_class=tokenizer,
            )
            trainer.train()
            input_ids = tokenized_dataset_with_mask["input_ids"]  # use original to compute expected
            non_padded_tokens_no_mask = (input_ids != tokenizer.pad_token_id).sum().item()
            self.assertEqual(trainer.state.num_input_tokens_seen, non_padded_tokens_no_mask)

            # Test case 3: "non_padding" with no padding info (fallback to numel)
            with self.assertLogs("transformers.trainer", level="WARNING") as cm:
                trainer = Trainer(
                    model=model,
                    args=args,
                    train_dataset=dataset_no_mask,  # still has input_ids
                    data_collator=data_collator,
                    processing_class=tokenizer_no_pad,  # tokenizer without pad token
                )
                trainer.train()
                self.assertTrue(
                    any("Could not determine method to count non-padding tokens" in log for log in cm.output)
                )
            total_tokens = input_ids.numel()
            self.assertEqual(trainer.state.num_input_tokens_seen, total_tokens)

            # Test case 4: "all"
            args.include_num_input_tokens_seen = "all"
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=dataset_with_mask,
                data_collator=data_collator,
                processing_class=tokenizer,
            )
            trainer.train()
            self.assertEqual(trainer.state.num_input_tokens_seen, total_tokens)

            # Test case 5: True (backward compatibility)
            args.include_num_input_tokens_seen = True
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=dataset_with_mask,
                data_collator=data_collator,
                processing_class=tokenizer,
            )
            trainer.train()
            self.assertEqual(trainer.state.num_input_tokens_seen, total_tokens)

    def test_get_num_trainable_parameters(self):
        model = nn.Sequential(nn.Linear(128, 64), nn.Linear(64, 32))
        # in_features * out_features + bias
        layer_1 = 128 * 64 + 64
        layer_2 = 64 * 32 + 32
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = Trainer(model=model, args=TrainingArguments(output_dir=tmp_dir))
            self.assertEqual(trainer.get_num_trainable_parameters(), layer_1 + layer_2)
            # Freeze the last layer
            for param in model[-1].parameters():
                param.requires_grad = False
            self.assertEqual(trainer.get_num_trainable_parameters(), layer_1)


# ---------------------------------------------------------------------------
# Step counting and training loss tests
# ---------------------------------------------------------------------------


@require_torch
class TrainerStepCountingTest(TestCasePlus):
    """Tests for training loss computation, step counting, and epoch handling."""

    def setUp(self):
        super().setUp()
        args = TrainingArguments("..")
        self.n_epochs = args.num_train_epochs
        self.batch_size = args.train_batch_size

    def test_training_loss(self):
        n_gpus = max(1, backend_device_count(torch_device))

        # With even logs
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(logging_steps=64 / (8 * n_gpus), output_dir=tmp_dir)
            trainer.train()
            log_history = trainer.state.log_history

            losses = [log["loss"] for log in log_history if "loss" in log]
            train_loss = log_history[-1]["train_loss"]
            self.assertAlmostEqual(sum(losses) / len(losses), train_loss, places=4)

        # With uneven logs
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(logging_steps=5, output_dir=tmp_dir)
            trainer.train()
            log_history = trainer.state.log_history

            # Training loss should be the same as before
            new_train_loss = log_history[-1]["train_loss"]
            self.assertAlmostEqual(train_loss, new_train_loss, places=4)

    def test_number_of_steps_in_training(self):
        # Regular training has n_epochs * len(train_dl) steps
        tmp_dir = self.get_auto_remove_tmp_dir()
        trainer = get_regression_trainer(learning_rate=0.1, output_dir=tmp_dir)
        train_output = trainer.train()
        self.assertEqual(train_output.global_step, self.n_epochs * 64 / self.batch_size)

        # Check passing num_train_epochs works (and a float version too):
        trainer = get_regression_trainer(learning_rate=0.1, num_train_epochs=1.5, output_dir=tmp_dir)
        train_output = trainer.train()
        self.assertEqual(train_output.global_step, int(1.5 * 64 / self.batch_size))

        # If we pass a max_steps, num_train_epochs is ignored
        trainer = get_regression_trainer(learning_rate=0.1, max_steps=10, output_dir=tmp_dir)
        train_output = trainer.train()
        self.assertEqual(train_output.global_step, 10)

    def test_num_train_epochs_in_training(self):
        # len(train_dl) < gradient_accumulation_steps shouldn't give ``ZeroDivisionError`` when ``max_steps`` is given.
        # It should give 1 update step for each epoch.
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                max_steps=3,
                train_len=64,
                per_device_train_batch_size=16,
                gradient_accumulation_steps=5,
                output_dir=tmp_dir,
            )
            train_output = trainer.train()
            self.assertEqual(train_output.global_step, 3)

            # Even ``max_steps`` is not specified, we still expect 1 update step for each epoch if
            # len(train_dl) < gradient_accumulation_steps.
            trainer = get_regression_trainer(
                train_len=64, per_device_train_batch_size=16, gradient_accumulation_steps=5, output_dir=tmp_dir
            )
            train_output = trainer.train()
            self.assertEqual(train_output.global_step, int(self.n_epochs))


# ---------------------------------------------------------------------------
# Reproducibility tests (pre-run training to check determinism across configs)
# ---------------------------------------------------------------------------


@require_torch
class TrainerIntegrationPrerunTest(TestCasePlus, TrainerIntegrationCommon):
    """
    Only tests that want to tap into the auto-pre-run 2 trainings:
    - self.default_trained_model
    - self.alternate_trained_model
    directly, or via check_trained_model
    """

    def setUp(self):
        super().setUp()
        args = TrainingArguments("..")
        self.n_epochs = args.num_train_epochs
        self.batch_size = args.train_batch_size

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(learning_rate=0.1, output_dir=tmp_dir)
            trainer.train()
            self.default_trained_model = (trainer.model.a, trainer.model.b)

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(learning_rate=0.1, seed=314, output_dir=tmp_dir)
            trainer.train()
            self.alternate_trained_model = (trainer.model.a, trainer.model.b)

    def check_trained_model(self, model, alternate_seed=False, **kwargs):
        (a, b) = self.alternate_trained_model if alternate_seed else self.default_trained_model
        torch.testing.assert_close(model.a, a, **kwargs)
        torch.testing.assert_close(model.b, b, **kwargs)

    def test_reproducible_training(self):
        # Checks that training worked, model trained and seed made a reproducible training.
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(learning_rate=0.1, output_dir=tmp_dir)
            trainer.train()
            self.check_trained_model(trainer.model)

        # Checks that a different seed gets different (reproducible) results.
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(learning_rate=0.1, seed=314, output_dir=tmp_dir)
            trainer.train()
            self.check_trained_model(trainer.model, alternate_seed=True)

    def test_trainer_with_datasets(self):
        np.random.seed(42)
        x = np.random.normal(size=(64,)).astype(np.float32)
        y = 2.0 * x + 3.0 + np.random.normal(scale=0.1, size=(64,)).astype(np.float32)
        train_dataset = datasets.Dataset.from_dict({"input_x": x, "label": y})

        # Base training. Should have the same results as test_reproducible_training
        model = RegressionModel()
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(tmp_dir, learning_rate=0.1)
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
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(tmp_dir, learning_rate=0.1)
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


# ---------------------------------------------------------------------------
# Torch compile tests
# ---------------------------------------------------------------------------


@require_torch
class TrainerTorchCompileTest(TestCasePlus):
    @pytest.mark.torch_compile_test
    def test_torch_compile_loss_func_compatibility(self):
        config = LlamaConfig(vocab_size=100, hidden_size=32, num_hidden_layers=3, num_attention_heads=4)
        tiny_llama = LlamaForCausalLM(config)

        x = torch.randint(0, 100, (128,))
        train_dataset = RepeatDataset(x)

        args = TrainingArguments(
            self.get_auto_remove_tmp_dir(),
            per_device_train_batch_size=2,
            torch_compile=True,
            max_steps=1,  # compile happens on the first step
        )
        trainer = Trainer(model=tiny_llama, args=args, train_dataset=train_dataset)  # noqa
        trainer.train()

    @require_peft
    @require_bitsandbytes
    @pytest.mark.torch_compile_test
    def test_bnb_compile(self):
        from peft import LoraConfig, get_peft_model

        # Simply tests if initializing a Trainer with a PEFT + compiled model works out of the box
        # QLoRA + torch compile is not really supported yet, but we should at least support the model
        # loading and let torch throw the
        tiny_model = AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-LlamaForCausalLM",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        )

        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        tiny_model = get_peft_model(tiny_model, peft_config)

        tiny_model = torch.compile(tiny_model)

        x = torch.randint(0, 100, (128,))
        train_dataset = RepeatDataset(x)

        args = TrainingArguments(
            self.get_auto_remove_tmp_dir(),
            learning_rate=1e-9,
            logging_steps=5,
        )
        with self.assertRaises(ValueError):
            _ = Trainer(tiny_model, args, train_dataset=train_dataset)  # noqa

    @require_torch_accelerator
    @pytest.mark.torch_compile_test
    def test_torch_compile_train(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(output_dir=tmp_dir)
            metrics = trainer.train()
            original_train_loss = metrics.training_loss

            trainer = get_regression_trainer(torch_compile=True, output_dir=tmp_dir)
            metrics = trainer.train()
            self.assertAlmostEqual(metrics.training_loss, original_train_loss)

    @require_torch_accelerator
    @pytest.mark.torch_compile_test
    def test_torch_compile_eval(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(output_dir=tmp_dir)
            metrics = trainer.evaluate()
            original_eval_loss = metrics["eval_loss"]

            trainer = get_regression_trainer(torch_compile=True, output_dir=tmp_dir)
            metrics = trainer.evaluate()

            self.assertAlmostEqual(metrics["eval_loss"], original_eval_loss, delta=1e-6)


# ---------------------------------------------------------------------------
# Early stopping tests
# ---------------------------------------------------------------------------


@require_torch
class TrainerEarlyStoppingTest(TestCasePlus):
    def test_early_stopping_callback(self):
        # early stopping stops training before num_training_epochs
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                output_dir=tmp_dir,
                num_train_epochs=20,
                gradient_accumulation_steps=1,
                per_device_train_batch_size=16,
                load_best_model_at_end=True,
                eval_strategy=IntervalStrategy.EPOCH,
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
                eval_strategy=IntervalStrategy.EPOCH,
                compute_metrics=AlmostAccuracy(),
                metric_for_best_model="accuracy",
            )
            trainer.add_callback(EarlyStoppingCallback(1))
            self.assertEqual(trainer.state.global_step, 0)
            try:
                trainer.train()
            except AssertionError:
                self.assertEqual(trainer.state.global_step, 0)

        # even if load_best_model_at_end is False, `best_model_checkpoint` should be set
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                output_dir=tmp_dir,
                num_train_epochs=20,
                gradient_accumulation_steps=1,
                per_device_train_batch_size=16,
                load_best_model_at_end=False,
                eval_strategy=IntervalStrategy.EPOCH,
                save_strategy=IntervalStrategy.EPOCH,
                compute_metrics=AlmostAccuracy(),
                metric_for_best_model="accuracy",
            )
            trainer.add_callback(EarlyStoppingCallback(1, 0.0001))
            train_output = trainer.train()
            self.assertIsNotNone(trainer.state.best_model_checkpoint)


# ---------------------------------------------------------------------------
# Liger kernel tests
# ---------------------------------------------------------------------------


@require_torch
class TrainerLigerKernelTest(TestCasePlus):
    @require_liger_kernel
    def test_use_liger_kernel_patching(self):
        # Ensure any monkey patching is cleaned up for subsequent tests
        with patch("transformers.models.llama.modeling_llama"):
            from liger_kernel.transformers import liger_rotary_pos_emb

            from transformers.models.llama import modeling_llama

            config = LlamaConfig(vocab_size=100, hidden_size=32, num_hidden_layers=3, num_attention_heads=4)
            tiny_llama = LlamaForCausalLM(config)

            # Spot check that modeling code and model instance variables are not yet patched
            self.assertNotEqual(modeling_llama.apply_rotary_pos_emb, liger_rotary_pos_emb)
            self.assertFalse("LigerRMSNorm" in tiny_llama.model.norm.__repr__())

            args = TrainingArguments(self.get_auto_remove_tmp_dir(), use_liger_kernel=True)
            Trainer(tiny_llama, args)

            # Spot check that modeling code and model instance variables are patched
            self.assertEqual(modeling_llama.apply_rotary_pos_emb, liger_rotary_pos_emb)
            self.assertTrue("LigerRMSNorm" in tiny_llama.model.norm.__repr__())

    @require_liger_kernel
    def test_use_liger_kernel_custom_config_patching(self):
        # Ensure any monkey patching is cleaned up for subsequent tests
        with patch("transformers.models.llama.modeling_llama"):
            from liger_kernel.transformers import LigerRMSNorm

            config = LlamaConfig(vocab_size=100, hidden_size=32, num_hidden_layers=3, num_attention_heads=4)
            tiny_llama = LlamaForCausalLM(config)

            args = TrainingArguments(
                self.get_auto_remove_tmp_dir(),
                use_liger_kernel=True,
                liger_kernel_config={"rms_norm": False},  # Don't apply Liger's RMSNorm
            )
            Trainer(tiny_llama, args)

            # Check that the RMSNorm kernel is not applied as specified in the config
            self.assertFalse(isinstance(tiny_llama.model.norm, LigerRMSNorm))

    @require_liger_kernel
    @require_torch_accelerator
    @require_torch_non_multi_accelerator  # Don't work with DP
    def test_use_liger_kernel_trainer(self):
        # Ensure any monkey patching is cleaned up for subsequent tests
        with patch("transformers.models.llama.modeling_llama"):
            config = LlamaConfig(vocab_size=100, hidden_size=32, num_hidden_layers=3, num_attention_heads=4)
            tiny_llama = LlamaForCausalLM(config)
            x = torch.randint(0, 100, (128,))
            train_dataset = RepeatDataset(x)
            args = TrainingArguments(
                self.get_auto_remove_tmp_dir(),
                learning_rate=1e-2,
                logging_steps=5,
                max_steps=20,
                use_liger_kernel=True,
            )
            Trainer(tiny_llama, args, train_dataset=train_dataset).train()

    @require_liger_kernel
    @require_torch_accelerator
    @require_torch_non_multi_accelerator  # don't work with DP
    def test_use_liger_kernel_custom_config_trainer(self):
        # Ensure any monkey patching is cleaned up for subsequent tests
        with patch("transformers.models.llama.modeling_llama"):
            config = LlamaConfig(vocab_size=100, hidden_size=32, num_hidden_layers=3, num_attention_heads=4)
            tiny_llama = LlamaForCausalLM(config)
            x = torch.randint(0, 100, (128,))
            train_dataset = RepeatDataset(x)
            args = TrainingArguments(
                self.get_auto_remove_tmp_dir(),
                learning_rate=1e-2,
                logging_steps=5,
                max_steps=20,
                use_liger_kernel=True,
                liger_kernel_config={"rms_norm": False, "cross_entropy": True, "fused_linear_cross_entropy": False},
            )
            Trainer(tiny_llama, args, train_dataset=train_dataset).train()


# ---------------------------------------------------------------------------
# Miscellaneous integration tests
# ---------------------------------------------------------------------------


@require_torch
class TrainerIntegrationTest(TestCasePlus):
    """Integration tests: compatibility, and e2e."""

    @slow
    @run_first
    @require_non_hpu
    @require_torch_multi_accelerator
    def test_end_to_end_example(self):
        # Tests that `translation.py` will run without issues
        script_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "..", "examples", "pytorch", "translation", "run_translation.py"
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            command = [
                "accelerate",
                "launch",
                script_path,
                "--model_name_or_path",
                "google-t5/t5-small",
                "--per_device_train_batch_size",
                "1",
                "--output_dir",
                tmpdir,
                "--do_train",
                "--max_train_samples",
                "64",
                "--num_train_epochs",
                "1",
                "--dataset_name",
                "wmt16",
                "--dataset_config",
                "ro-en",
                "--source_lang",
                "en",
                "--target_lang",
                "ro",
                "--do_predict",
                "--max_predict_samples",
                "64",
                "--predict_with_generate",
                "--ddp_timeout",
                "60",
            ]
            execute_subprocess_async(command)
            # successful return here == success - any errors would have caused an error or a timeout in the sub-call

    def test_special_token_alignment(self):
        """
        Tests that special token changes in the tokenizer result in model configs updates when using the trainer, to
        ensure special tokens are aligned across configs
        """

        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")

        # add new special tokens to tokenizer, so we can test that trainer aligns the model configs with the tokenizer
        tokenizer.eos_token = "<|im_end|>"
        tokenizer.pad_token = "<|im_end|>"
        tokenizer.bos_token = "<|im_start|>"
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_end|>", "<|im_start|>"]})

        # the model needs to have its embedding layer resized accordingly
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)

        # create a random dataset from the **new** vocab size
        x = torch.randint(0, len(tokenizer), (64,))
        dataset = RepeatDataset(x, length=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            training_args = TrainingArguments(output_dir=tmpdir, max_steps=1, per_device_train_batch_size=1)
            trainer = Trainer(
                model=model,
                args=training_args,
                processing_class=tokenizer,
                train_dataset=dataset,
            )

            # We haven't started training -> not yet aligned
            self.assertNotEqual(trainer.model.config.eos_token_id, tokenizer.eos_token_id)
            self.assertNotEqual(trainer.model.config.pad_token_id, tokenizer.pad_token_id)
            self.assertNotEqual(trainer.model.config.bos_token_id, tokenizer.bos_token_id)

            trainer.train()

            # Must be aligned as soon as we start training
            self.assertEqual(trainer.model.config.eos_token_id, tokenizer.eos_token_id)
            self.assertEqual(trainer.model.config.pad_token_id, tokenizer.pad_token_id)
            self.assertEqual(trainer.model.config.bos_token_id, tokenizer.bos_token_id)

    def test_trainer_works_without_model_config(self):
        """
        Tests that models without a `config` parameter can still be trained.
        This is useful for preserving compatibility with third parties that train different models using the
        transformers Trainer.

        If this test fails, it doesn't imply that there's issues with transformers, but perhaps with third
        parties.
        """

        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
        model = BasicTextGenerationModel(vocab_size=tokenizer.vocab_size, hidden_size=32)

        train_dataset = get_dataset(PATH_SAMPLE_TEXT, tokenizer, 100)

        with tempfile.TemporaryDirectory() as tmpdir:
            training_args = TrainingArguments(
                output_dir=tmpdir, max_steps=5, per_device_train_batch_size=1, use_cpu=True
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                processing_class=tokenizer,
                train_dataset=train_dataset,
            )
            trainer.train()

    def test_training_arguments_are_left_untouched(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        trainer = get_regression_trainer(output_dir=tmp_dir)
        trainer.train()
        args = TrainingArguments(tmp_dir)
        dict1, dict2 = args.to_dict(), trainer.args.to_dict()
        for key in dict1:
            self.assertEqual(dict1[key], dict2[key])

    def test_double_train_wrap_once(self):
        # test that we don't wrap the model more than once
        # since wrapping primarily happens on multi-gpu setup we want multiple gpus to test for
        # example DataParallel(DataParallel(model))

        trainer = get_regression_trainer(output_dir=self.get_auto_remove_tmp_dir())
        trainer.train()
        model_wrapped_before = trainer.model_wrapped
        trainer.train()
        model_wrapped_after = trainer.model_wrapped
        self.assertIs(model_wrapped_before, model_wrapped_after, "should be not wrapped twice")
