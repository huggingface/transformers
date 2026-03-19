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
Trainer checkpoint saving, loading, and resume tests: save intervals, sharded checkpoints,
auto batch size finder, resume with frozen params/gradient accumulation/different batch sizes,
checkpoint sorting and rotation, interrupted training recovery, JIT checkpointing (signal-based
checkpoint management), model/tokenizer/processor saving with best model selection, and Hub
push/tags/revision integration.
"""

import dataclasses
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import safetensors
import torch
from huggingface_hub import ModelCard, create_branch, list_repo_commits, list_repo_files
from torch import nn

from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Trainer,
    TrainerState,
    TrainingArguments,
    default_data_collator,
    is_torch_available,
)
from transformers.testing_utils import (
    ENDPOINT_STAGING,
    TOKEN,
    USER,
    CaptureLogger,
    TemporaryHubRepo,
    TestCasePlus,
    backend_device_count,
    evaluate_side_effect_factory,
    get_steps_per_epoch,
    is_staging_test,
    require_accelerate,
    require_deepspeed,
    require_non_hpu,
    require_peft,
    require_tensorboard,
    require_torch,
    require_torch_non_multi_accelerator,
    require_torch_up_to_2_accelerators,
    require_vision,
    run_first,
    run_test_using_subprocess,
    slow,
    torch_device,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    get_last_checkpoint,
    rotate_checkpoints,
    set_seed,
    sort_checkpoints,
)
from transformers.utils import SAFE_WEIGHTS_NAME, logging

from .trainer_test_utils import (
    PATH_SAMPLE_TEXT,
    AlmostAccuracy,
    MockCudaOOMCallback,
    RegressionDataset,
    RegressionModelConfig,
    RegressionPreTrainedModel,
    RegressionRandomPreTrainedModel,
    RegressionTrainingArguments,
    TrainerIntegrationCommon,
    get_dataset,
    get_language_model_trainer,
    get_regression_trainer,
)


if is_torch_available():
    from transformers.trainer_jit_checkpoint import CheckpointManager, JITCheckpointCallback


# ---------------------------------------------------------------------------
# Checkpoint save/load tests
# ---------------------------------------------------------------------------


@require_torch
class TrainerCheckpointSaveTest(TestCasePlus, TrainerIntegrationCommon):
    def setUp(self):
        super().setUp()
        args = TrainingArguments("..")
        self.n_epochs = args.num_train_epochs
        self.batch_size = args.train_batch_size

    def test_save_checkpoints(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        trainer = get_regression_trainer(output_dir=tmp_dir, save_steps=5)
        trainer.train()
        self.check_saved_checkpoints(tmp_dir, 5, int(self.n_epochs * 64 / self.batch_size))

        # With a regular model that is not a PreTrainedModel
        tmp_dir = self.get_auto_remove_tmp_dir()
        trainer = get_regression_trainer(output_dir=tmp_dir, save_steps=5, pretrained=False)
        trainer.train()
        self.check_saved_checkpoints(tmp_dir, 5, int(self.n_epochs * 64 / self.batch_size), False)

    def test_save_collator_tokenizer_by_default(self):
        class FakeCollator:
            def __init__(self):
                self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                self.tokenizer.add_tokens(["<NEW_TOKEN1>", "<NEW_TOKEN2>"])

            def __call__(self, features: list[Any], return_tensors="pt") -> dict[str, Any]:
                return default_data_collator(features, return_tensors)

        data_collator = FakeCollator()
        tmp_dir = self.get_auto_remove_tmp_dir()
        trainer = get_regression_trainer(output_dir=tmp_dir, save_steps=5, data_collator=data_collator)
        trainer.train()
        loaded_tokenizer = AutoTokenizer.from_pretrained(os.path.join(tmp_dir, os.listdir(tmp_dir)[0]))
        assert len(loaded_tokenizer) == len(trainer.data_collator.tokenizer), "Failed to load updated tokenizer"


# ---------------------------------------------------------------------------
# Resume from checkpoint tests
# ---------------------------------------------------------------------------


@require_torch
class TrainerResumeTrainingTest(TestCasePlus, TrainerIntegrationCommon):
    def setUp(self):
        super().setUp()
        args = TrainingArguments("..")
        self.n_epochs = args.num_train_epochs
        self.batch_size = args.train_batch_size

    @require_torch_non_multi_accelerator
    def test_can_resume_training(self):
        # This test will fail for more than 2 GPUs since the batch size will get bigger and with the number of
        # save_steps, the checkpoint will resume training at epoch 2 or more (so the data seen by the model
        # won't be the same since the training dataloader is shuffled).

        tmp_dir = self.get_auto_remove_tmp_dir()
        kwargs = {
            "output_dir": tmp_dir,
            "train_len": 128,
            "save_steps": 5,
            "learning_rate": 0.1,
            "logging_steps": 5,
        }
        trainer = get_regression_trainer(**kwargs)
        trainer.train()
        (a, b) = trainer.model.a.item(), trainer.model.b.item()
        state = dataclasses.asdict(trainer.state)

        checkpoint = os.path.join(tmp_dir, "checkpoint-5")

        # Reinitialize trainer
        trainer = get_regression_trainer(**kwargs)

        trainer.train(resume_from_checkpoint=checkpoint)
        (a1, b1) = trainer.model.a.item(), trainer.model.b.item()
        state1 = dataclasses.asdict(trainer.state)
        self.assertEqual(a, a1)
        self.assertEqual(b, b1)
        self.check_trainer_state_are_the_same(state, state1)

        # Now check with a later checkpoint that it also works when we span over one epoch
        checkpoint = os.path.join(tmp_dir, "checkpoint-15")

        # Reinitialize trainer and load model
        trainer = get_regression_trainer(**kwargs)

        trainer.train(resume_from_checkpoint=checkpoint)
        (a1, b1) = trainer.model.a.item(), trainer.model.b.item()
        state1 = dataclasses.asdict(trainer.state)
        self.assertEqual(a, a1)
        self.assertEqual(b, b1)
        self.check_trainer_state_are_the_same(state, state1)

        # With a regular model that is not a PreTrainedModel
        tmp_dir = self.get_auto_remove_tmp_dir()
        kwargs = {
            "output_dir": tmp_dir,
            "train_len": 128,
            "save_steps": 5,
            "learning_rate": 0.1,
            "pretrained": False,
        }

        trainer = get_regression_trainer(**kwargs)
        trainer.train()
        (a, b) = trainer.model.a.item(), trainer.model.b.item()
        state = dataclasses.asdict(trainer.state)

        checkpoint = os.path.join(tmp_dir, "checkpoint-5")

        # Reinitialize trainer and load model
        trainer = get_regression_trainer(**kwargs)

        trainer.train(resume_from_checkpoint=checkpoint)
        (a1, b1) = trainer.model.a.item(), trainer.model.b.item()
        state1 = dataclasses.asdict(trainer.state)
        self.assertEqual(a, a1)
        self.assertEqual(b, b1)
        self.check_trainer_state_are_the_same(state, state1)

        # Now check with a later checkpoint that it also works when we span over one epoch
        checkpoint = os.path.join(tmp_dir, "checkpoint-15")

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
        tmp_dir = self.get_auto_remove_tmp_dir()
        trainer = get_regression_trainer(output_dir=tmp_dir)
        with self.assertRaises(Exception) as context:
            trainer.train(resume_from_checkpoint=f"{checkpoint}-bogus")
        self.assertTrue("Can't find a valid checkpoint at" in str(context.exception))

        # 2. fail to find any checkpoint - due a fresh output_dir
        tmp_dir = self.get_auto_remove_tmp_dir()
        trainer = get_regression_trainer(output_dir=tmp_dir)
        with self.assertRaises(Exception) as context:
            trainer.train(resume_from_checkpoint=True)
        self.assertTrue("No valid checkpoint found in output directory" in str(context.exception))

    # require_torch_non_multi_accelerator is necessary because this worker blocks runs when using multiple GPUs, making
    # the test slower.
    @require_torch_non_multi_accelerator
    @run_test_using_subprocess
    @run_first
    @slow
    def test_can_resume_training_lm(self):
        # Check if it works for a simple language modeling example
        training_steps = 10
        resume_from_step = 8
        with tempfile.TemporaryDirectory() as tmpdir:
            kwargs = {
                "output_dir": tmpdir,
                "fp16": True,
                "max_steps": training_steps,
                "per_device_train_batch_size": 1,
                "learning_rate": 1e-5,
                "lr_scheduler_type": "cosine",
                "save_strategy": "steps",
                "save_steps": 1,
                "logging_strategy": "steps",
                "logging_steps": 1,
            }

            trainer = get_language_model_trainer(**kwargs)
            trainer.train(resume_from_checkpoint=False)
            # Get the parameter length of the model
            model_params = torch.cat([p.cpu().flatten() for p in trainer.model.parameters()])
            model_param_len = len(model_params)
            # Sample uniform indexes and save the values of the parameters (considering an unrolled vector with
            # all of them)
            indices = torch.randint(0, model_param_len, (1000,))
            # Save the values of the parameters for later comparison
            model_params_sample = model_params[indices].detach().clone()
            state1 = dataclasses.asdict(trainer.state)
            # Delete the reference
            del model_params, trainer
            # Checks if all checkpoints are there, +1 is necessary because range is 1-indexed
            self.check_saved_checkpoints(tmpdir, freq=1, total=training_steps + 1, is_pretrained=True, use_scaler=True)

            # Checkpoint at intermediate step
            checkpoint = os.path.join(tmpdir, f"checkpoint-{resume_from_step + 1}")
            trainer = get_language_model_trainer(**kwargs)
            trainer.train(resume_from_checkpoint=checkpoint)
            model_params = torch.cat([p.cpu().flatten() for p in trainer.model.parameters()])

            # Check that the parameters are the same
            self.assertTrue(torch.allclose(model_params[indices], model_params_sample))
            state2 = dataclasses.asdict(trainer.state)
            self.check_trainer_state_are_the_same(state1, state2)
            del model_params, trainer

    @unittest.skip(
        reason="@muellerzr: Fix once Trainer can take an accelerate configuration. Need to set `seedable_sampler=True`."
    )
    def test_resume_training_with_randomness(self):
        # For more than 1 GPUs, since the randomness is introduced in the model and with DataParallel (which is used
        # in this test for more than 2 GPUs), the calls to the torch RNG will happen in a random order (sometimes
        # GPU 0 will call first and sometimes GPU 1).
        random_torch = not torch.cuda.is_available() or backend_device_count(torch_device) <= 1

        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
        train_dataset = RegressionDataset(length=128)
        eval_dataset = RegressionDataset()

        with self.subTest("Test every step"):
            config = RegressionModelConfig(a=0, b=2, random_torch=random_torch)
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

            self.assertAlmostEqual(a, a1, delta=1e-5)
            self.assertAlmostEqual(b, b1, delta=1e-5)

        with self.subTest("Test every epoch"):
            config = RegressionModelConfig(a=0, b=2, random_torch=random_torch)
            model = RegressionRandomPreTrainedModel(config)

            tmp_dir = self.get_auto_remove_tmp_dir()
            args = RegressionTrainingArguments(tmp_dir, save_strategy="epoch", learning_rate=0.1)
            trainer = Trainer(model, args, train_dataset=train_dataset, eval_dataset=eval_dataset)

            trainer.train()
            (a, b) = trainer.model.a.item(), trainer.model.b.item()

            model = RegressionRandomPreTrainedModel(config)
            trainer = Trainer(model, args, train_dataset=train_dataset, eval_dataset=eval_dataset)

            checkpoints = [d for d in os.listdir(tmp_dir) if d.startswith("checkpoint-")]
            # There should be one checkpoint per epoch.
            self.assertEqual(len(checkpoints), 3)
            checkpoint_dir = min(checkpoints, key=lambda x: int(x.replace("checkpoint-", "")))

            trainer.train(resume_from_checkpoint=os.path.join(tmp_dir, checkpoint_dir))
            (a1, b1) = trainer.model.a.item(), trainer.model.b.item()

            self.assertAlmostEqual(a, a1, delta=1e-5)
            self.assertAlmostEqual(b, b1, delta=1e-5)

    def test_resume_training_with_different_batch_size(self):
        # Regression test for https://github.com/huggingface/transformers/issues/43708
        # When resuming from checkpoint without auto_find_batch_size, user's new batch size should be used
        train_dataset = RegressionDataset(length=64)

        config = RegressionModelConfig(a=0, b=2)
        model = RegressionRandomPreTrainedModel(config)

        tmp_dir = self.get_auto_remove_tmp_dir()

        # First training run with batch_size=2
        args = RegressionTrainingArguments(
            tmp_dir,
            do_train=True,
            max_steps=2,
            save_steps=1,
            per_device_train_batch_size=2,
            auto_find_batch_size=False,
        )
        trainer = Trainer(model, args, train_dataset=train_dataset)
        trainer.train()

        # Verify the checkpoint saved with the effective batch size (per_device * n_gpu)
        checkpoint = os.path.join(tmp_dir, "checkpoint-1")
        state = TrainerState.load_from_json(os.path.join(checkpoint, "trainer_state.json"))
        self.assertEqual(state.train_batch_size, args.train_batch_size)

        # Resume with a different batch_size=4 (without auto_find_batch_size)
        # The trainer should use the new batch_size, not the checkpoint's
        args2 = RegressionTrainingArguments(
            tmp_dir,
            do_train=True,
            max_steps=4,
            save_steps=1,
            per_device_train_batch_size=4,
            auto_find_batch_size=False,
        )
        trainer2 = Trainer(model, args2, train_dataset=train_dataset)
        trainer2.train(resume_from_checkpoint=checkpoint)

        # The trainer should be using the new batch size (4), not the checkpoint's (2)
        self.assertEqual(trainer2._train_batch_size, 4 * max(trainer2.args.n_gpu, 1))

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

    @require_torch_up_to_2_accelerators
    def test_resume_training_with_shard_checkpoint(self):
        # This test will fail for more than 2 GPUs since the batch size will get bigger and with the number of
        # save_steps, the checkpoint will resume training at epoch 2 or more (so the data seen by the model
        # won't be the same since the training dataloader is shuffled).

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(output_dir=tmpdir, train_len=128, save_steps=5, learning_rate=0.1)
            trainer.train()
            (a, b) = trainer.model.a.item(), trainer.model.b.item()
            state = dataclasses.asdict(trainer.state)

            checkpoint = os.path.join(tmpdir, "checkpoint-5")
            self.convert_to_sharded_checkpoint(checkpoint)

            # Reinitialize trainer
            trainer = get_regression_trainer(output_dir=tmpdir, train_len=128, save_steps=5, learning_rate=0.1)

            trainer.train(resume_from_checkpoint=checkpoint)
            (a1, b1) = trainer.model.a.item(), trainer.model.b.item()
            state1 = dataclasses.asdict(trainer.state)
            self.assertEqual(a, a1)
            self.assertEqual(b, b1)
            self.check_trainer_state_are_the_same(state, state1)

    @require_torch_up_to_2_accelerators
    def test_resume_training_with_checkpoint(self):
        # This test will fail for more than 2 GPUs since the batch size will get bigger and with the number of
        # save_steps, the checkpoint will resume training at epoch 2 or more (so the data seen by the model
        # won't be the same since the training dataloader is shuffled).

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                output_dir=tmpdir,
                train_len=128,
                save_steps=5,
                learning_rate=0.1,
            )
            trainer.train()
            (a, b) = trainer.model.a.item(), trainer.model.b.item()
            state = dataclasses.asdict(trainer.state)

            checkpoint = os.path.join(tmpdir, "checkpoint-5")
            self.convert_to_sharded_checkpoint(checkpoint)

            # Reinitialize trainer
            trainer = get_regression_trainer(output_dir=tmpdir, train_len=128, save_steps=5, learning_rate=0.1)

            trainer.train(resume_from_checkpoint=checkpoint)
            (a1, b1) = trainer.model.a.item(), trainer.model.b.item()
            state1 = dataclasses.asdict(trainer.state)
            self.assertEqual(a, a1)
            self.assertEqual(b, b1)
            self.check_trainer_state_are_the_same(state, state1)

    @require_torch_up_to_2_accelerators
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

    @require_torch_up_to_2_accelerators
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

    @require_peft
    def test_multiple_peft_adapters(self):
        from peft import LoraConfig, get_peft_model

        # Tests if resuming from checkpoint works if the model has multiple adapters

        MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tiny_model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

        peft_config = LoraConfig(
            r=4,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        tiny_model = get_peft_model(tiny_model, peft_config, "adapter1")
        tiny_model.add_adapter("adapter2", peft_config)

        train_dataset = get_dataset(PATH_SAMPLE_TEXT, tokenizer, 100)

        tokenizer.pad_token = tokenizer.eos_token

        tmp_dir = self.get_auto_remove_tmp_dir()
        args = TrainingArguments(
            tmp_dir,
            per_device_train_batch_size=1,
            learning_rate=1e-9,
            save_steps=5,
            logging_steps=5,
            max_steps=10,
            use_cpu=True,
        )
        trainer = Trainer(tiny_model, args, processing_class=tokenizer, train_dataset=train_dataset)

        trainer.train()
        parameters = dict(tiny_model.named_parameters())
        state = dataclasses.asdict(trainer.state)

        # Reinitialize trainer
        trainer = Trainer(tiny_model, args, processing_class=tokenizer, train_dataset=train_dataset)

        checkpoint = os.path.join(tmp_dir, "checkpoint-5")

        trainer.train(resume_from_checkpoint=checkpoint)
        parameters1 = dict(tiny_model.named_parameters())
        state1 = dataclasses.asdict(trainer.state)
        self.assertEqual(parameters, parameters1)
        self.check_trainer_state_are_the_same(state, state1)


# ---------------------------------------------------------------------------
# Auto batch size finder tests
# ---------------------------------------------------------------------------


@require_torch
class TrainerAutoBatchSizeTest(TestCasePlus, TrainerIntegrationCommon):
    def setUp(self):
        super().setUp()
        args = TrainingArguments("..")
        self.n_epochs = args.num_train_epochs
        self.batch_size = args.train_batch_size

    @slow
    @require_non_hpu
    @require_accelerate
    @require_torch_non_multi_accelerator
    def test_auto_batch_size_finder(self):
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True

        SRC_DIR = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "examples", "pytorch", "text-classification")
        )
        sys.path.append(SRC_DIR)
        import run_glue

        with tempfile.TemporaryDirectory() as tmpdir:
            testargs = f"""
                run_glue.py
                --model_name_or_path distilbert/distilbert-base-uncased
                --task_name mrpc
                --do_train
                --do_eval
                --max_seq_len 128
                --per_device_train_batch_size 4096
                --learning_rate 2e-5
                --num_train_epochs 1
                --output_dir {tmpdir}
                --auto_find_batch_size 0
                """.split()
            with self.assertRaises(RuntimeError):
                with patch.object(sys, "argv", testargs):
                    run_glue.main()

        testargs[-1] = "1"
        with patch.object(sys, "argv", testargs):
            run_glue.main()

    @require_deepspeed
    def test_auto_batch_size_with_deepspeed(self):
        train_dataset = RegressionDataset(length=128)

        config = RegressionModelConfig(a=0, b=2)
        model = RegressionRandomPreTrainedModel(config)

        tmp_dir = self.get_auto_remove_tmp_dir()

        for stage in [1, 2]:
            deepspeed = {
                "zero_optimization": {
                    "stage": stage,
                },
                "train_batch_size": "auto",
                "train_micro_batch_size_per_gpu": "auto",
            }

        args = RegressionTrainingArguments(
            tmp_dir,
            do_train=True,
            max_steps=2,
            save_strategy="no",
            per_device_train_batch_size=16,
            auto_find_batch_size=True,
            deepspeed=deepspeed,
        )
        trainer = Trainer(model, args, train_dataset=train_dataset, callbacks=[MockCudaOOMCallback()])
        trainer.train()
        self.assertEqual(trainer._train_batch_size, 14)

    def test_auto_batch_size_with_resume_from_checkpoint(self):
        train_dataset = RegressionDataset(length=128)

        config = RegressionModelConfig(a=0, b=2)
        model = RegressionRandomPreTrainedModel(config)

        tmp_dir = self.get_auto_remove_tmp_dir()

        args = RegressionTrainingArguments(
            tmp_dir,
            do_train=True,
            max_steps=2,
            save_steps=1,
            per_device_train_batch_size=16,
            auto_find_batch_size=True,
        )
        trainer = Trainer(model, args, train_dataset=train_dataset, callbacks=[MockCudaOOMCallback()])
        trainer.train()
        previous_batch_size = trainer._train_batch_size
        # Depends on the number of gpus so it is easier to just check that the batch_size decreased as expected
        self.assertEqual(trainer._train_batch_size < 16, True)

        # We can then make a new Trainer
        trainer = Trainer(model, args, train_dataset=train_dataset)
        # Check we are at 16 to start
        self.assertEqual(trainer._train_batch_size, 16 * max(trainer.args.n_gpu, 1))
        trainer.train(resume_from_checkpoint=True)
        # We should be back to 14 again, picking up based upon the last ran Trainer
        self.assertEqual(trainer._train_batch_size, previous_batch_size)


# ---------------------------------------------------------------------------
# Checkpoint sorting, rotation, and logging tests
# ---------------------------------------------------------------------------


@require_torch
class TrainerCheckpointRotationTest(TestCasePlus, TrainerIntegrationCommon):
    def setUp(self):
        super().setUp()
        args = TrainingArguments("..")
        self.n_epochs = args.num_train_epochs
        self.batch_size = args.train_batch_size

    def test_checkpoint_sorting(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create fake checkpoints in non-sorted order
            for n in [20, 5, 15, 25, 10]:
                os.makedirs(os.path.join(tmp_dir, f"{PREFIX_CHECKPOINT_DIR}-{n}"))

            # Test sorting by step number (oldest first)
            sorted_cps = sort_checkpoints(tmp_dir)
            values = [int(re.match(f".*{PREFIX_CHECKPOINT_DIR}-([0-9]+)", d).groups()[0]) for d in sorted_cps]
            self.assertEqual(values, [5, 10, 15, 20, 25])

            # Test with best_model_checkpoint - moved to second-to-last to protect from deletion
            best = os.path.join(tmp_dir, f"{PREFIX_CHECKPOINT_DIR}-5")
            sorted_cps = sort_checkpoints(tmp_dir, best_model_checkpoint=best)
            values = [int(re.match(f".*{PREFIX_CHECKPOINT_DIR}-([0-9]+)", d).groups()[0]) for d in sorted_cps]
            self.assertEqual(values, [10, 15, 20, 5, 25])

            # Test with best_model_checkpoint already at end (stays at end)
            best = os.path.join(tmp_dir, f"{PREFIX_CHECKPOINT_DIR}-25")
            sorted_cps = sort_checkpoints(tmp_dir, best_model_checkpoint=best)
            values = [int(re.match(f".*{PREFIX_CHECKPOINT_DIR}-([0-9]+)", d).groups()[0]) for d in sorted_cps]
            self.assertEqual(values, [5, 10, 15, 20, 25])

    def check_checkpoint_deletion(self, trainer, output_dir, expected):
        # Make fake checkpoints
        for n in [5, 10, 15, 20, 25]:
            os.makedirs(os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}-{n}"), exist_ok=True)
        rotate_checkpoints(
            output_dir=output_dir,
            save_total_limit=trainer.args.save_total_limit,
            best_model_checkpoint=trainer.state.best_model_checkpoint,
        )
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
                output_dir=tmp_dir, eval_strategy="steps", load_best_model_at_end=True, save_total_limit=2
            )
            trainer.state.best_model_checkpoint = os.path.join(tmp_dir, "checkpoint-5")
            self.check_checkpoint_deletion(trainer, tmp_dir, [5, 25])

            # Edge case: we don't always honor save_total_limit=1 if load_best_model_at_end=True to be able to resume
            # from checkpoint
            trainer = get_regression_trainer(
                output_dir=tmp_dir, eval_strategy="steps", load_best_model_at_end=True, save_total_limit=1
            )
            trainer.state.best_model_checkpoint = os.path.join(tmp_dir, "checkpoint-25")
            self.check_checkpoint_deletion(trainer, tmp_dir, [25])

            trainer.state.best_model_checkpoint = os.path.join(tmp_dir, "checkpoint-5")
            self.check_checkpoint_deletion(trainer, tmp_dir, [5, 25])

    def test_compare_trainer_and_checkpoint_args_logging(self):
        logger = logging.get_logger()

        with tempfile.TemporaryDirectory() as tmpdir, CaptureLogger(logger) as cl:
            trainer = get_regression_trainer(
                output_dir=tmpdir,
                train_len=128,
                eval_steps=5,
                gradient_accumulation_steps=2,
                per_device_train_batch_size=4,
                save_steps=5,
                learning_rate=0.1,
            )
            trainer.train()

            checkpoint = os.path.join(tmpdir, "checkpoint-5")
            checkpoint_trainer = get_regression_trainer(
                output_dir=tmpdir,
                train_len=256,
                eval_steps=10,
                gradient_accumulation_steps=4,
                per_device_train_batch_size=8,
                save_steps=10,
                learning_rate=0.1,
            )
            checkpoint_trainer.train(resume_from_checkpoint=checkpoint)

        self.assertIn("save_steps: 10 (from args) != 5 (from trainer_state.json)", cl.out)

        self.assertIn(
            "per_device_train_batch_size: 8 (from args) != 4 (from trainer_state.json)",
            cl.out,
        )
        self.assertIn(
            "eval_steps: 10 (from args) != 5 (from trainer_state.json)",
            cl.out,
        )


# ---------------------------------------------------------------------------
# Interrupted training and batch order tests
# ---------------------------------------------------------------------------


@require_torch
class TrainerInterruptedTrainingTest(TestCasePlus, TrainerIntegrationCommon):
    def setUp(self):
        super().setUp()
        args = TrainingArguments("..")
        self.n_epochs = args.num_train_epochs
        self.batch_size = args.train_batch_size

    def test_resume_from_interrupted_training(self):
        """
        Tests resuming training from a checkpoint after a simulated interruption.
        """

        # --- Helper classes and functions defined locally for this test ---
        class DummyModel(nn.Module):
            def __init__(self, input_dim=10, num_labels=2):
                super().__init__()
                self.linear = nn.Linear(input_dim, num_labels)

            def forward(self, input_ids=None, attention_mask=None, labels=None):
                logits = self.linear(input_ids.float())
                loss = None
                if labels is not None:
                    loss_fn = nn.CrossEntropyLoss()
                    loss = loss_fn(logits, labels)
                return {"loss": loss, "logits": logits}

        class DummyDictDataset(torch.utils.data.Dataset):
            def __init__(self, input_ids, attention_mask, labels):
                self.input_ids = input_ids
                self.attention_mask = attention_mask
                self.labels = labels

            def __len__(self):
                return len(self.input_ids)

            def __getitem__(self, idx):
                return {
                    "input_ids": self.input_ids[idx],
                    "attention_mask": self.attention_mask[idx],
                    "labels": self.labels[idx],
                }

        def create_dummy_dataset():
            """Creates a dummy dataset for this specific test."""
            num_samples = 100
            input_dim = 10
            dummy_input_ids = torch.rand(num_samples, input_dim)
            dummy_attention_mask = torch.ones(num_samples, input_dim)
            dummy_labels = torch.randint(0, 2, (num_samples,))
            return DummyDictDataset(dummy_input_ids, dummy_attention_mask, dummy_labels)

        # 1. Set up a dummy model and dataset
        model = DummyModel(input_dim=10, num_labels=2)
        dummy_dataset = create_dummy_dataset()

        # 2. First training phase (simulating an interruption)
        output_dir_initial = self.get_auto_remove_tmp_dir()
        training_args_initial = TrainingArguments(
            output_dir=output_dir_initial,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=3,
            save_strategy="steps",
            save_steps=1,  # Save at every step
            max_steps=2,  # Stop after step 2 to simulate interruption
        )

        trainer_initial = Trainer(
            model=model,
            args=training_args_initial,
            train_dataset=dummy_dataset,
        )
        trainer_initial.train()

        # 3. Verify that a checkpoint was created before the "interruption"
        checkpoint_path = os.path.join(output_dir_initial, "checkpoint-2")
        self.assertTrue(os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}")

        # 4. Second training phase (resuming from the checkpoint)
        output_dir_resumed = self.get_auto_remove_tmp_dir()
        # Total steps for one epoch is ceil(100 / (train_batch_size * 3)).
        # We stopped at step 2, so the resumed training should finish the remaining steps.
        training_args_resumed = TrainingArguments(
            output_dir=output_dir_resumed,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=3,
            save_strategy="steps",
            save_steps=1,
        )

        trainer_resumed = Trainer(
            model=model,
            args=training_args_resumed,
            train_dataset=dummy_dataset,
        )
        # Resume from the interrupted checkpoint and finish the remaining training
        trainer_resumed.train(resume_from_checkpoint=checkpoint_path)

        # 5. Assertions: Check if the training completed and the final model was saved
        # Total steps per epoch = ceil(num_samples / (train_batch_size * grad_accum))
        steps_per_epoch = math.ceil(
            100 / (training_args_resumed.train_batch_size * training_args_resumed.gradient_accumulation_steps)
        )
        self.assertEqual(trainer_resumed.state.global_step, steps_per_epoch)

        # Check that a checkpoint for the final step exists.
        final_checkpoint_path = os.path.join(output_dir_resumed, f"checkpoint-{steps_per_epoch}")
        self.assertTrue(os.path.exists(final_checkpoint_path))

        # Check if the model weights file exists in the final checkpoint directory.
        # Trainer saves non-PreTrainedModel models as `model.safetensors` by default if safetensors is available.
        final_model_path = os.path.join(final_checkpoint_path, SAFE_WEIGHTS_NAME)
        self.assertTrue(os.path.exists(final_model_path), "Final model checkpoint was not saved!")

    @require_torch_non_multi_accelerator
    def test_resume_batch_order(self):
        """
        Test that verifies dataloader order is reproducible when resuming from partial checkpoints.
        Tests resuming from checkpoint 7 (within epoch 1).
        """

        # --- Helper classes and functions defined locally for this test ---
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, size: int = 32):
                self.size = size
                self.data = torch.randn((size, 10))
                self.data[:, 0] = torch.arange(0, size)  # Encode the data order
                self.labels = torch.randint(0, 10, (size,))

            def __len__(self) -> int:
                return self.size

            def __getitem__(self, idx: int):
                return {"input_ids": self.data[idx], "labels": self.labels[idx]}

        class DummyModel(nn.Module):
            def __init__(self, size: int):
                super().__init__()
                self.fc = nn.Linear(10, 10, bias=False)
                # data_order logs the order of data points seen by the model
                self.register_buffer("data_order", torch.empty(0, dtype=torch.long))

            def load_state_dict(self, state_dict, strict=True):
                # Handle data_order buffer size mismatch during checkpoint loading
                if "data_order" in state_dict:
                    saved_data_order = state_dict["data_order"]
                    if hasattr(self, "data_order") and self.data_order.shape != saved_data_order.shape:
                        # Resize the buffer to match the saved state
                        self.data_order = saved_data_order.clone()

                return super().load_state_dict(state_dict, strict=strict)

            def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
                logits = self.fc(input_ids)
                loss = None
                if labels is not None:
                    loss_fn = nn.CrossEntropyLoss()
                    loss = loss_fn(logits, labels)

                # Log the data order for verification
                data_indices = input_ids[:, 0].int()
                self.data_order = torch.cat([self.data_order, data_indices.detach().clone()])

                return {"loss": loss, "logits": logits}

        # Scenario 1: Run baseline training to completion
        # 1.1 Run training to completion
        set_seed(42)
        train_dataset = DummyDataset(size=10)
        model_baseline = DummyModel(size=10)

        exp_dir_baseline = self.get_auto_remove_tmp_dir()
        args_baseline = TrainingArguments(
            output_dir=str(exp_dir_baseline),
            seed=42,
            learning_rate=0.1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            save_strategy="steps",
            save_steps=1,
            num_train_epochs=3,
            optim="sgd",
            disable_tqdm=True,
            dataloader_num_workers=0,  # Ensures that main process loads the data
        )

        trainer_baseline = Trainer(
            model=model_baseline,
            args=args_baseline,
            train_dataset=train_dataset,
        )

        trainer_baseline.train()

        # 1.2 Get the data order from the last saved checkpoint for the full run
        last_checkpoint_path = get_last_checkpoint(exp_dir_baseline)
        last_ckpt_num = int(os.path.basename(last_checkpoint_path).split("-")[1])  # Must be 15

        baseline_state_dict = safetensors.torch.load_file(
            os.path.join(exp_dir_baseline, f"checkpoint-{last_ckpt_num}", "model.safetensors")
        )
        baseline_data_order = baseline_state_dict["data_order"]

        # Scenario 2: Resume training from checkpoint in the middle of the second epoch
        # 2.1 Resume training from the second batch of epoch 1 (target_ckpt_num = 7)
        # 1 epoch consists of 10 points, so 5 steps with batch size 2
        target_ckpt_num = 7
        checkpoint_path = os.path.join(exp_dir_baseline, f"checkpoint-{target_ckpt_num - 1}")

        set_seed(42)
        model_resume = DummyModel(size=10)

        exp_dir_resume = self.get_auto_remove_tmp_dir()
        args_resume = TrainingArguments(
            output_dir=str(exp_dir_resume),
            seed=42,
            learning_rate=0.1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            save_strategy="steps",
            save_steps=1,
            num_train_epochs=3,
            optim="sgd",
            disable_tqdm=True,
            dataloader_num_workers=0,  # Ensures that main process loads the data
        )

        trainer_resume = Trainer(
            model=model_resume,
            args=args_resume,
            train_dataset=train_dataset,
        )

        trainer_resume.train(resume_from_checkpoint=checkpoint_path)

        # 2.2 Get the data order from the last saved checkpoint for the resumed run
        resumed_state_dict = safetensors.torch.load_file(
            os.path.join(exp_dir_resume, f"checkpoint-{last_ckpt_num}", "model.safetensors")
        )
        resumed_data_order = resumed_state_dict["data_order"]

        # 3. Compare results: the data order should be identical
        self.assertTrue(
            torch.equal(baseline_data_order, resumed_data_order),
            f"Data order mismatch after checkpoint deletion and resume.\n"
            f"Baseline: {baseline_data_order}\n"
            f"Resumed: {resumed_data_order}",
        )


# ---------------------------------------------------------------------------
# JIT checkpoint tests
# ---------------------------------------------------------------------------


@require_torch
class JITCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    def get_trainer(self, enable_jit=True):
        """Helper method to create a trainer with JIT checkpointing enabled."""
        from transformers import Trainer

        model_config = RegressionModelConfig(a=1.5, b=2.5)
        model = RegressionPreTrainedModel(model_config)

        args = TrainingArguments(
            output_dir=self.test_dir,
            enable_jit_checkpoint=enable_jit,
            per_device_train_batch_size=16,
            learning_rate=0.1,
            logging_steps=1,
            num_train_epochs=1,
            max_steps=10,
            save_steps=10,
        )

        train_dataset = RegressionDataset(length=64)

        return Trainer(model=model, args=args, train_dataset=train_dataset)

    def test_checkpoint_manager_initialization(self):
        """Test CheckpointManager initialization with different configurations."""
        trainer = self.get_trainer()

        # Test with default parameters
        manager = CheckpointManager(trainer)
        self.assertEqual(manager.trainer, trainer)
        self.assertEqual(manager.kill_wait, 3)
        self.assertFalse(manager.is_checkpoint_requested)

        # Test with custom parameters
        manager_custom = CheckpointManager(trainer, kill_wait=5)
        self.assertEqual(manager_custom.kill_wait, 5)

    def test_signal_handler_setup(self):
        """Test signal handler setup and restoration."""
        trainer = self.get_trainer()
        manager = CheckpointManager(trainer)

        # Store original handler
        original_handler = signal.signal(signal.SIGTERM, signal.SIG_DFL)

        try:
            # Setup JIT signal handler
            manager.setup_signal_handler()

            # Verify handler is set
            current_handler = signal.signal(signal.SIGTERM, signal.SIG_DFL)
            self.assertNotEqual(current_handler, signal.SIG_DFL)

            # Verify original handler is stored
            self.assertIsNotNone(manager._original_sigterm_handler)

        finally:
            # Restore original handler
            signal.signal(signal.SIGTERM, original_handler)

    @patch("threading.Timer")
    def test_sigterm_handler_flow(self, mock_timer):
        """Test SIGTERM handler execution flow."""
        trainer = self.get_trainer()
        manager = CheckpointManager(trainer, kill_wait=2)

        # Mock timer to prevent actual threading
        mock_timer_instance = Mock()
        mock_timer.return_value = mock_timer_instance

        # Test first SIGTERM call
        self.assertFalse(manager.is_checkpoint_requested)
        manager._sigterm_handler(signal.SIGTERM, None)

        # Verify checkpoint was NOT immediately requested (timer is used)
        self.assertFalse(manager.is_checkpoint_requested)

        # Verify timer was created with kill_wait period and correct callback
        mock_timer.assert_called_once_with(2, manager._enable_checkpoint)
        mock_timer_instance.start.assert_called_once()

        # Manually trigger the timer callback to test flag setting
        manager._enable_checkpoint()

        # Verify checkpoint is now requested
        self.assertTrue(manager.is_checkpoint_requested)

        # Test second SIGTERM call (should be ignored)
        mock_timer.reset_mock()
        manager._sigterm_handler(signal.SIGTERM, None)

        # Verify no additional timer was created
        mock_timer.assert_not_called()

    def test_toggle_checkpoint_flag(self):
        """Test the toggle checkpoint flag method."""
        trainer = self.get_trainer()
        manager = CheckpointManager(trainer)

        # Initially should not be requested
        self.assertFalse(manager.is_checkpoint_requested)

        # Toggle flag
        manager._enable_checkpoint()

        # Should now be requested
        self.assertTrue(manager.is_checkpoint_requested)

    def test_execute_jit_checkpoint(self):
        """Test the checkpoint execution logic with sentinel file."""
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

        trainer = self.get_trainer()
        manager = CheckpointManager(trainer)

        # Mock trainer's save checkpoint method
        trainer._save_checkpoint = Mock()
        trainer.state.global_step = 42

        # Set checkpoint requested flag
        manager.is_checkpoint_requested = True

        # Execute checkpoint
        manager.execute_jit_checkpoint()

        # Verify checkpoint was called
        trainer._save_checkpoint.assert_called_once_with(trainer.model, trial=None)

        # Verify checkpoint flag was reset
        self.assertFalse(manager.is_checkpoint_requested)

        # Verify sentinel file was removed (should be in checkpoint-42 folder)
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-42"
        sentinel_file = os.path.join(self.test_dir, checkpoint_folder, "checkpoint-is-incomplete.txt")
        self.assertFalse(os.path.exists(sentinel_file))

    def test_execute_jit_checkpoint_sentinel_file_cleanup(self):
        """Test that sentinel file is cleaned up after successful checkpoint."""
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

        trainer = self.get_trainer()
        manager = CheckpointManager(trainer)

        # Mock trainer's save checkpoint method
        trainer._save_checkpoint = Mock()
        trainer.state.global_step = 42

        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-42"
        sentinel_file = os.path.join(self.test_dir, checkpoint_folder, "checkpoint-is-incomplete.txt")

        # Execute checkpoint
        manager.execute_jit_checkpoint()

        # Verify sentinel file doesn't exist after successful checkpoint
        self.assertFalse(os.path.exists(sentinel_file))

    def test_execute_jit_checkpoint_with_exception(self):
        """Test checkpoint execution with exception handling."""
        trainer = self.get_trainer()
        manager = CheckpointManager(trainer)

        # Mock trainer's save checkpoint method to raise exception
        trainer._save_checkpoint = Mock(side_effect=Exception("Checkpoint failed"))
        trainer.state.global_step = 42

        # Test that exception is re-raised
        with self.assertRaises(Exception) as context:
            manager.execute_jit_checkpoint()

        self.assertEqual(str(context.exception), "Checkpoint failed")

        # Verify checkpoint flag was still reset to avoid multiple attempts
        self.assertFalse(manager.is_checkpoint_requested)

    def test_jit_checkpoint_callback_initialization(self):
        """Test JITCheckpointCallback initialization."""
        callback = JITCheckpointCallback()

        self.assertIsNone(callback.trainer)
        self.assertIsNone(callback.jit_manager)

    def test_jit_checkpoint_callback_set_trainer_enabled(self):
        """Test setting trainer with JIT checkpointing enabled."""
        trainer = self.get_trainer(enable_jit=True)
        callback = JITCheckpointCallback()

        with patch.object(CheckpointManager, "setup_signal_handler") as mock_setup:
            callback.set_trainer(trainer)

            self.assertEqual(callback.trainer, trainer)
            self.assertIsNotNone(callback.jit_manager)
            self.assertIsInstance(callback.jit_manager, CheckpointManager)
            mock_setup.assert_called_once()

    def test_jit_checkpoint_callback_set_trainer_disabled(self):
        """Test setting trainer with JIT checkpointing disabled."""
        trainer = self.get_trainer(enable_jit=False)
        callback = JITCheckpointCallback()

        callback.set_trainer(trainer)

        self.assertEqual(callback.trainer, trainer)
        self.assertIsNone(callback.jit_manager)

    def test_jit_checkpoint_callback_on_pre_optimizer_step(self):
        """Test callback behavior during pre-optimizer step."""
        trainer = self.get_trainer()
        callback = JITCheckpointCallback()
        callback.set_trainer(trainer)

        # Mock control object
        control = Mock()
        control.should_training_stop = False

        # Mock execute method
        with patch.object(callback.jit_manager, "execute_jit_checkpoint") as mock_execute:
            # Test when checkpoint not requested
            callback.jit_manager.is_checkpoint_requested = False
            callback.on_pre_optimizer_step(trainer.args, trainer.state, control)
            self.assertFalse(control.should_training_stop)
            mock_execute.assert_not_called()

            # Test when checkpoint requested
            callback.jit_manager.is_checkpoint_requested = True
            callback.on_pre_optimizer_step(trainer.args, trainer.state, control)
            self.assertTrue(control.should_training_stop)
            mock_execute.assert_called_once()

    def test_jit_checkpoint_callback_on_step_begin(self):
        """Test callback behavior at step begin."""
        trainer = self.get_trainer()
        callback = JITCheckpointCallback()
        callback.set_trainer(trainer)

        # Mock control object
        control = Mock()
        control.should_training_stop = False

        # Mock execute method
        with patch.object(callback.jit_manager, "execute_jit_checkpoint") as mock_execute:
            # Test when checkpoint not requested
            callback.jit_manager.is_checkpoint_requested = False
            callback.on_step_begin(trainer.args, trainer.state, control)
            self.assertFalse(control.should_training_stop)
            mock_execute.assert_not_called()

            # Test when checkpoint requested
            callback.jit_manager.is_checkpoint_requested = True
            callback.on_step_begin(trainer.args, trainer.state, control)
            self.assertTrue(control.should_training_stop)
            mock_execute.assert_called_once()

    def test_jit_checkpoint_callback_on_step_end(self):
        """Test callback behavior at step end."""
        trainer = self.get_trainer()
        callback = JITCheckpointCallback()
        callback.set_trainer(trainer)

        # Mock control object
        control = Mock()
        control.should_training_stop = False
        control.should_save = True

        # Mock execute method
        with patch.object(callback.jit_manager, "execute_jit_checkpoint") as mock_execute:
            # Test when checkpoint not requested
            callback.jit_manager.is_checkpoint_requested = False
            callback.on_step_end(trainer.args, trainer.state, control)
            self.assertFalse(control.should_training_stop)
            mock_execute.assert_not_called()

            # Reset control
            control.should_save = True

            # Test when checkpoint requested
            callback.jit_manager.is_checkpoint_requested = True
            callback.on_step_end(trainer.args, trainer.state, control)
            self.assertTrue(control.should_training_stop)
            self.assertFalse(control.should_save)
            mock_execute.assert_called_once()

    def test_jit_checkpoint_callback_on_epoch_end(self):
        """Test callback behavior at epoch end."""
        trainer = self.get_trainer()
        callback = JITCheckpointCallback()
        callback.set_trainer(trainer)

        # Mock control object
        control = Mock()
        control.should_save = True
        control.should_training_stop = False

        # Mock execute method
        with patch.object(callback.jit_manager, "execute_jit_checkpoint") as mock_execute:
            # Test when checkpoint not requested
            callback.jit_manager.is_checkpoint_requested = False
            callback.on_epoch_end(trainer.args, trainer.state, control)
            # should_save should remain unchanged when checkpoint not requested
            self.assertTrue(control.should_save)
            self.assertFalse(control.should_training_stop)
            mock_execute.assert_not_called()

            # Reset control
            control.should_save = True
            control.should_training_stop = False

            # Test when checkpoint requested
            callback.jit_manager.is_checkpoint_requested = True
            callback.on_epoch_end(trainer.args, trainer.state, control)
            self.assertFalse(control.should_save)
            self.assertTrue(control.should_training_stop)
            mock_execute.assert_called_once()

    def test_jit_checkpoint_callback_on_train_end(self):
        """Test signal handler restoration on training end."""
        trainer = self.get_trainer()
        callback = JITCheckpointCallback()

        # Store original SIGTERM handler
        original_handler = signal.signal(signal.SIGTERM, signal.SIG_DFL)

        try:
            callback.set_trainer(trainer)

            # Verify signal handler was set up
            self.assertIsNotNone(callback.jit_manager._original_sigterm_handler)

            # Mock control object
            control = Mock()

            # Call on_train_end
            callback.on_train_end(trainer.args, trainer.state, control)

            # Verify signal handler was restored
            current_handler = signal.signal(signal.SIGTERM, signal.SIG_DFL)
            self.assertEqual(current_handler, callback.jit_manager._original_sigterm_handler)

        finally:
            # Restore original handler for cleanup
            signal.signal(signal.SIGTERM, original_handler)

    @patch("threading.Timer")
    def test_kill_wait_period(self, mock_timer):
        """Test the kill wait period functionality."""
        trainer = self.get_trainer()
        manager = CheckpointManager(trainer, kill_wait=5)

        mock_timer_instance = Mock()
        mock_timer.return_value = mock_timer_instance

        manager._sigterm_handler(signal.SIGTERM, None)

        # Verify Timer was created with the correct kill_wait period and callback
        mock_timer.assert_called_once_with(5, manager._enable_checkpoint)
        mock_timer_instance.start.assert_called_once()

    def test_integration_with_trainer(self):
        """Test integration of JIT checkpointing with Trainer."""
        trainer = self.get_trainer(enable_jit=True)

        # Check that JIT callback was added
        jit_callbacks = [cb for cb in trainer.callback_handler.callbacks if isinstance(cb, JITCheckpointCallback)]
        self.assertEqual(len(jit_callbacks), 1)

        jit_callback = jit_callbacks[0]
        self.assertIsNotNone(jit_callback.jit_manager)
        self.assertEqual(jit_callback.trainer, trainer)


# ---------------------------------------------------------------------------
# Trainer saving tests (tokenizer, image processor, feature extractor, etc.)
# ---------------------------------------------------------------------------


@require_torch
class TrainerSavingTest(TestCasePlus, TrainerIntegrationCommon):
    def setUp(self):
        super().setUp()
        args = TrainingArguments("..")
        self.n_epochs = args.num_train_epochs
        self.batch_size = args.train_batch_size

    def test_trainer_saves_tokenizer(self):
        MODEL_ID = "google-bert/bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = RegressionModelConfig(a=1.5, b=2.5)
            trainer = Trainer(
                model=RegressionPreTrainedModel(config),
                args=TrainingArguments(output_dir=tmp_dir),
                processing_class=tokenizer,
            )
            trainer.save_model()

            reloaded_tokenizer = AutoTokenizer.from_pretrained(tmp_dir)

        # For tokenizers, there isn't a direct to_dict method and the properties stored in the configs e.g.
        # saved tokens change overtime, so we check that two tokenizers are equal by comparing their encoded outputs
        test_sentence = "This is a test sentence"
        self.assertListEqual(
            tokenizer(test_sentence, padding="max_length").input_ids,
            reloaded_tokenizer(test_sentence, padding="max_length").input_ids,
        )

    @require_vision
    def test_trainer_saves_image_processor(self):
        MODEL_ID = "openai/clip-vit-base-patch32"
        image_processor = AutoImageProcessor.from_pretrained(MODEL_ID)

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = RegressionModelConfig(a=1.5, b=2.5)
            trainer = Trainer(
                model=RegressionPreTrainedModel(config),
                args=TrainingArguments(output_dir=tmp_dir),
                processing_class=image_processor,
            )
            trainer.save_model()
            reloaded_image_processor = AutoImageProcessor.from_pretrained(tmp_dir)

        self.assertDictEqual(image_processor.to_dict(), reloaded_image_processor.to_dict())

    def test_trainer_saves_feature_extractor(self):
        MODEL_ID = "facebook/wav2vec2-base-960h"
        feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = RegressionModelConfig(a=1.5, b=2.5)
            trainer = Trainer(
                model=RegressionPreTrainedModel(config),
                args=TrainingArguments(output_dir=tmp_dir),
                processing_class=feature_extractor,
            )
            trainer.save_model()

            reloaded_feature_extractor = AutoFeatureExtractor.from_pretrained(tmp_dir)

        self.assertDictEqual(feature_extractor.to_dict(), reloaded_feature_extractor.to_dict())

    @require_vision
    def test_trainer_saves_processor(self):
        MODEL_ID = "openai/clip-vit-base-patch32"
        image_processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
        processor = AutoProcessor.from_pretrained(MODEL_ID)

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = RegressionModelConfig(a=1.5, b=2.5)
            trainer = Trainer(
                model=RegressionPreTrainedModel(config),
                args=TrainingArguments(output_dir=tmp_dir),
                processing_class=processor,
            )
            trainer.save_model()

            reloaded_processor = AutoProcessor.from_pretrained(tmp_dir)
            reloaded_image_processor = AutoImageProcessor.from_pretrained(tmp_dir)
            reloaded_tokenizer = AutoTokenizer.from_pretrained(tmp_dir)

        self.assertDictEqual(reloaded_processor.to_dict(), processor.to_dict())

        image_processor_dict = image_processor.to_dict()
        reloaded_image_processor_dict = reloaded_image_processor.to_dict()
        self.assertDictEqual(image_processor_dict, reloaded_image_processor_dict)

        # For tokenizers, there isn't a direct to_dict method and the properties stored in the configs e.g.
        # saved tokens change overtime, so we check that two tokenizers are equal by comparing their encoded outputs
        test_sentence = "This is a test sentence"
        self.assertListEqual(
            tokenizer(test_sentence, padding="max_length").input_ids,
            reloaded_tokenizer(test_sentence, padding="max_length").input_ids,
        )


# ---------------------------------------------------------------------------
# Best model selection and loading tests
# ---------------------------------------------------------------------------


@require_torch
class TrainerBestModelTest(TestCasePlus, TrainerIntegrationCommon):
    """Tests for best model selection, loading, and checkpoint behavior."""

    def setUp(self):
        super().setUp()
        args = TrainingArguments("..")
        self.n_epochs = args.num_train_epochs
        self.batch_size = args.train_batch_size

    def test_load_best_model_with_save(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        trainer = get_regression_trainer(
            output_dir=tmp_dir,
            save_steps=5,
            eval_strategy="steps",
            eval_steps=5,
            max_steps=9,
        )
        trainer.train()
        # Check that we have the last known step:
        assert os.path.exists(os.path.join(tmp_dir, f"checkpoint-{trainer.state.max_steps}")), (
            f"Could not find checkpoint-{trainer.state.max_steps}"
        )
        # And then check the last step
        assert os.path.exists(os.path.join(tmp_dir, "checkpoint-9")), "Could not find checkpoint-9"

        # Now test that using a limit works
        # Should result in:
        # - save at step 5 (but is deleted)
        # - save at step 10 (loaded in at the end when `load_best_model=True`)
        # - save at step 11
        tmp_dir = self.get_auto_remove_tmp_dir()
        trainer = get_regression_trainer(
            output_dir=tmp_dir,
            save_steps=5,
            eval_strategy="steps",
            eval_steps=5,
            load_best_model_at_end=True,
            save_total_limit=2,
            max_steps=11,
        )
        trainer.train()
        # Check that we have the last known step:
        assert os.path.exists(os.path.join(tmp_dir, "checkpoint-11")), "Could not find checkpoint-11"
        # And then check the last multiple
        assert os.path.exists(os.path.join(tmp_dir, "checkpoint-10")), "Could not find checkpoint-10"
        # Finally check that we don't have an old one
        assert not os.path.exists(os.path.join(tmp_dir, "checkpoint-5")), "Found checkpoint-5, limit not respected"

        # Finally check that the right model was loaded in, checkpoint-10
        # this goes by the last `eval` step check to do so, so it won't be
        # the last model *saved*
        model_state = trainer.model.state_dict()
        final_model_weights = safetensors.torch.load_file(os.path.join(tmp_dir, "checkpoint-10", "model.safetensors"))
        for k, v in model_state.items():
            assert torch.allclose(v, final_model_weights[k]), f"{k} is not the same"

    def test_save_best_checkpoint(self):
        freq = int(64 / self.batch_size)
        total = int(self.n_epochs * 64 / self.batch_size)

        # Case 1: args.metric_for_best_model == "accuracy".
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                a=1.5,
                b=2.5,
                output_dir=tmpdir,
                learning_rate=0.1,
                eval_strategy="epoch",
                save_strategy="best",
                metric_for_best_model="accuracy",
                compute_metrics=AlmostAccuracy(),
            )
            self.assertTrue(trainer.args.metric_for_best_model == "accuracy")

            with patch.object(
                trainer,
                "_evaluate",
                side_effect=[
                    {"eval_loss": 0.03, "eval_accuracy": 0.60, "epoch": 1.0},
                    {"eval_loss": 0.02, "eval_accuracy": 0.65, "epoch": 2.0},
                    {"eval_loss": 0.01, "eval_accuracy": 0.64, "epoch": 3.0},
                ],
            ):
                trainer.train()

                self.assertEqual(len(os.listdir(tmpdir)), 2)
                self.check_saved_checkpoints(
                    output_dir=tmpdir,
                    freq=freq,
                    total=total,
                )

        # Case 2: args.metric_for_best_model == "loss".
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                a=1.5,
                b=2.5,
                output_dir=tmpdir,
                learning_rate=0.1,
                eval_strategy="epoch",
                save_strategy="best",
                metric_for_best_model="loss",
                compute_metrics=AlmostAccuracy(),
            )
            self.assertTrue(trainer.args.metric_for_best_model == "loss")

            with patch.object(
                trainer,
                "_evaluate",
                side_effect=[
                    {"eval_loss": 0.03, "eval_accuracy": 0.60, "epoch": 1.0},
                    {"eval_loss": 0.02, "eval_accuracy": 0.65, "epoch": 2.0},
                    {"eval_loss": 0.03, "eval_accuracy": 0.66, "epoch": 3.0},
                ],
            ):
                trainer.train()

                self.assertEqual(len(os.listdir(tmpdir)), 2)
                self.check_saved_checkpoints(
                    output_dir=tmpdir,
                    freq=freq,
                    total=total,
                )

    def test_metric_for_best_model_behavior(self):
        # Case 1: Metric name not provided when `save_strategy == "best"`.
        # Should raise ValueError.
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError) as context:
                trainer = get_regression_trainer(
                    a=1.5,
                    b=2.5,
                    output_dir=tmpdir,
                    learning_rate=0.1,
                    eval_strategy="epoch",
                    save_strategy="best",
                    compute_metrics=AlmostAccuracy(),
                )
            self.assertIn("`args.metric_for_best_model` must be provided", str(context.exception))

        # Case 2: Metric name not provided when `load_best_model_at_end == True`.
        # `metric_for_best_model` should be set to `"loss"` by default.
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                a=1.5,
                b=2.5,
                output_dir=tmpdir,
                learning_rate=0.1,
                eval_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=True,
            )
            self.assertTrue(trainer.args.metric_for_best_model == "loss")

    def test_best_model_checkpoint_behavior(self):
        # Case 1. Never evaluated, save_total_limit > 1 and save_steps == 1.
        # Both best_metric and best_model_checkpoint should be None.
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                output_dir=tmpdir,
                eval_strategy="steps",
                save_strategy="steps",
                save_steps=1,
                metric_for_best_model="accuracy",
                greater_is_better=True,
            )
            trainer.train()

            assert trainer.state.best_metric is None
            assert trainer.state.best_model_checkpoint is None
            assert len(os.listdir(tmpdir)) == trainer.state.global_step

        # Case 2. Never evaluated and save_total_limit == 1.
        # Both best_metric and best_model_checkpoint should be None.
        # Only the last checkpoint should remain.
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                output_dir=tmpdir,
                eval_strategy="steps",
                save_strategy="steps",
                save_steps=1,
                metric_for_best_model="accuracy",
                greater_is_better=True,
                save_total_limit=1,
            )
            trainer.train()

            num_steps = trainer.state.global_step

            assert trainer.state.best_metric is None
            assert trainer.state.best_model_checkpoint is None
            assert len(os.listdir(tmpdir)) == 1

            ckpt = os.path.join(tmpdir, f"{PREFIX_CHECKPOINT_DIR}-{num_steps}")
            assert os.path.isdir(ckpt)
            assert os.listdir(tmpdir)[0] == f"{PREFIX_CHECKPOINT_DIR}-{num_steps}"

        # Case 3. eval_strategy == save_strategy.
        # best_model_checkpoint should be at epoch 1.
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                output_dir=tmpdir,
                eval_strategy="epoch",
                save_strategy="epoch",
                metric_for_best_model="accuracy",
                compute_metrics=AlmostAccuracy(),
                greater_is_better=True,
                load_best_model_at_end=False,
            )

            with patch.object(
                trainer,
                "_evaluate",
                side_effect=evaluate_side_effect_factory(
                    [
                        {"eval_accuracy": 0.59},
                        {"eval_accuracy": 0.57},
                        {"eval_accuracy": 0.55},
                    ]
                ),
            ):
                trainer.train()

            steps_per_epoch = get_steps_per_epoch(trainer)

            assert trainer.state.best_metric == 0.59
            assert trainer.state.best_global_step == steps_per_epoch

            best_ckpt = os.path.join(tmpdir, f"{PREFIX_CHECKPOINT_DIR}-{trainer.state.best_global_step}")
            assert trainer.state.best_model_checkpoint == best_ckpt

            assert len(os.listdir(tmpdir)) == trainer.state.num_train_epochs

        # Case 4. eval_strategy != save_strategy.
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                output_dir=tmpdir,
                eval_strategy="epoch",
                save_strategy="steps",
                save_steps=1,
                metric_for_best_model="accuracy",
                compute_metrics=AlmostAccuracy(),
                greater_is_better=True,
                load_best_model_at_end=False,
            )

            with patch.object(
                trainer,
                "_evaluate",
                side_effect=evaluate_side_effect_factory(
                    [
                        {"eval_accuracy": 0.59},
                        {"eval_accuracy": 0.57},
                        {"eval_accuracy": 0.55},
                    ]
                ),
            ):
                trainer.train()

            steps_per_epoch = get_steps_per_epoch(trainer)

            assert trainer.state.best_metric == 0.59
            assert trainer.state.best_global_step == steps_per_epoch

            best_ckpt = os.path.join(tmpdir, f"{PREFIX_CHECKPOINT_DIR}-{trainer.state.best_global_step}")
            assert trainer.state.best_model_checkpoint == best_ckpt

            assert len(os.listdir(tmpdir)) == trainer.state.global_step

        # Case 5. Multiple checkpoints, save_total_limit == 1.
        # Best metric is found at step 1 and that checkpoint should be saved.
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                output_dir=tmpdir,
                eval_strategy="steps",
                eval_steps=1,
                save_strategy="steps",
                save_steps=1,
                metric_for_best_model="accuracy",
                compute_metrics=AlmostAccuracy(),
                greater_is_better=True,
                save_total_limit=1,
            )

            with patch.object(
                trainer,
                "_evaluate",
                side_effect=evaluate_side_effect_factory(
                    [
                        {"eval_accuracy": 0.90},
                        {"eval_accuracy": 0.80},
                        {"eval_accuracy": 0.70},
                    ]
                ),
            ):
                trainer.train()

            assert trainer.state.best_metric == 0.90
            assert trainer.state.best_global_step == 1

            best_ckpt = os.path.join(tmpdir, f"{PREFIX_CHECKPOINT_DIR}-{trainer.state.best_global_step}")
            assert trainer.state.best_model_checkpoint == best_ckpt

            assert len(os.listdir(tmpdir)) == 1

        # Case 6. Saving happens more often and eval/save mismatch.
        # `best_model_checkpoint` should be None due to a step mismatch.
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                output_dir=tmpdir,
                eval_strategy="steps",
                eval_steps=3,
                save_strategy="steps",
                save_steps=2,
                metric_for_best_model="accuracy",
                compute_metrics=AlmostAccuracy(),
                greater_is_better=True,
            )

            with patch.object(
                trainer,
                "_evaluate",
                side_effect=evaluate_side_effect_factory(
                    [
                        {"eval_accuracy": 0.90},
                        {"eval_accuracy": 0.80},
                        {"eval_accuracy": 0.70},
                    ]
                ),
            ):
                trainer.train()

            assert trainer.state.best_metric == 0.90
            assert trainer.state.best_global_step == 3

            assert trainer.state.best_model_checkpoint is None

            assert len(os.listdir(tmpdir)) == trainer.state.global_step // 2

    def test_load_best_model_at_end(self):
        total = int(self.n_epochs * 64 / self.batch_size)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                a=1.5,
                b=2.5,
                output_dir=tmpdir,
                learning_rate=0.1,
                eval_steps=5,
                eval_strategy="steps",
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
                eval_strategy="steps",
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
                eval_strategy="epoch",
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
                eval_strategy="steps",
                save_steps=5,
                load_best_model_at_end=True,
                pretrained=False,
            )
            self.assertFalse(trainer.args.greater_is_better)
            trainer.train()
            self.check_saved_checkpoints(tmpdir, 5, total, is_pretrained=False)
            self.check_best_model_has_been_loaded(tmpdir, 5, total, trainer, "eval_loss", is_pretrained=False)

    def test_load_best_model_from_safetensors(self):
        total = int(self.n_epochs * 64 / self.batch_size)
        for pretrained in [False, True]:
            with tempfile.TemporaryDirectory() as tmpdir:
                trainer = get_regression_trainer(
                    a=1.5,
                    b=2.5,
                    output_dir=tmpdir,
                    learning_rate=0.1,
                    eval_steps=5,
                    eval_strategy="steps",
                    save_steps=5,
                    load_best_model_at_end=True,
                    pretrained=pretrained,
                )
                self.assertFalse(trainer.args.greater_is_better)
                trainer.train()
                self.check_saved_checkpoints(tmpdir, 5, total, is_pretrained=pretrained)
                self.check_best_model_has_been_loaded(tmpdir, 5, total, trainer, "eval_loss", is_pretrained=pretrained)


# ---------------------------------------------------------------------------
# Hub integration tests (push, tags, revision)
# ---------------------------------------------------------------------------


@require_torch
@is_staging_test
class TrainerIntegrationWithHubTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._token = TOKEN

    def test_push_to_hub(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            output_dir_name = tmp_repo.repo_name
            with tempfile.TemporaryDirectory() as tmp_dir:
                trainer = get_regression_trainer(
                    output_dir=os.path.join(tmp_dir, output_dir_name),
                    push_to_hub=True,
                    hub_token=self._token,
                )
                url = trainer.push_to_hub()

            # Extract repo_name from the url
            re_search = re.search(ENDPOINT_STAGING + r"/([^/]+/[^/]+)/", url)
            self.assertTrue(re_search is not None)
            repo_name = re_search.groups()[0]

            self.assertEqual(repo_name, f"{USER}/{output_dir_name}")

            model = RegressionPreTrainedModel.from_pretrained(repo_name)
            self.assertEqual(model.a.item(), trainer.model.a.item())
            self.assertEqual(model.b.item(), trainer.model.b.item())

    def test_push_to_hub_in_organization(self):
        with TemporaryHubRepo(namespace="valid_org", token=self._token) as tmp_repo:
            with tempfile.TemporaryDirectory() as tmp_dir:
                trainer = get_regression_trainer(output_dir=tmp_dir)
                trainer.save_model()
                output_dir_name = tmp_repo.repo_name
                trainer = get_regression_trainer(
                    output_dir=os.path.join(tmp_dir, output_dir_name),
                    push_to_hub=True,
                    hub_model_id=f"valid_org/{output_dir_name}",
                    hub_token=self._token,
                )
                url = trainer.push_to_hub()

            # Extract repo_name from the url
            re_search = re.search(ENDPOINT_STAGING + r"/([^/]+/[^/]+)/", url)
            self.assertTrue(re_search is not None)
            repo_name = re_search.groups()[0]
            self.assertEqual(repo_name, f"valid_org/{output_dir_name}")

            model = RegressionPreTrainedModel.from_pretrained(f"valid_org/{output_dir_name}")
            self.assertEqual(model.a.item(), trainer.model.a.item())
            self.assertEqual(model.b.item(), trainer.model.b.item())

    def get_commit_history(self, repo):
        commit_logs = subprocess.run(
            ["git", "log"],
            capture_output=True,
            check=True,
            encoding="utf-8",
            cwd=repo,
        ).stdout
        commits = commit_logs.split("\n\n")[1::2]
        return [commit.strip() for commit in commits]

    # TODO: @ydshieh or @SunMarc
    @unittest.skip("unknown failure reason, possibly staging hub issue")
    def test_push_to_hub_with_saves_each_epoch(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            with tempfile.TemporaryDirectory() as tmp_dir:
                with self.assertLogs(level="WARNING") as logs:
                    output_dir_name = tmp_repo.repo_name
                    trainer = get_regression_trainer(
                        output_dir=os.path.join(tmp_dir, output_dir_name),
                        push_to_hub=True,
                        hub_token=self._token,
                        # To avoid any flakiness if the training goes faster than the uploads.
                        hub_always_push=True,
                        save_strategy="epoch",
                    )
                    trainer.train()

            commits = list_repo_commits(f"{USER}/{output_dir_name}", token=self._token)
            commits = [c.title for c in commits]
            self.assertIn("initial commit", commits)
            self.assertIn("Training in progress, epoch 1", commits)
            self.assertIn("Training in progress, epoch 2", commits)
            # Epochs 3 and 4 are not guaranteed to be present (empty commits)
            self.assertTrue(any("Skipping to prevent empty commit." in record.message for record in logs.records))

    def test_push_to_hub_with_saves_each_n_steps(self):
        num_gpus = max(1, backend_device_count(torch_device))
        if num_gpus > 2:
            self.skipTest(reason="More than 2 GPUs available")

        with TemporaryHubRepo(token=self._token) as tmp_repo:
            with tempfile.TemporaryDirectory() as tmp_dir:
                with self.assertLogs(level="WARNING") as logs:
                    output_dir_name = tmp_repo.repo_name
                    trainer = get_regression_trainer(
                        output_dir=os.path.join(tmp_dir, output_dir_name),
                        push_to_hub=True,
                        hub_token=self._token,
                        # To avoid any flakiness if the training goes faster than the uploads.
                        hub_always_push=True,
                        save_strategy="steps",
                        save_steps=5,
                    )
                    trainer.train()

            commits = list_repo_commits(f"{USER}/{output_dir_name}", token=self._token)
            commits = [c.title for c in commits]
            self.assertIn("initial commit", commits)

            # Some commits are skipped if nothing has changed
            # We expect 1 commit per 5 epochs + 1 commit at the end
            nb_empty_commits = len(
                [record for record in logs.records if "Skipping to prevent empty commit." in record.message]
            )
            nb_epoch_commits = len([commit for commit in commits if "Training in progress, step" in commit])

            # max_steps depend on the number of available GPUs
            max_steps = math.ceil(trainer.args.num_train_epochs * len(trainer.get_train_dataloader()))
            nb_expected_commits = len(range(5, max_steps, 5))

            # '>=' since final commit might be an empty commit as well (not deterministic)
            self.assertGreaterEqual(nb_empty_commits + nb_epoch_commits, nb_expected_commits)

    @require_tensorboard
    def test_push_to_hub_with_tensorboard_logs(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            with tempfile.TemporaryDirectory() as tmp_dir:
                output_dir_name = tmp_repo.repo_name
                trainer = get_regression_trainer(
                    output_dir=os.path.join(tmp_dir, output_dir_name),
                    hub_token=self._token,
                    save_strategy="epoch",
                    report_to=["tensorboard"],
                )
                trainer.train()
                # Push the runs via `push_to_hub()`
                trainer.push_to_hub()

            files = list_repo_files(f"{USER}/{output_dir_name}", token=self._token)
            found_log = False
            for f in files:
                if len(f.split("runs")) > 1 and "events.out.tfevents" in f:
                    found_log = True

            assert found_log is True, "No tensorboard log found in repo"

    def test_push_to_hub_tags(self):
        # Checks if `trainer.push_to_hub()` works correctly by adding the desired
        # tag without having to pass `tags` in `push_to_hub`
        # see:
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            with tempfile.TemporaryDirectory() as tmp_dir:
                output_dir_name = tmp_repo.repo_name
                trainer = get_regression_trainer(
                    output_dir=os.path.join(tmp_dir, output_dir_name),
                    push_to_hub=True,
                    hub_token=self._token,
                )

                trainer.model.add_model_tags(["test-trainer-tags"])

                url = trainer.push_to_hub()

            # Extract repo_name from the url
            re_search = re.search(ENDPOINT_STAGING + r"/([^/]+/[^/]+)/", url)
            self.assertTrue(re_search is not None)
            repo_name = re_search.groups()[0]

            self.assertEqual(repo_name, f"{USER}/{output_dir_name}")

            model_card = ModelCard.load(repo_name)
            self.assertTrue("test-trainer-tags" in model_card.data.tags)

    def test_push_to_hub_with_revision(self):
        # Checks if `trainer.push_to_hub()` works correctly by adding revision
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            with tempfile.TemporaryDirectory() as tmp_dir:
                output_dir_name = tmp_repo.repo_name
                trainer = get_regression_trainer(
                    output_dir=os.path.join(tmp_dir, output_dir_name),
                    push_to_hub=True,
                    hub_token=self._token,
                )
                branch = "v1.0"
                create_branch(repo_id=trainer.hub_model_id, branch=branch, token=self._token, exist_ok=True)
                push_commit = trainer.push_to_hub(revision=branch)

            commits = list_repo_commits(repo_id=trainer.hub_model_id, revision=branch, token=self._token)
            self.assertEqual(commits[0].commit_id, push_commit.oid)
