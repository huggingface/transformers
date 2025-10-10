# tests/trainer/test_emergency_ckpt.py
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

import os
import tempfile
import unittest

from transformers import Trainer, TrainerCallback, TrainingArguments, is_torch_available
from transformers.testing_utils import TestCasePlus, require_torch

from .test_trainer import RegressionDataset, RegressionModel  # re-use fixtures


# if require_torch.is_torch_available():
if is_torch_available():
    pass


class _CrashAtStep(TrainerCallback):
    """Raise an exception at a specific global_step to simulate a crash."""

    def __init__(self, step: int = 5):
        self.step = step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.step:
            raise RuntimeError(f"💥 simulated failure at step {self.step}")


@require_torch
class EmergencyCheckpointTest(TestCasePlus):
    """
    Minimal tests for `enable_emergency_checkpoint`.

    We intentionally keep them CPU-only + tiny so they run in <10 s on CI.
    """

    def _build_trainer(
        self,
        tmp_dir: str,
        enable_flag: bool = True,
        callbacks=None,
        save_strategy="steps",
        save_steps=3,
    ):
        train_ds = RegressionDataset(length=32)
        model = RegressionModel()

        args = TrainingArguments(
            tmp_dir,
            num_train_epochs=1,
            per_device_train_batch_size=4,
            save_strategy=save_strategy,
            save_steps=save_steps,
            enable_emergency_checkpoint=enable_flag,
            report_to="none",
            logging_steps=1,
        )
        return Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            callbacks=callbacks or [],
        )

    # --------------------------------------------------------------------- #
    # 1) argument plumbing
    # --------------------------------------------------------------------- #
    def test_flag_roundtrip(self):
        args = TrainingArguments(
            output_dir="dummy",
            enable_emergency_checkpoint=True,
            report_to="none",
        )
        self.assertTrue(args.enable_emergency_checkpoint)

    # --------------------------------------------------------------------- #
    # 2) checkpoint is written and resume works
    # --------------------------------------------------------------------- #
    def test_emergency_ckpt_steps(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # first run – crash
            trainer = self._build_trainer(
                tmp_dir,
                enable_flag=True,
                callbacks=[_CrashAtStep(step=4)],
                save_strategy="steps",
                save_steps=2,
            )
            with self.assertRaises(RuntimeError):
                trainer.train()

            ckpt_dir = os.path.join(tmp_dir, "checkpoint-emergency")
            self.assertTrue(
                os.path.isdir(ckpt_dir),
                "Emergency checkpoint directory was not created!",
            )

            # second run – resume
            resume_args = TrainingArguments(
                tmp_dir,
                num_train_epochs=1,
                per_device_train_batch_size=4,
                report_to="none",
                enable_emergency_checkpoint=True,
                resume_from_checkpoint=ckpt_dir,
            )
            resume_trainer = Trainer(
                model=RegressionModel(),
                args=resume_args,
                train_dataset=RegressionDataset(length=32),
            )
            # should finish without raising
            resume_trainer.train()

    # --------------------------------------------------------------------- #
    # 3) opt-out should not create a folder
    # --------------------------------------------------------------------- #
    def test_no_ckpt_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = self._build_trainer(
                tmp_dir,
                enable_flag=False,
                callbacks=[_CrashAtStep(step=3)],
                save_strategy="steps",
                save_steps=1,
            )
            with self.assertRaises(RuntimeError):
                trainer.train()

            ckpt_dir = os.path.join(tmp_dir, "checkpoint-emergency")
            self.assertFalse(
                os.path.exists(ckpt_dir),
                "Checkpoint directory exists even though the feature was disabled!",
            )


if __name__ == "__main__":
    unittest.main()
