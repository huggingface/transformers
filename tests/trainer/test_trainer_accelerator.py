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
Trainer AcceleratorConfig tests: creation from dict/YAML/dataclass, partial overrides,
gradient accumulation settings, custom AcceleratorState, and validation.
"""

import dataclasses
import json
import tempfile
from pathlib import Path
from typing import Any

from accelerate import Accelerator
from accelerate.state import AcceleratorState

from transformers import Trainer, TrainingArguments
from transformers.testing_utils import TestCasePlus, require_torch
from transformers.trainer_pt_utils import AcceleratorConfig

from .trainer_test_utils import (
    RegressionModelConfig,
    RegressionPreTrainedModel,
    RegressionTrainingArguments,
    SampleIterableDataset,
)


@require_torch
class TrainerAcceleratorConfigTest(TestCasePlus):
    def test_accelerator_config_empty(self):
        # Checks that a config can be made with the defaults if not passed
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = RegressionModelConfig(a=1.5, b=2.5)
            model = RegressionPreTrainedModel(config)
            eval_dataset = SampleIterableDataset()

            # Leaves one option as something *not* basic
            args = RegressionTrainingArguments(output_dir=tmp_dir)
            trainer = Trainer(model=model, args=args, eval_dataset=eval_dataset)
            self.assertEqual(trainer.accelerator.split_batches, False)
            self.assertEqual(trainer.accelerator.dispatch_batches, None)
            self.assertEqual(trainer.accelerator.even_batches, True)
            self.assertEqual(trainer.accelerator.use_seedable_sampler, True)
            # gradient accumulation kwargs configures gradient_state
            self.assertNotIn("sync_each_batch", trainer.accelerator.gradient_state.plugin_kwargs)

    def test_accelerator_config_from_dict(self):
        # Checks that accelerator kwargs can be passed through
        # and the accelerator is initialized respectively
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = RegressionModelConfig(a=1.5, b=2.5)
            model = RegressionPreTrainedModel(config)
            eval_dataset = SampleIterableDataset()

            accelerator_config: dict[str, Any] = {
                "split_batches": True,
                "dispatch_batches": True,
                "even_batches": False,
                "use_seedable_sampler": True,
            }
            accelerator_config["gradient_accumulation_kwargs"] = {"sync_each_batch": True}

            # Leaves all options as something *not* basic
            args = RegressionTrainingArguments(output_dir=tmp_dir, accelerator_config=accelerator_config)
            trainer = Trainer(model=model, args=args, eval_dataset=eval_dataset)
            self.assertEqual(trainer.accelerator.split_batches, True)
            self.assertEqual(trainer.accelerator.dispatch_batches, True)
            self.assertEqual(trainer.accelerator.even_batches, False)
            self.assertEqual(trainer.accelerator.use_seedable_sampler, True)

    def test_accelerator_config_from_yaml(self):
        # Checks that accelerator kwargs can be passed through
        # and the accelerator is initialized respectively
        with tempfile.TemporaryDirectory() as tmp_dir:
            path_file = Path(tmp_dir) / "accelerator_config.json"
            with open(path_file, "w") as f:
                accelerator_config = {
                    "split_batches": True,
                    "dispatch_batches": True,
                    "even_batches": False,
                    "use_seedable_sampler": False,
                }
                json.dump(accelerator_config, f)
            config = RegressionModelConfig(a=1.5, b=2.5)
            model = RegressionPreTrainedModel(config)
            eval_dataset = SampleIterableDataset()

            # Leaves all options as something *not* basic
            args = RegressionTrainingArguments(output_dir=tmp_dir, accelerator_config=path_file)
            trainer = Trainer(model=model, args=args, eval_dataset=eval_dataset)
            self.assertEqual(trainer.accelerator.split_batches, True)
            self.assertEqual(trainer.accelerator.dispatch_batches, True)
            self.assertEqual(trainer.accelerator.even_batches, False)
            self.assertEqual(trainer.accelerator.use_seedable_sampler, False)

    def test_accelerator_config_from_dataclass(self):
        # Checks that accelerator kwargs can be passed through
        # and the accelerator is initialized respectively

        accelerator_config = AcceleratorConfig(
            split_batches=True,
            dispatch_batches=True,
            even_batches=False,
            use_seedable_sampler=False,
        )
        config = RegressionModelConfig(a=1.5, b=2.5)
        model = RegressionPreTrainedModel(config)
        eval_dataset = SampleIterableDataset()
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = RegressionTrainingArguments(output_dir=tmp_dir, accelerator_config=accelerator_config)
            trainer = Trainer(model=model, args=args, eval_dataset=eval_dataset)
            self.assertEqual(trainer.accelerator.split_batches, True)
            self.assertEqual(trainer.accelerator.dispatch_batches, True)
            self.assertEqual(trainer.accelerator.even_batches, False)
            self.assertEqual(trainer.accelerator.use_seedable_sampler, False)

    def test_accelerate_config_from_dataclass_grad_accum(self):
        # Checks that accelerator kwargs can be passed through
        # and the accelerator is initialized respectively

        grad_acc_kwargs = {
            "num_steps": 10,
            "adjust_scheduler": False,
            "sync_with_dataloader": False,
            "sync_each_batch": True,
        }
        accelerator_config = AcceleratorConfig(
            split_batches=True,
            dispatch_batches=True,
            even_batches=False,
            use_seedable_sampler=False,
            gradient_accumulation_kwargs=grad_acc_kwargs,
        )
        config = RegressionModelConfig(a=1.5, b=2.5)
        model = RegressionPreTrainedModel(config)
        eval_dataset = SampleIterableDataset()
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = RegressionTrainingArguments(output_dir=tmp_dir, accelerator_config=accelerator_config)
            trainer = Trainer(model=model, args=args, eval_dataset=eval_dataset)
            self.assertEqual(trainer.args.gradient_accumulation_steps, 10)

    def test_accelerator_config_from_partial(self):
        # Checks that accelerator kwargs can be passed through
        # and the accelerator is initialized respectively
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = RegressionModelConfig(a=1.5, b=2.5)
            model = RegressionPreTrainedModel(config)
            eval_dataset = SampleIterableDataset()

            # Leaves one option as something *not* basic
            args = RegressionTrainingArguments(
                output_dir=tmp_dir,
                accelerator_config={
                    "split_batches": True,
                },
            )
            trainer = Trainer(model=model, args=args, eval_dataset=eval_dataset)
            self.assertEqual(trainer.accelerator.split_batches, True)
            self.assertEqual(trainer.accelerator.dispatch_batches, None)
            self.assertEqual(trainer.accelerator.even_batches, True)
            self.assertEqual(trainer.accelerator.use_seedable_sampler, True)

    def test_accelerator_custom_state(self):
        AcceleratorState._reset_state(reset_partial_state=True)
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError) as cm:
                _ = RegressionTrainingArguments(output_dir=tmp_dir, accelerator_config={"use_configured_state": True})
                self.assertIn("Please define this beforehand", str(cm.warnings[0].message))
            _ = Accelerator()
            _ = RegressionTrainingArguments(output_dir=tmp_dir, accelerator_config={"use_configured_state": True})
        AcceleratorState._reset_state(reset_partial_state=True)

    def test_accelerator_config_from_dict_grad_accum_num_steps(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = RegressionModelConfig(a=1.5, b=2.5)
            model = RegressionPreTrainedModel(config)
            eval_dataset = SampleIterableDataset()

            # case - TrainingArguments.gradient_accumulation_steps == 1
            #      - gradient_accumulation_kwargs['num_steps] == 1
            # results in grad accum set to 1
            args = RegressionTrainingArguments(
                output_dir=tmp_dir,
                gradient_accumulation_steps=1,
                accelerator_config={
                    "gradient_accumulation_kwargs": {
                        "num_steps": 1,
                    }
                },
            )
            trainer = Trainer(model=model, args=args, eval_dataset=eval_dataset)
            self.assertEqual(trainer.accelerator.gradient_state.plugin_kwargs["num_steps"], 1)

            # case - TrainingArguments.gradient_accumulation_steps > 1
            #      - gradient_accumulation_kwargs['num_steps] specified
            # results in exception raised
            args = RegressionTrainingArguments(
                output_dir=tmp_dir,
                gradient_accumulation_steps=2,
                accelerator_config={
                    "gradient_accumulation_kwargs": {
                        "num_steps": 10,
                    }
                },
            )
            with self.assertRaises(Exception) as context:
                trainer = Trainer(model=model, args=args, eval_dataset=eval_dataset)
            self.assertTrue("The `AcceleratorConfig`'s `num_steps` is set but" in str(context.exception))

    def test_accelerator_config_not_instantiated(self):
        # Checks that accelerator kwargs can be passed through
        # and the accelerator is initialized respectively
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(NotImplementedError) as context:
                _ = RegressionTrainingArguments(
                    output_dir=tmp_dir,
                    accelerator_config=AcceleratorConfig,
                )
            self.assertTrue("Tried passing in a callable to `accelerator_config`" in str(context.exception))

        # Now test with a custom subclass
        @dataclasses.dataclass
        class CustomAcceleratorConfig(AcceleratorConfig):
            pass

        @dataclasses.dataclass
        class CustomTrainingArguments(TrainingArguments):
            accelerator_config: dict = dataclasses.field(
                default=CustomAcceleratorConfig,
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(NotImplementedError) as context:
                _ = CustomTrainingArguments(
                    output_dir=tmp_dir,
                )
            self.assertTrue("Tried passing in a callable to `accelerator_config`" in str(context.exception))
