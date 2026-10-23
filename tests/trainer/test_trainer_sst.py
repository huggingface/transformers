# Copyright 2026 the HuggingFace Inc. team. All rights reserved.
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

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch
    from torch import nn
    from torch.utils.data import Dataset

    from transformers import SST, Trainer, TrainingArguments

    class _TrainingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(6, 8)
            self.output = nn.Linear(8, 1)

        def forward(self, input_ids, labels=None):
            logits = self.output(torch.tanh(self.q_proj(input_ids.float()))).squeeze(-1)
            loss = None if labels is None else (logits - labels.float()).square().mean()
            return {"loss": loss, "logits": logits}

    class _Dataset(Dataset):
        def __init__(self):
            generator = torch.Generator().manual_seed(123)
            self.inputs = torch.randn(16, 6, generator=generator)
            self.labels = torch.randn(16, generator=generator)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            return {"input_ids": self.inputs[index], "labels": self.labels[index]}


@require_torch
class TrainerSSTTest(unittest.TestCase):
    def trainer(self, output_dir, optim="adamw_torch", refresh_every=None, **kwargs):
        torch.manual_seed(42)
        args = TrainingArguments(
            output_dir=output_dir,
            max_steps=4,
            learning_rate=0.01,
            optim=optim,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            save_steps=2,
            logging_strategy="no",
            report_to="none",
            disable_tqdm=True,
            dataloader_pin_memory=False,
            seed=42,
        )
        return Trainer(
            model=_TrainingModel(),
            args=args,
            train_dataset=_Dataset(),
            eval_dataset=_Dataset(),
            optimizer_wrapper=(
                SST,
                {"rank_fraction": 0.5, "target_modules": ["q_proj"], "refresh_every": refresh_every},
            ),
            **kwargs,
        )

    def assert_protected(self, trainer):
        item = trainer.optimizer.sst_state_dict()["protected_weights"][0]
        matrix = trainer.model.q_proj.weight.detach().cpu()
        left, right = item["left"], item["right"]
        delta = matrix - (left * item["singular_values"].unsqueeze(0)) @ right.mH
        self.assertLess((left.mH @ delta).abs().max().item(), 3e-6)
        self.assertLess((delta @ right).abs().max().item(), 3e-6)

    @parameterized.expand(["sgd", "adamw_torch", "adafactor"])
    def test_wraps_selected_optimizer_after_accelerate_and_trains(self, optim):
        expected = {"sgd": torch.optim.SGD, "adamw_torch": torch.optim.AdamW, "adafactor": "Adafactor"}
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.trainer(directory, optim=optim)
            trainer.train()
            self.assertIsInstance(trainer.optimizer, SST)
            inner = trainer.optimizer.optimizer.optimizer
            if isinstance(expected[optim], str):
                self.assertEqual(type(inner).__name__, expected[optim])
            else:
                self.assertIsInstance(inner, expected[optim])
            self.assertEqual(trainer.state.global_step, 4)
            self.assert_protected(trainer)
            state = torch.load(
                Path(directory) / "checkpoint-2" / "optimizer.pt", map_location="cpu", weights_only=True
            )
            self.assertIn("sst", state)
            self.assertFalse((Path(directory) / "checkpoint-2" / "sst_state.pt").exists())
            self.assertGreaterEqual(trainer.evaluate()["eval_loss"], 0.0)

    @parameterized.expand(
        [
            ("SGD", {"momentum": 0.9}),
            ("AdamW", {"betas": (0.8, 0.95)}),
            ("RMSprop", {"momentum": 0.9}),
        ]
    )
    def test_custom_optimizer_class_and_kwargs(self, name, optimizer_kwargs):
        optimizer_cls = getattr(torch.optim, name)
        optimizer_kwargs = {"lr": 0.01, **optimizer_kwargs}
        original_kwargs = copy.deepcopy(optimizer_kwargs)
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.trainer(directory, optimizer_cls_and_kwargs=(optimizer_cls, optimizer_kwargs))
            trainer.train()
            inner = trainer.optimizer.optimizer.optimizer
            self.assertIsInstance(inner, optimizer_cls)
            self.assertEqual(optimizer_kwargs, original_kwargs)
            for key, value in original_kwargs.items():
                self.assertEqual(inner.param_groups[0]["initial_lr" if key == "lr" else key], value)
            self.assert_protected(trainer)

    def test_checkpoint_resume_preserves_basis_without_svd(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.trainer(directory)
            trainer.train()
            final_weights = copy.deepcopy(trainer.model.state_dict())
            original_basis = copy.deepcopy(trainer.optimizer.sst_state_dict())
            resumed = self.trainer(str(Path(directory) / "resumed"))
            with patch("torch.linalg.svd", side_effect=AssertionError("Resume must restore the original SST basis")):
                resumed.train(resume_from_checkpoint=str(Path(directory) / "checkpoint-2"))
            self.assertEqual(resumed.state.global_step, 4)
            for name, value in final_weights.items():
                torch.testing.assert_close(value, resumed.model.state_dict()[name], rtol=0, atol=0)
            for first, second in zip(
                original_basis["protected_weights"], resumed.optimizer.sst_state_dict()["protected_weights"]
            ):
                for key in ("left", "singular_values", "right"):
                    torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
            self.assert_protected(resumed)

    @parameterized.expand([1, 3])
    def test_periodic_refresh_and_resume_crossing_a_refresh_boundary(self, refresh_every):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.trainer(directory, refresh_every=refresh_every)
            trainer.train()
            final_weights = copy.deepcopy(trainer.model.state_dict())
            final_state = copy.deepcopy(trainer.optimizer.sst_state_dict())
            self.assertEqual(trainer.optimizer.num_updates, 4)
            self.assertEqual(trainer.optimizer.num_refreshes, 4 // refresh_every)
            self.assert_protected(trainer)
            resumed = self.trainer(str(Path(directory) / "resumed"), refresh_every=10)
            with patch.object(
                SST, "initialize", side_effect=AssertionError("Resume must load, not initialize, a basis")
            ):
                resumed.train(resume_from_checkpoint=str(Path(directory) / "checkpoint-2"))
            self.assertEqual(resumed.optimizer.refresh_every, refresh_every)
            for name, value in final_weights.items():
                torch.testing.assert_close(value, resumed.model.state_dict()[name], rtol=0, atol=0)
            for key in ("num_updates", "num_refreshes", "last_refresh_step", "refresh_every"):
                self.assertEqual(final_state[key], resumed.optimizer.sst_state_dict()[key])
            for first, second in zip(
                final_state["protected_weights"], resumed.optimizer.sst_state_dict()["protected_weights"]
            ):
                for key in ("left", "singular_values", "right"):
                    torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
            self.assert_protected(resumed)

    def test_repeated_train_does_not_double_wrap(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.trainer(directory)
            trainer.train()
            optimizer = trainer.optimizer
            trainer.train()
            self.assertIs(trainer.optimizer, optimizer)
            self.assertNotIsInstance(trainer.optimizer.optimizer, SST)
            self.assert_protected(trainer)

    def test_prebuilt_optimizer_can_be_wrapped(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(42)
            model = _TrainingModel()
            inner = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            trainer = Trainer(
                model,
                TrainingArguments(
                    directory,
                    max_steps=2,
                    report_to="none",
                    save_strategy="no",
                    disable_tqdm=True,
                    dataloader_pin_memory=False,
                ),
                train_dataset=_Dataset(),
                optimizers=(inner, None),
                optimizer_wrapper=(SST, {"rank_fraction": 0.5, "target_modules": ["q_proj"]}),
            )
            trainer.train()
            self.assertIs(trainer.optimizer.optimizer.optimizer, inner)
            self.assert_protected(trainer)

    def test_invalid_optimizer_wrapper_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for wrapper in (SST, (SST,), (str, {}), (SST, None), (SST, {"model": _TrainingModel()})):
                with self.subTest(wrapper=wrapper), self.assertRaises((TypeError, ValueError)):
                    Trainer(
                        _TrainingModel(),
                        TrainingArguments(directory, report_to="none"),
                        optimizer_wrapper=wrapper,
                    )
