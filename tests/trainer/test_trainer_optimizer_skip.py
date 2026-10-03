# Copyright 2026 The HuggingFace Team. All rights reserved.
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

"""Regression tests for optimizer updates skipped by AMP gradient scaling."""

import json

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import IntervalStrategy
from transformers.utils import SAFE_WEIGHTS_NAME, logging


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, length=12, poison_id=4, poison_all=False):
        self.length = length
        self.poison_id = poison_id
        self.poison_all = poison_all

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        value = torch.tensor([index / 10], dtype=torch.float32)
        return {
            "input_ids": value,
            "labels": value / 2,
            "sample_id": index,
            "poison": self.poison_all or index == self.poison_id,
        }


class _Stream(torch.utils.data.IterableDataset):
    def __init__(self, length=20, poison_id=4, one_shot=False):
        self.dataset = _Dataset(length, poison_id=None)
        self.poison_ids = poison_id if isinstance(poison_id, tuple) else (poison_id,)
        self.iterator = self._samples() if one_shot else None

    def _samples(self):
        for index in range(len(self.dataset)):
            sample = self.dataset[index]
            sample["poison"] = index in self.poison_ids
            yield sample

    def __iter__(self):
        if self.iterator is not None:
            return self.iterator
        return self._samples()


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        self.linear.weight.register_hook(self._poison_gradient)
        self.poison_next_gradient = False
        self.seen_ids = []
        self.losses = []

    def _poison_gradient(self, gradient):
        if self.poison_next_gradient:
            gradient = gradient.clone()
            gradient.view(-1)[0] = float("inf")
        return gradient

    def forward(self, input_ids, labels, sample_id, poison):
        if self.training:
            self.seen_ids.extend(sample_id.tolist())
        self.poison_next_gradient = bool(poison.any())
        loss = nn.functional.mse_loss(self.linear(input_ids).float(), labels)
        if self.training:
            self.losses.append(loss.item())
        return {"loss": loss}


class _Recorder(TrainerCallback):
    def __init__(self):
        self.attempts = 0
        self.optimizer_steps = 0
        self.completed_steps = []
        self.logged_steps = []
        self.saved_steps = []
        self.evaluated_steps = []

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        self.attempts += 1

    def on_optimizer_step(self, args, state, control, **kwargs):
        self.optimizer_steps += 1

    def on_step_end(self, args, state, control, **kwargs):
        self.completed_steps.append(state.global_step)

    def on_log(self, args, state, control, logs, **kwargs):
        if "loss" in logs:
            self.logged_steps.append(state.global_step)

    def on_save(self, args, state, control, **kwargs):
        self.saved_steps.append(state.global_step)

    def on_evaluate(self, args, state, control, **kwargs):
        self.evaluated_steps.append(state.global_step)


class _RecordingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.skip_flags = []
        super().__init__(*args, **kwargs)

    def _get_learning_rate(self):
        self.skip_flags.append(self.accelerator.optimizer_step_was_skipped)
        return super()._get_learning_rate()


def _make_trainer(
    output_dir,
    *,
    length=12,
    poison_id=4,
    poison_all=False,
    accumulation=1,
    max_steps=6,
    save_steps=500,
    epochs=3,
    save_strategy=None,
    train_dataset=None,
    dispatch_batches=None,
    fp16=True,
):
    torch.manual_seed(0)
    model = _Model()
    recorder = _Recorder()
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=2 if accumulation == 1 else 1,
        gradient_accumulation_steps=accumulation,
        max_steps=max_steps,
        num_train_epochs=epochs,
        fp16=fp16,
        accelerator_config={"dispatch_batches": dispatch_batches},
        use_cpu=True,
        optim="adamw_torch",
        learning_rate=1e-2,
        train_sampling_strategy="sequential",
        save_strategy=save_strategy or ("steps" if save_steps < max_steps else "no"),
        save_steps=save_steps,
        logging_steps=1,
        disable_tqdm=True,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = _RecordingTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset if train_dataset is not None else _Dataset(length, poison_id, poison_all),
        callbacks=[recorder],
    )
    if fp16:
        trainer.accelerator.scaler = torch.amp.GradScaler("cpu", init_scale=16.0)
        trainer.accelerator.native_amp = True
    return trainer, recorder


def _applied_steps(trainer):
    return max(int(state["step"]) for state in trainer.optimizer.optimizer.state.values())


def test_skipped_update_does_not_count_toward_max_steps(tmp_path):
    trainer, recorder = _make_trainer(tmp_path)
    output = trainer.train()

    assert _applied_steps(trainer) == 6
    assert trainer.lr_scheduler.last_epoch == 6
    assert trainer.state.global_step == 6
    assert trainer.state.num_train_epochs == 2
    assert trainer.state.optimizer_step_attempts == 7
    assert recorder.attempts == 7
    assert recorder.optimizer_steps == 6
    assert recorder.completed_steps == list(range(1, 7))
    assert recorder.logged_steps == list(range(1, 7))
    assert trainer.skip_flags.count(True) == 1
    assert len(trainer.model.seen_ids) == 14
    assert output.training_loss == pytest.approx(sum(trainer.model.losses) / 7)
    assert output.metrics["train_samples_per_second"] * output.metrics["train_runtime"] == pytest.approx(14, abs=0.5)


def test_resume_after_skipped_update_uses_consumed_batch_position(tmp_path):
    baseline, _ = _make_trainer(tmp_path / "baseline", save_steps=3)
    baseline.train()
    checkpoint = tmp_path / "baseline" / "checkpoint-3"
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    assert state["optimizer_step_attempts"] == 4

    resumed, _ = _make_trainer(tmp_path / "resumed", save_steps=3)
    resumed_output = resumed.train(resume_from_checkpoint=str(checkpoint))

    assert resumed.model.seen_ids == baseline.model.seen_ids[8:]
    assert resumed.state.global_step == baseline.state.global_step == 6
    assert resumed.state.optimizer_step_attempts == baseline.state.optimizer_step_attempts == 7
    assert _applied_steps(resumed) == _applied_steps(baseline) == 6
    assert resumed.lr_scheduler.last_epoch == baseline.lr_scheduler.last_epoch == 6
    torch.testing.assert_close(resumed.model.linear.weight, baseline.model.linear.weight, rtol=0, atol=0)
    assert resumed_output.training_loss == pytest.approx(sum(resumed.model.losses) / 3)

    finished, _ = _make_trainer(tmp_path / "finished", save_steps=3)
    finished.train(resume_from_checkpoint=str(tmp_path / "baseline" / "checkpoint-6"))
    assert finished.state.global_step == 6
    assert finished.model.seen_ids == []


def test_resume_old_checkpoint_without_attempt_counter(tmp_path):
    baseline, _ = _make_trainer(tmp_path / "baseline", poison_id=None, save_steps=3)
    baseline.train()
    checkpoint = tmp_path / "baseline" / "checkpoint-3"
    state_path = checkpoint / "trainer_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    for field in ("optimizer_step_attempts", "train_dataloader_epoch", "train_dataloader_batches_seen"):
        del state[field]
    state_path.write_text(json.dumps(state), encoding="utf-8")

    resumed, _ = _make_trainer(tmp_path / "resumed", poison_id=None, save_steps=3)
    resumed.train(resume_from_checkpoint=str(checkpoint))

    assert resumed.state.optimizer_step_attempts == resumed.state.global_step == 6
    assert resumed.model.seen_ids == baseline.model.seen_ids[6:]
    torch.testing.assert_close(resumed.model.linear.weight, baseline.model.linear.weight, rtol=0, atol=0)


def test_resume_after_skip_with_partial_accumulation_group(tmp_path):
    settings = {"length": 7, "poison_id": 2, "accumulation": 2, "max_steps": 4, "save_steps": 2}
    baseline, _ = _make_trainer(tmp_path / "baseline", **settings)
    baseline.train()
    checkpoint = tmp_path / "baseline" / "checkpoint-2"
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    assert state["optimizer_step_attempts"] == 3

    resumed, _ = _make_trainer(tmp_path / "resumed", **settings)
    resumed.train(resume_from_checkpoint=str(checkpoint))

    assert resumed.model.seen_ids == baseline.model.seen_ids[6:]
    assert resumed.state.optimizer_step_attempts == baseline.state.optimizer_step_attempts == 5
    assert resumed.state.global_step == _applied_steps(resumed) == 4
    assert resumed.lr_scheduler.last_epoch == baseline.lr_scheduler.last_epoch == 4
    torch.testing.assert_close(resumed.model.linear.weight, baseline.model.linear.weight, rtol=0, atol=0)


@pytest.mark.parametrize("save_strategy", ["steps", "epoch"])
@pytest.mark.parametrize("poison_id", [None, 4])
def test_resume_at_epoch_boundary(tmp_path, save_strategy, poison_id):
    def random_gradient(gradient):
        return gradient * (0.5 + torch.rand((), device=gradient.device))

    boundary_step = 6 - int(poison_id is not None)
    settings = {
        "max_steps": boundary_step + 1,
        "save_strategy": save_strategy,
        "save_steps": boundary_step,
        "poison_id": poison_id,
    }
    baseline, _ = _make_trainer(tmp_path / "baseline", **settings)
    baseline.model.linear.weight.register_hook(random_gradient)
    baseline.train()
    checkpoint = tmp_path / "baseline" / f"checkpoint-{boundary_step}"
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    assert state["optimizer_step_attempts"] == 6

    resumed, _ = _make_trainer(tmp_path / "resumed", **settings)
    resumed.model.linear.weight.register_hook(random_gradient)
    resumed.train(resume_from_checkpoint=str(checkpoint))

    assert resumed.model.seen_ids == [0, 1]
    assert resumed.state.global_step == _applied_steps(resumed) == boundary_step + 1
    assert resumed.state.optimizer_step_attempts == 7
    torch.testing.assert_close(resumed.model.linear.weight, baseline.model.linear.weight, rtol=0, atol=0)


@pytest.mark.parametrize("poison_id, expected_attempts", [(2, 4), (None, 3)])
def test_skipped_update_with_partial_accumulation_group(tmp_path, poison_id, expected_attempts):
    trainer, recorder = _make_trainer(tmp_path, length=5, poison_id=poison_id, accumulation=2, max_steps=3)
    trainer.train()

    assert trainer.state.global_step == _applied_steps(trainer) == trainer.lr_scheduler.last_epoch == 3
    assert trainer.state.optimizer_step_attempts == recorder.attempts == expected_attempts
    assert recorder.optimizer_steps == 3
    assert len(trainer.model.seen_ids) == (7 if poison_id is not None else 5)


@pytest.mark.parametrize(
    "epochs, poison_id, attempts, applied_steps",
    [(1, None, 6, 6), (1, 4, 6, 5), (1, 10, 6, 5), (1.5, None, 9, 9), (1.5, 4, 9, 7), (1.5, 10, 9, 8)],
)
def test_epoch_based_training_stops_after_requested_epoch(tmp_path, epochs, poison_id, attempts, applied_steps):
    trainer, recorder = _make_trainer(
        tmp_path, max_steps=-1, epochs=epochs, poison_id=poison_id, save_strategy="steps", save_steps=2
    )
    trainer.args.eval_strategy = IntervalStrategy.STEPS
    trainer.args.eval_steps = 2
    trainer.eval_dataset = _Dataset(poison_id=None)
    trainer.train()

    assert trainer.state.global_step == _applied_steps(trainer) == applied_steps
    assert trainer.state.optimizer_step_attempts == recorder.attempts == attempts
    assert len(trainer.model.seen_ids) == 2 * attempts
    assert trainer.state.epoch == epochs
    assert trainer.control.should_training_stop
    expected_steps = sorted(set(range(2, applied_steps + 1, 2)) | {applied_steps})
    assert recorder.saved_steps == recorder.evaluated_steps == expected_steps
    checkpoint = tmp_path / f"checkpoint-{applied_steps}"
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    assert state["optimizer_step_attempts"] == attempts
    assert state["train_dataloader_epoch"] == (0 if epochs == 1 else 1)
    assert state["train_dataloader_batches_seen"] == (6 if epochs == 1 else 3)
    torch.testing.assert_close(
        load_file(str(checkpoint / SAFE_WEIGHTS_NAME))["linear.weight"],
        trainer.model.linear.weight.cpu(),
        rtol=0,
        atol=0,
    )
    finished, finished_recorder = _make_trainer(
        tmp_path / "finished", max_steps=-1, epochs=epochs, poison_id=poison_id, save_strategy="steps", save_steps=2
    )
    finished.train(resume_from_checkpoint=str(checkpoint))
    assert finished.model.seen_ids == finished_recorder.saved_steps == finished_recorder.evaluated_steps == []


def test_callback_can_stop_persistent_skips(tmp_path):
    class StopAfterThreeAttempts(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            if state.optimizer_step_attempts == 2:
                control.should_training_stop = True

    trainer, recorder = _make_trainer(tmp_path, poison_all=True)
    trainer.add_callback(StopAfterThreeAttempts())
    trainer.train()

    assert trainer.state.global_step == 0
    assert trainer.state.optimizer_step_attempts == recorder.attempts == 3


def test_persistent_skips_do_not_run_forever_with_max_steps(tmp_path):
    trainer, recorder = _make_trainer(tmp_path, poison_id=None)
    backward_calls = 0

    def poison_from_third_backward(gradient):
        nonlocal backward_calls
        backward_calls += 1
        if backward_calls >= 3:
            gradient = gradient.clone()
            gradient.view(-1)[0] = float("inf")
        return gradient

    trainer.model.linear.weight.register_hook(poison_from_third_backward)

    class StopRunawayTest(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            if state.optimizer_step_attempts >= 200:
                pytest.fail("Trainer did not stop after 200 optimizer attempts")

    trainer.add_callback(StopRunawayTest())
    with pytest.raises(RuntimeError, match="consecutive optimizer steps were skipped"):
        trainer.train()

    assert trainer.state.global_step == _applied_steps(trainer) == trainer.lr_scheduler.last_epoch == 2
    assert trainer.state.optimizer_step_attempts == recorder.attempts == backward_calls
    assert backward_calls < 200
    assert recorder.optimizer_steps == 2
    assert recorder.completed_steps == [1, 2]


@pytest.mark.parametrize("poison_id, expected_attempts", [(4, 6), (None, 5)])
def test_stream_continues_from_unread_samples(tmp_path, poison_id, expected_attempts):
    trainer, recorder = _make_trainer(tmp_path, max_steps=5, train_dataset=_Stream(poison_id=poison_id))
    trainer.train()

    assert trainer.model.seen_ids == list(range(2 * expected_attempts))
    assert trainer.state.global_step == _applied_steps(trainer) == trainer.lr_scheduler.last_epoch == 5
    assert trainer.state.optimizer_step_attempts == recorder.attempts == expected_attempts
    assert recorder.completed_steps == list(range(1, 6))


@pytest.mark.parametrize("one_shot", [False, True])
@pytest.mark.parametrize("dispatch_batches", [False, True])
def test_exhausted_stream_restarts_unless_one_shot(tmp_path, one_shot, dispatch_batches):
    trainer, recorder = _make_trainer(
        tmp_path,
        max_steps=5,
        train_dataset=_Stream(length=10, one_shot=one_shot),
        dispatch_batches=dispatch_batches,
    )
    with CaptureLogger(logging.get_logger("transformers.trainer")) as captured:
        trainer.train()

    assert trainer.model.seen_ids == list(range(10)) + ([] if one_shot else [0, 1])
    assert (
        trainer.state.global_step
        == _applied_steps(trainer)
        == trainer.lr_scheduler.last_epoch
        == (4 if one_shot else 5)
    )
    assert trainer.state.optimizer_step_attempts == recorder.attempts == (5 if one_shot else 6)
    if one_shot:
        assert "Training data exhausted at global_step=4 before reaching max_steps=5" in captured.out
    else:
        assert "Training data exhausted" not in captured.out


@pytest.mark.parametrize("dispatch_batches", [False, True])
def test_finite_stream_restarts_without_skipped_updates(tmp_path, dispatch_batches):
    trainer, recorder = _make_trainer(
        tmp_path,
        max_steps=5,
        train_dataset=_Stream(length=6, poison_id=None),
        dispatch_batches=dispatch_batches,
        fp16=False,
    )
    trainer.train()

    assert trainer.model.seen_ids == list(range(6)) + list(range(4))
    assert trainer.state.global_step == _applied_steps(trainer) == trainer.lr_scheduler.last_epoch == 5
    assert trainer.state.optimizer_step_attempts == recorder.attempts == 5


@pytest.mark.parametrize("dispatch_batches", [False, True])
@pytest.mark.parametrize("accumulation", [1, 2])
@pytest.mark.parametrize("save_steps", [2, 3])
def test_stream_resume_across_real_passes(tmp_path, dispatch_batches, accumulation, save_steps):
    settings = {
        "max_steps": 5,
        "save_steps": save_steps,
        "accumulation": accumulation,
        "dispatch_batches": dispatch_batches,
    }
    baseline, _ = _make_trainer(tmp_path / "baseline", train_dataset=_Stream(length=5, poison_id=2), **settings)
    baseline.train()
    checkpoint = tmp_path / "baseline" / f"checkpoint-{save_steps}"
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    assert state["train_dataloader_epoch"] == (0 if save_steps == 2 else 1)
    assert state["train_dataloader_batches_seen"] == (
        (3 if accumulation == 1 else 5) if save_steps == 2 else accumulation
    )

    resumed, _ = _make_trainer(tmp_path / "resumed", train_dataset=_Stream(length=5, poison_id=2), **settings)
    resumed.train(resume_from_checkpoint=str(checkpoint))

    assert resumed.model.seen_ids == baseline.model.seen_ids[5 if save_steps == 2 else 7 :]
    assert resumed.state.global_step == _applied_steps(resumed) == resumed.lr_scheduler.last_epoch == 5
    assert resumed.state.optimizer_step_attempts == baseline.state.optimizer_step_attempts == 7
    torch.testing.assert_close(resumed.model.linear.weight, baseline.model.linear.weight, rtol=0, atol=0)


@pytest.mark.parametrize("end_early", [False, True])
def test_stream_epoch_checkpoint_resumes_at_next_pass(tmp_path, end_early):
    class EndPass(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            control.should_epoch_stop = True

    def random_gradient(gradient):
        return gradient * (0.5 + torch.rand((), device=gradient.device))

    settings = {"max_steps": 5, "accumulation": 2, "save_strategy": "epoch"}
    baseline, _ = _make_trainer(tmp_path / "baseline", train_dataset=_Stream(length=5, poison_id=2), **settings)
    baseline.model.linear.weight.register_hook(random_gradient)
    if end_early:
        baseline.add_callback(EndPass())
    baseline.train()
    checkpoint = tmp_path / "baseline" / "checkpoint-2"
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    assert state["train_dataloader_epoch"] == (2 if end_early else 1)
    assert state["train_dataloader_batches_seen"] == 0

    resumed, _ = _make_trainer(tmp_path / "resumed", train_dataset=_Stream(length=5, poison_id=2), **settings)
    resumed.model.linear.weight.register_hook(random_gradient)
    if end_early:
        resumed.add_callback(EndPass())
    resumed.train(resume_from_checkpoint=str(checkpoint))

    assert resumed.model.seen_ids == baseline.model.seen_ids[4 if end_early else 5 :]
    torch.testing.assert_close(resumed.model.linear.weight, baseline.model.linear.weight, rtol=0, atol=0)


@pytest.mark.parametrize("dispatch_batches", [False, True])
def test_stream_dataset_errors_are_not_treated_as_exhaustion(tmp_path, dispatch_batches):
    class BrokenStream(torch.utils.data.IterableDataset):
        def __iter__(self):
            raise ValueError("dataset read failed")
            yield

    trainer, _ = _make_trainer(tmp_path, train_dataset=BrokenStream(), dispatch_batches=dispatch_batches)
    with pytest.raises(ValueError, match="dataset read failed"):
        trainer.train()


def test_stream_applies_partial_final_accumulation_group(tmp_path):
    trainer, recorder = _make_trainer(
        tmp_path, max_steps=3, accumulation=2, train_dataset=_Stream(length=7, poison_id=2)
    )
    trainer.train()

    assert trainer.model.seen_ids == list(range(7))
    assert trainer.state.global_step == _applied_steps(trainer) == trainer.lr_scheduler.last_epoch == 3
    assert trainer.state.optimizer_step_attempts == recorder.attempts == 4
    assert recorder.completed_steps == [1, 2, 3]


@pytest.mark.parametrize("accumulation", [1, 2])
def test_stream_resume_past_original_attempt_limit(tmp_path, accumulation):
    settings = {"max_steps": 5, "save_steps": 4, "accumulation": accumulation}
    baseline, _ = _make_trainer(
        tmp_path / "baseline", train_dataset=_Stream(length=40, poison_id=(4, 6, 8)), **settings
    )
    baseline.train()
    checkpoint = tmp_path / "baseline" / "checkpoint-4"
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    assert state["optimizer_step_attempts"] == 7

    resumed, _ = _make_trainer(tmp_path / "resumed", train_dataset=_Stream(length=40, poison_id=(4, 6, 8)), **settings)
    resumed.train(resume_from_checkpoint=str(checkpoint))

    assert resumed.model.seen_ids == baseline.model.seen_ids[14:]
    assert resumed.state.optimizer_step_attempts == baseline.state.optimizer_step_attempts == 8
    assert resumed.state.global_step == _applied_steps(resumed) == resumed.lr_scheduler.last_epoch == 5
    torch.testing.assert_close(resumed.model.linear.weight, baseline.model.linear.weight, rtol=0, atol=0)


def test_persistent_stream_skips_still_raise(tmp_path):
    trainer, _ = _make_trainer(tmp_path, train_dataset=_Stream(length=300, poison_id=tuple(range(4, 300))))
    with pytest.raises(RuntimeError, match="100 consecutive optimizer steps were skipped"):
        trainer.train()

    assert trainer.model.seen_ids == list(range(204))
    assert trainer.state.global_step == _applied_steps(trainer) == trainer.lr_scheduler.last_epoch == 2
    assert trainer.state.optimizer_step_attempts == 102
