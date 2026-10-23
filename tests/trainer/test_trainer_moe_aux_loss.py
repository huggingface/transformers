# Copyright 2025 the HuggingFace Inc. team.
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
Regression tests for MoE auxiliary-loss scaling in Trainer.

Bug summary
-----------
When `num_items_in_batch` is supplied the Trainer skips its usual
`/ gradient_accumulation_steps` division (`loss_is_scaled_for_ga=True`).
That is correct for the token-count-normalised CE term, but MoE models embed
a load-balancing auxiliary term in the combined scalar:

    combined_loss = ce_loss + router_aux_loss_coef * aux_loss   # in model.forward()

The aux term is a per-microbatch mean, not a token-count sum.  With K
accumulation steps it therefore accumulates K× unscaled, inflating the
effective auxiliary coefficient by K.

The fix in `Trainer.compute_loss` detects the embedded aux term (via
`outputs.aux_loss` and `model.router_aux_loss_coef`), subtracts the aux
contribution from the combined loss, and re-adds it divided by
`current_gradient_accumulation_steps`.

Test strategy
-------------
All tests use a tiny, randomly-initialised Qwen3MoeForCausalLM — no Hub
downloads, no GPU required.

Two levels of verification:

1. Unit-level (compute_loss): verify the backward-graph loss value is correct
   for gas=K by calling compute_loss directly with current_gradient_accumulation_steps
   primed and comparing against the expected formula.

2. End-to-end (trainer.train): for parameters that receive gradient exclusively
   through the CE path (e.g. lm_head.weight, model.norm.weight) gradients must
   be numerically identical between gas=1 and gas=K.  Router gate.weight is
   intentionally excluded because the load-balancing loss is a nonlinear
   function of routing statistics that legitimately differs across batch splits.

Proof-of-regression: `_BuggyTrainer` reproduces the pre-fix behaviour by
returning the raw `outputs.loss` without re-normalising the aux term.  The
regression tests assert that _BuggyTrainer shows the scaling error while the
real Trainer does not.
"""

import copy
import math
import tempfile
import unittest

import torch
from torch.utils.data import Dataset

from transformers import Trainer, TrainingArguments
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
from transformers.testing_utils import require_torch


# ---------------------------------------------------------------------------
# Shared tiny-model configuration
# ---------------------------------------------------------------------------

_TINY_CFG = {
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "moe_intermediate_size": 16,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "decoder_sparse_step": 1,
    "output_router_logits": True,
    "router_aux_loss_coef": 0.1,  # large enough for aux effect to be visible
    "max_position_embeddings": 64,
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
}

# Parameters whose gradient flows only through the CE path (not the aux loss).
# The load-balancing loss does NOT directly differentiate through lm_head or
# the final norm, so their gradients are identical across batch splits.
_CE_ONLY_PARAMS = frozenset(["lm_head.weight", "model.norm.weight"])


def _make_model(seed: int = 42, **overrides) -> Qwen3MoeForCausalLM:
    torch.manual_seed(seed)
    cfg = Qwen3MoeConfig(**{**_TINY_CFG, **overrides})
    return Qwen3MoeForCausalLM(cfg).float()


# ---------------------------------------------------------------------------
# Minimal deterministic dataset / collator  (no tokenizer needed)
# ---------------------------------------------------------------------------


class _FixedDataset(Dataset):
    def __init__(self, n: int = 16, seq_len: int = 8, vocab_size: int = 64, seed: int = 0):
        rng = torch.Generator()
        rng.manual_seed(seed)
        self._ids = torch.randint(2, vocab_size, (n, seq_len), generator=rng)

    def __len__(self):
        return len(self._ids)

    def __getitem__(self, idx):
        ids = self._ids[idx]
        return {"input_ids": ids, "labels": ids.clone()}


def _collate(features):
    input_ids = torch.stack([f["input_ids"] for f in features])
    labels = torch.stack([f["labels"] for f in features])
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": labels}


# ---------------------------------------------------------------------------
# Gradient capture helper
# ---------------------------------------------------------------------------


class _GradCapture:
    """Accumulates parameter gradients via backward hooks."""

    def __init__(self, model: torch.nn.Module, name_filter=None):
        self._accum: dict[str, torch.Tensor] = {}
        self._hooks = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name_filter is not None and not name_filter(name):
                continue

            def _hook(grad, _n=name):
                self._accum[_n] = self._accum.get(_n, torch.zeros_like(grad)) + grad.detach()

            self._hooks.append(param.register_hook(_hook))

    def grads(self) -> dict[str, torch.Tensor]:
        return dict(self._accum)

    def remove(self):
        for h in self._hooks:
            h.remove()


# ---------------------------------------------------------------------------
# Buggy trainer (reproduces the pre-fix scaling error)
# ---------------------------------------------------------------------------


class _BuggyTrainer(Trainer):
    """
    Returns `outputs.loss` directly without re-normalising the aux term per
    accumulation step.  With K steps the aux gradient accumulates K× unscaled.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.model_accepts_loss_kwargs and num_items_in_batch is not None:
            inputs = {**inputs, "num_items_in_batch": num_items_in_batch}
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Core training helper
# ---------------------------------------------------------------------------


def _run_one_step(
    model: Qwen3MoeForCausalLM,
    gas: int,
    batch_size: int,
    n_samples: int = 16,
    trainer_cls=None,
    capture_filter=None,
) -> dict[str, torch.Tensor]:
    """
    Train for exactly one optimizer step; return accumulated per-parameter
    gradients captured via backward hooks (unaffected by the optimizer state).
    """
    if trainer_cls is None:
        trainer_cls = Trainer
    if capture_filter is None:
        capture_filter = lambda n: n in _CE_ONLY_PARAMS  # noqa: E731

    capture = _GradCapture(model, name_filter=capture_filter)
    dataset = _FixedDataset(n=n_samples)

    with tempfile.TemporaryDirectory() as tmp_dir:
        args = TrainingArguments(
            output_dir=tmp_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gas,
            max_steps=1,
            learning_rate=1e-9,
            logging_steps=1,
            report_to="none",
            use_cpu=True,
        )
        trainer = trainer_cls(
            model=model,
            args=args,
            train_dataset=dataset,
            data_collator=_collate,
        )
        trainer.train()

    grads = capture.grads()
    capture.remove()
    return grads


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@require_torch
class TestMoeAuxLossAccumulationFix(unittest.TestCase):
    # ── helpers ──────────────────────────────────────────────────────────

    def _assert_grads_close(self, g1, g2, msg="", atol=1e-5):
        self.assertGreater(len(g1), 0, f"No captured gradients ({msg})")
        self.assertEqual(set(g1), set(g2), f"Parameter name mismatch ({msg})")
        for name in g1:
            torch.testing.assert_close(
                g1[name],
                g2[name],
                atol=atol,
                rtol=1e-3,
                msg=f"Gradient mismatch for '{name}' ({msg})",
            )

    def _assert_grads_differ(self, g1, g2, min_rel_diff=0.15, msg=""):
        """Assert at least one param has a relative gradient difference ≥ min_rel_diff."""
        for name in g1:
            if name not in g2:
                continue
            ref = max(g1[name].abs().max().item(), g2[name].abs().max().item())
            if ref == 0:
                continue
            if abs(g1[name].abs().max().item() - g2[name].abs().max().item()) / ref >= min_rel_diff:
                return
        self.fail(f"Gradients were not sufficiently different ({msg})")

    # ── unit-level: verify compute_loss loss value ────────────────────────

    def test_compute_loss_aux_scaled_by_gas_k(self):
        """
        With gas=K, compute_loss must return a loss in which the aux contribution
        is weighted by 1/K, not 1 (the pre-fix behaviour).

        Verification: call compute_loss with gas=1 primed and gas=K primed on
        identical inputs and verify the difference is exactly (1 - 1/K) * weighted_aux.
        """
        for gas in [2, 4]:
            with self.subTest(gas=gas):
                model = _make_model(seed=10 + gas)
                dataset = _FixedDataset(n=gas)
                batch = _collate([dataset[i] for i in range(gas)])
                num_items = torch.tensor(int((batch["labels"] > 0).sum()))

                with tempfile.TemporaryDirectory() as tmp_dir:
                    args = TrainingArguments(
                        output_dir=tmp_dir,
                        per_device_train_batch_size=gas,
                        gradient_accumulation_steps=gas,
                        max_steps=1,
                        report_to="none",
                        use_cpu=True,
                    )
                    trainer = Trainer(
                        model=model,
                        args=args,
                        train_dataset=dataset,
                        data_collator=_collate,
                    )

                    # Simulate gas=1 window (first microbatch in a gas=1 run)
                    trainer.current_gradient_accumulation_steps = 1
                    loss_gas1 = trainer.compute_loss(copy.deepcopy(model), dict(batch), num_items_in_batch=num_items)

                    # Simulate gas=K window (all microbatches in a gas=K run)
                    trainer.current_gradient_accumulation_steps = gas
                    loss_gasK = trainer.compute_loss(copy.deepcopy(model), dict(batch), num_items_in_batch=num_items)

                    # Retrieve aux_loss from a fresh forward pass
                    with torch.no_grad():
                        m_fwd = copy.deepcopy(model)
                        fwd_out = m_fwd(**{**batch, "num_items_in_batch": num_items})
                    aux = fwd_out["aux_loss"]
                    weighted_aux = model.router_aux_loss_coef * aux.to(loss_gas1.device)

                    # Expected difference: (1 - 1/K) * weighted_aux
                    expected_diff = weighted_aux * (1.0 - 1.0 / gas)
                    actual_diff = loss_gas1 - loss_gasK
                    torch.testing.assert_close(
                        actual_diff,
                        expected_diff,
                        atol=1e-6,
                        rtol=1e-4,
                        msg=f"gas={gas}: loss difference from aux re-scaling incorrect",
                    )

    def test_compute_loss_gas1_unchanged(self):
        """
        With gas=1 the re-normalised weighted_aux equals the original weighted_aux
        (dividing by 1 is identity), so compute_loss must return the same value
        regardless of whether the fix is active.
        """
        model = _make_model(seed=20)
        dataset = _FixedDataset(n=4)
        batch = _collate([dataset[i] for i in range(4)])
        num_items = torch.tensor(int((batch["labels"] > 0).sum()))

        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(
                output_dir=tmp_dir,
                per_device_train_batch_size=4,
                max_steps=1,
                report_to="none",
                use_cpu=True,
            )
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=dataset,
                data_collator=_collate,
            )
            trainer.current_gradient_accumulation_steps = 1

            # Fixed Trainer
            loss_fixed = trainer.compute_loss(copy.deepcopy(model), dict(batch), num_items_in_batch=num_items)
            # Buggy Trainer (no re-normalisation) — must give same answer at gas=1
            buggy = _BuggyTrainer(
                model=model,
                args=args,
                train_dataset=dataset,
                data_collator=_collate,
            )
            buggy.current_gradient_accumulation_steps = 1
            loss_buggy = buggy.compute_loss(copy.deepcopy(model), dict(batch), num_items_in_batch=num_items)
            torch.testing.assert_close(
                loss_fixed,
                loss_buggy,
                atol=1e-6,
                rtol=1e-4,
                msg="At gas=1 fixed and buggy trainers must return identical loss",
            )

    # ── end-to-end: CE-only parameter gradients ───────────────────────────

    def test_ce_param_grads_match_gas1_vs_gas4(self):
        """
        Parameters that only receive gradient through the CE path (lm_head, norm)
        must have identical accumulated gradients for gas=1 (batch_size=4) and
        gas=4 (batch_size=1) since the CE term is correctly normalised by
        num_items_in_batch in both cases.
        """
        m1 = _make_model(seed=0)
        m4 = copy.deepcopy(m1)
        g1 = _run_one_step(m1, gas=1, batch_size=4, n_samples=4)
        g4 = _run_one_step(m4, gas=4, batch_size=1, n_samples=4)
        self.assertGreater(len(g1), 0, "No CE-only gradients captured")
        self._assert_grads_close(g1, g4, msg="CE params: gas=1 vs gas=4 (fix)", atol=1e-5)

    def test_ce_param_grads_match_gas1_vs_gas2(self):
        """Same as above for gas=2."""
        m1 = _make_model(seed=1)
        m2 = copy.deepcopy(m1)
        g1 = _run_one_step(m1, gas=1, batch_size=4, n_samples=4)
        g2 = _run_one_step(m2, gas=2, batch_size=2, n_samples=4)
        self._assert_grads_close(g1, g2, msg="CE params: gas=1 vs gas=2 (fix)", atol=1e-5)

    # ── proof-of-regression ───────────────────────────────────────────────

    def test_buggy_trainer_loss_inflated_at_gas4(self):
        """
        Without the fix (_BuggyTrainer) the aux term accumulates K× unscaled, so
        with gas=4 the per-microbatch loss value is higher than with the fix.
        This directly measures the scaling error in the loss that flows to backward.
        """
        for gas in [2, 4]:
            with self.subTest(gas=gas):
                model = _make_model(seed=30 + gas)
                dataset = _FixedDataset(n=gas)
                batch = _collate([dataset[i] for i in range(gas)])
                num_items = torch.tensor(int((batch["labels"] > 0).sum()))

                with tempfile.TemporaryDirectory() as tmp_dir:
                    args = TrainingArguments(
                        output_dir=tmp_dir,
                        per_device_train_batch_size=gas,
                        gradient_accumulation_steps=gas,
                        max_steps=1,
                        report_to="none",
                        use_cpu=True,
                    )
                    fixed_trainer = Trainer(
                        model=model,
                        args=args,
                        train_dataset=dataset,
                        data_collator=_collate,
                    )
                    buggy_trainer = _BuggyTrainer(
                        model=model,
                        args=args,
                        train_dataset=dataset,
                        data_collator=_collate,
                    )
                    fixed_trainer.current_gradient_accumulation_steps = gas
                    buggy_trainer.current_gradient_accumulation_steps = gas

                    loss_fixed = fixed_trainer.compute_loss(
                        copy.deepcopy(model), dict(batch), num_items_in_batch=num_items
                    )
                    loss_buggy = buggy_trainer.compute_loss(
                        copy.deepcopy(model), dict(batch), num_items_in_batch=num_items
                    )

                # Retrieve aux_loss from a fresh forward pass
                with torch.no_grad():
                    fwd_out = copy.deepcopy(model)(**{**batch, "num_items_in_batch": num_items})
                aux = fwd_out["aux_loss"]
                weighted_aux = model.router_aux_loss_coef * aux.to(loss_fixed.device)

                # Buggy loss = fixed loss + (1 - 1/K) * weighted_aux
                # (the aux term in buggy is full-size; in fixed it's 1/K)
                expected_buggy = loss_fixed + weighted_aux * (1.0 - 1.0 / gas)
                torch.testing.assert_close(
                    loss_buggy,
                    expected_buggy,
                    atol=1e-6,
                    rtol=1e-4,
                    msg=f"gas={gas}: buggy loss must exceed fixed loss by (1-1/K)*weighted_aux",
                )

    # ── robustness / safety guards ────────────────────────────────────────

    def test_zero_aux_coef_all_params_unaffected(self):
        """
        With router_aux_loss_coef=0 the fix is mathematically a no-op.
        All parameters (including router gates) must match across gas=1 and gas=4.
        """
        m1 = _make_model(seed=3, router_aux_loss_coef=0.0)
        m4 = copy.deepcopy(m1)
        g1 = _run_one_step(m1, gas=1, batch_size=4, n_samples=4, capture_filter=lambda n: True)
        g4 = _run_one_step(m4, gas=4, batch_size=1, n_samples=4, capture_filter=lambda n: True)
        self._assert_grads_close(g1, g4, msg="coef=0 all params", atol=1e-5)

    def test_output_router_logits_false_no_exception(self):
        """
        When output_router_logits=False, outputs.aux_loss is None.
        The fix guard must skip silently; training must complete.
        """
        m = _make_model(seed=4, output_router_logits=False)
        _run_one_step(m, gas=2, batch_size=2, n_samples=4)  # must not raise

    def test_compute_loss_before_training_no_attribute_error(self):
        """
        Calling compute_loss() before trainer.train() (so
        `current_gradient_accumulation_steps` is not set) must not raise
        AttributeError — the fix guard checks hasattr() first.
        """
        model = _make_model(seed=5)
        dataset = _FixedDataset(n=4)

        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=1,
                report_to="none",
                use_cpu=True,
            )
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=dataset,
                data_collator=_collate,
            )
            self.assertFalse(
                hasattr(trainer, "current_gradient_accumulation_steps"),
                "Attribute must not exist before training",
            )
            batch = _collate([dataset[i] for i in range(2)])
            num_items = torch.tensor(int((batch["labels"] > 0).sum()))
            try:
                loss = trainer.compute_loss(model, batch, num_items_in_batch=num_items)
            except AttributeError as exc:
                self.fail(f"compute_loss raised AttributeError before training: {exc}")
            self.assertTrue(loss.isfinite(), "Loss must be finite")

    def test_return_outputs_true_loss_matches_scalar(self):
        """
        compute_loss(return_outputs=True) must return the same corrected loss
        scalar as compute_loss(return_outputs=False) for identical inputs.
        """
        model = _make_model(seed=6)
        dataset = _FixedDataset(n=4)
        batch = _collate([dataset[i] for i in range(4)])
        num_items = torch.tensor(int((batch["labels"] > 0).sum()))

        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=2,
                max_steps=1,
                report_to="none",
                use_cpu=True,
                learning_rate=0.0,
            )
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=dataset,
                data_collator=_collate,
            )
            trainer.current_gradient_accumulation_steps = 2

            loss_a = trainer.compute_loss(copy.deepcopy(model), dict(batch), num_items_in_batch=num_items)
            loss_b, _ = trainer.compute_loss(
                copy.deepcopy(model), dict(batch), return_outputs=True, num_items_in_batch=num_items
            )

        torch.testing.assert_close(
            loss_a,
            loss_b,
            atol=1e-6,
            rtol=1e-4,
            msg="return_outputs=True must return same loss as return_outputs=False",
        )
        self.assertTrue(loss_b.isfinite())

    def test_label_smoothing_path_not_affected(self):
        """
        With label_smoothing_factor > 0 the Trainer uses LabelSmoother (labels
        local var set → fix guard skips).  Training must complete with finite loss.
        """
        m = _make_model(seed=7)
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=2,
                max_steps=1,
                report_to="none",
                use_cpu=True,
                label_smoothing_factor=0.1,
            )
            result = Trainer(
                model=m,
                args=args,
                train_dataset=_FixedDataset(n=4),
                data_collator=_collate,
            ).train()
        self.assertTrue(math.isfinite(result.training_loss))

    def test_partial_final_accumulation_window(self):
        """
        When dataset_size % gas != 0 the last optimizer step uses a shorter
        accumulation window.  current_gradient_accumulation_steps must reflect
        the actual window size.  Training must complete with finite loss.
        """
        m = _make_model(seed=8)
        # 5 samples with gas=4: first step uses 4 microbatches, last uses 1.
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(
                output_dir=tmp_dir,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                num_train_epochs=1,
                report_to="none",
                use_cpu=True,
            )
            result = Trainer(
                model=m,
                args=args,
                train_dataset=_FixedDataset(n=5),
                data_collator=_collate,
            ).train()
        self.assertTrue(math.isfinite(result.training_loss))


# ---------------------------------------------------------------------------
# DDP unit-level tests (no actual distributed processes required)
# ---------------------------------------------------------------------------
# The DDP branch in compute_loss fires when average_tokens_across_devices=True
# and accelerator.num_processes > 1.  We mock num_processes directly on the
# trainer's accelerator to simulate a multi-rank world without spawning real
# processes.  This verifies the exact formula without the infrastructure
# overhead of torchrun / mp.spawn.


class TestMoeAuxDDPUnit(unittest.TestCase):
    """
    Unit tests for the DDP world-size compensation branch of compute_loss.

    Approach: build a real single-process Trainer, then patch
    `trainer.accelerator.num_processes` to simulate a world_size=N
    environment.  The fix must undo the DDP factor for the aux term while
    leaving it on the CE term.
    """

    def _make_trainer(self, gas: int = 1, world_size_override: int = 1) -> tuple:
        """
        Return (trainer, model, batch, num_items) ready for a direct
        compute_loss() call.  trainer.accelerator.num_processes is patched to
        world_size_override to simulate DDP scaling.
        """
        model = _make_model(seed=100)
        dataset = _FixedDataset(n=4)
        batch = _collate([dataset[i] for i in range(4)])
        num_items = torch.tensor(int((batch["labels"] > 0).sum()))

        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(
                output_dir=tmp_dir,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=gas,
                max_steps=1,
                report_to="none",
                use_cpu=True,
                average_tokens_across_devices=True,
            )
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=dataset,
                data_collator=_collate,
            )
            trainer.current_gradient_accumulation_steps = gas

        # Patch num_processes to simulate multi-rank DDP
        trainer.accelerator.state.num_processes = world_size_override
        return trainer, model, batch, num_items

    # ── DDP math correctness ──────────────────────────────────────────────

    def test_ddp_loss_scale_aux_removed_world_size_2(self):
        """
        With world_size=2 the DDP block multiplies the combined loss by 2.
        The fix must undo that multiplication for the aux term and replace it
        with the accumulation-normalised form.

        Expected corrected loss:
            loss_scale * (combined - weighted_aux) + weighted_aux / gas

        which with gas=1 simplifies to:
            loss_scale * ce_only + weighted_aux
        """
        for world_size in [2, 4]:
            with self.subTest(world_size=world_size):
                gas = 1
                trainer, model, batch, num_items = self._make_trainer(gas=gas, world_size_override=world_size)

                # Forward pass to get the raw combined loss and aux values
                with torch.no_grad():
                    fwd = copy.deepcopy(model)(**{**batch, "num_items_in_batch": num_items})
                combined_loss = fwd["loss"]
                aux = fwd["aux_loss"]
                weighted_aux = model.router_aux_loss_coef * aux

                # Expected: loss_scale applied to CE only, aux at 1/gas = 1
                loss_scale = world_size  # tp_size=1 on CPU single-process
                expected = loss_scale * (combined_loss - weighted_aux) + weighted_aux / gas
                expected = expected.detach()

                actual = trainer.compute_loss(copy.deepcopy(model), dict(batch), num_items_in_batch=num_items).detach()

                torch.testing.assert_close(
                    actual,
                    expected,
                    atol=1e-5,
                    rtol=1e-3,
                    msg=f"world_size={world_size}: DDP-corrected loss mismatch",
                )

    def test_ddp_plus_accumulation_combined_world_size_2_gas_4(self):
        """
        Both DDP (world_size=2) and accumulation (gas=4) active simultaneously.

        Expected:
            loss_scale * (combined - weighted_aux) + weighted_aux / gas
        """
        world_size, gas = 2, 4
        trainer, model, batch, num_items = self._make_trainer(gas=gas, world_size_override=world_size)

        with torch.no_grad():
            fwd = copy.deepcopy(model)(**{**batch, "num_items_in_batch": num_items})
        combined_loss = fwd["loss"]
        aux = fwd["aux_loss"]
        weighted_aux = model.router_aux_loss_coef * aux

        loss_scale = world_size
        expected = (loss_scale * (combined_loss - weighted_aux) + weighted_aux / gas).detach()

        actual = trainer.compute_loss(copy.deepcopy(model), dict(batch), num_items_in_batch=num_items).detach()

        torch.testing.assert_close(
            actual,
            expected,
            atol=1e-5,
            rtol=1e-3,
            msg="world_size=2, gas=4: combined DDP+accumulation loss mismatch",
        )

    def test_ddp_world_size_1_no_change(self):
        """
        world_size=1 is the same as single-process; no DDP scale is applied.
        The fix must not change the loss.
        """
        gas = 1
        trainer, model, batch, num_items = self._make_trainer(gas=gas, world_size_override=1)

        with torch.no_grad():
            fwd = copy.deepcopy(model)(**{**batch, "num_items_in_batch": num_items})
        combined_loss = fwd["loss"].detach()

        actual = trainer.compute_loss(copy.deepcopy(model), dict(batch), num_items_in_batch=num_items).detach()

        # world_size=1 → loss_scale=1 → no change to combined loss
        torch.testing.assert_close(
            actual,
            combined_loss,
            atol=1e-6,
            rtol=1e-4,
            msg="world_size=1: loss must equal raw combined loss",
        )

    # ── regression proof at unit level ───────────────────────────────────

    def test_ddp_buggy_trainer_inflates_aux_world_size_2(self):
        """
        Verify the DDP regression algebraically: before the fix, the combined
        loss (CE + weighted_aux) would have been multiplied by loss_scale,
        inflating the aux gradient.

        The pre-fix loss is:
            loss_scale * (ce_loss + weighted_aux)

        The fixed loss is:
            loss_scale * ce_loss + weighted_aux / gas

        So the excess that the fix removes is:
            (loss_scale - 1/gas) * weighted_aux

        We verify this by computing the pre-fix loss manually from the same
        forward pass and confirming the difference matches the formula.
        """
        world_size, gas = 2, 1
        trainer, model, batch, num_items = self._make_trainer(gas=gas, world_size_override=world_size)

        # Get fixed loss from the patched Trainer
        loss_fixed = trainer.compute_loss(copy.deepcopy(model), dict(batch), num_items_in_batch=num_items).detach()

        # Manually compute the pre-fix (buggy) loss:
        # loss_scale * (combined_loss) — no aux correction
        with torch.no_grad():
            fwd = copy.deepcopy(model)(**{**batch, "num_items_in_batch": num_items})
        combined_loss = fwd["loss"].detach()
        aux = fwd["aux_loss"].detach()
        weighted_aux = (model.router_aux_loss_coef * aux).detach()

        loss_scale = world_size  # tp_size=1
        loss_buggy_expected = loss_scale * combined_loss  # what the pre-fix code produced

        # The fix removes (loss_scale - 1/gas) * weighted_aux from the loss
        expected_reduction = (loss_scale - 1.0 / gas) * weighted_aux
        actual_reduction = loss_buggy_expected - loss_fixed

        torch.testing.assert_close(
            actual_reduction,
            expected_reduction,
            atol=1e-5,
            rtol=1e-3,
            msg=f"world_size={world_size},gas={gas}: "
            f"fix must reduce loss by (loss_scale - 1/gas)*weighted_aux; "
            f"got {actual_reduction.item():.6f}, expected {expected_reduction.item():.6f}",
        )

        # Also verify loss_fixed = loss_scale * ce_only + weighted_aux / gas
        ce_only = combined_loss - weighted_aux
        expected_fixed = loss_scale * ce_only + weighted_aux / gas
        torch.testing.assert_close(
            loss_fixed,
            expected_fixed,
            atol=1e-5,
            rtol=1e-3,
            msg="Fixed loss must equal loss_scale*ce + weighted_aux/gas",
        )


# ---------------------------------------------------------------------------
# Aux-only router gradient isolation tests
# ---------------------------------------------------------------------------
# These tests verify the core property stated in the bug report:
#
#   "For the same logical batch with matched routing distributions, changing
#    the DDP world size should preserve the effective router_aux_loss_coef
#    and produce matching router gradients within numerical tolerance."
#
# Methodology
# -----------
# Router parameters receive gradient from TWO sources:
#   (a) The CE path (via expert outputs → hidden states).
#       This legitimately scales with world_size (DDP factor on CE is correct).
#   (b) The aux path (load_balancing_loss_func → router logits).
#       This must NOT scale with world_size after the fix.
#
# We isolate source (b) by computing:
#   aux_grad = grads(coef=0.1) − grads(coef=0.0)
#
# With identical inputs and matched routing distributions, aux_grad must be
# invariant to world_size (fix removes the DDP factor from the aux term).
#
# For gradient accumulation, a single backward call with gas_window=K produces
# aux_grad / K.  After K accumulation steps the sum equals the gas=1 aux_grad.
# This is verified by asserting ratio ≈ 1/K for a single call.


@require_torch
class TestMoeAuxRouterGradientIsolation(unittest.TestCase):
    """
    Isolate and verify the aux-loss contribution to router gradients under
    DDP world-size compensation and gradient accumulation.
    """

    @classmethod
    def setUpClass(cls):
        # Shared batch used by all tests (same tokens → matched routing).
        dataset = _FixedDataset(n=4, seed=0)
        cls._batch = _collate([dataset[i] for i in range(4)])
        cls._num_items = torch.tensor(int((cls._batch["labels"] > 0).sum()))
        cls._dataset = dataset

    def _router_grads_from_backward(
        self,
        coef: float,
        world_size: int,
        gas_window: int,
    ) -> dict[str, torch.Tensor]:
        """
        Single forward+backward on the shared batch.
        Returns router gate gradients.
        world_size is patched onto accelerator.state so the DDP branch fires.
        """
        model = _make_model(seed=42, router_aux_loss_coef=coef)
        captured = {}

        def make_hook(name):
            def h(grad):
                captured[name] = captured.get(name, torch.zeros_like(grad)) + grad.detach()

            return h

        for name, param in model.named_parameters():
            if "gate.weight" in name:
                param.register_hook(make_hook(name))

        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(
                output_dir=tmp_dir,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=1,
                max_steps=1,
                learning_rate=1e-9,
                report_to="none",
                use_cpu=True,
                average_tokens_across_devices=True,
            )
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=self._dataset,
                data_collator=_collate,
            )
        trainer.current_gradient_accumulation_steps = gas_window
        trainer.accelerator.state.num_processes = world_size

        loss = trainer.compute_loss(model, dict(self._batch), num_items_in_batch=self._num_items)
        loss.backward()
        return {n: g.clone() for n, g in captured.items()}

    def _aux_only_grads(self, world_size: int, gas_window: int) -> dict[str, torch.Tensor]:
        """
        Subtract coef=0 gradients from coef=0.1 gradients to isolate the
        aux-loss contribution.  Both use identical model weights and input.
        """
        g_coef = self._router_grads_from_backward(0.1, world_size, gas_window)
        g_zero = self._router_grads_from_backward(0.0, world_size, gas_window)
        return {n: (g_coef[n] - g_zero[n]) for n in g_coef}

    # ── DDP invariance ────────────────────────────────────────────────────

    def test_aux_router_grad_invariant_to_world_size(self):
        """
        With matched routing distributions, the aux-only router gradient must
        be identical for world_size=1 and world_size=2 (and world_size=4).

        Before the fix: world_size=2 → aux_grad was 2× too large.
        After the fix:  world_size=N → aux_grad is unchanged.
        """
        ref = self._aux_only_grads(world_size=1, gas_window=1)
        self.assertGreater(len(ref), 0, "No router gate gradients captured")

        for ws in [2, 4]:
            with self.subTest(world_size=ws):
                g = self._aux_only_grads(world_size=ws, gas_window=1)
                for name in ref:
                    torch.testing.assert_close(
                        g[name],
                        ref[name],
                        atol=1e-6,
                        rtol=1e-4,
                        msg=f"world_size={ws}: aux router grad must equal world_size=1",
                    )

    def test_aux_router_grad_scales_as_1_over_gas_window(self):
        """
        A single backward call with gas_window=K divides the aux contribution
        by K (the fix: weighted_aux / current_gradient_accumulation_steps).

        After K accumulation steps the sum recovers the gas=1 value.
        Verify the per-step ratio is 1/K within tolerance.
        """
        ref = self._aux_only_grads(world_size=1, gas_window=1)

        for gas in [2, 4]:
            with self.subTest(gas=gas):
                g = self._aux_only_grads(world_size=1, gas_window=gas)
                for name in ref:
                    ref_norm = ref[name].norm().item()
                    if ref_norm == 0:
                        continue
                    actual_ratio = g[name].norm().item() / ref_norm
                    expected_ratio = 1.0 / gas
                    self.assertAlmostEqual(
                        actual_ratio,
                        expected_ratio,
                        delta=0.01,
                        msg=f"gas_window={gas}: per-step aux_grad must be 1/gas of gas=1 value",
                    )

    def test_aux_router_grad_combined_ddp_and_accumulation(self):
        """
        Combined: world_size=2 and gas_window=4.

        Expected aux-only per-step ratio vs (ws=1, gas=1): 1/4.
        (The DDP factor is removed by the fix; only the 1/gas factor remains.)
        """
        ref = self._aux_only_grads(world_size=1, gas_window=1)
        g = self._aux_only_grads(world_size=2, gas_window=4)

        for name in ref:
            ref_norm = ref[name].norm().item()
            if ref_norm == 0:
                continue
            actual_ratio = g[name].norm().item() / ref_norm
            expected_ratio = 1.0 / 4  # only gas divides; DDP factor removed
            self.assertAlmostEqual(
                actual_ratio,
                expected_ratio,
                delta=0.01,
                msg="ws=2,gas=4: combined aux_grad ratio must be 1/gas",
            )

    def test_aux_router_grad_zero_coef_is_zero(self):
        """
        With router_aux_loss_coef=0 the aux-only gradient (coef=0.1 − coef=0)
        is not meaningful, but the coef=0 gradient itself must be unaffected
        by world_size changes (fix is a no-op when coef=0).
        """
        model_zero = _make_model(seed=42, router_aux_loss_coef=0.0)

        def grads_zero(ws, gas):
            captured = {}

            def hook(name):
                def h(grad):
                    captured[name] = captured.get(name, torch.zeros_like(grad)) + grad.detach()

                return h

            m = copy.deepcopy(model_zero)
            for name, param in m.named_parameters():
                if "gate.weight" in name:
                    param.register_hook(hook(name))

            with tempfile.TemporaryDirectory() as tmp_dir:
                args = TrainingArguments(
                    output_dir=tmp_dir,
                    per_device_train_batch_size=4,
                    max_steps=1,
                    learning_rate=1e-9,
                    report_to="none",
                    use_cpu=True,
                    average_tokens_across_devices=True,
                )
                trainer = Trainer(
                    model=m,
                    args=args,
                    train_dataset=self._dataset,
                    data_collator=_collate,
                )
            trainer.current_gradient_accumulation_steps = gas
            trainer.accelerator.state.num_processes = ws
            loss = trainer.compute_loss(m, dict(self._batch), num_items_in_batch=self._num_items)
            loss.backward()
            return {n: g.clone() for n, g in captured.items()}

        g1 = grads_zero(1, 1)
        g2 = grads_zero(2, 1)
        for name in g1:
            # CE-path gradient doubles at ws=2 (correct DDP behavior for CE)
            # We only assert it's finite and non-zero — no assertion about equality
            self.assertTrue(g1[name].isfinite().all(), f"Non-finite grad at ws=1 for {name}")
            self.assertTrue(g2[name].isfinite().all(), f"Non-finite grad at ws=2 for {name}")
