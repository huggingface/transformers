# Copyright 2026 Nathan. Apache-2.0.
"""
Regression tests for the training loop's held-out evaluation.

These exist because the loop originally early-stopped on *training* loss while its own docstring
claimed validation. That is not a small mislabel: with a patience of 10 it halts the moment ten
consecutive minibatches fail to set a new global minimum, which on noisy data is a near certainty
early in a run. It would have stopped a real run within minutes and reported it as convergence, and
it would never once have looked at data the model had not seen.
"""

from __future__ import annotations

import pytest
import torch
from naturev1 import SURFACE_FIELDS, NatureConfig, NatureV1, Trainer, TrainSettings

from ihelix import fibonacci_sphere


@pytest.fixture(scope="module")
def pieces():
    config = NatureConfig(latent_points=128, num_layers=2, hidden_size=96, history_frames=1,
                          lead_times_hours=(6, 12))
    mesh = fibonacci_sphere(128, num_neighbours=12, cluster_size=32)
    source = fibonacci_sphere(64, num_neighbours=8, cluster_size=16)
    return config, mesh, source


def make_batches(config, source, count, seed=0):
    """Fixed batches, so a test measures the loop rather than the noise."""
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(count):
        batches.append({
            "analysis": torch.randn(1, config.history_frames, source.num_points,
                                    config.analysis_channels, generator=generator),
            "calendar": torch.zeros(1, config.history_frames, 6),
            "field_target": torch.randn(1, source.num_points, config.num_leads,
                                        len(SURFACE_FIELDS), generator=generator),
            "field_mask": torch.ones(1, source.num_points, config.num_leads, len(SURFACE_FIELDS)),
            "analysis_grid": source,
            "output_grid": source,
        })
    return batches


def settings(**overrides):
    base = dict(max_steps=4, warmup_steps=1, log_every=1000, precision="fp32",
                checkpoint_seconds=1e9, ema_decay=0.0, grad_clip=1.0)
    base.update(overrides)
    return base


def test_evaluate_does_not_change_the_weights(pieces, tmp_path):
    """A validation pass that trains on the validation set is not a validation pass."""
    config, mesh, source = pieces
    model = NatureV1(config, mesh)
    trainer = Trainer(model, TrainSettings(checkpoint_dir=str(tmp_path), **settings()), device="cpu")

    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    scores = trainer.evaluate(make_batches(config, source, 2), max_batches=2)

    assert all(torch.equal(value.detach(), before[name]) for name, value in model.named_parameters())
    assert "val_total" in scores and scores["val_total"] > 0


def test_evaluate_restores_training_mode(pieces, tmp_path):
    """Leaving the model in eval() would silently disable dropout for the rest of the run."""
    config, mesh, source = pieces
    model = NatureV1(config, mesh)
    trainer = Trainer(model, TrainSettings(checkpoint_dir=str(tmp_path), **settings()), device="cpu")

    model.train()
    trainer.evaluate(make_batches(config, source, 1), max_batches=1)
    assert model.training, "evaluate must put the model back the way it found it"

    model.eval()
    trainer.evaluate(make_batches(config, source, 1), max_batches=1)
    assert not model.training


def test_evaluate_respects_the_batch_limit(pieces, tmp_path):
    """A full sweep of a held-out decade every few hundred steps would cost more than the training."""
    config, mesh, source = pieces
    trainer = Trainer(NatureV1(config, mesh),
                      TrainSettings(checkpoint_dir=str(tmp_path), **settings(val_batches=2)),
                      device="cpu")

    class Counting(list):
        def __init__(self, items):
            super().__init__(items)
            self.served = 0

        def __iter__(self):
            for item in list.__iter__(self):
                self.served += 1
                yield item

    loader = Counting(make_batches(config, source, 8))
    trainer.evaluate(loader)
    assert loader.served == 2


def test_fit_reports_held_out_loss(pieces, tmp_path, capsys):
    config, mesh, source = pieces
    trainer = Trainer(NatureV1(config, mesh),
                      TrainSettings(checkpoint_dir=str(tmp_path), **settings(max_steps=4, val_every=2)),
                      device="cpu")
    trainer.fit(make_batches(config, source, 4), epochs=1,
                val_loader=make_batches(config, source, 2, seed=99))

    printed = capsys.readouterr().out
    assert "[val]" in printed and "held-out" in printed and "gap" in printed
    assert any("val_total" in entry for entry in trainer.state.history)


def test_early_stopping_never_fires_without_held_out_data(pieces, tmp_path):
    """
    The original bug. With patience on training loss, any run stops as soon as ten consecutive
    minibatches miss a new global minimum -- which says nothing about the model.
    """
    config, mesh, source = pieces
    trainer = Trainer(
        NatureV1(config, mesh),
        TrainSettings(checkpoint_dir=str(tmp_path),
                      **settings(max_steps=6, early_stopping_patience=1)),
        device="cpu",
    )
    state = trainer.fit(make_batches(config, source, 6), epochs=1)      # no val_loader
    assert state.step == 6, "without held-out data there is nothing to early-stop on"


def test_early_stopping_fires_on_held_out_loss(pieces, tmp_path):
    """With validation wired up, patience counts validation passes -- not minibatches."""
    config, mesh, source = pieces
    trainer = Trainer(
        NatureV1(config, mesh),
        TrainSettings(checkpoint_dir=str(tmp_path),
                      **settings(max_steps=20, val_every=1, early_stopping_patience=1,
                                 val_batches=1, learning_rate=5.0)),   # huge LR -> held-out worsens
        device="cpu",
    )
    state = trainer.fit(make_batches(config, source, 20), epochs=1,
                        val_loader=make_batches(config, source, 1, seed=7))
    assert state.step < 20, "a worsening held-out loss should have stopped the run"
