# Copyright 2026 Nathan. Apache-2.0.
"""
Training loop built for a runtime that will be interrupted.

The loop itself is ordinary. What is not ordinary is that it assumes every step might be its last: a
checkpoint goes down at a fixed wall-clock interval rather than every N steps, because what kills a
hosted notebook is time, not iterations, and a run that checkpoints every 500 steps loses everything
when step 499 is when the cell is reclaimed.

Restarting is meant to be the same command. It finds the newest usable checkpoint -- locally, or pulled
back down from the Hub if the machine itself is new -- restores the optimizer, schedule and random
state along with the weights, and carries on.
"""

from __future__ import annotations

import math
import signal
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .checkpoint import CheckpointManager, TrainingState
from .losses import total_loss
from .model import SURFACE_FIELDS, NatureConfig, NatureV1


@dataclass
class TrainSettings:
    """
    Args:
        learning_rate: peak learning rate after warmup.
        warmup_steps: linear warmup, then cosine decay.
        max_steps: total optimizer steps for the whole run, across restarts.
        grad_accum: micro-batches per optimizer step, for a larger effective batch than fits at once.
        grad_clip: gradient-norm clip. Weather targets are heavy-tailed and a single convective outlier
            can otherwise blow up a step.
        precision: ``"bf16"`` on Blackwell/Hopper/Ampere, ``"fp16"`` on older cards, ``"fp32"`` to debug.
            bf16 needs no loss scaling, which removes a whole class of silent divergence.
        checkpoint_seconds: wall time between checkpoints.
        log_every: steps between log lines.
        stage: ``"pretrain"`` trains everything on reanalysis, where the label is the next state and the
            corpus is effectively unlimited. ``"finetune"`` freezes the backbone and trains only the storm
            heads on best tracks. That split is what makes an 89M model safe to point at 55,230 track
            points: under a million parameters are ever fitted to them.
        ema_decay: exponential moving average of the weights, evaluated instead of the raw ones. Cheap,
            and it consistently helps on small fine-tuning sets where the last step is noisy.
        early_stopping_patience: validation evaluations without improvement before stopping. ``0`` disables.
    """

    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 500
    max_steps: int = 100_000
    grad_accum: int = 1
    grad_clip: float = 1.0
    precision: str = "bf16"
    checkpoint_seconds: float = 60.0
    checkpoint_dir: str = "./checkpoints"
    checkpoint_keep: int = 3
    hub_repo: str | None = None
    hub_push_seconds: float = 900.0
    log_every: int = 10
    stage: str = "pretrain"
    ema_decay: float = 0.999
    early_stopping_patience: int = 0


def build_scheduler(optimizer, settings: TrainSettings):
    """Linear warmup into a cosine decay, floored at 3% of peak so late training still moves."""

    def factor(step: int) -> float:
        if step < settings.warmup_steps:
            return (step + 1) / max(1, settings.warmup_steps)
        progress = (step - settings.warmup_steps) / max(1, settings.max_steps - settings.warmup_steps)
        return max(0.03, 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def move_batch(batch: dict, device: torch.device, dtype: torch.dtype | None = None) -> dict:
    """Move a batch to the device, leaving integer targets as integers."""
    moved = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            moved[key] = value
        elif value.is_floating_point():
            moved[key] = value.to(device, dtype=dtype) if dtype else value.to(device)
        else:
            moved[key] = value.to(device)
    return moved


class Trainer:
    """
    Args:
        model: the :class:`NatureV1` to train.
        settings: optimization and checkpointing settings.
        device: where to train. Defaults to CUDA when present.
    """

    def __init__(self, model: NatureV1, settings: TrainSettings, device: torch.device | str | None = None) -> None:
        self.model = model
        self.settings = settings
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

        decay, no_decay = [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            # Norms, biases and the learned scalars (radii, biases on the index) should not be decayed
            # toward zero: a receptive radius shrinking because of weight decay is not regularization.
            (no_decay if parameter.ndim <= 1 else decay).append(parameter)
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": settings.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=settings.learning_rate, betas=(0.9, 0.95), eps=1e-8,
        )
        self.scheduler = build_scheduler(self.optimizer, settings)
        self.use_amp = settings.precision in ("bf16", "fp16") and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if settings.precision == "bf16" else torch.float16
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=settings.precision == "fp16" and self.use_amp)

        self.checkpoints = CheckpointManager(
            settings.checkpoint_dir, every_seconds=settings.checkpoint_seconds, keep=settings.checkpoint_keep,
            repo_id=settings.hub_repo, push_every_seconds=settings.hub_push_seconds,
        )
        self.state = TrainingState()
        self._stop = False
        self._since_improvement = 0
        self.ema = None
        if settings.ema_decay and settings.ema_decay > 0:
            self.ema = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
        if settings.stage == "finetune":
            trainable, total = model.freeze_backbone(True)
            print(f"[train] fine-tuning stage: {trainable:,} of {total:,} parameters trainable "
                  f"({trainable / total:.1%}) -- the backbone is frozen", flush=True)
        self._install_signal_handlers()

    def _install_signal_handlers(self) -> None:
        """On SIGINT/SIGTERM, finish the current step and checkpoint rather than dying mid-update."""

        def handler(signum, frame):
            print(f"\n[train] signal {signum}: finishing this step and checkpointing", flush=True)
            self._stop = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass  # not the main thread, e.g. inside some notebook runtimes

    def resume(self) -> TrainingState:
        """Restore the newest checkpoint, pulling from the Hub first if this machine has none."""
        self.checkpoints.fetch_from_hub()
        restored = self.checkpoints.load(self.model, self.optimizer, self.scheduler, self.scaler,
                                         map_location=str(self.device))
        if restored is not None:
            self.state = restored
        else:
            print("[train] no checkpoint found; starting from scratch", flush=True)
        return self.state

    def save(self, force: bool = False):
        return self.checkpoints.save(
            self.model, self.optimizer, self.scheduler, self.scaler, self.state,
            config=self.model.config.to_dict(), force=force,
        )

    def fit(self, loader, epochs: int = 1) -> TrainingState:
        """
        Train until ``max_steps``, the epochs run out, or the process is asked to stop.

        ``loader`` yields dicts holding the model inputs and whichever targets are available; a batch
        missing a target simply does not contribute that term, so storm-centric and gridded data can be
        interleaved freely.
        """
        settings = self.settings
        self.model.train()
        started = time.time()
        micro = 0
        self.optimizer.zero_grad(set_to_none=True)

        for epoch in range(self.state.epoch, epochs):
            self.state.epoch = epoch
            for batch in loader:
                if self._stop or self.state.step >= settings.max_steps:
                    break
                batch = move_batch(batch, self.device)
                with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                    outputs = self.model(
                        satellite=batch.get("satellite"), satellite_grid=batch.get("satellite_grid"),
                        analysis=batch.get("analysis"), analysis_grid=batch.get("analysis_grid"),
                        calendar=batch["calendar"], output_grid=batch.get("output_grid"),
                    )
                    loss, parts = total_loss(outputs, batch, SURFACE_FIELDS)
                    loss = loss / settings.grad_accum

                self.scaler.scale(loss).backward() if self.scaler.is_enabled() else loss.backward()
                micro += 1
                if micro % settings.grad_accum:
                    continue

                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), settings.grad_clip)
                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                self.state.step += 1
                self.state.samples_seen += int(batch["calendar"].shape[0]) * settings.grad_accum
                self.state.wall_seconds = time.time() - started
                if parts["total"] < self.state.best_loss:
                    self.state.best_loss = parts["total"]
                    self._since_improvement = 0
                else:
                    self._since_improvement += 1
                if self.ema is not None:
                    decay = settings.ema_decay
                    with torch.no_grad():
                        for name, parameter in self.model.named_parameters():
                            if name in self.ema:
                                self.ema[name].mul_(decay).add_(parameter.detach(), alpha=1 - decay)
                if (settings.early_stopping_patience
                        and self._since_improvement >= settings.early_stopping_patience):
                    print(f"[train] no improvement for {self._since_improvement} steps; stopping", flush=True)
                    self._stop = True

                if self.state.step % settings.log_every == 0:
                    self.state.history.append({"step": self.state.step, **parts})
                    detail = "  ".join(f"{k}={v:.4f}" for k, v in parts.items() if k != "total")
                    print(
                        f"[train] step {self.state.step:>7}  loss {parts['total']:.4f}  "
                        f"lr {self.scheduler.get_last_lr()[0]:.2e}  |grad| {float(grad_norm):.2f}  {detail}",
                        flush=True,
                    )
                self.save()

            if self._stop or self.state.step >= settings.max_steps:
                break

        # Whatever ended the run -- a signal, the step limit, the data -- the last state is written down.
        self.save(force=True)
        print(f"[train] stopped at step {self.state.step} after {self.state.wall_seconds / 60:.1f} min", flush=True)
        return self.state


def apply_ema(model: NatureV1, ema: dict) -> None:
    """Copy the averaged weights into the model, for evaluation or export."""
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in ema:
                parameter.copy_(ema[name])


def next_state_targets(sequence: torch.Tensor, lead_steps: int = 1) -> dict:
    """
    Build self-supervised targets from a reanalysis sequence: the future is the label.

    This is what makes stage one unlimited. Given frames ``t-5..t``, the target is the state at ``t+k``,
    which needs no annotation and exists for every one of the 92,040 timesteps in the archive. No storm
    database is involved and none is needed -- the backbone is learning what the atmosphere does, not
    what a hurricane is called.

    Args:
        sequence: ``(B, T + lead_steps, N, C)`` a window of consecutive analysis states.
        lead_steps: how far ahead to predict.

    Returns:
        ``{"analysis": inputs, "field_target": future}``.
    """
    return {"analysis": sequence[:, :-lead_steps], "field_target": sequence[:, -1]}


def load_for_inference(
    checkpoint_dir: str | Path, latent_grid, device: torch.device | str | None = None, hub_repo: str | None = None
) -> NatureV1:
    """Rebuild a model from a checkpoint's own config, so inference cannot drift from what was trained."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    manager = CheckpointManager(checkpoint_dir, repo_id=hub_repo)
    manager.fetch_from_hub()
    for path in manager._candidates():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        config = NatureConfig.from_dict(payload["config"]) if payload.get("config") else NatureConfig()
        model = NatureV1(config, latent_grid)
        model.load_state_dict(payload["model"])
        return model.to(device).eval()
    raise FileNotFoundError(f"No checkpoint in {checkpoint_dir}")
