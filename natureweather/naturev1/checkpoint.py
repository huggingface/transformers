# Copyright 2026 Nathan. Apache-2.0.
"""
Checkpointing that assumes the process will be killed without warning.

On a hosted notebook that is not a worst case, it is the normal case: the cell is reclaimed, the runtime
recycles, the tab closes. So every save here is written to a temporary file and then renamed into place,
because rename is atomic on every filesystem that matters -- there is no instant at which the live
checkpoint is half-written. A process killed mid-save leaves a stray temporary file and an intact
previous checkpoint, which is the outcome you want.

Resuming restores everything that makes a run reproducible, not just the weights: optimizer moments,
the learning-rate schedule, the gradient scaler, the epoch and step counters, and the random states of
Python, NumPy and Torch. Restoring weights alone gives you a run that silently reshuffles its data and
restarts its schedule, which looks like it resumed and did not.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class TrainingState:
    """Everything needed to continue a run exactly where it stopped."""

    step: int = 0
    epoch: int = 0
    samples_seen: int = 0
    best_loss: float = float("inf")
    wall_seconds: float = 0.0
    history: list = None

    def __post_init__(self) -> None:
        if self.history is None:
            self.history = []

    def to_dict(self) -> dict:
        return {
            "step": self.step, "epoch": self.epoch, "samples_seen": self.samples_seen,
            "best_loss": self.best_loss, "wall_seconds": self.wall_seconds,
            "history": self.history[-500:],
        }


def _rng_state() -> dict:
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict) -> None:
    if not state:
        return
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu() if torch.is_tensor(state["torch"]) else state["torch"])
    if "numpy" in state:
        import numpy as np

        np.random.set_state(state["numpy"])
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
        except (RuntimeError, ValueError):
            # A different GPU count than the run that saved it: keep going with fresh CUDA seeds.
            pass


class CheckpointManager:
    """
    Periodic, crash-safe checkpoints with optional mirroring to the Hugging Face Hub.

    Args:
        directory: where checkpoints live. On Colab, point this at Google Drive so it outlives the VM.
        every_seconds: minimum wall time between saves. Saving is cheap next to a training step, so a
            minute is a reasonable default; the cost is bounded by the size of the model, not the data.
        keep: how many recent checkpoints to retain, oldest pruned first.
        repo_id: Hub repository to mirror to, e.g. ``"you/NatureV1"``. Needs ``HF_TOKEN`` in the
            environment or a prior ``huggingface-cli login``.
        push_every_seconds: uploads are slower than local writes, so they get their own, longer interval.
    """

    LATEST = "latest.pt"

    def __init__(
        self,
        directory: str | Path,
        every_seconds: float = 60.0,
        keep: int = 3,
        repo_id: str | None = None,
        push_every_seconds: float = 900.0,
        private: bool = False,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.every_seconds = every_seconds
        self.keep = keep
        self.repo_id = repo_id
        self.push_every_seconds = push_every_seconds
        self.private = private
        self._last_save = 0.0
        self._last_push = 0.0
        self._api = None

    # ------------------------------------------------------------------ saving --

    def due(self) -> bool:
        """Whether enough time has passed to warrant another save."""
        return (time.time() - self._last_save) >= self.every_seconds

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler=None,
        scaler=None,
        state: TrainingState | None = None,
        config: dict | None = None,
        force: bool = False,
    ) -> Path | None:
        """
        Write a checkpoint if one is due. Returns the path written, or None if it was skipped.

        The write goes to a temporary file first and is renamed into place, so an interrupted save can
        never leave a corrupt ``latest.pt`` behind.
        """
        if not force and not self.due():
            return None
        state = state or TrainingState()
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "state": state.to_dict(),
            "config": config or {},
            "rng": _rng_state(),
            "saved_at": time.time(),
            "format": 1,
        }

        target = self.directory / self.LATEST
        temporary = self.directory / f".{self.LATEST}.{os.getpid()}.tmp"
        torch.save(payload, temporary)
        os.replace(temporary, target)  # atomic; readers see either the old file or the new one

        numbered = self.directory / f"step_{state.step:09d}.pt"
        shutil.copy2(target, numbered)
        self._prune()
        self._last_save = time.time()

        (self.directory / "state.json").write_text(json.dumps(state.to_dict(), indent=2, default=str) + "\n")
        self._maybe_push(target, state)
        return target

    def _prune(self) -> None:
        numbered = sorted(self.directory.glob("step_*.pt"))
        for stale in numbered[: max(0, len(numbered) - self.keep)]:
            stale.unlink(missing_ok=True)

    def _maybe_push(self, path: Path, state: TrainingState) -> None:
        if not self.repo_id or (time.time() - self._last_push) < self.push_every_seconds:
            return
        try:
            from huggingface_hub import HfApi

            if self._api is None:
                self._api = HfApi()
                self._api.create_repo(self.repo_id, repo_type="model", exist_ok=True, private=self.private)
            self._api.upload_file(
                path_or_fileobj=str(path), path_in_repo="latest.pt", repo_id=self.repo_id, repo_type="model",
                commit_message=f"step {state.step}",
            )
            self._last_push = time.time()
        except Exception as error:  # a failed upload must never kill a training run
            print(f"[checkpoint] Hub push skipped: {type(error).__name__}: {error}", flush=True)

    # ----------------------------------------------------------------- loading --

    def load(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler=None,
        scaler=None,
        map_location: str = "cpu",
        strict: bool = True,
    ) -> TrainingState | None:
        """
        Restore the newest usable checkpoint, falling back through older ones if the newest is damaged.

        Returns the :class:`TrainingState` to continue from, or None when there is nothing to resume.
        """
        for path in self._candidates():
            try:
                payload = torch.load(path, map_location=map_location, weights_only=False)
            except Exception as error:
                print(f"[checkpoint] {path.name} unreadable ({type(error).__name__}), trying an older one", flush=True)
                continue
            model.load_state_dict(payload["model"], strict=strict)
            if optimizer is not None and payload.get("optimizer"):
                optimizer.load_state_dict(payload["optimizer"])
            if scheduler is not None and payload.get("scheduler"):
                scheduler.load_state_dict(payload["scheduler"])
            if scaler is not None and payload.get("scaler"):
                scaler.load_state_dict(payload["scaler"])
            _restore_rng(payload.get("rng", {}))
            state = TrainingState(**payload["state"])
            self._last_save = time.time()
            print(f"[checkpoint] resumed {path.name} at step {state.step} (epoch {state.epoch})", flush=True)
            return state
        return None

    def _candidates(self) -> list[Path]:
        latest = self.directory / self.LATEST
        numbered = sorted(self.directory.glob("step_*.pt"), reverse=True)
        return ([latest] if latest.exists() else []) + numbered

    def fetch_from_hub(self) -> bool:
        """Pull ``latest.pt`` from the Hub when the local directory is empty -- a fresh VM, same run."""
        if not self.repo_id or (self.directory / self.LATEST).exists():
            return False
        try:
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(repo_id=self.repo_id, filename="latest.pt", repo_type="model")
            shutil.copy2(downloaded, self.directory / self.LATEST)
            print(f"[checkpoint] pulled latest.pt from {self.repo_id}", flush=True)
            return True
        except Exception as error:
            print(f"[checkpoint] nothing to pull from the Hub ({type(error).__name__})", flush=True)
            return False
