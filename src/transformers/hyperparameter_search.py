from abc import ABC, abstractmethod
from typing import Dict

from .integrations import (
    is_optuna_available,
    is_ray_available,
    is_sigopt_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,
    run_hp_search_sigopt,
    run_hp_search_wandb,
)
from .trainer_utils import (
    HPSearchBackend,
    default_hp_space_optuna,
    default_hp_space_ray,
    default_hp_space_sigopt,
    default_hp_space_wandb,
)


class HyperParamSearchBackendBase(ABC):
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def is_available(self):
        raise NotImplementedError

    @abstractmethod
    def run(self, trainer, n_trials: int, direction: str, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def default_hp_space(self, trial):
        raise NotImplementedError

    def ensure_available(self):
        if not self.is_available():
            raise RuntimeError(
                f"You picked the {self.name()} backend, but it is not installed. "
                f"Use `pip install {self.pip_install}`."
            )

    def pip_install(self):
        return self.name()


class OptunaBackend(HyperParamSearchBackendBase):
    def name(self) -> str:
        return "optuna"

    def is_available(self):
        return is_optuna_available()

    def run(self, trainer, n_trials: int, direction: str, **kwargs):
        return run_hp_search_optuna(trainer, n_trials, direction, **kwargs)

    def default_hp_space(self, trial):
        return default_hp_space_optuna(trial)


class RayTuneBackend(HyperParamSearchBackendBase):
    def name(self) -> str:
        return "ray"

    def pip_install(self):
        return "'ray[tune]'"

    def is_available(self):
        return is_ray_available()

    def run(self, trainer, n_trials: int, direction: str, **kwargs):
        return run_hp_search_ray(trainer, n_trials, direction, **kwargs)

    def default_hp_space(self, trial):
        return default_hp_space_ray(trial)


class SigOptBackend(HyperParamSearchBackendBase):
    def name(self) -> str:
        return "sigopt"

    def is_available(self):
        return is_sigopt_available()

    def run(self, trainer, n_trials: int, direction: str, **kwargs):
        return run_hp_search_sigopt(trainer, n_trials, direction, **kwargs)

    def default_hp_space(self, trial):
        return default_hp_space_sigopt(trial)


class WandbBackend(HyperParamSearchBackendBase):
    def name(self) -> str:
        return "wandb"

    def is_available(self):
        return is_wandb_available()

    def run(self, trainer, n_trials: int, direction: str, **kwargs):
        return run_hp_search_wandb(trainer, n_trials, direction, **kwargs)

    def default_hp_space(self, trial):
        return default_hp_space_wandb(trial)


all_backends: Dict[str, HyperParamSearchBackendBase] = {
    backend.name(): backend for backend in [OptunaBackend(), RayTuneBackend(), SigOptBackend(), WandbBackend()]
}

assert list(all_backends) == list(HPSearchBackend)


def default_hp_search_backend() -> str:
    available_backends = [backend for backend in all_backends.values() if backend.is_available()]
    if available_backends:
        # TODO warn if len(available_backends) > 1 ?
        return available_backends[0].name()
    raise RuntimeError(
        "No hyperparameter search backend available.\n"
        + "\n".join(
            f" - To install {backend.name()} run `pip install {backend.pip_install()}`"
            for backend in all_backends.values()
        )
    )
