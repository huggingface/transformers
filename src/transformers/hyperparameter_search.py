from typing import Callable, Dict, Type

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


class HyperParamSearchBackendBase:
    name: HPSearchBackend
    pip_package: str = None
    is_available: Callable[[], bool]
    run: Callable
    default_hp_space: Callable

    def ensure_available(self):
        if not self.is_available():
            raise RuntimeError(
                f"You picked the {self.name} backend, but it is not installed. Run {self.pip_install()}."
            )

    @classmethod
    def pip_install(cls):
        return f"`pip install {cls.pip_package or cls.name}`"


class OptunaBackend(HyperParamSearchBackendBase):
    is_available = staticmethod(is_optuna_available)
    run = staticmethod(run_hp_search_optuna)
    default_hp_space = staticmethod(default_hp_space_optuna)


class RayTuneBackend(HyperParamSearchBackendBase):
    pip_package = "'ray[tune]'"
    is_available = staticmethod(is_ray_available)
    run = staticmethod(run_hp_search_ray)
    default_hp_space = staticmethod(default_hp_space_ray)


class SigOptBackend(HyperParamSearchBackendBase):
    is_available = staticmethod(is_sigopt_available)
    run = staticmethod(run_hp_search_sigopt)
    default_hp_space = staticmethod(default_hp_space_sigopt)


class WandbBackend(HyperParamSearchBackendBase):
    is_available = staticmethod(is_wandb_available)
    run = staticmethod(run_hp_search_wandb)
    default_hp_space = staticmethod(default_hp_space_wandb)


all_backends: Dict[HPSearchBackend, Type[HyperParamSearchBackendBase]] = {
    HPSearchBackend.OPTUNA: OptunaBackend,
    HPSearchBackend.RAY: RayTuneBackend,
    HPSearchBackend.SIGOPT: SigOptBackend,
    HPSearchBackend.WANDB: WandbBackend,
}

assert list(all_backends) == list(HPSearchBackend)

for _name, _backend in all_backends.items():
    _backend.name = _name


def default_hp_search_backend() -> str:
    available_backends = [backend for backend in all_backends.values() if backend.is_available()]
    if available_backends:
        # TODO warn if len(available_backends) > 1 ?
        return available_backends[0].name
    raise RuntimeError(
        "No hyperparameter search backend available.\n"
        + "\n".join(f" - To install {backend.name} run {backend.pip_install()}" for backend in all_backends.values())
    )
