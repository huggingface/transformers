"""Resolve model types to local config/model classes without checkpoint loading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ResolvedArchitecture:
    model_type: str
    config: Any
    model: Any
    entrypoints: dict[str, Any]
    warnings: list[str]


def resolve_architecture(model_type: str) -> ResolvedArchitecture:
    try:
        import torch

        from transformers.initialization import no_init_weights
        from transformers.models.auto.configuration_auto import AutoConfig
        from transformers.models.auto.modeling_auto import AutoModel
    except Exception as error:
        raise RuntimeError(
            "Architecture IR generation requires a local Transformers development environment with torch available."
        ) from error

    config = AutoConfig.for_model(model_type)

    with torch.device("meta"), no_init_weights():
        model = AutoModel.from_config(config)

    return ResolvedArchitecture(
        model_type=model_type,
        config=config,
        model=model,
        entrypoints={
            "config_class": {
                "name": config.__class__.__name__,
                "module": config.__class__.__module__,
            },
            "model_class": {
                "name": model.__class__.__name__,
                "module": model.__class__.__module__,
            },
            "auto_config": "transformers.models.auto.configuration_auto.AutoConfig.for_model",
            "auto_model": "transformers.models.auto.modeling_auto.AutoModel.from_config",
        },
        warnings=[],
    )
