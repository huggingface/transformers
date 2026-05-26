# coding=utf-8
# Copyright 2026 The HuggingFace Inc. team and the Rio2 contributors.
# Licensed under the Apache License, Version 2.0.
"""Processor for Rio2.

The policy mostly reuses the MolmoAct2 processor for image+instruction inputs and
keeps robot state tensors as-is. This lightweight processor delegates image/text
preprocessing to a wrapped base processor when provided.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ...processing_utils import ProcessorMixin
from ...utils import logging


logger = logging.get_logger(__name__)


class Rio2Processor(ProcessorMixin):
    attributes = []
    optional_attributes = []

    def __init__(self, base_processor=None, base_model_id: Optional[str] = None, **kwargs):
        self.base_processor = base_processor
        self.base_model_id = base_model_id
        self.chat_template = kwargs.pop("chat_template", None)

    @classmethod
    def from_base_model_id(cls, base_model_id: str, **kwargs):
        from transformers import AutoProcessor

        base_processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True, **kwargs)
        return cls(base_processor=base_processor, base_model_id=base_model_id)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        path = Path(pretrained_model_name_or_path)
        base_model_id = kwargs.pop("base_model_id", None)
        load_base_processor = bool(kwargs.pop("load_base_processor", False))
        hub_kwargs = {
            key: kwargs.get(key)
            for key in ["cache_dir", "force_download", "proxies", "token", "revision", "local_files_only", "subfolder"]
            if key in kwargs
        }
        if path.exists():
            cfg_path = path / "processor_config.json"
            model_cfg_path = path / "config.json"
            if cfg_path.exists():
                data = json.loads(cfg_path.read_text(encoding="utf-8"))
                base_model_id = base_model_id or data.get("base_model_id")
            if base_model_id is None and model_cfg_path.exists():
                data = json.loads(model_cfg_path.read_text(encoding="utf-8"))
                base_model_id = data.get("base_model_id")
        else:
            try:
                from transformers.utils import cached_file

                cfg_file = cached_file(pretrained_model_name_or_path, "processor_config.json", **hub_kwargs)
                if cfg_file:
                    data = json.loads(Path(cfg_file).read_text(encoding="utf-8"))
                    base_model_id = base_model_id or data.get("base_model_id")
            except Exception:
                pass
            if base_model_id is None:
                try:
                    from transformers.utils import cached_file

                    cfg_file = cached_file(pretrained_model_name_or_path, "config.json", **hub_kwargs)
                    if cfg_file:
                        data = json.loads(Path(cfg_file).read_text(encoding="utf-8"))
                        base_model_id = data.get("base_model_id")
                except Exception:
                    pass

        base_processor = None
        if base_model_id and load_base_processor:
            try:
                from transformers import AutoProcessor

                trust_remote_code = kwargs.pop("trust_remote_code", True)
                base_processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=trust_remote_code, **kwargs)
            except Exception as exc:
                logger.warning("Could not load base processor %s: %s", base_model_id, exc)
        return cls(base_processor=base_processor, base_model_id=base_model_id)

    def save_pretrained(self, save_directory, **kwargs):
        out = Path(save_directory)
        out.mkdir(parents=True, exist_ok=True)
        data = {
            "processor_class": self.__class__.__name__,
            "base_model_id": self.base_model_id,
            "auto_map": {"AutoProcessor": "processing_rio2.Rio2Processor"},
        }
        (out / "processor_config.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        if self.base_processor is not None and kwargs.pop("save_base_processor", False):
            base_dir = out / "base_processor"
            self.base_processor.save_pretrained(base_dir)
        return [str(out / "processor_config.json")]

    def __call__(
        self,
        images=None,
        instruction: Optional[str] = None,
        state: Optional[torch.Tensor] = None,
        state_history: Optional[torch.Tensor] = None,
        action_history: Optional[torch.Tensor] = None,
        target_actions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.base_processor is not None and images is not None and instruction is not None:
            out.update(self.base_processor(images=images, text=instruction, return_tensors="pt", **kwargs))
        else:
            if images is not None:
                out["images"] = images
            if instruction is not None:
                out["instruction"] = instruction
        if state is not None:
            out["state"] = state
        if state_history is not None:
            out["state_history"] = state_history
        if action_history is not None:
            out["action_history"] = action_history
        if target_actions is not None:
            out["target_actions"] = target_actions
        return out


__all__ = ["Rio2Processor"]
