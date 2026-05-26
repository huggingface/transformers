# coding=utf-8
# Copyright 2026 The HuggingFace Inc. team and the Rio2 contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""PyTorch RIO-2 model.

This upstream-style module keeps the RIO-2 Transformers registration while
switching the default implementation to a *weight-preserved* MolmoAct2 path.
The original `allenai/MolmoAct2-SO100_101` weights remain the primary S2/S1
source. RIO-2 adds only a two-rate runtime, optional residual adapters, and
small tuning utilities.

Runtime modes:
  - `refresh_s2(images, instruction)`: low-frequency context refresh.
  - `act_fast(state, ...)`: high-frequency action generation.
  - `forward(..., s2_tokens=...)`: cached-token fallback used for tests and
    for adapter-only training when MolmoAct2 internals are unavailable.
"""

from __future__ import annotations

import copy
import inspect
import math
import time
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_rio2 import Rio2Config


logger = logging.get_logger(__name__)

ImageLike = Union[Image.Image, np.ndarray, torch.Tensor]


@dataclass
class Rio2Output(ModelOutput):
    """Output type for RIO-2."""

    loss: Optional[torch.FloatTensor] = None
    actions: Optional[torch.FloatTensor] = None
    s2_tokens: Optional[torch.FloatTensor] = None
    loss_flow_mse: Optional[torch.FloatTensor] = None
    loss_flow_l1: Optional[torch.FloatTensor] = None
    loss_diffusion: Optional[torch.FloatTensor] = None
    loss_consistency: Optional[torch.FloatTensor] = None
    loss_smooth: Optional[torch.FloatTensor] = None
    loss_jepa: Optional[torch.FloatTensor] = None
    loss_jepa_prior: Optional[torch.FloatTensor] = None
    pred_action_latent: Optional[torch.FloatTensor] = None
    target_action_latent: Optional[torch.FloatTensor] = None
    runtime_path: Optional[str] = None


def _torch_dtype_from_string(dtype_name: str) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return table.get(str(dtype_name).lower(), torch.bfloat16)


def _to_pil_list(images: Union[ImageLike, List[ImageLike], Tuple[ImageLike, ...]]) -> List[Image.Image]:
    if isinstance(images, (list, tuple)):
        return [_to_pil_list(x)[0] for x in images]
    if isinstance(images, Image.Image):
        return [images.convert("RGB")]
    if isinstance(images, np.ndarray):
        arr = images
        if arr.ndim == 4:
            return [_to_pil_list(a)[0] for a in arr]
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 1) if arr.max() <= 1.5 else np.clip(arr, 0, 255)
            arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.5 else arr.astype(np.uint8)
        return [Image.fromarray(arr).convert("RGB")]
    if torch.is_tensor(images):
        x = images.detach().cpu()
        if x.ndim == 4:
            return [_to_pil_list(xx)[0] for xx in x]
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = x.permute(1, 2, 0)
        arr = x.numpy()
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 1) if arr.max() <= 1.5 else np.clip(arr, 0, 255)
            arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.5 else arr.astype(np.uint8)
        return [Image.fromarray(arr).convert("RGB")]
    raise TypeError(f"Unsupported image type: {type(images)}")


def _move_to_device(batch: Any, device: torch.device, dtype: Optional[torch.dtype] = None) -> Any:
    if torch.is_tensor(batch):
        if batch.is_floating_point() and dtype is not None:
            return batch.to(device=device, dtype=dtype)
        return batch.to(device=device)
    if isinstance(batch, dict):
        return {k: _move_to_device(v, device, dtype) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(_move_to_device(v, device, dtype) for v in batch)
    return batch


def _first_existing_attr(obj: Any, names: Iterable[str]) -> Optional[Any]:
    for name in names:
        cur = obj
        ok = True
        for part in name.split("."):
            if hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                ok = False
                break
        if ok:
            return cur
    return None


def _safe_signature_accepts(fn: Any, name: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return True
    if name in sig.parameters:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


class Rio2RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(input_dtype)


class Rio2SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim == 0:
            timesteps = timesteps[None]
        half_dim = self.dim // 2
        freqs = torch.exp(
            torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
            * -(math.log(10000.0) / max(half_dim - 1, 1))
        )
        args = timesteps.float()[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.mlp(emb.to(dtype=self.mlp[0].weight.dtype))


class Rio2S1MoEResidualExpert(nn.Module):
    def __init__(self, width: int, hidden_dim: int, flat_action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, flat_action_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)


class Rio2S1MoEResidualBank(nn.Module):
    def __init__(self, config: Rio2Config, width: int):
        super().__init__()
        self.config = config
        self.flat_action_dim = int(config.action_horizon * config.action_dim)
        self.num_experts = int(config.s1_moe_num_experts)
        self.top_k = max(1, min(int(config.s1_moe_top_k), self.num_experts))
        hidden_dim = int(config.s1_moe_expert_hidden_dim)
        self.router = nn.Linear(width, self.num_experts)
        self.experts = nn.ModuleList(
            Rio2S1MoEResidualExpert(width, hidden_dim, self.flat_action_dim)
            for _ in range(self.num_experts)
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        logits = self.router(context)
        weights, indices = torch.topk(logits, k=self.top_k, dim=-1)
        weights = torch.softmax(weights, dim=-1).to(dtype=context.dtype)
        out = context.new_zeros(context.shape[0], self.flat_action_dim)
        for slot in range(self.top_k):
            slot_indices = indices[:, slot]
            slot_weights = weights[:, slot]
            for expert_id, expert in enumerate(self.experts):
                mask = slot_indices == expert_id
                if not bool(mask.any()):
                    continue
                out[mask] = out[mask] + slot_weights[mask, None] * expert(context[mask])
        return out.view(context.shape[0], self.config.action_horizon, self.config.action_dim)


class Rio2S2ContextCompressor(nn.Module):
    """Fallback compressor for cached-token training.

    Weight-preserved inference prefers the original MolmoAct2 S2/S1 bridge.
    This compressor remains useful for small adapter training, tests, and for
    base versions whose action expert cannot be split cleanly.
    """

    def __init__(self, config: Rio2Config):
        super().__init__()
        self.config = config
        self.in_proj = nn.Linear(config.s2_input_width, config.s2_width)
        self.query = nn.Parameter(torch.randn(config.s2_token_count, config.s2_width) / math.sqrt(config.s2_width))
        layer = nn.TransformerEncoderLayer(
            d_model=config.s2_width,
            nhead=max(1, min(8, config.s2_width // 64)),
            dim_feedforward=config.s2_width * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.refiner = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = Rio2RMSNorm(config.s2_width)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context.shape[-1] != self.config.s2_input_width:
            raise ValueError(
                f"S2 context width mismatch: got {context.shape[-1]}, expected {self.config.s2_input_width}."
            )
        hidden_states = self.in_proj(context)
        query = self.query.unsqueeze(0).expand(hidden_states.shape[0], -1, -1)
        scores = (query @ hidden_states.transpose(-1, -2)) / math.sqrt(hidden_states.shape[-1])
        attn = torch.softmax(scores, dim=-1)
        tokens = attn @ hidden_states
        tokens = self.refiner(tokens)
        return self.norm(tokens)


class Rio2MolmoAct2Core(nn.Module):
    """Weight-preserved wrapper around `allenai/MolmoAct2-SO100_101`.

    The original MolmoAct2 object is loaded once and kept as the source of truth
    for both S2 and S1. `refresh_s2()` extracts cache/context when possible;
    `act_original()` first tries a split action-expert call and falls back to
    `base.predict_action()` for exact original behavior.
    """

    VLM_CANDIDATES = (
        "vlm",
        "language_model",
        "molmo",
        "backbone",
        "model",
        "text_model",
    )
    ACTION_CANDIDATES = (
        "action_expert",
        "flow_head",
        "action_head",
        "continuous_action_expert",
        "flow_matching_head",
        "policy_head",
        "robot_action_head",
    )

    def __init__(self, config: Rio2Config):
        super().__init__()
        self.config = config
        self.base = None
        self.processor = None
        self.s2_module = None
        self.s1_module = None
        self.compressor = Rio2S2ContextCompressor(config)
        self.last_pil_images: Optional[List[Image.Image]] = None
        self.last_instruction: Optional[str] = None
        self.last_base_outputs: Optional[Any] = None
        self.last_s2_cache: Optional[Any] = None
        self.last_compact_tokens: Optional[torch.Tensor] = None
        self.last_refresh_time: float = 0.0
        self.last_runtime_path: str = "uninitialized"

    @property
    def base_device(self) -> torch.device:
        if self.base is None:
            return next(self.compressor.parameters()).device
        return next(self.base.parameters()).device

    def load_base(self, device: Optional[Union[str, torch.device]] = None, device_map: Optional[str] = None):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = _torch_dtype_from_string(self.config.torch_dtype)
        self.processor = AutoProcessor.from_pretrained(
            self.config.base_model_id,
            trust_remote_code=self.config.trust_remote_code,
        )
        kwargs = {"trust_remote_code": self.config.trust_remote_code, "dtype": dtype}
        if device_map is not None:
            kwargs["device_map"] = device_map
        try:
            self.base = AutoModelForImageTextToText.from_pretrained(self.config.base_model_id, **kwargs)
        except TypeError:
            kwargs.pop("dtype", None)
            kwargs["torch_dtype"] = dtype
            self.base = AutoModelForImageTextToText.from_pretrained(self.config.base_model_id, **kwargs)
        if device is not None and device_map is None:
            self.base.to(device)
        self.base.eval()
        self.s2_module = _first_existing_attr(self.base, self.VLM_CANDIDATES)
        self.s1_module = _first_existing_attr(self.base, self.ACTION_CANDIDATES)
        if self.s2_module is None:
            logger.warning("RIO-2 could not locate a named MolmoAct2 S2/VLM module; full base forward will be used.")
        if self.s1_module is None:
            logger.warning("RIO-2 could not locate a named MolmoAct2 action expert; predict_action fallback will be used.")
        return self

    def freeze_base(self):
        if self.base is not None:
            self.base.eval()
            for param in self.base.parameters():
                param.requires_grad = False

    def unfreeze_action_expert(self):
        if self.s1_module is None:
            return 0
        count = 0
        for param in self.s1_module.parameters():
            param.requires_grad = True
            count += param.numel()
        return count

    def unfreeze_adapters_only(self):
        self.freeze_base()
        for param in self.compressor.parameters():
            param.requires_grad = True

    def _extract_sequence_context(self, outputs: Any) -> Optional[torch.Tensor]:
        if outputs is None:
            return None
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            return outputs.hidden_states[-1]
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state
        if isinstance(outputs, dict):
            if outputs.get("hidden_states") is not None:
                return outputs["hidden_states"][-1]
            if outputs.get("last_hidden_state") is not None:
                return outputs["last_hidden_state"]
        if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
            chunks = []
            for layer in outputs.past_key_values:
                if isinstance(layer, (tuple, list)) and len(layer) >= 2:
                    key, value = layer[0], layer[1]
                    chunks.append(key.float().mean(dim=(-3, -2)))
                    chunks.append(value.float().mean(dim=(-3, -2)))
            if chunks:
                return torch.stack(chunks, dim=1).to(dtype=chunks[0].dtype)
        return None

    def _extract_cache(self, outputs: Any) -> Any:
        if outputs is None:
            return None
        for name in ("past_key_values", "kv_cache", "cache", "action_cache", "vlm_cache"):
            if hasattr(outputs, name) and getattr(outputs, name) is not None:
                return getattr(outputs, name)
            if isinstance(outputs, dict) and outputs.get(name) is not None:
                return outputs[name]
        return outputs

    @torch.no_grad()
    def refresh_s2(self, images: Union[ImageLike, List[ImageLike]], instruction: str, force: bool = False) -> torch.Tensor:
        if self.base is None or self.processor is None:
            raise RuntimeError("MolmoAct2 base is not loaded. Call model.load_s2_base() first.")
        age = time.time() - self.last_refresh_time
        if (
            not force
            and self.last_compact_tokens is not None
            and self.last_instruction == instruction
            and age < self.config.max_s2_cache_age_s
        ):
            return self.last_compact_tokens

        pil_images = _to_pil_list(images)
        inputs = self.processor(images=pil_images, text=instruction, return_tensors="pt")
        inputs = _move_to_device(inputs, self.base_device, _torch_dtype_from_string(self.config.torch_dtype))
        try:
            outputs = self.base(**inputs, use_cache=True, output_hidden_states=True, return_dict=True)
        except TypeError:
            outputs = self.base(**inputs, return_dict=True)

        self.last_base_outputs = outputs
        self.last_s2_cache = self._extract_cache(outputs)
        self.last_pil_images = pil_images
        self.last_instruction = instruction
        self.last_refresh_time = time.time()
        sequence_context = self._extract_sequence_context(outputs)
        if sequence_context is not None:
            sequence_context = sequence_context.to(
                device=next(self.compressor.parameters()).device,
                dtype=next(self.compressor.parameters()).dtype,
            )
            try:
                self.last_compact_tokens = self.compressor(sequence_context).detach()
            except Exception as exc:
                logger.warning("RIO-2 compact-token compression failed: %s", exc)
                self.last_compact_tokens = torch.zeros(
                    1,
                    self.config.s2_token_count,
                    self.config.s2_width,
                    device=next(self.compressor.parameters()).device,
                    dtype=next(self.compressor.parameters()).dtype,
                )
        else:
            self.last_compact_tokens = torch.zeros(
                1,
                self.config.s2_token_count,
                self.config.s2_width,
                device=next(self.compressor.parameters()).device,
                dtype=next(self.compressor.parameters()).dtype,
            )
        return self.last_compact_tokens

    def _try_split_action_expert(
        self,
        state: torch.Tensor,
        state_history: Optional[torch.Tensor],
        action_history: Optional[torch.Tensor],
        num_steps: int,
    ) -> Optional[torch.Tensor]:
        if not self.config.prefer_split_action_expert or self.s1_module is None:
            return None
        candidates = [self.s1_module]
        for method_name in ("predict_action", "sample", "generate_actions", "forward"):
            if hasattr(self.s1_module, method_name):
                candidates.append(getattr(self.s1_module, method_name))
        for fn in candidates:
            try:
                kwargs = {}
                if _safe_signature_accepts(fn, "state"):
                    kwargs["state"] = state
                if _safe_signature_accepts(fn, "states"):
                    kwargs["states"] = state
                if _safe_signature_accepts(fn, "vlm_kv_cache"):
                    kwargs["vlm_kv_cache"] = self.last_s2_cache
                if _safe_signature_accepts(fn, "past_key_values"):
                    kwargs["past_key_values"] = self.last_s2_cache
                if _safe_signature_accepts(fn, "s2_cache"):
                    kwargs["s2_cache"] = self.last_s2_cache
                if _safe_signature_accepts(fn, "state_history"):
                    kwargs["state_history"] = state_history
                if _safe_signature_accepts(fn, "action_history"):
                    kwargs["action_history"] = action_history
                if _safe_signature_accepts(fn, "num_steps"):
                    kwargs["num_steps"] = num_steps
                if _safe_signature_accepts(fn, "num_flow_steps"):
                    kwargs["num_flow_steps"] = num_steps
                out = fn(**kwargs) if kwargs else fn(state)
                actions = self._coerce_actions(out, state)
                if actions is not None:
                    self.last_runtime_path = "split_original_action_expert"
                    return actions
            except Exception as exc:
                logger.debug("RIO-2 split action expert attempt failed for %s: %s", fn, exc)
        return None

    def _coerce_actions(self, out: Any, state: torch.Tensor) -> Optional[torch.Tensor]:
        if out is None:
            return None
        if torch.is_tensor(out):
            actions = out
        elif hasattr(out, "actions"):
            actions = torch.as_tensor(out.actions, device=state.device)
        elif isinstance(out, dict) and out.get("actions") is not None:
            actions = torch.as_tensor(out["actions"], device=state.device)
        else:
            return None
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        return actions.to(device=state.device, dtype=state.dtype if state.is_floating_point() else torch.float32)

    @torch.no_grad()
    def predict_action_fallback(self, state: torch.Tensor, num_steps: int) -> torch.Tensor:
        if self.base is None or self.processor is None:
            raise RuntimeError("MolmoAct2 base is not loaded.")
        if self.last_pil_images is None or self.last_instruction is None:
            raise RuntimeError("S2 cache is empty. Call refresh_s2(images, instruction) first.")
        if not hasattr(self.base, "predict_action"):
            raise RuntimeError("MolmoAct2 base has no predict_action method and split action expert was unavailable.")
        state_np = state.detach().float().cpu().numpy()
        out = self.base.predict_action(
            processor=self.processor,
            images=self.last_pil_images,
            task=self.last_instruction,
            state=state_np,
            norm_tag=self.config.norm_tag,
            action_mode=self.config.action_mode,
            num_steps=num_steps,
        )
        actions = torch.as_tensor(out.actions, device=state.device, dtype=state.dtype if state.is_floating_point() else torch.float32)
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        self.last_runtime_path = "predict_action_fallback_exact"
        return actions

    @torch.no_grad()
    def act_original(
        self,
        state: torch.Tensor,
        state_history: Optional[torch.Tensor] = None,
        action_history: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        steps = int(num_steps or self.config.molmoact_num_steps)
        split_actions = self._try_split_action_expert(state, state_history, action_history, steps)
        if split_actions is not None:
            return split_actions
        if self.config.fallback_to_predict_action:
            return self.predict_action_fallback(state, steps)
        raise RuntimeError("No callable original S1/action path was found and fallback_to_predict_action=False.")


class Rio2FastS1FlowActionExpert(nn.Module):
    """Small fallback S1 for cached-token training.

    In weight-preserved RIO-2, this is not the preferred runtime path. It remains
    as an adapter/student fallback and for upstream tests without downloading
    MolmoAct2.
    """

    def __init__(self, config: Rio2Config):
        super().__init__()
        self.config = config
        width = config.s1_width
        self.s2_proj = nn.Linear(config.s2_width, width)
        self.state_proj = nn.Linear(config.state_dim, width)
        self.state_hist_proj = nn.Linear(config.state_dim, width)
        self.action_hist_proj = nn.Linear(config.action_dim, width)
        self.noisy_action_proj = nn.Linear(config.action_dim, width)
        self.time_emb = Rio2SinusoidalTimeEmbedding(width)
        self.type_emb = nn.Parameter(torch.randn(5, width) / math.sqrt(width))
        self.memory_proj = nn.Linear(config.s2_width, width)
        self.memory_type_emb = nn.Parameter(torch.randn(1, width) / math.sqrt(width))
        self.memory_gate = nn.Parameter(torch.tensor(-2.0))
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=config.s1_heads,
            dim_feedforward=width * 4,
            dropout=config.s1_dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=config.s1_layers)
        self.norm = Rio2RMSNorm(width)
        self.action_head = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, config.action_dim))
        self.noise_head = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, config.action_dim))

        hidden = int(config.jepa_hidden_dim)
        latent = int(config.jepa_latent_dim)
        self.jepa_s2_proj = nn.Linear(config.s2_width, hidden)
        self.jepa_memory_proj = nn.Linear(config.s2_width, hidden)
        self.jepa_state_proj = nn.Linear(config.state_dim, hidden)
        self.jepa_action_hist_proj = nn.Linear(config.action_dim, hidden)
        self.jepa_norm = Rio2RMSNorm(hidden)
        self.jepa_predictor = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, latent))
        flat_action_dim = config.action_horizon * config.action_dim
        self.action_encoder = nn.Sequential(nn.Linear(flat_action_dim, hidden), nn.SiLU(), nn.Linear(hidden, latent))
        self.target_action_encoder = copy.deepcopy(self.action_encoder)
        for param in self.target_action_encoder.parameters():
            param.requires_grad = False
        self.jepa_to_action_prior = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.SiLU(),
            nn.Linear(hidden, flat_action_dim),
        )
        self.consistency_head = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.SiLU(),
            nn.Linear(hidden, flat_action_dim),
        )
        self.jepa_condition_proj = nn.Linear(latent, width)
        nn.init.zeros_(self.jepa_to_action_prior[-1].weight)
        nn.init.zeros_(self.jepa_to_action_prior[-1].bias)
        nn.init.zeros_(self.jepa_condition_proj.weight)
        nn.init.zeros_(self.jepa_condition_proj.bias)
        self.moe_residual = Rio2S1MoEResidualBank(config, width) if bool(config.enable_s1_moe) else None

    def default_task_memory_from_s2(self, s2_tokens):
        if s2_tokens.ndim == 2:
            s2_tokens = s2_tokens.unsqueeze(0)
        batch_size, token_count, width = s2_tokens.shape
        slots = max(1, int(self.config.task_memory_slots))
        if token_count >= slots:
            return s2_tokens[:, :slots]
        pad_value = s2_tokens.mean(dim=1, keepdim=True).expand(batch_size, slots - token_count, width)
        return torch.cat([s2_tokens, pad_value], dim=1)

    def _prepare_task_memory(self, task_memory, s2_tokens, batch_size, device, dtype):
        if not bool(self.config.task_memory_enabled):
            return None
        if task_memory is None:
            task_memory = self.default_task_memory_from_s2(s2_tokens)
        if task_memory.ndim == 2:
            task_memory = task_memory.unsqueeze(0)
        task_memory = task_memory.to(device=device, dtype=dtype)
        if task_memory.shape[0] == 1 and batch_size > 1:
            task_memory = task_memory.expand(batch_size, -1, -1)
        elif task_memory.shape[0] != batch_size:
            task_memory = task_memory[:1].expand(batch_size, -1, -1)
        slots = max(1, int(self.config.task_memory_slots))
        if task_memory.shape[1] < slots:
            pad = task_memory.mean(dim=1, keepdim=True).expand(batch_size, slots - task_memory.shape[1], task_memory.shape[2])
            task_memory = torch.cat([task_memory, pad], dim=1)
        return task_memory[:, :slots]

    def _prepare_hist(self, values, length, dim, batch_size, device, dtype):
        if values is None:
            return torch.zeros(batch_size, length, dim, device=device, dtype=dtype)
        if values.ndim == 2:
            values = values.unsqueeze(0)
        values = values.to(device=device, dtype=dtype)
        if values.shape[1] < length:
            pad = torch.zeros(values.shape[0], length - values.shape[1], values.shape[2], device=device, dtype=dtype)
            values = torch.cat([pad, values], dim=1)
        return values[:, -length:]

    def _decode(self, s2_tokens, state, state_history, action_history, noisy_actions, timesteps, head, jepa_latent=None, task_memory=None):
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if noisy_actions.ndim == 2:
            noisy_actions = noisy_actions.unsqueeze(0)
        if s2_tokens.ndim == 2:
            s2_tokens = s2_tokens.unsqueeze(0)
        batch_size = state.shape[0]
        device = state.device
        dtype = state.dtype if state.is_floating_point() else torch.float32
        state = state.to(device=device, dtype=dtype)
        noisy_actions = noisy_actions.to(device=device, dtype=dtype)
        s2_tokens = s2_tokens.to(device=device, dtype=dtype)
        state_history = self._prepare_hist(state_history, self.config.state_history_len, self.config.state_dim, batch_size, device, dtype)
        action_history = self._prepare_hist(action_history, self.config.action_history_len, self.config.action_dim, batch_size, device, dtype)
        task_memory = self._prepare_task_memory(task_memory, s2_tokens, batch_size, device, dtype)
        s2_tok = self.s2_proj(s2_tokens) + self.type_emb[0]
        token_chunks = [s2_tok]
        if task_memory is not None:
            gate = torch.sigmoid(self.memory_gate).to(dtype=s2_tok.dtype)
            mem_tok = gate * float(self.config.task_memory_alpha) * self.memory_proj(task_memory) + self.memory_type_emb
            token_chunks.append(mem_tok)
        state_tok = self.state_proj(state).unsqueeze(1) + self.type_emb[1]
        state_hist_tok = self.state_hist_proj(state_history) + self.type_emb[2]
        action_hist_tok = self.action_hist_proj(action_history) + self.type_emb[3]
        action_tok = self.noisy_action_proj(noisy_actions) + self.type_emb[4]
        action_tok = action_tok + self.time_emb(timesteps).unsqueeze(1)
        if jepa_latent is not None and bool(self.config.enable_jepa_diffusion):
            cond = self.jepa_condition_proj(jepa_latent.to(device=device, dtype=dtype)).unsqueeze(1)
            action_tok = action_tok + float(self.config.jepa_condition_alpha) * cond.to(dtype=action_tok.dtype)
        token_chunks.extend([state_tok, state_hist_tok, action_hist_tok, action_tok])
        tokens = torch.cat(token_chunks, dim=1)
        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)
        return head(tokens[:, -self.config.action_horizon :])

    def velocity(self, s2_tokens, state, state_history, action_history, noisy_actions, timesteps, jepa_latent=None, task_memory=None):
        return self._decode(s2_tokens, state, state_history, action_history, noisy_actions, timesteps, self.action_head, jepa_latent, task_memory)

    def diffusion_noise(self, s2_tokens, state, state_history, action_history, noisy_actions, timesteps, jepa_latent=None, task_memory=None):
        return self._decode(s2_tokens, state, state_history, action_history, noisy_actions, timesteps, self.noise_head, jepa_latent, task_memory)

    def predict_action_latent(self, s2_tokens, state, action_history=None, task_memory=None):
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if s2_tokens.ndim == 2:
            s2_tokens = s2_tokens.unsqueeze(0)
        batch_size = state.shape[0]
        device = state.device
        dtype = state.dtype if state.is_floating_point() else torch.float32
        action_history = self._prepare_hist(action_history, self.config.action_history_len, self.config.action_dim, batch_size, device, dtype)
        s2_summary = s2_tokens.to(device=device, dtype=dtype).mean(dim=1)
        task_memory = self._prepare_task_memory(task_memory, s2_tokens, batch_size, device, dtype)
        memory_summary = torch.zeros_like(s2_summary) if task_memory is None else task_memory.mean(dim=1)
        hist_summary = action_history.mean(dim=1)
        memory_scale = torch.sigmoid(self.memory_gate).to(dtype=s2_summary.dtype) * float(self.config.task_memory_alpha)
        context = (
            self.jepa_s2_proj(s2_summary)
            + memory_scale * self.jepa_memory_proj(memory_summary)
            + self.jepa_state_proj(state.to(dtype=dtype))
            + self.jepa_action_hist_proj(hist_summary)
        )
        return self.jepa_predictor(self.jepa_norm(context))

    def moe_action_residual(self, s2_tokens, state, action_history=None, task_memory=None):
        if self.moe_residual is None:
            return None
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if s2_tokens.ndim == 2:
            s2_tokens = s2_tokens.unsqueeze(0)
        batch_size = state.shape[0]
        device = state.device
        dtype = state.dtype if state.is_floating_point() else torch.float32
        action_history = self._prepare_hist(action_history, self.config.action_history_len, self.config.action_dim, batch_size, device, dtype)
        s2_tokens = s2_tokens.to(device=device, dtype=dtype)
        task_memory = self._prepare_task_memory(task_memory, s2_tokens, batch_size, device, dtype)
        context = (
            self.s2_proj(s2_tokens).mean(dim=1)
            + self.state_proj(state.to(dtype=dtype))
            + self.action_hist_proj(action_history).mean(dim=1)
        )
        if task_memory is not None:
            gate = torch.sigmoid(self.memory_gate).to(dtype=context.dtype)
            context = context + gate * float(self.config.task_memory_alpha) * self.memory_proj(task_memory).mean(dim=1)
        return self.moe_residual(context).to(dtype=dtype)

    def encode_action_latent(self, actions, target=False):
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        flat = actions.reshape(actions.shape[0], -1)
        encoder = self.target_action_encoder if target else self.action_encoder
        return F.normalize(encoder(flat).float(), dim=-1).to(dtype=flat.dtype)

    def action_prior_from_latent(self, latent, dtype):
        prior = self.jepa_to_action_prior(latent).view(latent.shape[0], self.config.action_horizon, self.config.action_dim)
        return prior.to(dtype=dtype)

    def consistency_action_from_latent(self, latent, dtype):
        actions = self.consistency_head(latent).view(latent.shape[0], self.config.action_horizon, self.config.action_dim)
        return actions.to(dtype=dtype)

    def jepa_diffusion_sample(self, s2_tokens, state, state_history=None, action_history=None, steps=None, task_memory=None):
        if state.ndim == 1:
            state = state.unsqueeze(0)
        batch_size = state.shape[0]
        dtype = state.dtype if state.is_floating_point() else torch.float32
        jepa_latent = self.predict_action_latent(s2_tokens, state, action_history, task_memory)
        x = self.consistency_action_from_latent(jepa_latent, dtype)
        if float(self.config.jepa_action_prior_alpha) != 0.0:
            x = x + float(self.config.jepa_action_prior_alpha) * self.action_prior_from_latent(jepa_latent, dtype)
        moe_residual = self.moe_action_residual(s2_tokens, state, action_history, task_memory)
        if moe_residual is not None:
            x = x + float(self.config.s1_moe_residual_scale) * moe_residual
        denoise_steps = int(steps if steps is not None else self.config.diffusion_inference_steps)
        denoise_steps = max(0, denoise_steps)
        if denoise_steps > 0:
            x = x + torch.randn_like(x) * float(self.config.s1_sampling_noise_scale) / float(denoise_steps + 1)
        for i in range(denoise_steps):
            frac = float(denoise_steps - i) / float(max(denoise_steps, 1))
            timesteps = torch.full((batch_size,), frac, device=state.device, dtype=dtype)
            eps = self.diffusion_noise(s2_tokens, state, state_history, action_history, x, timesteps, jepa_latent, task_memory)
            x = x - eps / float(denoise_steps + 1)
        return x

    @torch.no_grad()
    def update_target_encoder(self, decay=None):
        decay = float(self.config.jepa_ema_decay if decay is None else decay)
        for online, target in zip(self.action_encoder.parameters(), self.target_action_encoder.parameters()):
            target.data.mul_(decay).add_(online.data, alpha=1.0 - decay)

    def freeze_target_encoder(self):
        for param in self.target_action_encoder.parameters():
            param.requires_grad = False

    def training_loss(self, s2_tokens, state, state_history, action_history, target_actions, task_memory=None):
        if target_actions.ndim == 2:
            target_actions = target_actions.unsqueeze(0)
        batch_size = target_actions.shape[0]
        jepa_latent = self.predict_action_latent(s2_tokens, state, action_history, task_memory) if bool(self.config.enable_jepa_diffusion) else None
        x0 = torch.randn_like(target_actions)
        x1 = target_actions
        timesteps = torch.rand(batch_size, device=target_actions.device, dtype=target_actions.dtype)
        xt = (1.0 - timesteps[:, None, None]) * x0 + timesteps[:, None, None] * x1
        target_velocity = x1 - x0
        pred_velocity = self.velocity(s2_tokens, state, state_history, action_history, xt, timesteps, jepa_latent, task_memory)
        loss_flow_mse = F.mse_loss(pred_velocity, target_velocity)
        loss_flow_l1 = F.l1_loss(pred_velocity, target_velocity)
        if bool(self.config.enable_jepa_diffusion) and float(self.config.diffusion_loss_weight) > 0:
            diffusion_t = torch.rand(batch_size, device=target_actions.device, dtype=target_actions.dtype)
            eps = torch.randn_like(target_actions)
            alpha = torch.cos(diffusion_t[:, None, None] * (math.pi / 2.0))
            sigma = torch.sin(diffusion_t[:, None, None] * (math.pi / 2.0))
            noisy = alpha * target_actions + sigma * eps
            pred_eps = self.diffusion_noise(s2_tokens, state, state_history, action_history, noisy, diffusion_t, jepa_latent, task_memory)
            loss_diffusion = F.mse_loss(pred_eps, eps)
        else:
            loss_diffusion = target_actions.new_tensor(0.0)
        if bool(self.config.enable_jepa_diffusion) and float(self.config.jepa_loss_weight) > 0:
            pred_latent = F.normalize(jepa_latent.float(), dim=-1)
            with torch.no_grad():
                target_latent = self.encode_action_latent(target_actions, target=True).float()
            loss_jepa = F.mse_loss(pred_latent, target_latent)
        else:
            loss_jepa = target_actions.new_tensor(0.0)
        if bool(self.config.enable_jepa_diffusion) and float(self.config.jepa_action_prior_weight) > 0:
            prior_actions = self.action_prior_from_latent(jepa_latent, target_actions.dtype)
            loss_jepa_prior = F.mse_loss(prior_actions, target_actions)
        else:
            loss_jepa_prior = target_actions.new_tensor(0.0)
        if bool(self.config.enable_jepa_diffusion) and float(self.config.consistency_loss_weight) > 0:
            consistency_actions = self.consistency_action_from_latent(jepa_latent, target_actions.dtype)
            moe_residual = self.moe_action_residual(s2_tokens, state, action_history, task_memory)
            if moe_residual is not None:
                consistency_actions = consistency_actions + float(self.config.s1_moe_residual_scale) * moe_residual
            loss_consistency = F.mse_loss(consistency_actions, target_actions)
        else:
            loss_consistency = target_actions.new_tensor(0.0)
        loss_smooth = (target_actions[:, 1:] - target_actions[:, :-1]).pow(2).mean() if target_actions.shape[1] > 1 else target_actions.new_tensor(0.0)
        loss = (
            self.config.flow_loss_weight * (loss_flow_mse + self.config.action_l1_weight * loss_flow_l1)
            + self.config.smooth_loss_weight * loss_smooth
            + self.config.diffusion_loss_weight * loss_diffusion
            + self.config.consistency_loss_weight * loss_consistency
            + self.config.jepa_loss_weight * loss_jepa.to(loss_flow_mse.dtype)
            + self.config.jepa_action_prior_weight * loss_jepa_prior
        )
        return {
            "loss": loss,
            "loss_flow_mse": loss_flow_mse,
            "loss_flow_l1": loss_flow_l1,
            "loss_diffusion": loss_diffusion,
            "loss_consistency": loss_consistency,
            "loss_jepa": loss_jepa,
            "loss_jepa_prior": loss_jepa_prior,
            "loss_smooth": loss_smooth,
        }

    @torch.no_grad()
    def sample(self, s2_tokens, state, state_history=None, action_history=None, steps=None, task_memory=None):
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if self.config.s1_policy_mode == "jepa_diffusion" and bool(self.config.enable_jepa_diffusion):
            x = self.jepa_diffusion_sample(s2_tokens, state, state_history, action_history, steps=steps, task_memory=task_memory)
            if self.config.action_clip > 0:
                x = x.clamp(-self.config.action_clip, self.config.action_clip)
            return x
        batch_size = state.shape[0]
        steps = steps or self.config.flow_inference_steps
        dtype = state.dtype if state.is_floating_point() else torch.float32
        jepa_latent = self.predict_action_latent(s2_tokens, state, action_history, task_memory) if bool(self.config.enable_jepa_diffusion) else None
        x = torch.randn(batch_size, self.config.action_horizon, self.config.action_dim, device=state.device, dtype=dtype)
        x = x * float(self.config.s1_sampling_noise_scale)
        if jepa_latent is not None and float(self.config.jepa_action_prior_alpha) != 0.0:
            x = x + float(self.config.jepa_action_prior_alpha) * self.action_prior_from_latent(jepa_latent, dtype)
        moe_residual = self.moe_action_residual(s2_tokens, state, action_history, task_memory)
        if moe_residual is not None:
            x = x + float(self.config.s1_moe_residual_scale) * moe_residual
        for i in range(steps):
            timesteps = torch.full((batch_size,), float(i) / max(steps, 1), device=state.device, dtype=x.dtype)
            x = x + self.velocity(s2_tokens, state, state_history, action_history, x, timesteps, jepa_latent, task_memory) / float(steps)
        if self.config.action_clip > 0:
            x = x.clamp(-self.config.action_clip, self.config.action_clip)
        return x




class Rio2JepaS1ActionExpert(nn.Module):
    """JEPA-style S1 that preserves the online S1 policy weights.

    This module does **not** replace the original S1 policy with an unrelated
    world model. Instead it wraps the existing fast flow S1 as `online_s1` and
    adds a small latent prediction side objective:

    - online_s1: action generator; initialized and trained exactly like the
      existing RIO-2 S1 path, so old S1 checkpoints can be remapped into it.
    - jepa_context_encoder + predictor: predicts future action latent from
      S2 tokens, current state, and action history.
    - target_action_encoder: EMA target encoder for the future action chunk.
    - latent_to_action_delta: optional zero-initialized residual head.

    Inference defaults to the online S1 policy. JEPA affects actions only when
    `config.use_jepa_action_residual=True` and `config.jepa_action_alpha > 0`.
    """

    def __init__(self, config: Rio2Config):
        super().__init__()
        self.config = config
        self.online_s1 = Rio2FastS1FlowActionExpert(config)

        hidden = int(config.jepa_hidden_dim)
        latent = int(config.jepa_latent_dim)
        self.s2_jepa_proj = nn.Linear(config.s2_width, hidden)
        self.state_jepa_proj = nn.Linear(config.state_dim, hidden)
        self.action_hist_jepa_proj = nn.Linear(config.action_dim, hidden)
        self.type_emb = nn.Parameter(torch.randn(3, hidden) / math.sqrt(hidden))

        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=max(1, int(config.jepa_heads)),
            dim_feedforward=hidden * 4,
            dropout=config.s1_dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.jepa_context_encoder = nn.TransformerEncoder(layer, num_layers=max(1, int(config.jepa_layers)))
        self.jepa_norm = Rio2RMSNorm(hidden)
        self.jepa_predictor = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent),
        )

        flat_action_dim = config.action_horizon * config.action_dim
        self.action_encoder = nn.Sequential(
            nn.Linear(flat_action_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent),
        )
        self.target_action_encoder = copy.deepcopy(self.action_encoder)
        for param in self.target_action_encoder.parameters():
            param.requires_grad = False

        self.latent_to_action_delta = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.SiLU(),
            nn.Linear(hidden, flat_action_dim),
        )
        nn.init.zeros_(self.latent_to_action_delta[-1].weight)
        nn.init.zeros_(self.latent_to_action_delta[-1].bias)

    def _prepare_action_history(self, action_history, batch_size, device, dtype):
        if action_history is None:
            return torch.zeros(batch_size, self.config.action_history_len, self.config.action_dim, device=device, dtype=dtype)
        if action_history.ndim == 2:
            action_history = action_history.unsqueeze(0)
        action_history = action_history.to(device=device, dtype=dtype)
        if action_history.shape[1] < self.config.action_history_len:
            pad = torch.zeros(
                action_history.shape[0],
                self.config.action_history_len - action_history.shape[1],
                action_history.shape[2],
                device=device,
                dtype=dtype,
            )
            action_history = torch.cat([pad, action_history], dim=1)
        return action_history[:, -self.config.action_history_len :]

    def encode_context(self, s2_tokens, state, action_history=None):
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if s2_tokens.ndim == 2:
            s2_tokens = s2_tokens.unsqueeze(0)
        batch_size = state.shape[0]
        device = state.device
        dtype = state.dtype if state.is_floating_point() else torch.float32
        s2_tokens = s2_tokens.to(device=device, dtype=dtype)
        state = state.to(device=device, dtype=dtype)
        action_history = self._prepare_action_history(action_history, batch_size, device, dtype)

        s2_tok = self.s2_jepa_proj(s2_tokens) + self.type_emb[0]
        state_tok = self.state_jepa_proj(state).unsqueeze(1) + self.type_emb[1]
        hist_tok = self.action_hist_jepa_proj(action_history) + self.type_emb[2]
        tokens = torch.cat([s2_tok, state_tok, hist_tok], dim=1)
        hidden = self.jepa_context_encoder(tokens)
        hidden = self.jepa_norm(hidden)
        return hidden.mean(dim=1)

    def predict_action_latent(self, s2_tokens, state, action_history=None):
        context = self.encode_context(s2_tokens, state, action_history)
        return self.jepa_predictor(context)

    def encode_action_latent(self, actions: torch.Tensor, target: bool = False) -> torch.Tensor:
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        flat = actions.reshape(actions.shape[0], -1)
        encoder = self.target_action_encoder if target else self.action_encoder
        latent = encoder(flat)
        return F.normalize(latent.float(), dim=-1).to(dtype=flat.dtype)

    @torch.no_grad()
    def update_target_encoder(self, decay: Optional[float] = None):
        decay = float(self.config.jepa_ema_decay if decay is None else decay)
        for online, target in zip(self.action_encoder.parameters(), self.target_action_encoder.parameters()):
            target.data.mul_(decay).add_(online.data, alpha=1.0 - decay)

    def freeze_target_encoder(self):
        if hasattr(self.online_s1, "freeze_target_encoder"):
            self.online_s1.freeze_target_encoder()
        for param in self.target_action_encoder.parameters():
            param.requires_grad = False

    def training_loss(self, s2_tokens, state, state_history, action_history, target_actions, task_memory=None):
        base_losses = self.online_s1.training_loss(s2_tokens, state, state_history, action_history, target_actions, task_memory=task_memory)
        pred_latent = F.normalize(self.predict_action_latent(s2_tokens, state, action_history).float(), dim=-1)
        with torch.no_grad():
            target_latent = self.encode_action_latent(target_actions, target=True).float()
        loss_jepa = F.mse_loss(pred_latent, target_latent)
        loss = base_losses["loss"] + float(self.config.jepa_loss_weight) * loss_jepa.to(base_losses["loss"].dtype)
        return {
            **base_losses,
            "loss": loss,
            "loss_jepa": loss_jepa,
            "pred_action_latent": pred_latent,
            "target_action_latent": target_latent,
        }

    @torch.no_grad()
    def sample(self, s2_tokens, state, state_history=None, action_history=None, steps=None, task_memory=None):
        actions = self.online_s1.sample(s2_tokens, state, state_history, action_history, steps=steps, task_memory=task_memory)
        if bool(self.config.use_jepa_action_residual) and float(self.config.jepa_action_alpha) != 0.0:
            pred_latent = self.predict_action_latent(s2_tokens, state, action_history).to(actions.dtype)
            delta = self.latent_to_action_delta(pred_latent).view(
                actions.shape[0], self.config.action_horizon, self.config.action_dim
            )
            actions = actions + float(self.config.jepa_action_alpha) * delta
            if self.config.action_clip > 0:
                actions = actions.clamp(-self.config.action_clip, self.config.action_clip)
        return actions


class Rio2ResidualAdapter(nn.Module):
    """Tiny correction head. Initial output is zero when residual_alpha=0."""

    def __init__(self, config: Rio2Config):
        super().__init__()
        width = min(256, max(64, config.s1_width))
        self.net = nn.Sequential(
            nn.Linear(config.state_dim, width),
            nn.SiLU(),
            nn.Linear(width, config.action_horizon * config.action_dim),
        )
        self.config = config
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        delta = self.net(state).view(state.shape[0], self.config.action_horizon, self.config.action_dim)
        return delta


class Rio2PreTrainedModel(PreTrainedModel):
    config_class = Rio2Config
    base_model_prefix = "rio2"
    supports_gradient_checkpointing = False
    _no_split_modules = ["Rio2FastS1FlowActionExpert", "Rio2MolmoAct2Core"]

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, Rio2RMSNorm):
            module.weight.data.fill_(1.0)


class Rio2Model(Rio2PreTrainedModel):
    """RIO-2 weight-preserved SO101 policy integrated as a Transformers model."""

    def __init__(self, config: Rio2Config):
        super().__init__(config)
        self.molmoact = Rio2MolmoAct2Core(config)
        if bool(config.enable_jepa_s1):
            self.s1_student = Rio2JepaS1ActionExpert(config)
        else:
            self.s1_student = Rio2FastS1FlowActionExpert(config)
        self.residual_adapter = Rio2ResidualAdapter(config) if config.enable_residual_adapter else None
        self._s2_cache: Optional[torch.Tensor] = None
        self._s2_cache_time: float = 0.0
        self._cached_instruction: Optional[str] = None
        self._action_chunk_history: List[Tuple[torch.Tensor, int]] = []
        self._task_memory_cache: Optional[torch.Tensor] = None
        self.post_init()
        if config.load_base_on_init:
            logger.warning("config.load_base_on_init=True loads MolmoAct2 during construction; prefer load_s2_base().")
            self.load_s2_base()
        self.apply_finetuning_policy()

    @property
    def s2(self):
        """Backward-compatible alias without duplicate module registration."""
        return self.molmoact

    @property
    def s1(self):
        """Backward-compatible alias without duplicate module registration."""
        return self.s1_student

    def load_s2_base(self, device: Optional[Union[str, torch.device]] = None, device_map: Optional[str] = None):
        self.molmoact.load_base(device=device, device_map=device_map)
        self.apply_finetuning_policy()
        return self

    def freeze_s2_base(self):
        self.molmoact.freeze_base()
        return self

    @torch.no_grad()
    def reset_temporal_ensemble(self):
        self._action_chunk_history.clear()
        return self

    @torch.no_grad()
    def reset_task_memory(self):
        self._task_memory_cache = None
        return self

    @torch.no_grad()
    def update_task_memory(self, s2_tokens: torch.Tensor, reset: bool = False):
        if not bool(self.config.task_memory_enabled):
            self._task_memory_cache = None
            return None
        device = next(self.s1_student.parameters()).device
        dtype = next(self.s1_student.parameters()).dtype
        if hasattr(self.s1_student, "default_task_memory_from_s2"):
            candidate = self.s1_student.default_task_memory_from_s2(s2_tokens.to(device=device, dtype=dtype)).detach()
        elif hasattr(self.s1_student, "online_s1"):
            candidate = self.s1_student.online_s1.default_task_memory_from_s2(s2_tokens.to(device=device, dtype=dtype)).detach()
        else:
            return None
        if (
            reset
            or self._task_memory_cache is None
            or tuple(self._task_memory_cache.shape) != tuple(candidate.shape)
        ):
            memory = candidate
        else:
            memory = float(self.config.task_memory_ema) * self._task_memory_cache.to(device=device, dtype=dtype)
            memory = memory + (1.0 - float(self.config.task_memory_ema)) * candidate
        max_norm = float(self.config.task_memory_max_norm)
        if max_norm > 0:
            norms = memory.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            memory = memory * (max_norm / norms).clamp(max=1.0)
        self._task_memory_cache = memory.detach()
        return self._task_memory_cache

    @torch.no_grad()
    def _apply_temporal_ensemble(self, actions: torch.Tensor, enabled: Optional[bool] = None) -> torch.Tensor:
        use_ensemble = self.config.temporal_ensemble_enabled if enabled is None else enabled
        if not use_ensemble or actions.ndim != 3:
            return actions
        if self._action_chunk_history and self._action_chunk_history[0][0].shape != actions.shape:
            self.reset_temporal_ensemble()
        aged = []
        for chunk, age in self._action_chunk_history:
            next_age = age + 1
            if next_age < actions.shape[1]:
                aged.append((chunk, next_age))
        max_chunks = int(max(1, self.config.temporal_ensemble_max_chunks))
        self._action_chunk_history = [(actions.detach(), 0)] + aged[: max_chunks - 1]
        blended = []
        for offset in range(actions.shape[1]):
            weighted_sum = None
            weight_sum = 0.0
            for chunk, age in self._action_chunk_history:
                idx = age + offset
                if idx >= actions.shape[1]:
                    continue
                weight = math.exp(-float(self.config.temporal_ensemble_decay) * age)
                value = chunk[:, idx]
                weighted_sum = value * weight if weighted_sum is None else weighted_sum + value * weight
                weight_sum += weight
            blended.append(weighted_sum / max(weight_sum, 1e-8))
        return torch.stack(blended, dim=1)

    def apply_finetuning_policy(self):
        """Apply the default small-tuning policy.

        Base MolmoAct2 weights are frozen by default. Trainable parameters are
        compressor/student/residual-adapter parameters, and optionally the
        detected original action expert when the user explicitly unfreezes it.
        """
        if self.config.train_adapters_only:
            if self.molmoact.base is not None:
                self.molmoact.freeze_base()
            for param in self.molmoact.compressor.parameters():
                param.requires_grad = True
            for param in self.s1_student.parameters():
                param.requires_grad = True
            if hasattr(self.s1_student, "freeze_target_encoder"):
                self.s1_student.freeze_target_encoder()
            if self.residual_adapter is not None:
                for param in self.residual_adapter.parameters():
                    param.requires_grad = bool(self.config.residual_trainable)
        return self

    def unfreeze_original_s1(self):
        return self.molmoact.unfreeze_action_expert()

    def trainable_parameter_names(self) -> List[str]:
        return [name for name, param in self.named_parameters() if param.requires_grad]

    @torch.no_grad()
    def update_jepa_target_encoder(self, decay: Optional[float] = None):
        if hasattr(self.s1_student, "update_target_encoder"):
            self.s1_student.update_target_encoder(decay=decay)
        return self

    @torch.no_grad()
    def refresh_s2(self, images: Union[ImageLike, List[ImageLike]], instruction: str, force: bool = False) -> torch.Tensor:
        tokens = self.molmoact.refresh_s2(images, instruction, force=force)
        if instruction != self._cached_instruction or force:
            self.reset_temporal_ensemble()
            self.update_task_memory(tokens, reset=instruction != self._cached_instruction)
        else:
            self.update_task_memory(tokens, reset=False)
        self._s2_cache = tokens.detach()
        self._s2_cache_time = time.time()
        self._cached_instruction = instruction
        return self._s2_cache

    @torch.no_grad()
    def act_fast(
        self,
        state: torch.Tensor,
        state_history: Optional[torch.Tensor] = None,
        action_history: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        use_original: Optional[bool] = None,
        temporal_ensemble: Optional[bool] = None,
    ) -> torch.Tensor:
        use_original = self.config.use_original_s1 if use_original is None else use_original
        device = next(self.parameters()).device
        state = state.to(device)
        state_history = None if state_history is None else state_history.to(device)
        action_history = None if action_history is None else action_history.to(device)

        if use_original and self.molmoact.base is not None:
            actions = self.molmoact.act_original(state, state_history, action_history, num_steps=steps)
        else:
            if self._s2_cache is None:
                raise RuntimeError("S2 cache is empty. Call refresh_s2() or pass s2_tokens to forward().")
            s2_tokens = self._s2_cache.to(device=device, dtype=state.dtype if state.is_floating_point() else torch.float32)
            task_memory = None if self._task_memory_cache is None else self._task_memory_cache.to(device=device, dtype=s2_tokens.dtype)
            actions = self.s1_student.sample(s2_tokens, state, state_history, action_history, steps=steps, task_memory=task_memory)

        if self.residual_adapter is not None and float(self.config.residual_alpha) != 0.0:
            actions = actions + float(self.config.residual_alpha) * self.residual_adapter(state).to(actions.dtype)
        if self.config.action_clip > 0:
            actions = actions.clamp(-self.config.action_clip, self.config.action_clip)
        return self._apply_temporal_ensemble(actions, enabled=temporal_ensemble)

    def forward_from_s2_tokens(
        self,
        s2_tokens: torch.Tensor,
        state: torch.Tensor,
        state_history: Optional[torch.Tensor] = None,
        action_history: Optional[torch.Tensor] = None,
        target_actions: Optional[torch.Tensor] = None,
        s1_steps: Optional[int] = None,
        task_memory: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Rio2Output]:
        return self.forward(
            state=state,
            s2_tokens=s2_tokens,
            state_history=state_history,
            action_history=action_history,
            target_actions=target_actions,
            s1_steps=s1_steps,
            task_memory=task_memory,
            return_dict=return_dict,
            use_original=False,
        )

    def forward(
        self,
        state: torch.Tensor,
        s2_tokens: Optional[torch.Tensor] = None,
        state_history: Optional[torch.Tensor] = None,
        action_history: Optional[torch.Tensor] = None,
        target_actions: Optional[torch.Tensor] = None,
        images: Optional[Union[ImageLike, List[ImageLike]]] = None,
        instruction: Optional[str] = None,
        refresh_s2: bool = False,
        s1_steps: Optional[int] = None,
        task_memory: Optional[torch.Tensor] = None,
        use_original: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Rio2Output]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_original = self.config.use_original_s1 if use_original is None else use_original

        if refresh_s2:
            if images is None or instruction is None:
                raise ValueError("`images` and `instruction` are required when refresh_s2=True.")
            s2_tokens = self.refresh_s2(images, instruction, force=True)
        elif s2_tokens is None:
            s2_tokens = self._s2_cache

        device = next(self.parameters()).device
        state = state.to(device)
        state_history = None if state_history is None else state_history.to(device)
        action_history = None if action_history is None else action_history.to(device)

        # Training path: use cached-token/student path by default because the
        # original MolmoAct2 action expert is usually frozen and remote-code
        # signatures may not expose target-action training directly.
        if target_actions is not None:
            if s2_tokens is None:
                raise ValueError("Training requires `s2_tokens` or refresh_s2=True.")
            s2_tokens = s2_tokens.to(device=device, dtype=state.dtype if state.is_floating_point() else torch.float32)
            task_memory = None if task_memory is None else task_memory.to(device=device, dtype=s2_tokens.dtype)
            target_actions = target_actions.to(device=device, dtype=state.dtype if state.is_floating_point() else torch.float32)
            losses = self.s1_student.training_loss(s2_tokens, state, state_history, action_history, target_actions, task_memory=task_memory)
            output = Rio2Output(
                loss=losses["loss"],
                s2_tokens=s2_tokens,
                loss_flow_mse=losses["loss_flow_mse"],
                loss_flow_l1=losses["loss_flow_l1"],
                loss_diffusion=losses.get("loss_diffusion"),
                loss_consistency=losses.get("loss_consistency"),
                loss_smooth=losses["loss_smooth"],
                loss_jepa=losses.get("loss_jepa"),
                loss_jepa_prior=losses.get("loss_jepa_prior"),
                pred_action_latent=losses.get("pred_action_latent"),
                target_action_latent=losses.get("target_action_latent"),
                runtime_path="jepa_s1_training" if "loss_jepa" in losses else "student_adapter_training",
            )
            return tuple(v for v in output.to_tuple() if v is not None) if not return_dict else output

        if use_original and self.molmoact.base is not None:
            actions = self.act_fast(state, state_history, action_history, steps=s1_steps, use_original=True)
            runtime_path = self.molmoact.last_runtime_path
            tokens = self._s2_cache
        else:
            if s2_tokens is None:
                raise ValueError("Pass `s2_tokens`, call refresh_s2(), or set refresh_s2=True.")
            s2_tokens = s2_tokens.to(device=device, dtype=state.dtype if state.is_floating_point() else torch.float32)
            if task_memory is None and self._task_memory_cache is not None:
                task_memory = self._task_memory_cache
            task_memory = None if task_memory is None else task_memory.to(device=device, dtype=s2_tokens.dtype)
            actions = self.s1_student.sample(s2_tokens, state, state_history, action_history, steps=s1_steps, task_memory=task_memory)
            if self.residual_adapter is not None and float(self.config.residual_alpha) != 0.0:
                actions = actions + float(self.config.residual_alpha) * self.residual_adapter(state).to(actions.dtype)
            runtime_path = "student_cached_tokens"
            tokens = s2_tokens

        output = Rio2Output(actions=actions, s2_tokens=tokens, runtime_path=runtime_path)
        return (actions, tokens) if not return_dict else output


__all__ = [
    "Rio2Config",
    "Rio2Model",
    "Rio2Output",
    "Rio2PreTrainedModel",
    "Rio2S1MoEResidualBank",
    "Rio2JepaS1ActionExpert",
]
