#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import inspect
import json
import logging
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate.utils import DistributedType
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import LoraLayer
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Sampler

from transformers import (
    Gliner2ForSchemaExtraction,
    Gliner2Processor,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)


logger = logging.getLogger(__name__)

ENCODER_PATTERNS = ("query", "key", "value", "dense")
TASK_HEADS = (
    "span_rep",
    "classifier",
    "count_embed",
    "count_pred",
    "boundary_head",
    "record_decoder",
    "relation_scorer",
)
DEFAULT_LORA_TARGETS = ("encoder", *TASK_HEADS)


def gold_injection_probability(
    progress: float, start: float = 1.0, end: float = 0.25, hold_fraction: float = 0.15
) -> float:
    """Hold the start rate, then decay linearly to the end rate."""
    if progress <= hold_fraction:
        return float(start)
    span = max(1.0 - hold_fraction, 1e-12)
    fraction = min(max((progress - hold_fraction) / span, 0.0), 1.0)
    return float(start + (end - start) * fraction)


def consistency_scale(step: int, warmup_steps: int) -> float:
    """Ramp consistency loss from zero to one across warmup steps."""
    if warmup_steps <= 0:
        return 1.0
    return min(step / warmup_steps, 1.0)


def soft_iou_anneal_scale(step: int, anneal_steps: int) -> float:
    """Decay the soft-IoU weight linearly to exact zero."""
    if anneal_steps <= 0:
        return 0.0
    return max(1.0 - step / anneal_steps, 0.0)


def recall_gate_exit_code(metrics: dict[str, float], overall_gate: float = 0.97, long_gate: float = 0.93) -> int:
    """Return 1 when oracle recall or long-span recall misses its gate."""
    overall = float(metrics.get("dry_run_proposal_oracle_recall", 0.0))
    long_recall = float(metrics.get("dry_run_recall_length_9_plus", 1.0))
    return int(overall < overall_gate or long_recall < long_gate)


def proposal_recall_rates(counts: dict[str, float], prefix: str = "dry_run") -> dict[str, float]:
    """Turn accumulated proposal counts into the published recall ratios."""

    def scalar(name: str) -> float:
        value = counts.get(name, 0.0)
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu())
        return float(value)

    result: dict[str, float] = {}
    gold_total = scalar("proposal_gold_total")
    if gold_total > 0:
        result[f"{prefix}_proposal_oracle_recall"] = scalar("proposal_gold_hit") / gold_total
    boundary_total = scalar("boundary_total")
    if boundary_total > 0:
        result[f"{prefix}_start_recall"] = scalar("start_hit") / boundary_total
        result[f"{prefix}_end_recall"] = scalar("end_hit") / boundary_total
    valid_queries = scalar("valid_queries")
    if valid_queries > 0:
        result[f"{prefix}_candidates_per_query"] = scalar("unique_candidates") / valid_queries
    for label in ("1", "2", "3_4", "5_8", "9_plus"):
        total = scalar(f"length_{label}_total")
        if total > 0:
            result[f"{prefix}_recall_length_{label}"] = scalar(f"length_{label}_hit") / total
    return result


def _metric_scalar(value: Any) -> float | None:
    """Read one proposal count, skipping rates and non-scalars."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu())
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _proposal_counts(outputs) -> dict[str, float]:
    """Read proposal count tensors from a schema-extraction forward."""
    boundary = outputs.get("boundary") if isinstance(outputs, dict) else getattr(outputs, "boundary", None)
    metrics = None if boundary is None else getattr(boundary, "metrics", None)
    if metrics is None and isinstance(boundary, dict):
        metrics = boundary.get("metrics")
    counts = {}
    for key, value in (metrics or {}).items():
        if key.endswith("_recall") or key.startswith("recall_") or key == "candidates_per_query":
            continue
        scalar = _metric_scalar(value)
        if scalar is not None:
            counts[key] = scalar
    return counts


def _nonempty_eval_splits(dataset) -> list[str | None]:
    """Return eval split names that contain rows. None means the single dataset."""
    if dataset is None:
        return []
    if isinstance(dataset, dict):
        return [name for name, split in dataset.items() if split is not None and len(split) > 0]
    try:
        if len(dataset) == 0:
            return []
    except TypeError:
        return []
    return [None]


def _mention(value: Any) -> Any:
    """Keep a string for every match, or one character offset."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and "text" in value:
        if "start" not in value and "end" not in value:
            return value["text"]
        span = {"text": value["text"]}
        if "start" in value:
            span["start"] = value["start"]
        if "end" in value:
            span["end"] = value["end"]
        return span
    raise ValueError(f"mention must be a string or {{text, start, end}}, got {value!r}")


def _split_field(value: Any) -> tuple[Any, Any]:
    """Split one structure field into its schema spec and supervised value."""
    if isinstance(value, dict) and "choices" in value:
        spec: dict[str, Any] = {"choices": list(value["choices"])}
        if "dtype" in value:
            spec["dtype"] = value["dtype"]
        return spec, value.get("value")
    if isinstance(value, list):
        labels = [_mention(item) if isinstance(item, (str, dict)) else item for item in value]
        return "list", labels
    if isinstance(value, dict) and "text" in value:
        return "str", _mention(value)
    if isinstance(value, str):
        return "str", value
    return "str", value


def convert_legacy_record(record: dict[str, Any]) -> dict[str, Any]:
    """Map one legacy ``input``/``output`` row onto text, schema, and labels."""
    output = record.get("output") or {}
    schema: dict[str, Any] = {}
    labels: dict[str, Any] = {}
    entities = output.get("entities") or {}
    if entities:
        schema["entities"] = list(entities)
        labels["entities"] = {name: [_mention(item) for item in mentions] for name, mentions in entities.items()}
    if output.get("entity_descriptions"):
        schema["entity_descriptions"] = output["entity_descriptions"]
    classifications = output.get("classifications") or []
    if classifications:
        schema["classifications"] = [
            {key: value for key, value in item.items() if key != "true_label"} for item in classifications
        ]
        classified = {}
        for item in classifications:
            true_label = item.get("true_label", [])
            classified[item["task"]] = [true_label] if isinstance(true_label, str) else list(true_label)
        labels["classifications"] = classified
    structures = output.get("json_structures") or []
    if structures:
        schema_by_parent: dict[str, dict[str, Any]] = {}
        label_by_parent: dict[str, list[dict[str, Any]]] = {}
        for item in structures:
            for parent, fields in item.items():
                spec = schema_by_parent.setdefault(parent, {})
                instance = {}
                for field_name, value in fields.items():
                    field_spec, field_label = _split_field(value)
                    spec.setdefault(field_name, field_spec)
                    if field_label is not None:
                        instance[field_name] = field_label
                label_by_parent.setdefault(parent, []).append(instance)
        schema["json_structures"] = [{parent: fields} for parent, fields in schema_by_parent.items()]
        labels["json_structures"] = label_by_parent
    if output.get("json_descriptions"):
        schema["json_descriptions"] = output["json_descriptions"]
    if output.get("record_metadata"):
        schema["record_metadata"] = output["record_metadata"]
    relations = output.get("relations") or []
    if relations:
        schema_fields: dict[str, dict[str, str]] = {}
        edges: dict[str, list[dict[str, Any]]] = {}
        for item in relations:
            for name, fields in item.items():
                spec = schema_fields.setdefault(name, {})
                edge = {}
                for field_name, value in fields.items():
                    spec.setdefault(field_name, "")
                    edge[field_name] = _mention(value) if isinstance(value, (str, dict)) else value
                edges.setdefault(name, []).append(edge)
        schema["relations"] = [{name: fields} for name, fields in schema_fields.items()]
        labels["relations"] = edges
    if output.get("relation_descriptions"):
        schema["relation_descriptions"] = output["relation_descriptions"]
    return {"text": record["input"], "schema": schema, "labels": labels}


def convert_row(row: dict[str, Any]) -> dict[str, Any]:
    """Accept a canonical row or convert one legacy row."""
    if "text" in row and "schema" in row and "labels" in row:
        return {"text": row["text"], "schema": row["schema"], "labels": row["labels"]}
    if "input" in row and "output" in row:
        return convert_legacy_record(row)
    raise ValueError("row must contain text/schema/labels or input/output")


def load_rows(path: str, seed: int, shuffle: bool, max_samples: int | None = None) -> Dataset:
    """Load JSONL, convert legacy rows once, then apply the load-time shuffle."""
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(convert_row(json.loads(line)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path} line {line_number}") from exc
    if not rows:
        raise ValueError(f"no rows in {path}")
    if shuffle:
        random.seed(seed)
        random.shuffle(rows)
    if max_samples is not None and max_samples > 0:
        rows = rows[:max_samples]
    return Dataset.from_list(rows)


class SeededShuffleSampler(Sampler[int]):
    """Permute dataset indexes with ``seed + epoch``."""

    def __init__(self, data_source, seed: int):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        yield from torch.randperm(len(self.data_source), generator=generator).tolist()

    def __len__(self) -> int:
        return len(self.data_source)


class LengthGroupedSampler(Sampler[int]):
    """Shuffle, sort each window by length, then keep a partial batch last."""

    def __init__(self, lengths, batch_size: int, window_batches: int = 50, seed: int = 0):
        if batch_size <= 0 or window_batches <= 0:
            raise ValueError("batch_size and window_batches must be > 0")
        self.lengths = tuple(int(length) for length in lengths)
        self.batch_size = int(batch_size)
        self.window_batches = int(window_batches)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _batches(self) -> list[list[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.lengths), generator=generator).tolist()
        window = self.batch_size * self.window_batches
        batches: list[list[int]] = []
        for start in range(0, len(order), window):
            group = order[start : start + window]
            group.sort(key=self.lengths.__getitem__)
            batches.extend(
                group[offset : offset + self.batch_size] for offset in range(0, len(group), self.batch_size)
            )
        if not batches:
            return batches
        full = [batch for batch in batches if len(batch) == self.batch_size]
        # Partial batch stays last; DataLoader drop_last removes it.
        partial = [batch for batch in batches if len(batch) != self.batch_size]
        batch_order = torch.randperm(len(full), generator=generator).tolist()
        return [full[index] for index in batch_order] + partial

    def __iter__(self):
        return iter(index for batch in self._batches() for index in batch)

    def __len__(self) -> int:
        return len(self.lengths)


class DistributedLengthGroupedSampler(LengthGroupedSampler):
    """Shard full length-grouped batches without padding or repeated indexes."""

    def __init__(
        self,
        lengths,
        batch_size: int,
        num_replicas: int | None = None,
        rank: int | None = None,
        window_batches: int = 50,
        seed: int = 0,
    ):
        if num_replicas is None or rank is None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("distributed process group is not initialized")
            num_replicas = dist.get_world_size() if num_replicas is None else num_replicas
            rank = dist.get_rank() if rank is None else rank
        if num_replicas <= 0 or rank < 0 or rank >= num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        super().__init__(lengths, batch_size, window_batches=window_batches, seed=seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def _rank_batches(self) -> list[list[int]]:
        batches = [batch for batch in self._batches() if len(batch) == self.batch_size]
        usable = len(batches) - (len(batches) % self.num_replicas)
        return batches[:usable][self.rank :: self.num_replicas]

    def __iter__(self):
        return iter(index for batch in self._rank_batches() for index in batch)

    def __len__(self) -> int:
        return sum(len(batch) for batch in self._rank_batches())


def _unwrap(model: nn.Module) -> nn.Module:
    """Return the GLiNER2 module under DDP and PEFT wrappers."""
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    return model


def _boundary_head(model: nn.Module | None):
    """Return the boundary head when this checkpoint has one."""
    if model is None:
        return None
    return getattr(_unwrap(model), "boundary_head", None)


def _boundary_config(model: nn.Module):
    """Return the boundary config object or mapping."""
    config = getattr(_unwrap(model), "config", None)
    return getattr(config, "boundary_config", None)


def _config_value(config, name: str, default: int) -> int:
    """Read one boundary schedule length from an object or dict."""
    if config is None:
        return default
    if isinstance(config, dict):
        return int(config.get(name, default) or 0)
    return int(getattr(config, name, default) or 0)


def _has_module(model: nn.Module, name: str) -> bool:
    """Return whether ``name`` is a module or the prefix of one."""
    return any(module_name == name or module_name.startswith(f"{name}.") for module_name, _ in model.named_modules())


def _alias_prefixes(model: nn.Module, alias: str) -> tuple[str, ...]:
    """Expand one public LoRA alias into module prefixes that exist."""
    if alias == "all_task_heads":
        return tuple(name for name in TASK_HEADS if _has_module(model, name))
    if alias == "classification_head":
        return ("classifier",) if _has_module(model, "classifier") else ()
    if alias == "extractive_head":
        if _has_module(model, "boundary_head"):
            return ("boundary_head",)
        return tuple(name for name in ("span_rep", "count_embed", "count_pred") if _has_module(model, name))
    if alias == "relation_head":
        return ("relation_scorer",) if _has_module(model, "relation_scorer") else ()
    if alias == "record_head":
        return ("record_decoder",) if _has_module(model, "record_decoder") else ()
    if alias in TASK_HEADS and _has_module(model, alias):
        return (alias,)
    return ()


def resolve_lora_targets(model: nn.Module, targets: list[str]) -> list[str]:
    """Expand encoder and task aliases to concrete ``nn.Linear`` paths."""
    prefixes = {prefix for target in targets for prefix in _alias_prefixes(model, target)}
    selected = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        encoder_match = False
        if name.startswith("encoder."):
            for target in targets:
                if target == "encoder" and any(pattern in leaf for pattern in ENCODER_PATTERNS):
                    encoder_match = True
                elif target.startswith("encoder.") and target.split(".", 1)[1] in leaf:
                    encoder_match = True
        head_match = any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
        if encoder_match or head_match:
            selected.append(name)
    return sorted(set(selected))


def _cast_lora_dtype(model: nn.Module) -> None:
    """Cast LoRA A/B weights to the dtype of their base layer."""
    for module in model.modules():
        if not isinstance(module, LoraLayer):
            continue
        base = module.get_base_layer()
        if base is None:
            continue
        dtype = next(base.parameters()).dtype
        for key in list(module.lora_A):
            module.lora_A[key] = module.lora_A[key].to(dtype=dtype)
        for key in list(module.lora_B):
            module.lora_B[key] = module.lora_B[key].to(dtype=dtype)


def apply_lora(model: nn.Module, rank: int, alpha: float, dropout: float, targets: list[str], use_dora: bool):
    """Attach PEFT LoRA and keep adapter weights in the base dtype."""
    resolved = resolve_lora_targets(model, targets)
    if not resolved:
        raise ValueError("LoRA target list matched no Linear modules")
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=resolved,
        bias="none",
        use_dora=use_dora,
    )
    # autocast_adapter_dtype=False keeps adapters in the base dtype.
    peft_model = get_peft_model(model, config, autocast_adapter_dtype=False)
    _cast_lora_dtype(peft_model)
    return peft_model


def _forward_batch(processor, model, texts, schemas, labels, architecture: str) -> dict[str, Any]:
    """Tokenize a batch and pass labels only through an explicit processor hook."""
    accepted = inspect.signature(processor.__call__).parameters
    kwargs = {"architecture": architecture} if "architecture" in accepted else {}
    if "labels" in accepted:
        kwargs["labels"] = labels
    batch = processor(texts, schemas, **kwargs)
    forward_keys = set(inspect.signature(_unwrap(model).forward).parameters)
    data = {key: value for key, value in dict(batch).items() if key in forward_keys}
    if "labels" in forward_keys and "labels" not in data:
        data["labels"] = labels
    return data


class SchemaCollator:
    """Build model inputs from text, schema, and labels rows."""

    def __init__(self, processor, model, architecture: str, training: bool):
        self.processor = processor
        self.model = model
        self.architecture = architecture
        self.training = training

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if not self.training:
            _unwrap(self.model).eval()
            head = _boundary_head(self.model)
            if head is not None:
                head.set_gold_injection_prob(0.0)
        texts = [feature["text"] for feature in features]
        schemas = [feature["schema"] for feature in features]
        labels = [feature.get("labels") for feature in features]
        return _forward_batch(self.processor, self.model, texts, schemas, labels, self.architecture)


def _apply_schedule(head, setter_name: str, value: float, warned: set[str]) -> None:
    """Call a boundary schedule setter when the library provides it."""
    setter = getattr(head, setter_name, None)
    if setter is not None:
        setter(value)
        return
    if setter_name not in warned:
        warned.add(setter_name)
        logger.warning("%s is not implemented on BoundaryHead.", setter_name)


class Gliner2ScheduleCallback(TrainerCallback):
    """Apply gold injection, consistency warmup, and soft-IoU annealing."""

    def __init__(self, start: float, end: float, hold_fraction: float):
        self.start = start
        self.end = end
        self.hold_fraction = hold_fraction
        self._warned: set[str] = set()

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        head = _boundary_head(model)
        if head is None:
            return control
        progress = state.global_step / max(state.max_steps, 1)
        head.set_gold_injection_prob(gold_injection_probability(progress, self.start, self.end, self.hold_fraction))
        config = _boundary_config(model)
        _apply_schedule(
            head,
            "set_consistency_scale",
            consistency_scale(state.global_step, _config_value(config, "consistency_warmup_steps", 0)),
            self._warned,
        )
        _apply_schedule(
            head,
            "set_soft_iou_scale",
            soft_iou_anneal_scale(state.global_step, _config_value(config, "soft_iou_anneal_steps", 0)),
            self._warned,
        )
        return control


def _loss_tensor(outputs) -> torch.Tensor:
    """Read the scalar loss returned by forward."""
    if isinstance(outputs, dict) and "loss" in outputs:
        loss = outputs["loss"]
    else:
        loss = getattr(outputs, "loss", None)
    if not isinstance(loss, torch.Tensor):
        raise RuntimeError("Gliner2ForSchemaExtraction.forward does not return loss.")
    return loss


def _text_lengths(dataset) -> list[int]:
    """Estimate length with ``len(text.split())`` for boundary grouping."""
    if hasattr(dataset, "column_names") and "text" in dataset.column_names:
        texts = dataset["text"]
    else:
        texts = [row["text"] for row in dataset]
    return [len(str(text).split()) for text in texts]


def _relax_rank_shard(loader) -> None:
    """Keep a rank-local length sampler from being sharded a second time."""
    shard = getattr(loader, "batch_sampler", None)
    inner = getattr(shard, "batch_sampler", None)
    sampler = getattr(inner, "sampler", None)
    if isinstance(sampler, DistributedLengthGroupedSampler) and hasattr(shard, "num_processes"):
        shard.num_processes = 1
        shard.process_index = 0


class Gliner2Trainer(Trainer):
    """Trainer policies that reproduce gliner2 optimization and sampling."""

    def __init__(
        self,
        *args,
        encoder_lr: float = 1e-5,
        task_lr: float = 5e-4,
        scheduler_type: str = "linear",
        num_cycles: float = 0.5,
        gold_injection_start: float = 1.0,
        gold_injection_end: float = 0.25,
        gold_injection_hold_frac: float = 0.15,
        length_group_window_batches: int = 50,
        use_lora: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if scheduler_type not in {"linear", "cosine", "cosine_restarts", "constant"}:
            raise ValueError(f"unknown scheduler_type {scheduler_type!r}")
        self.encoder_lr = encoder_lr
        self.task_lr = task_lr
        self.scheduler_type = scheduler_type
        self.num_cycles = num_cycles
        self.length_group_window_batches = length_group_window_batches
        self.use_lora = use_lora
        self.args.remove_unused_columns = False
        self._loss_finite_flag = None
        self.add_callback(Gliner2ScheduleCallback(gold_injection_start, gold_injection_end, gold_injection_hold_frac))
        self._install_finite_grad_hooks()

    def _architecture(self) -> str:
        """Return ``span`` or ``boundary`` from the loaded config."""
        config = getattr(_unwrap(self.model), "config", None)
        return getattr(config, "architecture", "span")

    def _install_finite_grad_hooks(self) -> None:
        """Zero a micro-batch gradient when the consensus flag is false."""

        def sanitize(gradient):
            flag = self._loss_finite_flag
            if flag is None:
                return gradient
            return torch.where(flag, gradient, torch.zeros_like(gradient))

        self._finite_handles = [
            parameter.register_hook(sanitize) for parameter in self.model.parameters() if parameter.requires_grad
        ]

    def _get_train_sampler(self, train_dataset=None):
        """Use a seeded shuffle for span and length grouping for boundary."""
        dataset = self.train_dataset if train_dataset is None else train_dataset
        if dataset is None:
            return None
        batch_size = self.args.per_device_train_batch_size
        if self._architecture() != "boundary":
            return SeededShuffleSampler(dataset, self.args.seed)
        lengths = _text_lengths(dataset)
        if self.args.world_size > 1:
            return DistributedLengthGroupedSampler(
                lengths,
                batch_size,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                window_batches=self.length_group_window_batches,
                seed=self.args.seed,
            )
        return LengthGroupedSampler(
            lengths,
            batch_size,
            window_batches=self.length_group_window_batches,
            seed=self.args.seed,
        )

    def _swap_loader_args(self, training: bool):
        """Point the collator at train or eval and apply Darwin worker limits."""
        previous = (
            self.data_collator,
            self.args.dataloader_num_workers,
            self.args.dataloader_prefetch_factor,
            self.args.dataloader_drop_last,
        )
        self.data_collator = SchemaCollator(self.processing_class, self.model, self._architecture(), training)
        if sys.platform == "darwin":
            self.args.dataloader_num_workers = 0
            self.args.dataloader_prefetch_factor = None
        return previous

    def get_train_dataloader(self):
        """Drop a partial train batch when the dataset is longer than one batch."""
        previous = self._swap_loader_args(True)
        batch_size = self.args.per_device_train_batch_size
        self.args.dataloader_drop_last = len(self.train_dataset) > batch_size
        try:
            loader = super().get_train_dataloader()
        finally:
            (
                self.data_collator,
                self.args.dataloader_num_workers,
                self.args.dataloader_prefetch_factor,
                self.args.dataloader_drop_last,
            ) = previous
        _relax_rank_shard(loader)
        return loader

    def get_eval_dataloader(self, eval_dataset=None):
        """Build eval targets while the model is in eval and injection is off."""
        previous = self._swap_loader_args(False)
        self.args.dataloader_drop_last = False
        try:
            return super().get_eval_dataloader(eval_dataset)
        finally:
            (
                self.data_collator,
                self.args.dataloader_num_workers,
                self.args.dataloader_prefetch_factor,
                self.args.dataloader_drop_last,
            ) = previous

    def create_optimizer(self, model=None):
        """AdamW groups with decay on every parameter, including bias and LayerNorm."""
        if self.optimizer is not None:
            return self.optimizer
        opt_model = self.model if model is None else model
        trainable = [param for param in opt_model.parameters() if param.requires_grad]
        if not trainable:
            raise ValueError("no trainable parameters")
        decay = self.args.weight_decay
        if self.use_lora:
            groups = [{"params": trainable, "lr": self.task_lr, "weight_decay": decay}]
        else:
            encoder, task = [], []
            for name, param in opt_model.named_parameters():
                if not param.requires_grad:
                    continue
                if "encoder" in name:
                    encoder.append(param)
                else:
                    task.append(param)
            groups = []
            if encoder:
                groups.append({"params": encoder, "lr": self.encoder_lr, "weight_decay": decay})
            if task:
                groups.append({"params": task, "lr": self.task_lr, "weight_decay": decay})
        kwargs = {"betas": (0.9, 0.999), "eps": 1e-8, "lr": self.task_lr}
        if self.args.device.type == "cuda":
            kwargs["fused"] = True
        else:
            kwargs["foreach"] = True
        self.optimizer = AdamW(groups, **kwargs)
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """Build a gliner2 LambdaLR. Trainer skips it when GradScaler skips the update."""
        optimizer = self.optimizer if optimizer is None else optimizer
        warmup = self.args.get_warmup_steps(num_training_steps)
        cycles = self.num_cycles
        total = num_training_steps

        def linear(step):
            if step < warmup:
                return float(step) / float(max(1, warmup))
            remain = float(max(1, total - warmup))
            return max(0.0, float(total - step) / remain)

        def cosine(step):
            if step < warmup:
                return float(step) / float(max(1, warmup))
            progress = float(step - warmup) / float(max(1, total - warmup))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        def cosine_restarts(step):
            if step < warmup:
                return float(step) / float(max(1, warmup))
            progress = float(step - warmup) / float(max(1, total - warmup))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((cycles * progress) % 1.0))))

        def constant(step):
            if step < warmup:
                return float(step) / float(max(1, warmup))
            return 1.0

        schedules = {
            "linear": linear,
            "cosine": cosine,
            "cosine_restarts": cosine_restarts,
            "constant": constant,
        }
        self.lr_scheduler = LambdaLR(optimizer, schedules[self.scheduler_type])
        self._created_lr_scheduler = True
        return self.lr_scheduler

    def _zero_nonfinite_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Zero a non-finite micro-batch after a distributed MAX consensus."""
        finite = torch.isfinite(loss.detach())
        if finite.ndim > 0:
            finite = finite.all()
        if dist.is_available() and dist.is_initialized():
            bad = (~finite).to(dtype=torch.float32)
            dist.all_reduce(bad, op=dist.ReduceOp.MAX)
            finite = bad == 0
        self._loss_finite_flag = finite
        return torch.where(finite, loss, torch.zeros_like(loss))

    def _rescale_short_accumulation(self, model: nn.Module, accum: int) -> None:
        """Undo the full-window divisor when the last window is short."""
        current = getattr(self, "current_gradient_accumulation_steps", accum)
        syncing = getattr(self.accelerator.gradient_state, "sync_gradients", False)
        if not syncing or not 0 < current < accum:
            return
        scale = accum / current
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Divide the micro-batch loss by gradient_accumulation_steps once."""
        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = _loss_tensor(model(**inputs))
        loss = self._zero_nonfinite_loss(loss)
        accum = max(self.args.gradient_accumulation_steps, 1)
        # Accelerate gradient accumulation is 1, so divide here once.
        loss = loss / accum
        self.loss_is_scaled_for_ga = True
        if self.args.n_gpu > 1:
            loss = loss.mean()
        backward_kwargs = {}
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            backward_kwargs["scale_wrt_gas"] = False
        self.accelerator.backward(loss, **backward_kwargs)
        self._rescale_short_accumulation(model, accum)
        return loss.detach()

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Run eval with gold injection forced off."""
        head = _boundary_head(model)
        if head is not None:
            head.set_gold_injection_prob(0.0)
        model.eval()
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

    def run_proposal_recall_gate(self, overall_gate: float = 0.97, long_gate: float = 0.93) -> int:
        """Score one injection-off eval pass and return the recall exit code."""
        if self._architecture() != "boundary":
            raise ValueError("dry_run_recall_steps requires architecture='boundary'")
        splits = _nonempty_eval_splits(self.eval_dataset)
        if not splits:
            return 0
        head = _boundary_head(self.model)
        previous_injection = getattr(head, "_gold_injection_prob", None) if head is not None else None
        if head is not None:
            head.set_gold_injection_prob(0.0)
        was_training = self.model.training
        self.model.eval()
        counts: dict[str, float] = {}
        saw_batch = False
        try:
            with torch.no_grad():
                for split in splits:
                    loader = self.get_eval_dataloader() if split is None else self.get_eval_dataloader(split)
                    if len(loader) == 0:
                        continue
                    for batch in loader:
                        saw_batch = True
                        outputs = self.model(**self._prepare_inputs(batch))
                        for key, value in _proposal_counts(outputs).items():
                            counts[key] = counts.get(key, 0.0) + value
        finally:
            if was_training:
                self.model.train()
            if head is not None and previous_injection is not None:
                head.set_gold_injection_prob(previous_injection)
        if not saw_batch:
            return 0
        rates = proposal_recall_rates(counts, "dry_run")
        logger.info(
            "Oracle recall dry run (gold injection=0): overall=%s long=%s",
            f"{rates['dry_run_proposal_oracle_recall']:.6g}" if "dry_run_proposal_oracle_recall" in rates else "n/a",
            f"{rates['dry_run_recall_length_9_plus']:.6g}" if "dry_run_recall_length_9_plus" in rates else "n/a",
        )
        return recall_gate_exit_code(rates, overall_gate, long_gate)


@dataclass
class ModelArguments:
    """Checkpoint to fine-tune."""

    model_name_or_path: str = field(metadata={"help": "Hub id or local GLiNER2 checkpoint."})


@dataclass
class DataArguments:
    """JSONL files of text, schema, and labels rows."""

    train_file: str = field(metadata={"help": "Training JSONL. Legacy input/output rows are converted once."})
    validation_file: str | None = field(default=None, metadata={"help": "Optional evaluation JSONL."})
    max_train_samples: int | None = field(
        default=None, metadata={"help": "Truncate the training split after shuffling."}
    )
    max_eval_samples: int | None = field(default=None, metadata={"help": "Truncate the evaluation split."})


@dataclass
class Gliner2Arguments:
    """GLiNER2 optimizer, schedule, and LoRA choices."""

    encoder_lr: float = field(
        default=1e-5, metadata={"help": "Learning rate for parameters whose name contains encoder."}
    )
    task_lr: float = field(
        default=5e-4, metadata={"help": "Learning rate for task heads and for the single LoRA group."}
    )
    scheduler_type: str = field(default="linear", metadata={"help": "linear, cosine, cosine_restarts, or constant."})
    num_cycles: float = field(default=0.5, metadata={"help": "Cycle count for cosine_restarts."})
    gold_injection_start: float = field(default=1.0, metadata={"help": "Gold-injection rate during the hold."})
    gold_injection_end: float = field(default=0.25, metadata={"help": "Gold-injection rate after the linear decay."})
    gold_injection_hold_frac: float = field(
        default=0.15, metadata={"help": "Fraction of steps that hold the start rate."}
    )
    length_group_window_batches: int = field(
        default=50, metadata={"help": "Boundary length-group window, in batches."}
    )
    use_lora: bool = field(default=False, metadata={"help": "Train PEFT LoRA adapters at task_lr."})
    lora_r: int = field(default=16, metadata={"help": "LoRA rank."})
    lora_alpha: float = field(default=32.0, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "LoRA dropout."})
    lora_use_dora: bool = field(default=False, metadata={"help": "Enable PEFT DoRA."})
    lora_target_modules: str = field(
        default=",".join(DEFAULT_LORA_TARGETS),
        metadata={"help": "Comma-separated LoRA aliases. bias is none."},
    )
    dry_run_recall_steps: int = field(
        default=0, metadata={"help": "Run the proposal-recall gate and skip training when > 0."}
    )
    gate_recall: float = field(default=0.97, metadata={"help": "Minimum proposal oracle recall."})
    gate_long_recall: float = field(default=0.93, metadata={"help": "Minimum recall for spans of length 9 or more."})


def main():
    """Fine-tune GLiNER2 through the standard Trainer loop."""
    parser = HfArgumentParser((ModelArguments, DataArguments, Gliner2Arguments, TrainingArguments))
    model_args, data_args, policy_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    train_dataset = load_rows(
        data_args.train_file, training_args.seed, shuffle=True, max_samples=data_args.max_train_samples
    )
    eval_dataset = None
    if data_args.validation_file:
        eval_dataset = load_rows(
            data_args.validation_file, training_args.seed, shuffle=False, max_samples=data_args.max_eval_samples
        )
    model = Gliner2ForSchemaExtraction.from_pretrained(model_args.model_name_or_path)
    processor = Gliner2Processor.from_pretrained(model_args.model_name_or_path)
    use_lora = policy_args.use_lora
    if use_lora:
        targets = [item.strip() for item in policy_args.lora_target_modules.split(",") if item.strip()]
        model = apply_lora(
            model,
            policy_args.lora_r,
            policy_args.lora_alpha,
            policy_args.lora_dropout,
            targets,
            policy_args.lora_use_dora,
        )
    trainer = Gliner2Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        data_collator=SchemaCollator(processor, model, getattr(model.config, "architecture", "span"), True),
        encoder_lr=policy_args.encoder_lr,
        task_lr=policy_args.task_lr,
        scheduler_type=policy_args.scheduler_type,
        num_cycles=policy_args.num_cycles,
        gold_injection_start=policy_args.gold_injection_start,
        gold_injection_end=policy_args.gold_injection_end,
        gold_injection_hold_frac=policy_args.gold_injection_hold_frac,
        length_group_window_batches=policy_args.length_group_window_batches,
        use_lora=use_lora,
    )
    if policy_args.dry_run_recall_steps > 0:
        exit_code = trainer.run_proposal_recall_gate(policy_args.gate_recall, policy_args.gate_long_recall)
        raise SystemExit(exit_code)
    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
    if training_args.do_eval:
        trainer.evaluate()


if __name__ == "__main__":
    main()
