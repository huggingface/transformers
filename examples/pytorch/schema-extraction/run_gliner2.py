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
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch.nn as nn
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import LoraLayer
from torch.optim import AdamW

from transformers import (
    Gliner2ForSchemaExtraction,
    Gliner2Processor,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


logger = logging.getLogger(__name__)

DEFAULT_LORA_TARGETS = ("query", "key", "value", "dense")


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


def convert_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return one text, schema, and labels row."""
    if "text" in row and "schema" in row and "labels" in row:
        return {"text": row["text"], "schema": row["schema"], "labels": row["labels"]}
    raise ValueError("row must contain text, schema, and labels")


def load_rows(path: str, seed: int, shuffle: bool, max_samples: int | None = None) -> Dataset:
    """Load JSONL rows, then apply the load-time shuffle."""
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = convert_row(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path} line {line_number}") from exc
            row["length"] = len(str(row["text"]).split())
            rows.append(row)
    if not rows:
        raise ValueError(f"no rows in {path}")
    if shuffle:
        random.seed(seed)
        random.shuffle(rows)
    if max_samples is not None and max_samples > 0:
        rows = rows[:max_samples]
    return Dataset.from_list(rows)


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
    """Attach PEFT LoRA using target_modules as given."""
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(targets),
        bias="none",
        use_dora=use_dora,
    )
    # autocast_adapter_dtype=False keeps adapters in the base dtype.
    peft_model = get_peft_model(model, config, autocast_adapter_dtype=False)
    _cast_lora_dtype(peft_model)
    return peft_model


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
        batch = self.processor(texts, schemas, architecture=self.architecture, labels=labels)
        forward_keys = set(inspect.signature(_unwrap(self.model).forward).parameters)
        data = {key: value for key, value in dict(batch).items() if key in forward_keys}
        if "labels" in forward_keys and "labels" not in data:
            data["labels"] = labels
        return data


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


class Gliner2Trainer(Trainer):
    """Encoder and task learning rates, with Trainer's scheduler and sampler."""

    def __init__(
        self,
        *args,
        encoder_lr: float = 1e-5,
        task_lr: float = 5e-4,
        gold_injection_start: float = 1.0,
        gold_injection_end: float = 0.25,
        gold_injection_hold_frac: float = 0.15,
        use_lora: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.encoder_lr = encoder_lr
        self.task_lr = task_lr
        self.use_lora = use_lora
        self.args.remove_unused_columns = False
        self.add_callback(Gliner2ScheduleCallback(gold_injection_start, gold_injection_end, gold_injection_hold_frac))

    def _architecture(self) -> str:
        """Return ``span`` or ``boundary`` from the loaded config."""
        config = getattr(_unwrap(self.model), "config", None)
        return getattr(config, "architecture", "span")

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
            return super().get_train_dataloader()
        finally:
            (
                self.data_collator,
                self.args.dataloader_num_workers,
                self.args.dataloader_prefetch_factor,
                self.args.dataloader_drop_last,
            ) = previous

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

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Run eval with gold injection forced off."""
        head = _boundary_head(model)
        if head is not None:
            head.set_gold_injection_prob(0.0)
        model.eval()
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)


@dataclass
class ModelArguments:
    """Checkpoint to fine-tune."""

    model_name_or_path: str = field(metadata={"help": "Hub id or local GLiNER2 checkpoint."})


@dataclass
class DataArguments:
    """JSONL files of text, schema, and labels rows."""

    train_file: str = field(metadata={"help": "Training JSONL of text, schema, and labels rows."})
    validation_file: str | None = field(default=None, metadata={"help": "Optional evaluation JSONL."})
    max_train_samples: int | None = field(
        default=None, metadata={"help": "Truncate the training split after shuffling."}
    )
    max_eval_samples: int | None = field(default=None, metadata={"help": "Truncate the evaluation split."})


@dataclass
class Gliner2Arguments:
    """GLiNER2 optimizer and LoRA choices."""

    encoder_lr: float = field(
        default=1e-5, metadata={"help": "Learning rate for parameters whose name contains encoder."}
    )
    task_lr: float = field(
        default=5e-4, metadata={"help": "Learning rate for task heads and for the single LoRA group."}
    )
    gold_injection_start: float = field(default=1.0, metadata={"help": "Gold-injection rate during the hold."})
    gold_injection_end: float = field(default=0.25, metadata={"help": "Gold-injection rate after the linear decay."})
    gold_injection_hold_frac: float = field(
        default=0.15, metadata={"help": "Fraction of steps that hold the start rate."}
    )
    use_lora: bool = field(default=False, metadata={"help": "Train PEFT LoRA adapters at task_lr."})
    lora_r: int = field(default=16, metadata={"help": "LoRA rank."})
    lora_alpha: float = field(default=32.0, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "LoRA dropout."})
    lora_use_dora: bool = field(default=False, metadata={"help": "Enable PEFT DoRA."})
    lora_target_modules: str = field(
        default=",".join(DEFAULT_LORA_TARGETS),
        metadata={"help": "Comma-separated PEFT target_modules. Passed through unchanged. bias is none."},
    )


def main():
    """Fine-tune GLiNER2 through the standard Trainer loop."""
    parser = HfArgumentParser((ModelArguments, DataArguments, Gliner2Arguments, TrainingArguments))
    model_args, data_args, policy_args, training_args = parser.parse_args_into_dataclasses()
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
        gold_injection_start=policy_args.gold_injection_start,
        gold_injection_end=policy_args.gold_injection_end,
        gold_injection_hold_frac=policy_args.gold_injection_hold_frac,
        use_lora=use_lora,
    )
    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
    if training_args.do_eval:
        trainer.evaluate()


if __name__ == "__main__":
    main()
