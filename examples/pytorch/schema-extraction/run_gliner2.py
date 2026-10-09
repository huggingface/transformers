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
from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import LoraLayer
from torch.optim import AdamW

from transformers import (
    Gliner2ForSchemaExtraction,
    Gliner2Processor,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)


DEFAULT_LORA_TARGETS = ("query", "key", "value", "dense")


def _unwrap(model: nn.Module) -> nn.Module:
    """Return the GLiNER2 module under DDP and PEFT wrappers.

    Args:
        model: Trainer model, possibly wrapped.

    Returns:
        The underlying `Gliner2ForSchemaExtraction` module.
    """
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    return model


def _cast_lora_dtype(model: nn.Module) -> None:
    """Cast LoRA A/B weights to the dtype of their base layer.

    Args:
        model: PEFT model whose adapters should match the frozen base dtype.
    """
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
    """Attach PEFT LoRA using `target_modules` as given.

    Args:
        model: Model to wrap.
        rank: LoRA rank.
        alpha: LoRA alpha.
        dropout: LoRA dropout.
        targets: Module names passed through to PEFT. Aliases are not expanded.
        use_dora: Enable DoRA when true.

    Returns:
        The PEFT model.
    """
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

    def __init__(self, processor, model, architecture: str):
        """Store the processor and the architecture passed to it.

        Args:
            processor: `Gliner2Processor`.
            model: Model whose `forward` selects which processor keys are kept.
            architecture: `"span"` or `"boundary"`.
        """
        self.processor = processor
        self.model = model
        self.architecture = architecture

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """Tokenize one batch with labels.

        Args:
            features: Rows with `text`, `schema`, and `labels`.

        Returns:
            A dict of tensors accepted by `forward`.
        """
        texts = [feature["text"] for feature in features]
        schemas = [feature["schema"] for feature in features]
        labels = [feature.get("labels") for feature in features]
        batch = self.processor(texts, schemas, architecture=self.architecture, labels=labels)
        forward_keys = set(inspect.signature(_unwrap(self.model).forward).parameters)
        data = {key: value for key, value in dict(batch).items() if key in forward_keys}
        if "labels" in forward_keys and "labels" not in data:
            data["labels"] = labels
        return data


class Gliner2Trainer(Trainer):
    """Encoder and task learning rates, with Trainer's scheduler and sampler."""

    def __init__(self, *args, encoder_lr: float = 1e-5, task_lr: float = 5e-4, use_lora: bool = False, **kwargs):
        """Keep schema columns and store the two learning rates.

        Args:
            encoder_lr: Learning rate for parameters whose name contains `encoder`.
            task_lr: Learning rate for the remaining parameters, and for LoRA.
            use_lora: Use one optimizer group at `task_lr`.
        """
        super().__init__(*args, **kwargs)
        self.encoder_lr = encoder_lr
        self.task_lr = task_lr
        self.use_lora = use_lora
        self.args.remove_unused_columns = False

    def create_optimizer(self, model=None):
        """AdamW groups with decay on every parameter, including bias and LayerNorm.

        Args:
            model: Optional model override. Defaults to `self.model`.

        Returns:
            The AdamW optimizer.
        """
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


@dataclass
class ModelArguments:
    """Checkpoint to fine-tune."""

    model_name_or_path: str = field(metadata={"help": "Hub id or local GLiNER2 checkpoint."})


@dataclass
class DataArguments:
    """JSONL files of text, schema, and labels rows."""

    train_file: str = field(metadata={"help": "Training JSONL of text, schema, and labels rows."})
    validation_file: str | None = field(default=None, metadata={"help": "Optional evaluation JSONL."})
    max_train_samples: int | None = field(default=None, metadata={"help": "Truncate the training split."})
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
    data_files = {"train": data_args.train_file}
    if data_args.validation_file:
        data_files["validation"] = data_args.validation_file
    raw_datasets = load_dataset("json", data_files=data_files)
    if "length" in raw_datasets["train"].column_names:
        raw_datasets = raw_datasets.remove_columns("length")
    train_dataset = raw_datasets["train"]
    if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
        train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))
    eval_dataset = None
    if data_args.validation_file:
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None and data_args.max_eval_samples > 0:
            eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples)))
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
    architecture = getattr(model.config, "architecture", "span")
    trainer = Gliner2Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        data_collator=SchemaCollator(processor, model, architecture),
        encoder_lr=policy_args.encoder_lr,
        task_lr=policy_args.task_lr,
        use_lora=use_lora,
    )
    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
    if training_args.do_eval:
        trainer.evaluate()


if __name__ == "__main__":
    main()
