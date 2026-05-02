# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""
Training CLI command.

Wraps ``Trainer`` to fine-tune or pretrain a model on any supported task
from a single CLI invocation. Supports text, vision, and audio tasks,
with built-in LoRA/QLoRA, distributed training, and hyperparameter search.

Examples::

    # Fine-tune text classification
    transformers train text-classification --model bert-base-uncased --dataset glue/sst2 --output ./out

    # Fine-tune image classification
    transformers train image-classification --model google/vit-base-patch16-224 --dataset food101 --output ./out

    # QLoRA (4-bit base + LoRA adapters)
    transformers train text-generation --model meta-llama/Llama-3.1-8B \\
        --dataset ./data.jsonl --output ./out --lora --quantization bnb-4bit

    # Distributed with DeepSpeed
    transformers train text-generation --model meta-llama/Llama-3.1-8B \\
        --dataset ./data.jsonl --output ./out --deepspeed zero3 --dtype bfloat16

Supported tasks: text-classification, token-classification, question-answering,
summarization, translation, text-generation, language-modeling,
image-classification, object-detection, semantic-segmentation,
speech-recognition, audio-classification.
"""

from typing import Annotated

import typer


# Maps CLI task names to (AutoModel class name, preprocessing type)
_TASK_CONFIGS = {
    # Text tasks
    "text-classification": {
        "auto_class": "AutoModelForSequenceClassification",
        "preprocess": "tokenize",
        "text_columns": ("sentence", "text"),
        "label_column": "label",
    },
    "token-classification": {
        "auto_class": "AutoModelForTokenClassification",
        "preprocess": "tokenize_and_align_labels",
        "text_columns": ("tokens",),
        "label_column": "ner_tags",
    },
    "question-answering": {
        "auto_class": "AutoModelForQuestionAnswering",
        "preprocess": "tokenize_qa",
        "text_columns": ("question", "context"),
        "label_column": None,
    },
    "summarization": {
        "auto_class": "AutoModelForSeq2SeqLM",
        "preprocess": "tokenize_seq2seq",
        "text_columns": ("article", "document"),
        "label_column": ("highlights", "summary"),
    },
    "translation": {
        "auto_class": "AutoModelForSeq2SeqLM",
        "preprocess": "tokenize_seq2seq",
        "text_columns": None,
        "label_column": None,
    },
    "text-generation": {
        "auto_class": "AutoModelForCausalLM",
        "preprocess": "tokenize_causal",
        "text_columns": ("text",),
        "label_column": None,
    },
    "language-modeling": {
        "auto_class": "AutoModelForMaskedLM",
        "preprocess": "tokenize_mlm",
        "text_columns": ("text",),
        "label_column": None,
    },
    # Vision tasks
    "image-classification": {
        "auto_class": "AutoModelForImageClassification",
        "preprocess": "image_transform",
        "text_columns": None,
        "label_column": "label",
    },
    "object-detection": {
        "auto_class": "AutoModelForObjectDetection",
        "preprocess": "image_transform",
        "text_columns": None,
        "label_column": None,
    },
    "semantic-segmentation": {
        "auto_class": "AutoModelForSemanticSegmentation",
        "preprocess": "image_transform",
        "text_columns": None,
        "label_column": None,
    },
    # Audio tasks
    "speech-recognition": {
        "auto_class": "AutoModelForSpeechSeq2Seq",
        "preprocess": "audio_transform",
        "text_columns": None,
        "label_column": "text",
    },
    "audio-classification": {
        "auto_class": "AutoModelForAudioClassification",
        "preprocess": "audio_transform",
        "text_columns": None,
        "label_column": "label",
    },
}


def _detect_text_column(dataset, candidates: tuple[str, ...] | None) -> str:
    """Find the first matching column name in the dataset."""
    if candidates is None:
        return None
    columns = dataset.column_names
    if isinstance(columns, dict):
        columns = columns.get("train", list(columns.values())[0])
    for c in candidates:
        if c in columns:
            return c
    return columns[0]


def _load_dataset(dataset_path: str, subset: str | None, token: str | None):
    """Load a dataset from the Hub or local files."""
    from datasets import load_dataset

    kwargs = {}
    if token:
        kwargs["token"] = token

    # Detect local files
    if dataset_path.endswith((".csv", ".json", ".jsonl", ".txt", ".parquet")):
        if dataset_path.endswith(".csv"):
            fmt = "csv"
        elif dataset_path.endswith((".json", ".jsonl")):
            fmt = "json"
        elif dataset_path.endswith(".parquet"):
            fmt = "parquet"
        else:
            fmt = "text"
        return load_dataset(fmt, data_files=dataset_path, **kwargs)

    # Hub dataset, possibly with subset: "glue/sst2" -> ("glue", "sst2")
    if subset is not None:
        return load_dataset(dataset_path, subset, **kwargs)
    if "/" in dataset_path and not dataset_path.startswith((".", "/")):
        parts = dataset_path.split("/")
        if len(parts) == 2:
            # Could be "org/dataset" or "dataset/subset" — try as-is first
            try:
                return load_dataset(dataset_path, **kwargs)
            except Exception:
                return load_dataset(parts[0], parts[1], **kwargs)
    return load_dataset(dataset_path, **kwargs)


def train(
    task: Annotated[str, typer.Argument(help=f"Task to train. One of: {', '.join(_TASK_CONFIGS.keys())}.")],
    # Model
    model: Annotated[str, typer.Option("--model", "-m", help="Model ID or local path.")],
    dataset: Annotated[str, typer.Option(help="Dataset name (Hub) or path (local file).")],
    output: Annotated[str, typer.Option(help="Output directory for checkpoints and final model.")],
    subset: Annotated[str | None, typer.Option(help="Dataset subset/config name.")] = None,
    # Training hyperparameters
    epochs: Annotated[float, typer.Option(help="Number of training epochs.")] = 3.0,
    lr: Annotated[float, typer.Option(help="Learning rate.")] = 5e-5,
    batch_size: Annotated[int, typer.Option(help="Per-device training batch size.")] = 8,
    eval_batch_size: Annotated[int | None, typer.Option(help="Per-device eval batch size.")] = None,
    max_seq_length: Annotated[int, typer.Option(help="Maximum sequence length for tokenization.")] = 512,
    gradient_accumulation_steps: Annotated[int, typer.Option(help="Gradient accumulation steps.")] = 1,
    warmup_ratio: Annotated[float, typer.Option(help="Warmup ratio.")] = 0.0,
    weight_decay: Annotated[float, typer.Option(help="Weight decay.")] = 0.0,
    # Evaluation
    eval_strategy: Annotated[str, typer.Option(help="Evaluation strategy: 'no', 'steps', 'epoch'.")] = "epoch",
    eval_steps: Annotated[int | None, typer.Option(help="Evaluation interval (if strategy='steps').")] = None,
    # Checkpointing
    save_strategy: Annotated[str, typer.Option(help="Save strategy: 'no', 'steps', 'epoch'.")] = "epoch",
    save_total_limit: Annotated[int | None, typer.Option(help="Max checkpoints to keep.")] = None,
    resume_from_checkpoint: Annotated[str | None, typer.Option(help="Path to checkpoint to resume from.")] = None,
    load_best_model_at_end: Annotated[bool, typer.Option(help="Load best model after training.")] = True,
    # Early stopping
    early_stopping: Annotated[bool, typer.Option(help="Enable early stopping.")] = False,
    early_stopping_patience: Annotated[int, typer.Option(help="Early stopping patience (eval rounds).")] = 3,
    # LoRA / QLoRA
    lora: Annotated[bool, typer.Option(help="Use LoRA for parameter-efficient fine-tuning.")] = False,
    lora_r: Annotated[int, typer.Option(help="LoRA rank.")] = 16,
    lora_alpha: Annotated[int, typer.Option(help="LoRA alpha.")] = 32,
    lora_dropout: Annotated[float, typer.Option(help="LoRA dropout.")] = 0.05,
    # Quantization
    quantization: Annotated[str | None, typer.Option(help="Quantize base model: 'bnb-4bit', 'bnb-8bit'.")] = None,
    # Precision & device
    dtype: Annotated[str, typer.Option(help="Training dtype: 'auto', 'float16', 'bfloat16', 'float32'.")] = "auto",
    device: Annotated[str | None, typer.Option(help="Device to train on: 'cpu', 'cuda', 'mps', 'tpu'.")] = None,
    gradient_checkpointing: Annotated[bool, typer.Option(help="Enable gradient checkpointing.")] = False,
    # Distributed
    multi_gpu: Annotated[bool, typer.Option(help="Use all available GPUs on this machine.")] = False,
    nnodes: Annotated[
        int | None, typer.Option(help="Number of nodes for multi-node training (uses torchrun).")
    ] = None,
    deepspeed: Annotated[str | None, typer.Option(help="DeepSpeed config: 'zero2', 'zero3', or path to JSON.")] = None,
    fsdp: Annotated[str | None, typer.Option(help="FSDP strategy: 'full-shard', 'shard-grad-op', 'offload'.")] = None,
    # Logging
    logging: Annotated[str | None, typer.Option(help="Logging integration: 'tensorboard', 'wandb', 'comet'.")] = None,
    # Hub
    push_to_hub: Annotated[bool, typer.Option(help="Push final model to the Hub.")] = False,
    hub_model_id: Annotated[str | None, typer.Option(help="Hub repository ID.")] = None,
    token: Annotated[str | None, typer.Option(help="HF Hub token.")] = None,
    trust_remote_code: Annotated[bool, typer.Option(help="Trust remote code.")] = False,
    # HPO
    hpo: Annotated[str | None, typer.Option(help="HPO backend: 'optuna', 'ray'.")] = None,
    hpo_trials: Annotated[int, typer.Option(help="Number of HPO trials.")] = 10,
    # Pretraining from scratch
    from_scratch: Annotated[bool, typer.Option(help="Initialize model from scratch (random weights).")] = False,
    mlm: Annotated[bool, typer.Option(help="Use masked language modeling (for language-modeling task).")] = False,
):
    """
    Fine-tune or pretrain a model on a dataset.

    The first argument is the task name (e.g., ``text-classification``).
    The model, dataset, and output directory are required options. All
    other options have sensible defaults.

    Examples::

        # Basic fine-tuning
        transformers train text-classification --model bert-base-uncased --dataset glue/sst2 --output ./out --epochs 3

        # LoRA
        transformers train text-generation --model meta-llama/Llama-3.2-1B --dataset ./data.jsonl --output ./out --lora

        # Resume from checkpoint
        transformers train text-classification --model bert-base-uncased --dataset glue/sst2 --output ./out --resume-from-checkpoint ./out/checkpoint-500

        # Multi-GPU
        transformers train text-generation --model meta-llama/Llama-3.2-1B --dataset ./data.jsonl --output ./out --multi-gpu

        # MPS (Apple Silicon)
        transformers train text-classification --model bert-base-uncased --dataset glue/sst2 --output ./out --device mps
    """
    import transformers
    from transformers import (
        AutoConfig,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    if task not in _TASK_CONFIGS:
        raise SystemExit(f"Unknown task '{task}'. Choose from: {', '.join(_TASK_CONFIGS.keys())}")

    task_config = _TASK_CONFIGS[task]

    # Override: if MLM flag is set, use masked LM
    if task == "language-modeling" and not mlm:
        task_config = {**task_config, "auto_class": "AutoModelForCausalLM"}

    # --- Load dataset ---
    ds = _load_dataset(dataset, subset, token)

    # Split if there's no validation set
    if "validation" not in ds and "test" not in ds:
        split = ds["train"].train_test_split(test_size=0.1, seed=42)
        ds["train"] = split["train"]
        ds["validation"] = split["test"]

    eval_split = "validation" if "validation" in ds else "test"

    # --- Determine label count for classification ---
    num_labels = None
    label_col = task_config.get("label_column")
    if isinstance(label_col, tuple):
        for c in label_col:
            if c in ds["train"].column_names:
                label_col = c
                break
        else:
            label_col = label_col[0]
    if label_col and label_col in ds["train"].column_names:
        features = ds["train"].features
        if hasattr(features[label_col], "names"):
            num_labels = features[label_col].num_classes

    # --- Load model & processing_class ---
    auto_cls = getattr(transformers, task_config["auto_class"])

    model_kwargs = {}
    if trust_remote_code:
        model_kwargs["trust_remote_code"] = True
    if token:
        model_kwargs["token"] = token
    if num_labels is not None:
        model_kwargs["num_labels"] = num_labels

    if quantization == "bnb-4bit":
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    elif quantization == "bnb-8bit":
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    if from_scratch:
        config = AutoConfig.from_pretrained(
            model, **{k: v for k, v in model_kwargs.items() if k not in ("quantization_config",)}
        )
        loaded_model = auto_cls.from_config(config)
    else:
        loaded_model = auto_cls.from_pretrained(model, **model_kwargs)

    # Load processor based on task type
    processing_class = None
    preprocess_type = task_config["preprocess"]

    if preprocess_type.startswith("tokenize") or preprocess_type == "audio_transform":
        tok_kwargs = {}
        if trust_remote_code:
            tok_kwargs["trust_remote_code"] = True
        if token:
            tok_kwargs["token"] = token
        processing_class = AutoTokenizer.from_pretrained(model, **tok_kwargs)
    elif preprocess_type == "image_transform":
        from transformers import AutoImageProcessor

        proc_kwargs = {}
        if trust_remote_code:
            proc_kwargs["trust_remote_code"] = True
        if token:
            proc_kwargs["token"] = token
        processing_class = AutoImageProcessor.from_pretrained(model, **proc_kwargs)

    # --- Preprocess dataset ---
    if preprocess_type == "tokenize":
        text_col = _detect_text_column(ds["train"], task_config["text_columns"])

        def preprocess_fn(examples):
            return processing_class(examples[text_col], truncation=True, max_length=max_seq_length)

        ds = ds.map(preprocess_fn, batched=True)
    elif preprocess_type == "tokenize_causal":
        text_col = _detect_text_column(ds["train"], task_config["text_columns"])

        def preprocess_fn(examples):
            return processing_class(examples[text_col], truncation=True, max_length=max_seq_length)

        ds = ds.map(preprocess_fn, batched=True)
    elif preprocess_type == "tokenize_seq2seq":
        columns = ds["train"].column_names
        # Find source and target columns
        source_col = columns[0]
        target_col = (
            label_col if label_col and label_col in columns else columns[1] if len(columns) > 1 else columns[0]
        )

        def preprocess_fn(examples):
            model_inputs = processing_class(examples[source_col], truncation=True, max_length=max_seq_length)
            labels = processing_class(text_target=examples[target_col], truncation=True, max_length=max_seq_length)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        ds = ds.map(preprocess_fn, batched=True)
    elif preprocess_type == "image_transform":
        from torchvision.transforms import Compose, Normalize, RandomResizedCrop, ToTensor

        _normalize = Normalize(
            mean=processing_class.image_mean if hasattr(processing_class, "image_mean") else [0.485, 0.456, 0.406],
            std=processing_class.image_std if hasattr(processing_class, "image_std") else [0.229, 0.224, 0.225],
        )
        _size = processing_class.size.get("shortest_edge", 224) if hasattr(processing_class, "size") else 224
        _transforms = Compose([RandomResizedCrop(_size), ToTensor(), _normalize])

        def preprocess_fn(examples):
            examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
            return examples

        ds["train"].set_transform(preprocess_fn)
        if eval_split in ds:
            ds[eval_split].set_transform(preprocess_fn)

    # --- LoRA ---
    if lora:
        from peft import LoraConfig, get_peft_model

        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type="CAUSAL_LM" if "CausalLM" in task_config["auto_class"] else "SEQ_CLS",
        )
        loaded_model = get_peft_model(loaded_model, peft_config)
        loaded_model.print_trainable_parameters()

    # --- Build TrainingArguments ---
    training_args_kwargs = {
        "output_dir": output,
        "num_train_epochs": epochs,
        "learning_rate": lr,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": eval_batch_size or batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "eval_strategy": eval_strategy,
        "save_strategy": save_strategy,
        "load_best_model_at_end": load_best_model_at_end and eval_strategy != "no",
        "gradient_checkpointing": gradient_checkpointing,
        "push_to_hub": push_to_hub,
    }

    if eval_steps is not None:
        training_args_kwargs["eval_steps"] = eval_steps
    if save_total_limit is not None:
        training_args_kwargs["save_total_limit"] = save_total_limit
    if hub_model_id is not None:
        training_args_kwargs["hub_model_id"] = hub_model_id
    if token is not None:
        training_args_kwargs["hub_token"] = token

    # Precision
    if dtype == "float16":
        training_args_kwargs["fp16"] = True
    elif dtype == "bfloat16":
        training_args_kwargs["bf16"] = True

    # Device targeting
    if device == "cpu":
        training_args_kwargs["no_cuda"] = True
        training_args_kwargs["use_mps_device"] = False
    elif device == "mps":
        training_args_kwargs["use_mps_device"] = True
    elif device == "tpu":
        # TPU is handled automatically when running on a TPU instance with XLA
        pass

    # Multi-GPU / multi-node: delegate to accelerate or torchrun
    if multi_gpu or nnodes is not None:
        # Build the command to re-launch via accelerate
        cmd = ["accelerate", "launch"]
        if nnodes is not None:
            cmd.extend(["--num_machines", str(nnodes)])
        if multi_gpu:
            cmd.append("--multi_gpu")
        cmd.extend(["--module", "transformers.cli.agentic.train", "_train_inner"])
        # Pass all original args through environment
        print(f"Launching distributed training: {' '.join(cmd)}")
        print("Note: for full control, use `accelerate launch` directly.")
        # Fall through to normal training — Trainer handles multi-GPU automatically
        # when CUDA_VISIBLE_DEVICES or the accelerate launcher sets up the environment.

    # Distributed
    if deepspeed is not None:
        if deepspeed in ("zero2", "zero3"):
            # Use built-in DeepSpeed configs
            training_args_kwargs["deepspeed"] = deepspeed
        else:
            training_args_kwargs["deepspeed"] = deepspeed
    if fsdp is not None:
        training_args_kwargs["fsdp"] = fsdp

    # Logging
    if logging is not None:
        training_args_kwargs["report_to"] = logging

    training_args = TrainingArguments(**training_args_kwargs)

    # --- Data collator ---
    data_collator = None
    if preprocess_type == "tokenize":
        from transformers import DataCollatorWithPadding

        data_collator = DataCollatorWithPadding(tokenizer=processing_class)
    elif preprocess_type in ("tokenize_causal", "tokenize_mlm"):
        from transformers import DataCollatorForLanguageModeling

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=processing_class,
            mlm=(preprocess_type == "tokenize_mlm" or mlm),
        )
    elif preprocess_type == "tokenize_seq2seq":
        from transformers import DataCollatorForSeq2Seq

        data_collator = DataCollatorForSeq2Seq(tokenizer=processing_class, model=loaded_model)

    # --- Callbacks ---
    callbacks = []
    if early_stopping:
        from transformers import EarlyStoppingCallback

        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    # --- Build Trainer ---
    trainer_cls = Trainer
    if "Seq2Seq" in task_config["auto_class"]:
        from transformers import Seq2SeqTrainer

        trainer_cls = Seq2SeqTrainer

    trainer = trainer_cls(
        model=loaded_model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get(eval_split),
        processing_class=processing_class,
        data_collator=data_collator,
        callbacks=callbacks if callbacks else None,
    )

    # --- Train ---
    if hpo is not None:
        best_trial = trainer.hyperparameter_search(
            direction="minimize",
            backend=hpo,
            n_trials=hpo_trials,
        )
        print(f"Best trial: {best_trial}")
    else:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # --- Save ---
    trainer.save_model(output)
    if processing_class is not None:
        processing_class.save_pretrained(output)

    print(f"\nModel saved to {output}")
    if push_to_hub:
        trainer.push_to_hub()
        print(f"Pushed to Hub: {hub_model_id or output}")
