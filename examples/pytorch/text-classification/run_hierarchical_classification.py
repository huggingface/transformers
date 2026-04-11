#!/usr/bin/env python
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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

# /// script
# dependencies = [
#     "transformers @ git+https://github.com/huggingface/transformers.git",
#     "accelerate >= 0.12.0",
#     "datasets >= 1.8.0",
#     "sentencepiece != 0.1.92",
#     "scikit-learn",
#     "torch >= 1.3",
#     "evaluate",
# ]
# ///

"""
Hierarchical Text Classification using Transformers.

This script demonstrates how to implement hierarchical classification where:
- Level 1: Coarse categories (e.g., Science, Sports, Politics)
- Level 2: Fine-grained categories (e.g., Physics, Chemistry within Science)

The model uses predictions from earlier levels to inform later level predictions.
"""

import logging
import os
import random
import sys
from dataclasses import dataclass, field

import datasets
import evaluate
import numpy as np
import torch
import torch.nn as nn

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.modeling_outputs import ModelOutput
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


check_min_version("4.57.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/text-classification/requirements.txt",
)


logger = logging.getLogger(__name__)


HIERARCHY_DEFINITION = {
    "Technology": ["Software", "Hardware", "AI", "Security", "Gaming"],
    "Science": ["Physics", "Chemistry", "Biology", "Astronomy", "Mathematics"],
    "Sports": ["Football", "Basketball", "Tennis", "Baseball", "Soccer"],
    "Business": ["Finance", "Marketing", "Management", "Economics", "Startup"],
    "Entertainment": ["Movies", "Music", "Television", "Gaming", "Books"],
}

PARENT_LABELS = list(HIERARCHY_DEFINITION.keys())
CHILD_LABELS = []
PARENT_TO_CHILD_IDX = {}
for parent_idx, parent in enumerate(PARENT_LABELS):
    PARENT_TO_CHILD_IDX[parent_idx] = []
    for child in HIERARCHY_DEFINITION[parent]:
        CHILD_LABELS.append(child)
        PARENT_TO_CHILD_IDX[parent_idx].append(len(CHILD_LABELS) - 1)

NUM_PARENT_LABELS = len(PARENT_LABELS)
NUM_CHILD_LABELS = len(CHILD_LABELS)


SAMPLE_TEXTS = {
    "Software": [
        "The new Python framework simplifies API development with automatic documentation generation.",
        "Microsoft released a new version of VS Code with improved IntelliSense features.",
    ],
    "Hardware": [
        "NVIDIA announced next-generation GPUs with 2x performance improvement.",
        "AMD's new processor architecture offers better power efficiency.",
    ],
    "AI": [
        "Researchers developed a new transformer model achieving state-of-the-art results.",
        "Deep learning approach solves protein folding prediction problem.",
    ],
    "Security": [
        "Critical vulnerability discovered in popular authentication library.",
        "New ransomware attack targets enterprise cloud infrastructure.",
    ],
    "Gaming": [
        "New open-world RPG features ray tracing and realistic physics simulation.",
        "The latest game engine update brings photorealistic graphics capabilities.",
    ],
    "Physics": [
        "Scientists observe new particle behavior at CERN accelerator.",
        "Quantum entanglement experiments show faster-than-light correlation.",
    ],
    "Chemistry": [
        "New catalyst enables efficient water splitting for hydrogen production.",
        "Researchers synthesize material with unprecedented electrical conductivity.",
    ],
    "Biology": [
        "Gene therapy technique successfully treats hereditary blindness.",
        "CRISPR technology enables precise genome editing in human cells.",
    ],
    "Astronomy": [
        "James Webb telescope captures detailed images of distant galaxy formation.",
        "New exoplanet discovered in habitable zone of nearby star system.",
    ],
    "Mathematics": [
        "Mathematicians prove long-standing conjecture in number theory.",
        "New algorithm achieves optimal complexity for matrix multiplication.",
    ],
    "Football": [
        "Super Bowl LIX sets new viewership record with thrilling overtime finish.",
        "College football playoffs expand to include more conference champions.",
    ],
    "Basketball": [
        "NBA draft lottery reveals surprising top picks for next season.",
        "Golden State Warriors announce major roster changes for championship run.",
    ],
    "Tennis": [
        "Wimbledon finals feature intense five-set match between top players.",
        "New tennis technique revolutionizes serve optimization.",
    ],
    "Baseball": [
        "MLB announces experimental rule changes for faster game pace.",
        "Youngest player in history hits record-breaking home run.",
    ],
    "Soccer": [
        "World Cup qualification matches determine tournament brackets.",
        "European Championship delivers unexpected group stage upsets.",
    ],
    "Finance": [
        "Federal Reserve announces interest rate policy adjustments.",
        "Stock market indices reach new all-time highs amid strong earnings.",
    ],
    "Marketing": [
        "Social media platform introduces new targeted advertising features.",
        "Influencer marketing ROI exceeds traditional advertising benchmarks.",
    ],
    "Management": [
        "Remote work policies evolve with hybrid team coordination tools.",
        "New leadership training program emphasizes emotional intelligence.",
    ],
    "Economics": [
        "GDP growth exceeds expectations in quarterly economic report.",
        "Inflation metrics show stabilization following policy interventions.",
    ],
    "Startup": [
        "Venture capital funding reaches record levels for AI startups.",
        "Unicorn company announces strategic acquisition of competitor.",
    ],
    "Movies": [
        "Award-winning director announces new epic science fiction film.",
        "Streaming platform releases critically acclaimed documentary series.",
    ],
    "Music": [
        "Grammy nominations highlight emerging artists across multiple genres.",
        "Virtual concert technology enables global interactive performances.",
    ],
    "Television": [
        "Streaming series breaks viewership records with season finale.",
        "New drama receives critical acclaim for innovative storytelling.",
    ],
    "Books": [
        "Bestselling author announces highly anticipated sequel release date.",
        "Literary prize recognizes debut novel for experimental narrative.",
    ],
}


@dataclass
class HierarchicalClassificationOutput(ModelOutput):
    parent_logits: torch.FloatTensor = None
    child_logits: torch.FloatTensor = None
    loss: torch.FloatTensor | None = None


class HierarchicalClassificationModel(transformers.PreTrainedModel):
    def __init__(
        self,
        config,
        num_parent_labels=5,
        num_child_labels=25,
        hierarchy_alpha=0.5,
    ):
        super().__init__(config)
        self.num_parent_labels = num_parent_labels
        self.num_child_labels = num_child_labels
        self.hierarchy_alpha = hierarchy_alpha

        self.encoder = AutoModel.from_config(config)

        hidden_size = config.hidden_size
        classifier_dropout = (
            config.classifier_dropout
            if hasattr(config, "classifier_dropout")
            else config.hidden_dropout_prob
        )

        self.dropout = nn.Dropout(classifier_dropout)

        self.parent_classifier = nn.Linear(hidden_size, num_parent_labels)

        self.child_classifier = nn.Linear(
            hidden_size + num_parent_labels, num_child_labels
        )

        self.loss_fct = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        parent_labels=None,
        child_labels=None,
        **kwargs,
    ):
        outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        parent_logits = self.parent_classifier(pooled_output)

        parent_probs = torch.softmax(parent_logits, dim=-1)
        child_input = torch.cat([pooled_output, parent_probs], dim=-1)
        child_logits = self.child_classifier(child_input)

        loss = None
        if parent_labels is not None and child_labels is not None:
            parent_loss = self.loss_fct(parent_logits, parent_labels)

            parent_probs_detached = parent_probs.detach()
            constrained_child_logits = self._apply_hierarchy_constraints(
                child_logits, parent_probs_detached
            )
            child_loss = self.loss_fct(constrained_child_logits, child_labels)

            loss = (
                self.hierarchy_alpha * parent_loss
                + (1 - self.hierarchy_alpha) * child_loss
            )

        return HierarchicalClassificationOutput(
            parent_logits=parent_logits,
            child_logits=child_logits,
            loss=loss,
        )

    def _apply_hierarchy_constraints(self, child_logits, parent_probs):
        constrained_logits = child_logits.clone()
        batch_size = child_logits.size(0)

        for i in range(batch_size):
            for parent_idx, child_indices in PARENT_TO_CHILD_IDX.items():
                if parent_probs[i, parent_idx] < 0.3:
                    for child_idx in child_indices:
                        constrained_logits[i, child_idx] = float("-inf")

        return constrained_logits


@dataclass
class DataTrainingArguments:
    dataset_name: str | None = field(
        default=None, metadata={"help": "The name of the dataset to use."}
    )
    max_seq_length: int = field(
        default=128, metadata={"help": "Max sequence length for tokenization."}
    )
    preprocessing_num_workers: int | None = field(
        default=None, metadata={"help": "Number of workers for preprocessing."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite cached datasets."}
    )
    max_train_samples: int | None = field(
        default=None, metadata={"help": "Max training samples."}
    )
    max_eval_samples: int | None = field(
        default=None, metadata={"help": "Max eval samples."}
    )
    hierarchy_alpha: float = field(
        default=0.5, metadata={"help": "Weight for parent loss vs child loss."}
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier."}
    )
    config_name: str | None = field(
        default=None, metadata={"help": "Pretrained config name or path."}
    )
    tokenizer_name: str | None = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path."}
    )
    cache_dir: str | None = field(
        default=None, metadata={"help": "Cache directory for models."}
    )
    use_fast_tokenizer: bool = field(
        default=True, metadata={"help": "Use fast tokenizer."}
    )


def generate_synthetic_dataset(
    num_samples=500, val_samples=100, test_samples=100
):
    all_samples = []
    for child_label, texts in SAMPLE_TEXTS.items():
        parent_label = None
        for parent, children in HIERARCHY_DEFINITION.items():
            if child_label in children:
                parent_label = parent
                break

        for text in texts:
            for _ in range(num_samples // len(SAMPLE_TEXTS)):
                all_samples.append(
                    {
                        "text": text,
                        "parent_label": parent_label,
                        "child_label": child_label,
                    }
                )

    random.shuffle(all_samples)

    for sample in all_samples[:num_samples]:
        sample["parent_label"] = PARENT_LABELS.index(sample["parent_label"])
        sample["child_label"] = CHILD_LABELS.index(sample["child_label"])

    train_samples = all_samples[:num_samples]
    val_samples = all_samples[num_samples : num_samples + val_samples]
    test_samples = all_samples[
        num_samples + val_samples : num_samples + val_samples + test_samples
    ]

    def to_dataset(samples, split_name):
        return (
            datasets.Dataset.from_list(samples)
            .rename_column("parent_label", "parent_labels")
            .rename_column("child_label", "child_labels")
            .train_test_split(test_size=0.0, seed=42)
        )

    train_dataset = (
        datasets.Dataset.from_list(train_samples)
        .rename_column("parent_label", "parent_labels")
        .rename_column("child_label", "child_labels")
    )
    val_dataset = (
        datasets.Dataset.from_list(val_samples)
        .rename_column("parent_label", "parent_labels")
        .rename_column("child_label", "child_labels")
    )
    test_dataset = (
        datasets.Dataset.from_list(test_samples)
        .rename_column("parent_label", "parent_labels")
        .rename_column("child_label", "child_labels")
    )

    dataset_dict = datasets.DatasetDict(
        {
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset,
        }
    )

    return dataset_dict


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = (
            parser.parse_args_into_dataclasses()
        )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    logger.warning(f"Training/evaluation parameters {training_args}")

    set_seed(training_args.seed)

    logger.info("Generating synthetic hierarchical classification dataset...")
    raw_datasets = generate_synthetic_dataset()

    logger.info(
        f"Dataset sizes: train={len(raw_datasets['train'])}, val={len(raw_datasets['validation'])}, test={len(raw_datasets['test'])}"
    )
    logger.info(f"Number of parent labels: {NUM_PARENT_LABELS}")
    logger.info(f"Number of child labels: {NUM_CHILD_LABELS}")

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )

    model = HierarchicalClassificationModel(
        config,
        num_parent_labels=NUM_PARENT_LABELS,
        num_child_labels=NUM_CHILD_LABELS,
        hierarchy_alpha=data_args.hierarchy_alpha,
    )

    def preprocess_function(examples):
        result = tokenizer(
            examples["text"],
            padding=False,
            max_length=data_args.max_seq_length,
            truncation=True,
        )
        result["parent_labels"] = examples["parent_labels"]
        result["child_labels"] = examples["child_labels"]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Tokenizing dataset",
        )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(
                len(train_dataset), data_args.max_train_samples
            )
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(
                len(eval_dataset), data_args.max_eval_samples
            )
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    accuracy_metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)
    f1_metric = evaluate.load("f1", cache_dir=model_args.cache_dir)

    def compute_metrics(p):
        parent_preds = np.argmax(p.predictions[0], axis=1)
        child_preds = np.argmax(p.predictions[1], axis=1)

        parent_result = accuracy_metric.compute(
            predictions=parent_preds, references=p.label_ids[0]
        )
        child_result = accuracy_metric.compute(
            predictions=child_preds, references=p.label_ids[1]
        )
        child_f1_result = f1_metric.compute(
            predictions=child_preds, references=p.label_ids[1], average="macro"
        )

        return {
            "parent_accuracy": parent_result["accuracy"],
            "child_accuracy": child_result["accuracy"],
            "child_f1_macro": child_f1_result["f1"],
        }

    def data_collator(batch):
        input_ids = torch.tensor([example["input_ids"] for example in batch])
        attention_mask = torch.tensor(
            [example["attention_mask"] for example in batch]
        )
        parent_labels = torch.tensor(
            [example["parent_labels"] for example in batch]
        )
        child_labels = torch.tensor(
            [example["child_labels"] for example in batch]
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "parent_labels": parent_labels,
            "child_labels": child_labels,
        }

    class HierarchicalTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = (inputs.get("parent_labels"), inputs.get("child_labels"))
            outputs = model(
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
                parent_labels=inputs.get("parent_labels"),
                child_labels=inputs.get("child_labels"),
            )
            loss = outputs.loss
            return (
                (loss, (outputs.parent_logits, outputs.child_logits, labels))
                if return_outputs
                else loss
            )

        def prediction_step(
            self, model, inputs, prediction_only_output=False, ignore_keys=None
        ):
            outputs = model(
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
            )
            return ((outputs.parent_logits, outputs.child_logits),)

    trainer = HierarchicalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_model()
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(raw_datasets["validation"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        predictions = trainer.predict(raw_datasets["test"])
        parent_preds = np.argmax(predictions.predictions[0], axis=1)
        child_preds = np.argmax(predictions.predictions[1], axis=1)

        output_predict_file = os.path.join(
            training_args.output_dir, "hierarchical_predictions.txt"
        )
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                writer.write(
                    "index\tparent_pred\tchild_pred\tparent_true\tchild_true\n"
                )
                for i, (pp, cp) in enumerate(zip(parent_preds, child_preds)):
                    pt = raw_datasets["test"][i]["parent_labels"]
                    ct = raw_datasets["test"][i]["child_labels"]
                    writer.write(
                        f"{i}\t{PARENT_LABELS[pp]}\t{CHILD_LABELS[cp]}\t{PARENT_LABELS[pt]}\t{CHILD_LABELS[ct]}\n"
                    )
            logger.info(f"Predict results saved at {output_predict_file}")

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "text-classification",
        "task_type": "hierarchical",
    }

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    main()
