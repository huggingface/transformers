import argparse
import glob
import logging
import os
from argparse import Namespace
from importlib import import_module

import numpy as np
import torch
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset

from lightning_base import BaseTransformer, add_generic_args, generic_train
from utils_ner import TokenClassificationTask


logger = logging.getLogger(__name__)


class NERTransformer(BaseTransformer):
    """
    A training module for NER. See BaseTransformer for the core options.
    """

    mode = "token-classification"

    def __init__(self, hparams):
        if type(hparams) == dict:
            hparams = Namespace(**hparams)
        module = import_module("tasks")
        try:
            token_classification_task_clazz = getattr(module, hparams.task_type)
            self.token_classification_task: TokenClassificationTask = token_classification_task_clazz()
        except AttributeError:
            raise ValueError(
                f"Task {hparams.task_type} needs to be defined as a TokenClassificationTask subclass in {module}. "
                f"Available tasks classes are: {TokenClassificationTask.__subclasses__()}"
            )
        self.labels = self.token_classification_task.get_labels(hparams.labels)
        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        super().__init__(hparams, len(self.labels), self.mode)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_num):
        "Compute loss and log."
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        if self.config.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch[2] if self.config.model_type in ["bert", "xlnet"] else None
            )  # XLM and RoBERTa don"t use token_type_ids

        outputs = self(**inputs)
        loss = outputs[0]
        # tensorboard_logs = {"loss": loss, "rate": self.lr_scheduler.get_last_lr()[-1]}
        return {"loss": loss}

    def prepare_data(self):
        "Called to initialize data. Use the call to construct features"
        args = self.hparams
        for mode in ["train", "dev", "test"]:
            cached_features_file = self._feature_file(mode)
            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                logger.info("Loading features from cached file %s", cached_features_file)
                features = torch.load(cached_features_file)
            else:
                logger.info("Creating features from dataset file at %s", args.data_dir)
                examples = self.token_classification_task.read_examples_from_file(args.data_dir, mode)
                features = self.token_classification_task.convert_examples_to_features(
                    examples,
                    self.labels,
                    args.max_seq_length,
                    self.tokenizer,
                    cls_token_at_end=bool(self.config.model_type in ["xlnet"]),
                    cls_token=self.tokenizer.cls_token,
                    cls_token_segment_id=2 if self.config.model_type in ["xlnet"] else 0,
                    sep_token=self.tokenizer.sep_token,
                    sep_token_extra=False,
                    pad_on_left=bool(self.config.model_type in ["xlnet"]),
                    pad_token=self.tokenizer.pad_token_id,
                    pad_token_segment_id=self.tokenizer.pad_token_type_id,
                    pad_token_label_id=self.pad_token_label_id,
                )
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

    def get_dataloader(self, mode: int, batch_size: int, shuffle: bool = False) -> DataLoader:
        "Load datasets. Called after prepare data."
        cached_features_file = self._feature_file(mode)
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        if features[0].token_type_ids is not None:
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        else:
            all_token_type_ids = torch.tensor([0 for f in features], dtype=torch.long)
            # HACK(we will not use this anymore soon)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        return DataLoader(
            TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids), batch_size=batch_size
        )

    def validation_step(self, batch, batch_nb):
        """Compute validation""" ""
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        if self.config.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch[2] if self.config.model_type in ["bert", "xlnet"] else None
            )  # XLM and RoBERTa don"t use token_type_ids
        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()
        return {"val_loss": tmp_eval_loss.detach().cpu(), "pred": preds, "target": out_label_ids}

    def _eval_end(self, outputs):
        "Evaluation called for both Val and Test"
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        preds = np.concatenate([x["pred"] for x in outputs], axis=0)
        preds = np.argmax(preds, axis=2)
        out_label_ids = np.concatenate([x["target"] for x in outputs], axis=0)

        label_map = {i: label for i, label in enumerate(self.labels)}
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != self.pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        results = {
            "val_loss": val_loss_mean,
            "accuracy_score": accuracy_score(out_label_list, preds_list),
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }

        ret = {k: v for k, v in results.items()}
        ret["log"] = results
        return ret, preds_list, out_label_list

    def validation_epoch_end(self, outputs):
        # when stable
        ret, preds, targets = self._eval_end(outputs)
        logs = ret["log"]
        return {"val_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

    def test_epoch_end(self, outputs):
        # updating to test_epoch_end instead of deprecated test_end
        ret, predictions, targets = self._eval_end(outputs)

        # Converting to the dict required by pl
        # https://github.com/PyTorchLightning/pytorch-lightning/blob/master/\
        # pytorch_lightning/trainer/logging.py#L139
        logs = ret["log"]
        # `val_loss` is the key returned by `self._eval_end()` but actually refers to `test_loss`
        return {"avg_test_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        # Add NER specific options
        BaseTransformer.add_model_specific_args(parser, root_dir)
        parser.add_argument(
            "--task_type", default="NER", type=str, help="Task type to fine tune in training (e.g. NER, POS, etc)"
        )
        parser.add_argument(
            "--max_seq_length",
            default=128,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )

        parser.add_argument(
            "--labels",
            default="",
            type=str,
            help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",
        )
        parser.add_argument(
            "--gpus",
            default=0,
            type=int,
            help="The number of GPUs allocated for this, it is by default 0 meaning none",
        )

        parser.add_argument(
            "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
        )

        return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_args(parser, os.getcwd())
    parser = NERTransformer.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()
    model = NERTransformer(args)
    trainer = generic_train(model, args)

    if args.do_predict:
        # See https://github.com/huggingface/transformers/issues/3159
        # pl use this format to create a checkpoint:
        # https://github.com/PyTorchLightning/pytorch-lightning/blob/master\
        # /pytorch_lightning/callbacks/model_checkpoint.py#L169
        checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "checkpointepoch=*.ckpt"), recursive=True)))
        model = model.load_from_checkpoint(checkpoints[-1])
        trainer.test(model)
