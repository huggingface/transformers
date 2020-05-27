import glob
import logging
import os
from argparse import ArgumentParser, Namespace
from typing import List, Union

import nlp
import numpy as np
import pandas as pd
import pyarrow
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from lightning_base import BaseTransformer, LoggingCallback, set_seed
from transformers.data import glue_tasks_num_labels


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Glue tasks in nlp library have names without dashes, so we remove them
glue_tasks_num_labels = {k.replace("-", ""): v for k, v in glue_tasks_num_labels.items()}


class GLUETransformer(BaseTransformer):

    mode = "sequence-classification"

    def __init__(self, hparams: Namespace):
        """A Pytorch Lightning Module for training/evaluating transformers models on the GLUE benchmark

        Args:
            hparams (Namespace): Supplied CLI arguments to configure the run.
        """
        num_labels = glue_tasks_num_labels[hparams.task]
        hparams.glue_output_mode = "classification" if num_labels > 1 else "regression"

        super().__init__(hparams, num_labels, self.mode)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):

        outputs = self(**batch)
        loss = outputs[0]

        tensorboard_logs = {"loss": loss, "rate": self.lr_scheduler.get_last_lr()[-1]}
        return {"loss": loss, "log": tensorboard_logs}

    def prepare_data(self):
        """
        Use this to download and prepare data.
        In distributed (GPU, TPU), this will only be called once.
        This is called before requesting the dataloaders.
        """
        self.dataset = nlp.load_dataset("glue", name=self.hparams.task)
        self.metric = nlp.load_metric("glue", name=self.hparams.task)

        cached_dataset_file = self._feature_file(self.hparams.task)

        if os.path.exists(cached_dataset_file) and not self.hparams.overwrite_cache:
            logger.info("Loading from cached dataloader file ", cached_dataset_file)
            self.dataset = torch.load(cached_dataset_file)
        else:
            logger.info("No cache file found. Creating one at", cached_dataset_file)

            # Make the directory to hold cached dataset if it doesn't already exist
            os.makedirs(os.path.dirname(cached_dataset_file), exist_ok=True)

            # We don't know names of text field(s) so we find that here. If multiple, we tokenize text pairs.
            text_fields = [field.name for field in self.dataset["train"].schema if pyarrow.types.is_string(field.type)]

            def convert_to_features(example_batch):
                """Function to be mapped across GLUE datasets. Tokenizes text(s) and renames label column.

                Args:
                    example_batch (Dict[List]): Dict with feature names as keys and batch values as values

                Returns:
                    Dict[List]: The desired updates to the batch (keys are feature names, values are batch values)
                """

                # Either encode single sentence or sentence pairs
                if len(text_fields) > 1:
                    texts_or_text_pairs = list(zip(example_batch[text_fields[0]], example_batch[text_fields[1]]))
                else:
                    texts_or_text_pairs = example_batch[text_fields[0]]

                # Tokenize the text/text pairs
                features = self.tokenizer.batch_encode_plus(
                    texts_or_text_pairs, max_length=self.hparams.max_seq_length, pad_to_max_length=True
                )

                # Rename 'label' to 'labels' so we can directly unpack batches into model call function
                features["labels"] = example_batch["label"]

                return features

            # TODO - check if this is still right or if it needs updating
            if self.config.model_type in ["bert", "xlnet", "albert"]:
                cols_to_keep = ["input_ids", "attention_mask", "token_type_ids", "labels"]
            else:
                cols_to_keep = ["input_ids", "attention_mask", "labels"]

            # Splits are train, val, and test, but for MNLI there will be 2 val and 2 test datasets
            splits = ["train", "validation", "test"]
            if self.hparams.task == "mnli":
                splits = ["train"] + [s + mnli_split for mnli_split in ("_matched", "_mismatched") for s in splits[1:]]

            # Process each dataset inplace and set the format to torch tensors
            for split in splits:
                logger.info(f"Preparing {self.hparams.task} - Split: {split}")
                self.dataset[split] = self.dataset[split].map(convert_to_features, batched=True)
                self.dataset[split].set_format(
                    type="torch", columns=cols_to_keep if not split.startswith("test") else cols_to_keep + ["idx"]
                )

            # Save the processed data to cache file so we don't have to process same data more than once
            # TODO - is there a way to do this with just the nlp library?
            torch.save(self.dataset, cached_dataset_file)

    def load_dataset(self, mode: str, batch_size: int) -> Union[List[DataLoader], DataLoader]:
        """Get DataLoader(s) corresponding to the given split 'mode'

        Args:
            mode (str): The dataset split ('train', 'validation', 'test')
            batch_size (int): Number of examples to feed to model on each step.

        Returns:
            Union[List[DataLoader], DataLoader]: Single loader or list of loaders if MNLI
        """

        # Return two dataloaders for val/test datasets if MNLI
        if self.hparams.task == "mnli" and mode in ["validation", "test"]:
            return [
                DataLoader(
                    self.dataset[mode + "_matched"], batch_size=batch_size, num_workers=self.hparams.num_workers,
                ),
                DataLoader(
                    self.dataset[mode + "_mismatched"], batch_size=batch_size, num_workers=self.hparams.num_workers,
                ),
            ]

        # Otherwise, just return a single dataset for the given split mode
        return DataLoader(
            self.dataset[mode],
            batch_size=batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=bool(mode == "train"),
        )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._eval_step(batch, batch_idx, split="val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        idx = batch.pop("idx").detach().cpu().numpy()
        return {"idx": idx, **self._eval_step(batch, batch_idx, split="test")}

    def validation_epoch_end(self, outputs: list) -> dict:

        # For MNLI, we have to run self._eval_end twice for matched and mismatched splits
        if self.hparams.task == "mnli":
            logs = {}
            for i, split in enumerate(["matched", "mismatched"]):
                split_logs, _ = self._eval_end(outputs[i])
                # Convert 'validation_loss' to 'validation_loss_matched' or 'validation_loss_mismatched'
                logs.update({k + f"_{split}": v for k, v in split_logs.items()})

            # HACK - pytorch lightning is looking for 'val_loss' as a key in returned dict.
            # Here, we take mean of the matched and mismatched losses and return that as val_loss
            # Is there a better way to do this?
            logs["val_loss"] = torch.mean(torch.stack((logs["val_loss_matched"], logs["val_loss_mismatched"])))

        # For all other datasets, we can simply run self._eval_end once
        else:
            logs, _ = self._eval_end(outputs)

        # Update return dict by adding tensorboard logs and progress bar updates
        logs.update({"log": {**logs}, "progress_bar": {**logs}})
        return logs

    def test_epoch_end(self, outputs):

        # For MNLI, store predictions as Dict[List] where keys are matched and mismatched
        if self.hparams.task == "mnli":
            self.predictions = {}
            self.idxs = {}
            for i, split in enumerate(["matched", "mismatched"]):
                _, split_preds, split_idxs = self._eval_end(outputs[i], split="test")
                self.predictions[split] = split_preds
                self.idxs[split] = split_idxs

        # Otherwise, store predictions as List
        else:
            _, self.predictions, self.idxs = self._eval_end(outputs, split="test")

        return {}

    def _eval_step(self, batch, batch_idx, split="val"):
        outputs = self(**batch)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()
        out_label_ids = batch["labels"].detach().cpu().numpy()

        return {f"{split}_loss": tmp_eval_loss.detach().cpu(), "pred": preds, "target": out_label_ids}

    def _eval_end(self, outputs, split="val"):
        val_loss_mean = torch.stack([x[f"{split}_loss"] for x in outputs]).mean().detach().cpu()
        preds = np.concatenate([x["pred"] for x in outputs], axis=0)

        if self.hparams.glue_output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif self.hparams.glue_output_mode == "regression":
            preds = np.squeeze(preds)

        out_label_ids = np.concatenate([x["target"] for x in outputs], axis=0)

        results = {f"{split}_loss": val_loss_mean}

        to_return = (results, preds)

        # For validation dataset, include metric results.
        if split != "test":
            # HACK - the .tolist() call here is to prevent an error:
            # pyarrow.lib.ArrowInvalid: Floating point value truncated error
            results.update(self.metric.compute(preds.tolist(), out_label_ids.tolist()))

        # Test dataset should include idxs for submission
        else:
            idxs = np.concatenate([x["idx"] for x in outputs], axis=0)
            to_return += (idxs,)

        return to_return

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--max_seq_length",
            default=128,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--task", type=str, required=True, help="The GLUE task to run",
        )
        parser.add_argument(
            "--data_dir", default="./glue_dir", type=str, help="Directory to save/load processed cache data"
        )
        parser.add_argument(
            "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
        )
        return parser


def parse_args(args=None):
    parser = ArgumentParser()

    # add some script specific args
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir", type=str, default="", help="Directory to write outputs to or load checkpoint from"
    )
    parser.add_argument("--do_train", action="store_true", help="Run training loop")
    parser.add_argument("--do_predict", action="store_true", help="Run test loop")

    # enable all trainer args
    parser = Trainer.add_argparse_args(parser)

    # add the base module args
    parser = BaseTransformer.add_model_specific_args(parser)

    # add the glue module args
    parser = GLUETransformer.add_model_specific_args(parser)

    # cook them all up :)
    args = parser.parse_args(args)

    return args


def main(args):

    # Init Model
    model = GLUETransformer(args)

    # Init Trainer
    trainer = Trainer.from_argparse_args(args)

    # Include custom callback for logging results
    trainer.callbacks.append(LoggingCallback())

    # Run training and reload best model from checkpoint
    if args.do_train:

        # Fit the model
        trainer.fit(model)

        # Reload best model from current experiment run's checkpoint directory
        experiment_ckpt_dir = model.trainer.weights_save_path
        checkpoints = list(sorted(glob.glob(os.path.join(experiment_ckpt_dir, "epoch=*.ckpt"), recursive=True)))
        trainer.resume_from_checkpoint = checkpoints[-1]

    # Predict on test split and write to submission files
    if args.do_predict:

        # If we didn't train and and output directory is not supplied, raise an exception
        if not args.do_train and not bool(args.output_dir):
            raise RuntimeError("No output_dir is specified for writing results. Try setting --output_dir flag")
        # If we did train, but an output_dir was supplied, use the output_dir
        elif bool(args.output_dir):
            output_dir = args.output_dir
        # If we did train and no output_dir was supplied, use lightning_logs/version_x directory
        elif args.do_train and not bool(args.output_dir):
            output_dir = os.path.dirname(experiment_ckpt_dir)

        # Make the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Run on test data
        trainer.test(model)

        # Have to split MNLI submission into two files for matched and mismatched
        if args.task == "mnli":
            df_matched = pd.DataFrame({"idx": model.idxs["matched"], "prediction": model.predictions["matched"]})
            df_mismatched = pd.DataFrame(
                {"idx": model.idxs["mismatched"], "prediction": model.predictions["mismatched"]}
            )
            df_matched.to_csv(os.path.join(output_dir, "mnli_matched_submission.csv"), index=False)
            df_mismatched.to_csv(os.path.join(output_dir, "mnli_mismatched_submission.csv"), index=False)

        # All other tasks have single submission files
        else:
            df = pd.DataFrame({"idx": model.idxs, "prediction": model.predictions})
            df.to_csv(os.path.join(output_dir, f"{args.task}_submission.csv"), index=False)


if __name__ == "__main__":
    args = parse_args()
    set_seed(args)
    main(args)
