import argparse
import glob
import logging
import os

import numpy as np
import torch
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from transformer_base import BaseTransformer, add_generic_args, generic_train
from utils_ner import convert_examples_to_features, get_labels, read_examples_from_file


logger = logging.getLogger(__name__)


class NERTransformer(BaseTransformer):
    """
    A training module for NER. See BaseTransformer for the core options.
    """

    def __init__(self, hparams):
        self.labels = get_labels(hparams.labels)
        num_labels = len(self.labels)
        super(NERTransformer, self).__init__(hparams, num_labels)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_num):
        "Compute loss"
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        if self.hparams.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch[2] if self.hparams.model_type in ["bert", "xlnet"] else None
            )  # XLM and RoBERTa don"t use segment_ids

        outputs = self.forward(**inputs)
        loss = outputs[0]

        tensorboard_logs = {"loss": loss, "rate": self.lr_scheduler.get_last_lr()[-1]}
        return {"loss": loss, "log": tensorboard_logs}

    def load_dataset(self, mode, batch_size):
        labels = get_labels(self.hparams.labels)
        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        dataset = self.load_and_cache_examples(labels, self.pad_token_label_id, mode)
        if mode == "train":
            if self.hparams.n_gpu > 1:
                sampler = DistributedSampler(dataset)
            else:
                sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        return dataloader

    def validation_step(self, batch, batch_nb):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        if self.hparams.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch[2] if self.hparams.model_type in ["bert", "xlnet"] else None
            )  # XLM and RoBERTa don"t use segment_ids
        outputs = self.forward(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()

        return {"val_loss": tmp_eval_loss, "pred": preds, "target": out_label_ids}

    def _eval_end(self, outputs):
        "Task specific validation"
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
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }

        if self.is_logger():
            logger.info(self.proc_rank)
            logger.info("***** Eval results *****")
            for key in sorted(results.keys()):
                logger.info("  %s = %s", key, str(results[key]))

        tensorboard_logs = results
        ret = {k: v for k, v in results.items()}
        ret["log"] = tensorboard_logs
        return ret, preds_list, out_label_list

    def validation_end(self, outputs):
        ret, preds, targets = self._eval_end(outputs)
        return ret

    def test_end(self, outputs):
        ret, predictions, targets = self._eval_end(outputs)

        if self.is_logger():
            # Write output to a file:
            # Save results
            output_test_results_file = os.path.join(self.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(ret.keys()):
                    if key != "log":
                        writer.write("{} = {}\n".format(key, str(ret[key])))
            # Save predictions
            output_test_predictions_file = os.path.join(self.hparams.output_dir, "test_predictions.txt")
            with open(output_test_predictions_file, "w") as writer:
                with open(os.path.join(self.hparams.data_dir, "test.txt"), "r") as f:
                    example_id = 0
                    for line in f:
                        if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                            writer.write(line)
                            if not predictions[example_id]:
                                example_id += 1
                        elif predictions[example_id]:
                            output_line = line.split()[0] + " " + predictions[example_id].pop(0) + "\n"
                            writer.write(output_line)
                        else:
                            logger.warning(
                                "Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0]
                            )
        return ret

    def load_and_cache_examples(self, labels, pad_token_label_id, mode):
        args = self.hparams
        tokenizer = self.tokenizer
        if self.proc_rank not in [-1, 0] and mode == "train":
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}".format(
                mode, list(filter(None, args.model_name_or_path.split("/"))).pop(), str(args.max_seq_length)
            ),
        )
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s", args.data_dir)
            examples = read_examples_from_file(args.data_dir, mode)
            features = convert_examples_to_features(
                examples,
                labels,
                args.max_seq_length,
                tokenizer,
                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                sep_token=tokenizer.sep_token,
                sep_token_extra=bool(args.model_type in ["roberta"]),
                pad_on_left=bool(args.model_type in ["xlnet"]),
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                pad_token_label_id=pad_token_label_id,
            )
            if self.proc_rank in [-1, 0]:
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

        if self.proc_rank == 0 and mode == "train":
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        return dataset

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        # Add NER specific options
        BaseTransformer.add_model_specific_args(parser, root_dir)
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
            "--data_dir",
            default=None,
            type=str,
            required=True,
            help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
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
        checkpoints = list(sorted(glob.glob(args.output_dir + "/checkpoint_*.ckpt", recursive=True)))
        NERTransformer.load_from_checkpoint(checkpoints[-1])
        trainer.test(model)
