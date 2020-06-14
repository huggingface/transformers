import argparse
import glob
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import git
import numpy as np
import pytorch_lightning as pl
import torch
from rouge_score import rouge_scorer, scoring
from torch import nn
from torch.utils.data import DataLoader

from lightning_base import BaseTransformer, add_generic_args, generic_train
from transformers import AutoModelWithLMHead, get_linear_schedule_with_warmup


try:
    from .utils import SummarizationDataset, lmap, flatten_list, pickle_save
    from .callbacks import Seq2SeqLoggingCallback, get_rouge2_checkpoint_callback
except ImportError:
    from utils import SummarizationDataset, lmap, flatten_list, pickle_save
    from callbacks import Seq2SeqLoggingCallback, get_rouge2_checkpoint_callback

logger = logging.getLogger(__name__)

ROUGE_KEYS = ["rouge1", "rouge2", "rougeL"]


def save_git_info(folder_path: str):
    """
    Log commit info.
    """
    repo_infos = get_git_info()

    with open(os.path.join(folder_path, "git_log.json"), "w") as f:
        json.dump(repo_infos, f, indent=4)


def get_git_info():
    repo = git.Repo(search_parent_directories=True)
    repo_infos = {
        "repo_id": str(repo),
        "repo_sha": str(repo.head.object.hexsha),
        "repo_branch": str(repo.active_branch),
    }
    return repo_infos


def calculate_rouge(output_lns: List[str], reference_lns: List[str]) -> Dict:
    scorer = rouge_scorer.RougeScorer(ROUGE_KEYS, use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    for reference_ln, output_ln in zip(reference_lns, output_lns):
        scores = scorer.score(reference_ln, output_ln)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    return {k: v.mid.fmeasure for k, v in result.items()}


def dictify(rouge_obj) -> List:
    records = []
    for k, rouge_measurement in rouge_obj.items():
        if k == "rouge1":
            continue
        for k1 in ["low", "mid", "high"]:
            if k1 != "mid":
                continue
            v1 = getattr(rouge_measurement, k1)
            for k2 in ["precision", "recall", "fmeasure"]:
                records.append([k, k1, k2, getattr(v1, k2)])

    return records


def freeze_params(model: nn.Module):
    for par in model.parameters():
        par.requires_grad = False


def grad_status(model: nn.Module) -> Iterable:
    return (par.requires_grad for par in model.parameters())


def any_requires_grad(model: nn.Module) -> bool:
    return any(grad_status(model))


def assert_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert not any(model_grads), f"{n_require_grad/npars:.1%} of {npars} weights require grad"


def assert_not_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    npars = len(model_grads)
    assert any(model_grads), f"none of {npars} weights require grad"


class SummarizationTrainer(BaseTransformer):
    mode = "language-modeling"
    loss_names = ["loss"]

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, num_labels=None, mode=self.mode, **kwargs)
        save_git_info(self.hparams.output_dir)
        self.model: AutoModelWithLMHead
        self.metrics_save_path = Path(self.output_dir) / "metrics.pkl"
        self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        self.step_count = 0
        self.metrics = {"train": [], "val": [], "test": []}

        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            # overwrite_cache=self.hparams.no_cache,
            prefix=self.model.config.prefix or "",
        )
        base_nobs = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
        }

        self.target_lens = {
            "train": self.hparams.max_target_length,
            "val": self.hparams.val_max_target_length,
            "test": self.hparams.test_max_target_length,
        }
        assert self.target_lens["train"] <= self.target_lens["val"], f"target_lens: {self.target_lens}"
        assert self.target_lens["train"] <= self.target_lens["test"], f"target_lens: {self.target_lens}"
        self.n_obs = {k: v if v >= 0 else None for k, v in base_nobs.items()}
        if self.hparams.freeze_embeds:
            self.freeze_embeds()
        if self.hparams.freeze_encoder:
            freeze_params(self.model.model.encoder)
        self.hparams.git_sha = get_git_info()["repo_sha"]
        self.num_workers = 4 if self.hparams.gpus <= 1 else None

    def freeze_embeds(self):
        if self.model.config.model_type == "bart":
            freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)
        else:
            freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                freeze_params(d.embed_tokens)

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

    def _step(self, batch: dict) -> Tuple:
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"]
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone()
        lm_labels[y[:, 1:] == pad_token_id] = -100
        outputs = self(source_ids, attention_mask=source_mask, decoder_input_ids=y_ids, labels=lm_labels,)
        loss = outputs[0]
        return (loss,)

    def training_step(self, batch, batch_idx) -> Dict:
        loss_tensors = self._step(batch)
        logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        return {"loss": loss_tensors[0], "log": logs}

    def validation_step(self, batch, batch_idx) -> Dict:
        return self._generative_step(batch)

    def validation_end(self, outputs, prefix="val") -> Dict:
        self.step_count += 1
        losses = {k: torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names}
        loss = losses["loss"]
        rouges = {k: np.array([x[k] for x in outputs]).mean() for k in ROUGE_KEYS + ["gen_time"]}
        rouge: torch.FloatTensor = torch.tensor(rouges["rouge2"]).type_as(loss)
        rouges.update({k: v.item() for k, v in losses.items()})
        losses.update(rouges)
        metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        metrics["step_count"] = self.step_count
        self.save_metrics(metrics, prefix)
        preds = flatten_list([x["preds"] for x in outputs])
        ret_dict = {"log": metrics, "preds": preds}
        ret_dict[f"{prefix}_loss"] = loss
        ret_dict[f"{prefix}_rouge"] = rouge
        return ret_dict

    def save_metrics(self, metrics, prefix) -> None:
        self.metrics[prefix].append(metrics)
        pickle_save(self.metrics, self.metrics_save_path)

    def _generative_step(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = SummarizationDataset.trim_seq2seq_batch(batch, pad_token_id)
        # TODO(SS): task specific params

        t0 = time.time()
        generated_ids = self.model.generate(input_ids=source_ids, attention_mask=source_mask, use_cache=True,)
        gen_time = time.time() - t0
        preds = self.ids_to_clean_text(generated_ids)
        target = self.ids_to_clean_text(y)
        loss_tensors = self._step(batch)
        base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        rouge: Dict = calculate_rouge(preds, target)
        summ_len = np.mean(lmap(len, generated_ids))
        base_metrics.update(gen_time=gen_time, summ_len=summ_len, preds=preds, target=target, **rouge)
        return base_metrics

    def test_step(self, batch, batch_idx):
        return self._generative_step(batch)

    def test_end(self, outputs):
        return self.validation_end(outputs, prefix="test")

    def test_epoch_end(self, outputs):
        output_test_predictions_file = os.path.join(self.hparams.output_dir, "test_predictions.txt")
        output_test_targets_file = os.path.join(self.hparams.output_dir, "test_targets.txt")
        # write predictions and targets for later rouge evaluation.
        with open(output_test_predictions_file, "w+") as p_writer, open(output_test_targets_file, "w+") as t_writer:
            for output_batch in outputs:
                p_writer.writelines(s + "\n" for s in output_batch["preds"])
                t_writer.writelines(s + "\n" for s in output_batch["target"])
            p_writer.close()
            t_writer.close()

        return self.test_end(outputs)

    def validation_epoch_end(self, outputs):
        self.validation_end(outputs, "val")

    def get_dataset(self, type_path) -> SummarizationDataset:
        n_obs = self.n_obs[type_path]
        max_target_length = self.target_lens[type_path]
        dataset = SummarizationDataset(
            self.tokenizer,
            type_path=type_path,
            n_obs=n_obs,
            max_target_length=max_target_length,
            **self.dataset_kwargs,
        )
        return dataset

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = self.get_dataset(type_path)
        sampler = None
        if self.hparams.sortish_sampler and type_path == "train":
            assert self.hparams.gpus <= 1  # TODO: assert earlier
            sampler = dataset.make_sortish_sampler(batch_size)
            shuffle = False

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            num_workers=self.num_workers,
            sampler=sampler,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.gpus)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)
        add_generic_args(parser, root_dir)
        parser.add_argument(
            "--max_source_length",
            default=1024,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--max_target_length",
            default=56,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--val_max_target_length",
            default=142,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--test_max_target_length",
            default=142,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )

        parser.add_argument(
            "--data_dir",
            default=None,
            type=str,
            required=True,
            help="The input data dir. Should contain Should contain train.source, train.target, val.source, val.target, test.source, test.target",
        )
        parser.add_argument(
            "--freeze_encoder", action="store_true",
        )
        parser.add_argument(
            "--freeze_embeds", action="store_true",
        )
        parser.add_argument("--n_train", type=int, default=-1, required=False)
        parser.add_argument("--n_val", type=int, default=500, required=False)
        parser.add_argument("--n_test", type=int, default=-1, required=False)
        parser.add_argument("--sortish_sampler", action="store_true", default=False)
        return parser


def main(args, model=None):
    Path(args.output_dir).mkdir(exist_ok=True)
    if len(os.listdir(args.output_dir)) > 3 and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if model is None:
        model: BaseTransformer = SummarizationTrainer(args)
    trainer: pl.Trainer = generic_train(
        model,
        args,
        early_stopping_callback=True,
        logging_callback=Seq2SeqLoggingCallback(),
        checkpoint_callback=get_rouge2_checkpoint_callback(args.output_dir),
    )
    if not args.do_predict:
        return model

    model.hparams.test_checkpoint = ""
    checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "*.ckpt"), recursive=True)))
    if checkpoints:
        model.hparams.test_checkpoint = checkpoints[-1]
        trainer.resume_from_checkpoint = checkpoints[-1]
    trainer.logger.log_hyperparams(model.hparams)
    trainer.test(model)  # NOTE(SS): this will break in DDP, known lightning issue. See evaluate_checkpoint
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = SummarizationTrainer.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()

    main(args)
