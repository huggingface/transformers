import argparse
import glob
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from rouge_score import rouge_scorer, scoring
from torch import nn
from torch.utils.data import DataLoader

from durbango import lmap, pickle_load, pickle_save
from lightning_base import BaseTransformer, add_generic_args, generic_train, get_linear_schedule_with_warmup
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer
from transformers.modeling_bart import invert_mask


try:
    from .utils import SummarizationDataset, flatten_list
    from .bart_distiller import init_student, copy_layers
except ImportError:
    from utils import SummarizationDataset, flatten_list
    from bart_distiller import init_student, copy_layers

logger = logging.getLogger(__name__)

ROUGE_KEYS = ["rouge1", "rouge2", "rougeL"]


def calculate_rouge(output_lns: List[str], reference_lns: List[str]) -> Dict:
    # score_file = Path(score_path).open("w")
    scorer = rouge_scorer.RougeScorer(ROUGE_KEYS, use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    for reference_ln, output_ln in zip(reference_lns, output_lns):
        scores = scorer.score(reference_ln, output_ln)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    return {k: v.mid.fmeasure for k, v in result.items()}


class SummarizationTrainer(BaseTransformer):
    mode = "language-modeling"
    loss_names = ["loss"]

    def __init__(self, hparams, **kwargs):
        tokenizer = BartTokenizer.from_pretrained("bart-large")
        super().__init__(hparams, num_labels=None, mode=self.mode, tokenizer=tokenizer, **kwargs)
        self.model: BartForConditionalGeneration
        self.metrics_save_path = Path(self.output_dir) / "metrics.pkl"
        assert Path(self.output_dir).exists()
        if os.path.exists(self.metrics_save_path):
            self.metrics = pickle_load(self.metrics_save_path)
        else:
            self.metrics = {"train": [], "val": [], "test": []}

        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            overwrite_cache=self.hparams.no_cache,
            tgt_suffix=self.hparams.tgt_suffix,
        )
        base_nobs = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
        }

        self.target_lens = {
            "train": self.hparams.max_target_length,
            "val": self.hparams.val_mtl,
            "test": self.hparams.test_mtl,
        }
        assert self.target_lens["train"] <= self.target_lens["val"], f"target_lens: {self.target_lens}"
        assert self.target_lens["train"] <= self.target_lens["test"], f"target_lens: {self.target_lens}"
        self.n_obs = {k: v if v >= 0 else None for k, v in base_nobs.items()}
        self.freeze_stuff()
        freeze_part(self.model.model.encoder)

    def freeze_stuff(self):

        freeze_part(self.model.model.shared)
        for d in [self.model.model.encoder, self.model.model.decoder]:
            freeze_part(d.embed_positions)
            freeze_part(d.embed_tokens)

    @property
    def metrics_df(self):
        return pd.DataFrame(self.metrics)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, lm_labels=None):
        return self.model(
            input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, lm_labels=lm_labels,
        )

    def _step(self, batch: dict) -> Tuple:
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"]
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone()
        lm_labels[y[:, 1:] == pad_token_id] = -100
        outputs = self(source_ids, attention_mask=source_mask, decoder_input_ids=y_ids, lm_labels=lm_labels,)

        loss = outputs[0]

        return (loss,)

    def training_step(self, batch, batch_idx) -> Dict:
        loss_tensors = self._step(batch)
        logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        return {"loss": loss_tensors[0], "log": logs}

    def validation_step(self, batch, batch_idx) -> Dict:
        return self._generative_step(batch)

    def validation_end(self, outputs, prefix="val") -> Dict:
        losses = {k: torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names}
        loss = losses["loss"]
        rouges = {k: np.array([x[k] for x in outputs]).mean() for k in ROUGE_KEYS + ["gen_time"]}
        rouges.update({k: v.item() for k, v in losses.items()})
        losses.update(rouges)
        metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        self.metrics[prefix].append(metrics)

        pickle_save(self.metrics, self.metrics_save_path)
        preds = flatten_list([x["preds"] for x in outputs])
        ret_dict = {"log": metrics, "preds": preds}
        ret_dict[f"{prefix}_loss"] = loss
        return ret_dict

    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

    def _generative_step(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = SummarizationDataset.trim_seq2seq_batch(batch, pad_token_id)
        cfg = self.config
        # {'max_length': cfg.max_length, 'min_length': cfg.min_length, 'num_beams': cfg.num_beams}

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

    def get_dataset(self, type_path) -> SummarizationDataset:
        n_obs = self.n_obs[type_path]
        max_target_length = self.target_lens[type_path]
        dataset = SummarizationDataset.from_raw_data(
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
        if self.hparams.grouped_sampler and type_path == "train":
            sampler = dataset.make_sampler(self.hparams)
            shuffle = False
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            num_workers=4,
            sampler=sampler,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
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
        # Add BART specific options
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
            "--val_mtl",
            default=142,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--test_mtl",
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
            help="The input data dir. Should contain the dataset files for the CNN/DM summarization task.",
        )
        parser.add_argument(
            "--no_cache", action="store_true",
        )
        parser.add_argument("--tgt_suffix", type=str, default="", required=False)
        parser.add_argument("--n_train", type=int, default=-1, required=False)
        parser.add_argument("--n_val", type=int, default=-1, required=False)
        parser.add_argument("--n_test", type=int, default=-1, required=False)
        parser.add_argument("--grouped_sampler", action="store_true", default=False)

        return parser


def freeze_part(model: nn.Module):
    for par in model.parameters():
        par.requires_grad = False


def is_frozen(model):
    return not any(p.requires_grad for p in model.parameters())


def get_layers_to_copy(n_to_get, tot):
    all_layers = list(range(tot))
    if tot == 12:  # Alternating for special cases
        base = {6: [0, 2, 4, 7, 9, 11], 1: [11], 3: [0, 6, 11], 2: [0, 11],
                9: [0, 1, 2, 4, 5, 7, 9, 10, 11],
                12: all_layers}
        return base[n_to_get]
    else:
        return all_layers[:n_to_get]


class SummarizationDistiller(SummarizationTrainer):
    loss_names = ["loss", "ce_loss", "mlm_loss"]

    def __init__(self, hparams):
        assert Path(hparams.data_dir).exists()

        # Dump empty student model at a path, then call from_pretrained on it
        teacher = BartForConditionalGeneration.from_pretrained(hparams.teacher).eval()

        student_updates = {
            "decoder_layers": hparams.student_decoder_layers,
            "encoder_layers": hparams.student_encoder_layers,
        }
        d_layers_to_copy = get_layers_to_copy(student_updates["decoder_layers"], teacher.config.decoder_layers)
        e_layers_to_copy: List = get_layers_to_copy(student_updates["encoder_layers"], teacher.config.encoder_layers)
        hparams.layer_to_copy = d_layers_to_copy
        hparams.e_layer_to_copy = e_layers_to_copy

        kw = teacher.config.to_diff_dict()
        kw.update(student_updates)
        # Copy weights
        student_cfg = BartConfig(**kw)
        student = BartForConditionalGeneration(student_cfg)

        student, _ = init_student(student, teacher)
        self.different_encoder: bool = hparams.student_encoder_layers != teacher.config.encoder_layers
        self.different_decoder = hparams.student_decoder_layers != teacher.config.decoder_layers
        if self.different_decoder:
            copy_layers(teacher.model.decoder.layers, student.model.decoder.layers, d_layers_to_copy)
        if self.different_encoder:
            copy_layers(teacher.model.encoder.layers, student.model.encoder.layers, e_layers_to_copy)
        Path(hparams.output_dir).mkdir(exist_ok=True)

        super().__init__(hparams, model=student, config=student_cfg)
        if torch.cuda.is_available() and hparams.fp16:
            teacher = teacher.to(self.device).half()
        # assert len(student.model.encoder.layers) == 12
        if not self.different_encoder:
            freeze_part(self.model.model.encoder)
            teacher.model.encoder = None
        if not self.different_decoder:
            freeze_part(self.model.model.decoder)

        assert len(self.model.model.decoder.layers) == len(d_layers_to_copy)
        self.model.teacher = teacher
        self.refreeze()
        self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
        self.temperature = 2.0
        self.alpha_mlm = hparams.alpha_mlm
        self.alpha_ce = hparams.alpha_ce

    def refreeze(self):
        freeze_part(self.model.model.shared)
        d = self.model.model.decoder
        e = self.model.model.encoder
        freeze_part(d.embed_positions)
        freeze_part(d.embed_tokens)
        freeze_part(e.embed_positions)
        freeze_part(e.embed_tokens)
        freeze_part(self.model.teacher)

    def get_dataset(self, type_path) -> SummarizationDataset:
        n_obs = self.n_obs[type_path]
        dataset = SummarizationDataset.from_raw_data(
            self.tokenizer, type_path=type_path, n_obs=n_obs, **self.dataset_kwargs
        )
        return dataset

    def _step(self, batch):
        # assert is_frozen(self.model.teacher)
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"]
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone()
        lm_labels[y[:, 1:] == pad_token_id] = -100
        # noinspection PyCallingNonCallable
        sloss, slogits, enc_outputs = self(
            source_ids, attention_mask=source_mask, decoder_input_ids=y_ids, lm_labels=lm_labels,
        )
        if self.different_encoder:
            enc_outputs = None
        else:
            enc_outputs = (enc_outputs,)

        with torch.no_grad():
            tloss, tlogits, *trash = self.model.teacher(
                source_ids,
                attention_mask=source_mask,
                encoder_outputs=enc_outputs,
                decoder_input_ids=y_ids,
                lm_labels=lm_labels,
            )
        loss_ce, s_logits_slct, t_logits_slct = self.calc_ce_loss(self.model.model.last_padding_mask, slogits, tlogits)
        blended_loss = loss_ce * self.alpha_ce + self.alpha_mlm * sloss
        return blended_loss, loss_ce, sloss

    def calc_ce_loss(self, mask, s_logits, t_logits):
        if mask is not None:
            # mask has True at padding_idx
            mask = invert_mask(mask)
            sel_mask = mask[:, :, None].expand_as(s_logits)
            s_logits_slct = torch.masked_select(
                s_logits, sel_mask
            )  # (bs * seq_length * voc_size) modulo the 1s in mask
            t_logits_slct = torch.masked_select(
                t_logits, sel_mask
            )  # (bs * seq_length * voc_size) modulo the 1s in mask
        else:

            t_logits_slct = t_logits
            s_logits_slct = s_logits  # (bs * seq_length * voc_size) modulo the 1s in mask
        s_logits_slct = s_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
        t_logits_slct = t_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
        assert t_logits_slct.size() == s_logits_slct.size()
        loss_ce = (
            self.ce_loss_fct(
                F.log_softmax(s_logits_slct / self.temperature, dim=-1),
                F.softmax(t_logits_slct / self.temperature, dim=-1),
            )
            * (self.temperature) ** 2
        )
        return loss_ce, s_logits_slct, t_logits_slct

    def calc_cos_loss(self, attention_mask, s_hidden_states, t_hidden_states):
        s_hidden_states = s_hidden_states[-1]  # (bs, seq_length, dim)
        t_hidden_states = t_hidden_states[-1]  # (bs, seq_length, dim)
        mask = attention_mask.unsqueeze(-1).expand_as(s_hidden_states)  # (bs, seq_length, dim)
        assert s_hidden_states.size() == t_hidden_states.size()
        dim = s_hidden_states.size(-1)
        s_hidden_states_slct = torch.masked_select(s_hidden_states, mask)  # (bs * seq_length * dim)
        s_hidden_states_slct = s_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)
        t_hidden_states_slct = torch.masked_select(t_hidden_states, mask)  # (bs * seq_length * dim)
        t_hidden_states_slct = t_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)
        target = s_hidden_states_slct.new(s_hidden_states_slct.size(0)).fill_(1)  # (bs * seq_length,)
        loss_cos = self.cosine_loss_fct(s_hidden_states_slct, t_hidden_states_slct, target)
        return loss_cos

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        SummarizationTrainer.add_model_specific_args(parser, root_dir)
        parser.add_argument(
            "--teacher", default=None, type=str, required=True,
        )
        parser.add_argument(
            "--alpha_ce", default=0.8, type=float,
        )
        parser.add_argument(
            "--alpha_mlm", default=0.2, type=float,
        )
        parser.add_argument(
            "--student_decoder_layers", default=6, type=int, required=False,
        )
        parser.add_argument(
            "--student_encoder_layers", default=12, type=int, required=False,
        )
        parser.add_argument(
            "--no_teacher", action="store_true", default=False,
        )

        return parser


def main(args):
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not args.output_dir:
        args.output_dir = os.path.join("./results", f"dbart_{time.strftime('%Y%m%d_%H%M%S')}",)
        os.makedirs(args.output_dir)
    module_cls = SummarizationTrainer if args.no_teacher else SummarizationDistiller
    model: SummarizationTrainer = module_cls(args)
    trainer: pl.Trainer = generic_train(model, args, early_stopping_callback=True)
    checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "checkpointepoch=*.ckpt"), recursive=True)))
    if checkpoints:
        model = model.load_from_checkpoint(checkpoints[-1])
    # if not args.do_train:

    trainer.test(model)
    # model.metrics_df.to_csv(Path(model.output_dir)/'metrics.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_args(parser, os.getcwd())
    parser = SummarizationDistiller.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()

    main(args)
