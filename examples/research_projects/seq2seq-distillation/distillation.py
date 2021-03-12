#!/usr/bin/env python

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import List

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from finetune import SummarizationModule, TranslationModule
from finetune import main as ft_main
from make_student import create_student_by_copying_alternating_layers, get_layers_to_supervise
from transformers import AutoModelForSeq2SeqLM, MBartTokenizer, T5ForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right
from utils import calculate_bleu, check_output_dir, freeze_params, label_smoothed_nll_loss, use_task_specific_params


# need the parent dir module
sys.path.insert(2, str(Path(__file__).resolve().parents[1]))
from lightning_base import generic_train  # noqa


class SummarizationDistiller(SummarizationModule):
    """Supports T5, Bart, Pegasus and other models that inherit from Bart."""

    loss_names = ["loss", "ce_loss", "mlm_loss", "hid_loss_enc", "hid_loss_dec"]

    def __init__(self, hparams):
        assert Path(hparams.data_dir).exists()
        self.output_dir = Path(hparams.output_dir)
        self.output_dir.mkdir(exist_ok=True)

        save_dir = self.output_dir.joinpath("student")

        hparams.model_name_or_path = str(save_dir)  # Tell lightning we are training the student
        teacher = AutoModelForSeq2SeqLM.from_pretrained(hparams.teacher).eval()
        use_task_specific_params(teacher, hparams.task)  # We copy good generation parameters to student by default
        if hparams.student is not None:
            student = AutoModelForSeq2SeqLM.from_pretrained(hparams.student)
            use_task_specific_params(student, hparams.task)
            e_layer_ids, d_layer_ids = None, None
        else:
            student, e_layer_ids, d_layer_ids = create_student_by_copying_alternating_layers(
                teacher, e=hparams.student_encoder_layers, d=hparams.student_decoder_layers, save_path=save_dir
            )

        if hparams.length_penalty != -1:
            student.config.length_penalty = hparams.length_penalty
        hparams.tokenizer_name = hparams.teacher  # Use teacher's tokenizer
        super().__init__(hparams, model=student, config=student.config)
        assert (
            student.config.model_type == teacher.config.model_type
        ), f"teacher, student model types should be the same, got {student.config.model_type} != {teacher.config.model_type}"

        if student.config.model_type == "t5":
            student_encoder_layers = len(student.get_encoder().block)
            student_decoder_layers = len(student.get_decoder().block)
            teacher_encoder_layers = len(teacher.get_encoder().block)
            teacher_decoder_layers = len(teacher.get_decoder().block)
        else:
            student_encoder_layers = student.config.encoder_layers
            student_decoder_layers = student.config.decoder_layers
            teacher_encoder_layers = teacher.config.encoder_layers
            teacher_decoder_layers = teacher.config.decoder_layers

        self.different_base_models = not (hparams.student is None or hparams.teacher == hparams.student)
        self.do_calc_hidden_loss = (not self.different_base_models) and hparams.alpha_hid > 0
        self.different_encoder = self.different_base_models or (student_encoder_layers != teacher_encoder_layers)
        # self.different_encoder determines whether we need to run the teacher encoder
        self.teacher = teacher
        freeze_params(self.teacher)

        if not self.different_encoder:  # To save RAM, delete teacher encoder and freeze student encoder.
            try:
                del self.teacher.model.encoder
            except AttributeError:  # T5
                del self.teacher.encoder

        if e_layer_ids is None:
            e_layer_ids = list(range(student_encoder_layers))
        if d_layer_ids is None:
            d_layer_ids = list(range(student_decoder_layers))

        self.e_layer_ids, self.d_layer_ids = e_layer_ids, d_layer_ids  # type: List[int], List[int]

        if self.do_calc_hidden_loss:  # Intermediate supervision: Decide which layers to supervise
            if hparams.supervise_forward:
                self.e_matches = get_layers_to_supervise(
                    n_student=len(self.e_layer_ids), n_teacher=teacher_encoder_layers
                )
                self.d_matches = get_layers_to_supervise(
                    n_student=len(self.d_layer_ids), n_teacher=teacher_decoder_layers
                )
            else:  # student layer should emulate hidden states of the teacher layer it was copied from
                self.e_matches = self.e_layer_ids
                self.d_matches = self.d_layer_ids
        else:
            self.e_matches = None
            self.d_matches = None

        self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
        self.temperature = 2.0
        self.alpha_mlm = hparams.alpha_mlm
        self.alpha_ce = hparams.alpha_ce
        self.alpha_hid = hparams.alpha_hid
        gc.collect()
        torch.cuda.empty_cache()

    def calc_ce_loss(self, mask, s_logits, t_logits):
        """Copy pasted from distillbert (transformers/examples/distillation/)"""
        # mask has False at padding_idx
        sel_mask = mask[:, :, None].expand_as(s_logits)
        vocab_size = s_logits.size(-1)
        s_logits_slct = torch.masked_select(s_logits, sel_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        t_logits_slct = torch.masked_select(t_logits, sel_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        s_logits_slct = s_logits_slct.view(-1, vocab_size)  # (bs * seq_length, voc_size) modulo the 1s in mask
        t_logits_slct = t_logits_slct.view(-1, vocab_size)  # (bs * seq_length, voc_size) modulo the 1s in mask
        assert t_logits_slct.size() == s_logits_slct.size()
        loss_ce = (
            self.ce_loss_fct(
                F.log_softmax(s_logits_slct / self.temperature, dim=-1),
                F.softmax(t_logits_slct / self.temperature, dim=-1),
            )
            * (self.temperature) ** 2
        )
        return loss_ce

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        SummarizationModule.add_model_specific_args(parser, root_dir)
        add_distill_args(parser)
        return parser

    def _step(self, batch: dict) -> tuple:
        """Compute the loss for a batch"""
        pad_token_id = self.tokenizer.pad_token_id
        input_ids, src_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        if isinstance(self.model, T5ForConditionalGeneration):
            decoder_input_ids = self.model._shift_right(labels)
        else:
            decoder_input_ids = shift_tokens_right(labels, pad_token_id)

        # noinspection PyCallingNonCallable
        student_outputs = self(
            input_ids,
            attention_mask=src_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=self.do_calc_hidden_loss,
            output_attentions=False,
            use_cache=False,
        )
        lm_logits = student_outputs["logits"]

        # Same cross entropy vs. label smoothing logic as finetune.py
        assert lm_logits.shape[-1] == self.model.config.vocab_size
        if self.hparams.label_smoothing == 0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
            student_lm_loss = loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
        else:
            lprobs = F.log_softmax(lm_logits, dim=-1)
            student_lm_loss, _ = label_smoothed_nll_loss(
                lprobs, labels, self.hparams.label_smoothing, ignore_index=pad_token_id
            )

        def zero_tensor():
            return torch.tensor(0.0).type_as(student_lm_loss)

        teacher_enc_outputs = student_outputs[
            "encoder_last_hidden_state"
        ]  # use this unless self.different_base_models
        hid_loss_enc, hid_loss_dec = zero_tensor(), zero_tensor()
        if self.different_encoder:  # compute encoder hidden state loss
            all_teacher_encoder_outputs = self.teacher.get_encoder()(
                input_ids,
                attention_mask=src_mask,
                output_hidden_states=self.do_calc_hidden_loss,
            )
            if self.different_base_models:
                teacher_enc_outputs = all_teacher_encoder_outputs["last_hidden_state"]
            elif self.do_calc_hidden_loss:
                hid_loss_enc = self.calc_hidden_loss(
                    src_mask,
                    student_outputs["encoder_hidden_states"],
                    all_teacher_encoder_outputs["hidden_states"],
                    self.e_matches,
                    normalize_hidden=self.hparams.normalize_hidden,
                )

        teacher_outputs = self.teacher(
            input_ids,
            attention_mask=src_mask,
            encoder_outputs=(teacher_enc_outputs,),
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=self.do_calc_hidden_loss,
            use_cache=False,  # since we are not passing labels, never let this default to True
        )
        dec_mask = decoder_input_ids.ne(pad_token_id)
        loss_ce = self.calc_ce_loss(dec_mask, lm_logits, teacher_outputs["logits"])
        if self.do_calc_hidden_loss:  # Intermediate supervision of decoder hidden states
            hid_loss_dec = self.calc_hidden_loss(
                dec_mask,
                student_outputs["decoder_hidden_states"],
                teacher_outputs["decoder_hidden_states"],
                self.d_matches,
                normalize_hidden=self.hparams.normalize_hidden,
            )

        blended_loss = (
            self.alpha_ce * loss_ce
            + self.alpha_mlm * student_lm_loss
            + self.hparams.alpha_hid * (hid_loss_enc + hid_loss_dec)
        )
        return blended_loss, loss_ce, student_lm_loss, hid_loss_enc, hid_loss_dec

    @staticmethod
    def calc_hidden_loss(attention_mask, hidden_states, hidden_states_T, matches, normalize_hidden):
        """MSE(student_hid, teacher_hid[matches]). Called "Intermediate supervision" in paper. Inspired by TinyBERT."""
        msg = "expected list or tuple for hidden_states, got tensor of shape: "
        assert not isinstance(hidden_states, torch.Tensor), f"{msg}{hidden_states.shape}"
        assert not isinstance(hidden_states_T, torch.Tensor), f"{msg}{hidden_states_T.shape}"
        mask = attention_mask.to(hidden_states[0])
        valid_count = mask.sum() * hidden_states[0].size(-1)
        student_states = torch.stack([hidden_states[i] for i in range(len(matches))])
        teacher_states = torch.stack([hidden_states_T[j] for j in matches])
        assert student_states.shape == teacher_states.shape, f"{student_states.shape} != {teacher_states.shape}"
        if normalize_hidden:
            student_states = F.layer_norm(student_states, student_states.shape[1:])
            teacher_states = F.layer_norm(teacher_states, teacher_states.shape[1:])
        mse = F.mse_loss(student_states, teacher_states, reduction="none")
        masked_mse = (mse * mask.unsqueeze(0).unsqueeze(-1)).sum() / valid_count
        return masked_mse


def add_distill_args(parser):
    # NOTE: if --student argument was specified and the teacher and student base models
    # are different, the models still have to have the same tokenizer, specified by
    # --tokenizer_name. So, for example, you can distill from t5_large to t5_small but not
    # from bart to t5. This s because if the tokenizers are different, the output space
    # for the two models is also different and their logits are not comparable.
    parser.add_argument("--teacher", type=str)
    parser.add_argument("--alpha_ce", default=0.8, type=float)
    parser.add_argument("--alpha_mlm", default=0.2, type=float)
    parser.add_argument("--alpha_hid", default=0.0, type=float, required=False)
    parser.add_argument("--student", type=str, required=False)
    parser.add_argument("--student_decoder_layers", default=12, type=int, required=False)
    parser.add_argument("--student_encoder_layers", default=12, type=int, required=False)
    parser.add_argument("--no_teacher", action="store_true", default=False)
    parser.add_argument("--length_penalty", type=float, default=-1)
    parser.add_argument("--supervise_forward", action="store_true", default=False)
    parser.add_argument("--normalize_hidden", action="store_true", default=False)


class TranslationDistiller(SummarizationDistiller):
    """Supports T5, mBART, Marian, other models that inherit from Bart."""

    mode = "translation"
    metric_names = ["bleu"]
    default_val_metric = "bleu"

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        assert hparams.src_lang is not None
        assert hparams.tgt_lang is not None
        self.dataset_kwargs["src_lang"] = hparams.src_lang
        self.dataset_kwargs["tgt_lang"] = hparams.tgt_lang
        if self.model.config.decoder_start_token_id is None and isinstance(self.tokenizer, MBartTokenizer):
            self.decoder_start_token_id = self.tokenizer.lang_code_to_id[hparams.tgt_lang]

    def calc_generative_metrics(self, preds, target) -> dict:
        return calculate_bleu(preds, target)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        TranslationModule.add_model_specific_args(parser, root_dir)
        add_distill_args(parser)
        return parser


def create_module(args):
    if args.no_teacher:
        module_cls = TranslationModule if "translation" in args.task else SummarizationModule
    else:  # DISTILL WITH TEACHER
        module_cls = TranslationDistiller if "translation" in args.task else SummarizationDistiller
    args.setup_cls: str = module_cls.__name__
    print(f"using module {args.setup_cls}")
    model = module_cls(args)
    return model


def distill_main(args):
    Path(args.output_dir).mkdir(exist_ok=True)
    check_output_dir(args, expected_items=3)

    model = create_module(args)
    return ft_main(args, model=model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SummarizationDistiller.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()

    distill_main(args)
