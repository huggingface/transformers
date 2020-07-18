import argparse
import gc
import os
from pathlib import Path
from typing import List

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from lightning_base import generic_train
from transformers import AdamW, BartConfig, BartForConditionalGeneration, T5Config, T5ForConditionalGeneration


try:
    from .finetune import SummarizationModule
    from .finetune import main as ft_main
    from .initialization_utils import init_student, copy_layers
    from .utils import use_task_specific_params, pickle_load, freeze_params, assert_all_frozen, any_requires_grad

except ImportError:
    from finetune import SummarizationModule
    from finetune import main as ft_main
    from initialization_utils import init_student, copy_layers
    from utils import use_task_specific_params, pickle_load, freeze_params, assert_all_frozen, any_requires_grad


class BartSummarizationDistiller(SummarizationModule):
    loss_names = ["loss", "ce_loss", "mlm_loss", "enc_mse_loss", "hid_loss_enc", "hid_loss_dec"]

    def __init__(self, hparams):
        assert Path(hparams.data_dir).exists()
        student, student_cfg, teacher = self.pre_init(hparams)

        super().__init__(hparams, model=student, config=student_cfg)
        self.teacher = teacher
        use_task_specific_params(self.teacher, "summarization")
        freeze_params(self.teacher)
        self.sanity_check_gradients()
        self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
        self.temperature = 2.0
        self.alpha_mlm = hparams.alpha_mlm
        self.alpha_ce = hparams.alpha_ce
        self.alpha_hid = hparams.alpha_hid
        # self.alpha_cos = hparams.alpha_cos
        self.alpha_encoder_loss = self.hparams.alpha_encoder_loss
        gc.collect()
        torch.cuda.empty_cache()

    def sanity_check_gradients(self):
        assert_all_frozen(self.teacher)
        assert_all_frozen(self.model.model.decoder.embed_tokens)
        assert_all_frozen(self.model.model.encoder.embed_tokens)
        if self.different_encoder:
            assert any_requires_grad(self.model.model.encoder)
        else:
            freeze_params(self.model.model.encoder)
            del self.teacher.model.encoder

    def pre_init(self, hparams):
        self.output_dir = Path(hparams.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        teacher = BartForConditionalGeneration.from_pretrained(hparams.teacher).eval()
        student_updates = {
            "decoder_layers": hparams.student_decoder_layers,
            "encoder_layers": hparams.student_encoder_layers,
        }
        if hparams.length_penalty != -1:
            student_updates["length_penalty"] = hparams.length_penalty
        d_layers_to_copy = get_layers_to_copy(student_updates["decoder_layers"], teacher.config.decoder_layers)
        e_layers_to_copy: List = get_layers_to_copy(student_updates["encoder_layers"], teacher.config.encoder_layers)
        hparams.d_layer_to_copy = d_layers_to_copy
        hparams.e_layer_to_copy = e_layers_to_copy
        kw = teacher.config.to_diff_dict()
        kw.update(student_updates)
        # Copy weights
        student_cfg = BartConfig(**kw)
        student = BartForConditionalGeneration(student_cfg)
        student, _ = init_student(student, teacher)
        save_dir = self.output_dir.joinpath("student")
        self.copy_to_student(d_layers_to_copy, e_layers_to_copy, hparams, student, teacher)
        student.save_pretrained(save_dir)
        hparams.model_name_or_path = str(save_dir)
        return student, student_cfg, teacher

    def copy_to_student(self, d_layers_to_copy, e_layers_to_copy, hparams, student, teacher):
        if teacher.config.model_type == "t5":
            return self.copy_t5_to_student(d_layers_to_copy, e_layers_to_copy, hparams, student, teacher)
        self.different_encoder: bool = hparams.student_encoder_layers != teacher.config.encoder_layers
        self.different_decoder = hparams.student_decoder_layers != teacher.config.decoder_layers
        if self.different_decoder:
            copy_layers(teacher.model.decoder.layers, student.model.decoder.layers, d_layers_to_copy)
        if self.different_encoder:
            copy_layers(teacher.model.encoder.layers, student.model.encoder.layers, e_layers_to_copy)

    def copy_t5_to_student(self, d_layers_to_copy, e_layers_to_copy, hparams, student, teacher):
        self.different_encoder: bool = hparams.student_encoder_layers != teacher.config.num_layers
        self.different_decoder = hparams.student_decoder_layers != teacher.config.num_layers
        if self.different_decoder:
            copy_layers(teacher.decoder.block, student.decoder.block, d_layers_to_copy)
        if self.different_encoder:
            copy_layers(teacher.encoder.block, student.encoder.block, e_layers_to_copy)

    def calc_mse_loss(self, teacher_outputs: torch.Tensor, student_outputs: torch.Tensor, mask) -> torch.FloatTensor:
        if mask is not None:
            # mask has False at padding_idx
            sel_mask = mask[:, :, None].expand_as(student_outputs).bool()
            s_logits_slct = torch.masked_select(student_outputs, sel_mask)
            t_logits_slct = torch.masked_select(teacher_outputs, sel_mask)
        else:
            t_logits_slct = teacher_outputs
            s_logits_slct = student_outputs
        return F.mse_loss(s_logits_slct, t_logits_slct)

    def calc_ce_loss(self, mask, s_logits, t_logits):
        if mask is not None:
            # mask has False at padding_idx
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

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        SummarizationModule.add_model_specific_args(parser, root_dir)
        parser.add_argument("--teacher", default="facebook/bart-large-cnn", type=str)
        parser.add_argument("--alpha_ce", default=0.8, type=float)
        parser.add_argument("--alpha_mlm", default=0.2, type=float)
        # parser.add_argument("--alpha_cos", default=0.0, type=float)
        parser.add_argument("--alpha_encoder_loss", default=0.0, type=float)
        parser.add_argument("--alpha_hid", default=0.0, type=float, required=False)
        parser.add_argument("--student_decoder_layers", default=12, type=int, required=False)
        parser.add_argument("--student_encoder_layers", default=12, type=int, required=False)
        parser.add_argument("--no_teacher", action="store_true", default=False)
        parser.add_argument("--length_penalty", type=float, default=-1)

        return parser

    def _step(self, batch):
        # assert is_frozen(self.teacher)
        pad_token_id = self.tokenizer.pad_token_id
        input_ids, src_mask, y = batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"]
        decoder_input_ids = y[:, :-1].contiguous()
        labels = y[:, 1:].clone()
        labels[y[:, 1:] == pad_token_id] = -100
        # noinspection PyCallingNonCallable
        sloss, slogits, dec_hidden, enc_outputs, enc_hidden_state = self(
            input_ids,
            attention_mask=src_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            output_hidden_states=True,
            output_attentions=False,
        )

        def zero_tensor():
            return torch.tensor(0.0).type_as(sloss)

        loss_encoder, hid_loss_enc, hid_loss_dec = zero_tensor(), zero_tensor(), zero_tensor()
        if self.different_encoder:
            with torch.no_grad():
                teacher_enc_outputs, teacher_enc_hid, _ = self.teacher.model.encoder(
                    input_ids, attention_mask=src_mask, output_hidden_states=True
                )
            if self.hparams.alpha_encoder_loss > 0:
                loss_encoder = self.calc_mse_loss(enc_outputs, teacher_enc_outputs, src_mask)

            hid_loss_enc = self.calc_hidden_loss(
                src_mask, enc_hidden_state, teacher_enc_hid, self.hparams.e_layer_to_copy
            )

        teacher_enc_outputs = (enc_outputs,)
        assert isinstance(teacher_enc_outputs, tuple), type(teacher_enc_outputs)

        with torch.no_grad():
            tloss, tlogits, tdec_hidden, _ = self.teacher(
                input_ids,
                attention_mask=src_mask,
                encoder_outputs=teacher_enc_outputs,
                decoder_input_ids=decoder_input_ids,
                lm_labels=labels,
                output_hidden_states=True,
            )
        dec_mask = decoder_input_ids.ne(pad_token_id)
        loss_ce, s_logits_slct, t_logits_slct = self.calc_ce_loss(dec_mask, slogits, tlogits)
        if self.alpha_hid > 0:
            hid_loss_dec = self.calc_hidden_loss(dec_mask, dec_hidden, tdec_hidden, self.hparams.d_layer_to_copy)

        blended_loss = (
            self.alpha_ce * loss_ce
            + self.alpha_mlm * sloss
            + self.hparams.alpha_encoder_loss * loss_encoder
            + self.hparams.alpha_hid * (hid_loss_enc + hid_loss_dec)
        )
        return blended_loss, loss_ce, sloss, loss_encoder, hid_loss_enc, hid_loss_dec

    def calc_hidden_loss(self, attention_mask, hidden_states, hidden_states_T, matches):
        assert not isinstance(
            hidden_states, torch.Tensor
        ), f"expected list or tuple for hidden_states, got tensor of shape {hidden_states.shape}"
        assert not isinstance(
            hidden_states_T, torch.Tensor
        ), f"expected list or tuple for hidden_states_T, got tensor of shape {hidden_states_T.shape}"
        mask = attention_mask.to(hidden_states[0])
        valid_count = mask.sum() * hidden_states[0].size(-1)
        hidden_losses = [
            (F.mse_loss(hidden_states[i], hidden_states_T[j], reduction="none") * mask.unsqueeze(-1)).sum()
            / valid_count
            for i, j in enumerate(matches)
        ]
        return sum(hidden_losses)


class T5SummarizationDistiller(BartSummarizationDistiller):
    def pre_init(self, hparams):
        raise NotImplementedError("T5 Distillation does not work yet")
        self.output_dir = Path(hparams.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        teacher = T5ForConditionalGeneration.from_pretrained(hparams.teacher)
        n_layer = hparams.student_decoder_layers
        assert n_layer == hparams.student_encoder_layers  # TODO(SS): relax this constraint so that we can do 12-6.
        d_layers_to_copy = get_layers_to_copy(n_layer, len(teacher.decoder.block))
        e_layers_to_copy: List = get_layers_to_copy(n_layer, len(teacher.encoder.block))
        student_updates = {"num_layers": n_layer}
        hparams.d_layer_to_copy = d_layers_to_copy
        hparams.e_layer_to_copy = e_layers_to_copy
        kw = teacher.config.to_diff_dict()

        kw.update(student_updates)
        # Copy weights
        student_cfg = T5Config(**kw)
        student = T5ForConditionalGeneration(student_cfg)
        student, _ = init_student(student, teacher)
        self.copy_to_student(d_layers_to_copy, e_layers_to_copy, hparams, student, teacher)
        Path(hparams.output_dir).mkdir(exist_ok=True)
        task_specific_params = student.config.task_specific_params
        if task_specific_params is not None:
            student.config.update(task_specific_params.get("summarization", {}))  # TODO: dont hardcode
        save_dir = self.output_dir.joinpath("student")
        save_dir.mkdir(exist_ok=True)

        student.save_pretrained(save_dir)
        hparams.model_name_or_path = str(save_dir)
        return student, student_cfg, teacher

    def freeze_embeds(self):
        freeze_params(self.model.shared)
        for d in [self.model.encoder, self.model.decoder]:
            freeze_params(d.embed_tokens)

    def sanity_check_gradients(self):
        """T5"""
        assert_all_frozen(self.teacher)
        assert_all_frozen(self.model.decoder.embed_tokens)
        assert_all_frozen(self.model.encoder.embed_tokens)
        if self.different_encoder:
            assert any_requires_grad(self.model.encoder)
        else:
            freeze_params(self.model.encoder)
            del self.teacher.model.encoder
        if self.different_decoder:
            assert any_requires_grad(self.model.decoder)
        else:
            freeze_params(self.model.decoder)  # TODO(SS): very suspicious

    def _step(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"]
        decoder_input_ids = y[:, :-1].contiguous()
        labels = y[:, 1:].clone()
        labels[y[:, 1:] == pad_token_id] = -100
        # noinspection PyCallingNonCallable
        dec_mask = decoder_input_ids.ne(pad_token_id)

        sloss, slogits, dec_hidden, enc_outputs, enc_hidden_state = self(
            source_ids,
            attention_mask=source_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
        )

        def zero_tensor():
            return torch.tensor(0.0).type_as(sloss)

        loss_encoder, hid_loss_enc, hid_loss_dec = zero_tensor(), zero_tensor(), zero_tensor()
        if self.different_encoder:
            with torch.no_grad():
                teacher_enc_outputs, teacher_enc_hid = self.teacher.encoder(
                    source_ids, attention_mask=source_mask, output_hidden_states=True, use_cache=False,
                )
            if self.hparams.alpha_encoder_loss > 0:
                loss_encoder = self.calc_mse_loss(enc_outputs, teacher_enc_outputs, source_mask)

            hid_loss_enc = self.calc_hidden_loss(
                source_mask, enc_hidden_state, teacher_enc_hid, self.hparams.e_layer_to_copy
            )

        teacher_enc_outputs = (enc_outputs,)
        assert isinstance(teacher_enc_outputs, tuple), type(teacher_enc_outputs)

        with torch.no_grad():
            tloss, tlogits, tdec_hidden, _ = self.teacher(
                source_ids,
                attention_mask=source_mask,
                encoder_outputs=teacher_enc_outputs,
                decoder_input_ids=decoder_input_ids,
                lm_labels=labels,
                output_hidden_states=True,
                use_cache=False,
            )

        loss_ce, s_logits_slct, t_logits_slct = self.calc_ce_loss(dec_mask, slogits, tlogits)
        if self.alpha_hid > 0:
            hid_loss_dec = self.calc_hidden_loss(dec_mask, dec_hidden, tdec_hidden, self.hparams.d_layer_to_copy)

        blended_loss = (
            self.alpha_ce * loss_ce
            + self.alpha_mlm * sloss
            + self.hparams.alpha_encoder_loss * loss_encoder
            + self.hparams.alpha_hid * (hid_loss_enc + hid_loss_dec)
        )
        return blended_loss, loss_ce, sloss, loss_encoder, hid_loss_enc, hid_loss_dec


def create_module(args):
    t5 = "t5" in args.model_name_or_path
    if args.no_teacher:
        assert not args.enc_only
        module_cls = SummarizationModule
    elif t5:
        module_cls = T5SummarizationDistiller
    elif args.enc_only:
        raise ValueError("Deleted that")
    else:
        module_cls = BartSummarizationDistiller
    args.setup_cls: str = module_cls.__name__
    model = module_cls(args)
    return model


def evaluate_checkpoint(ckpt_path: Path, dest_dir=None):
    exp_dir = ckpt_path.parent
    if dest_dir is None:
        dest_dir = exp_dir
    clash = list(dest_dir.glob("test_generations*"))
    if clash:
        print(f"SKIPPING to avoid overwriting {clash}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "hparams" in ckpt:
        args = argparse.Namespace(**ckpt["hparams"])
    else:
        args = argparse.Namespace(**pickle_load(exp_dir / "hparams.pkl"))
    args.resume_from_checkpoint = str(ckpt_path)
    args.do_train = False
    args.output_dir = str(dest_dir)
    args.n_gpu = 1
    args.eval_batch_size = 16
    Path(args.output_dir).mkdir(exist_ok=True)
    model = create_module(args)
    trainer: pl.Trainer = generic_train(model, args, early_stopping_callback=False)
    trainer.test(model)


def get_layers_to_copy(n_to_get, tot):
    all_layers = list(range(tot))
    if tot == 12:  # Alternating for special cases
        layers_to_copy = {  # maps  num layers in student -> which teacher layers to copy
            1: [0],
            2: [0, 6],
            3: [0, 6, 11],
            4: [0, 4, 8, 11],
            6: [0, 2, 4, 7, 9, 11],
            9: [0, 1, 2, 4, 5, 7, 9, 10, 11],
            12: all_layers,
        }
        return layers_to_copy[n_to_get]
    else:
        return all_layers[:n_to_get]  # TODO: better version on theseus-bart branch


def distill_main(args):
    Path(args.output_dir).mkdir(exist_ok=True)
    if len(os.listdir(args.output_dir)) > 3 and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    model = create_module(args)
    return ft_main(args, model=model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = BartSummarizationDistiller.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()

    distill_main(args)
