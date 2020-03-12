import argparse
import glob
import logging
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import AdamW, BartTokenizer, get_linear_schedule_with_warmup
from transformers.modeling_bart import BartForConditionalGeneration
from utils import CnnDailyMailDataset, add_generic_args


logger = logging.getLogger(__name__)


class BartSystem(pl.LightningModule):
    def __init__(self, hparams):
        super(BartSystem, self).__init__()
        self.hparams = hparams
        self.bart = BartForConditionalGeneration.from_pretrained("bart-large", output_past=True)

        self.tokenizer = BartTokenizer.from_pretrained("bart-large")
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None):
        return self.bart(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )

    def _step(self, batch):
        outputs = self.forward(
            batch["source_ids"], attention_mask=batch["source_mask"], decoder_input_ids=batch["target_ids"]
        )

        out = outputs[0]

        logits = F.log_softmax(out, dim=-1)
        y = batch["target_ids"]
        norm = (y != self.tokenizer.pad_token_id).data.sum()

        targets = y.clone()
        targets[y == self.tokenizer.pad_token_id] = -100
        loss = self.criterion(logits.contiguous().view(-1, logits.size(-1)), targets.contiguous().view(-1)) / norm
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        generated_ids = self.bart.generate(
            batch["input_ids"].cuda(),
            attention_mask=batch["attention_mask"].cuda(),
            num_beams=5,
            max_length=40,
            repetition_penalty=3.0,
        )
        preds = [
            self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for g in generated_ids
        ]
        target = [
            self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for t in batch["target_ids"]
        ]
        return {"val_loss": loss, "preds": preds, "target": target}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_end(self, outputs):
        return self.validation_end(outputs)

    def test_epoch_end(self, outputs):
        output_test_predictions_file = os.path.join(self.hparams.output_dir, "test_predictions.txt")
        output_test_targets_file = os.path.join(self.hparams.output_dir, "test_targets.txt")
        # write predictions and targets for later rouge evaluation.
        with open(output_test_predictions_file, "w") as p_writer, open(output_test_targets_file, "w") as t_writer:
            for output_batch in outputs:
                p_writer.writelines(output_batch["preds"])
                t_writer.writelines(output_batch["target"])

        return self.test_end(outputs)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def configure_optimizers(self):
        t_total = len(self.train_dataloader())
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return [optimizer]

    def prepare_data(self):
        self.train_dataset = CnnDailyMailDataset(
            self.tokenizer, data_dir=self.hparams.data_dir, block_size=self.hparams.max_seq_length
        )
        self.val_dataset = CnnDailyMailDataset(
            self.tokenizer, data_dir=self.hparams.data_dir, type_path="val", block_size=self.hparams.max_seq_length
        )
        self.test_dataset = CnnDailyMailDataset(
            self.tokenizer, data_dir=self.hparams.data_dir, type_path="test", block_size=self.hparams.max_seq_length
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.train_batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.eval_batch_size)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        # Add BART specific options
        parser.add_argument(
            "--max_seq_length",
            default=1024,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )

        parser.add_argument(
            "--data_dir",
            default=None,
            type=str,
            required=True,
            help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
        )

        parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument(
            "--num_train_epochs", default=1, type=int, help="Total number of training epochs to perform."
        )

        parser.add_argument("--train_batch_size", default=4, type=int)
        parser.add_argument("--eval_batch_size", default=4, type=int)

        return parser


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def generic_train(model, args):
    # init model
    set_seed(args)

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=5
    )

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        early_stop_callback=False,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=checkpoint_callback,
        progress_bar_refresh_rate=1,  # progress bar is slow to update otherwise
    )

    if args.fp16:
        train_params["use_amp"] = args.fp16
        train_params["amp_level"] = args.fp16_opt_level

    if args.n_gpu > 1:
        train_params["distributed_backend"] = "ddp"

    trainer = pl.Trainer(**train_params)

    if args.do_train:
        trainer.fit(model)

    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_args(parser, os.getcwd())
    parser = BartSystem.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()
    model = BartSystem(args)
    trainer = generic_train(model, args)

    if args.do_predict:
        # See https://github.com/huggingface/transformers/issues/3159
        # pl use this format to create a checkpoint:
        # https://github.com/PyTorchLightning/pytorch-lightning/blob/master\
        # /pytorch_lightning/callbacks/model_checkpoint.py#L169
        checkpoints = list(sorted(glob.glob(args.output_dir + "/checkpointepoch=*.ckpt", recursive=True)))
        BartSystem.load_from_checkpoint(checkpoints[-1])
        trainer.test(model)
