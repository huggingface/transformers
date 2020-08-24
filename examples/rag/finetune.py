import argparse
import glob
import logging
import os
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from examples.lightning_base import BaseTransformer, add_generic_args, generic_train
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
    RagSequenceModel,
    RagTokenModel,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup,
)


try:
    from .utils import (
        assert_all_frozen,
        use_task_specific_params,
        lmap,
        flatten_list,
        pickle_save,
        save_git_info,
        save_json,
        freeze_params,
        calculate_rouge,
        calculate_exact_match,
        get_git_info,
        ROUGE_KEYS,
        calculate_bleu_score,
        Seq2SeqDataset,
        label_smoothed_nll_loss,
        is_rag_model,
    )

    from .callbacks import Seq2SeqLoggingCallback, get_checkpoint_callback, get_early_stopping_callback
except ImportError:
    from utils import (
        Seq2SeqDataset,
        assert_all_frozen,
        use_task_specific_params,
        lmap,
        flatten_list,
        pickle_save,
        save_git_info,
        save_json,
        freeze_params,
        calculate_rouge,
        calculate_exact_match,
        get_git_info,
        ROUGE_KEYS,
        calculate_bleu_score,
        label_smoothed_nll_loss,
        is_rag_model,
    )
    from callbacks import Seq2SeqLoggingCallback, get_checkpoint_callback, get_early_stopping_callback

logger = logging.getLogger(__name__)


class GenerativeQAModule(BaseTransformer):
    mode = "generative_qa"
    loss_names = ["loss"]
    # metric_names = ROUGE_KEYS + ["em"]
    metric_names = ["em"]
    val_metric = "em"  # piktus

    def __init__(self, hparams, **kwargs):
        print("hparams", hparams)
        if hparams.model_type == "rag_sequence":
            model_type = RagSequenceModel
        elif hparams.model_type == "rag_token":
            model_type = RagTokenModel
        else:
            model_type = BartForConditionalGeneration

        model = model_type.from_pretrained(hparams.model_name_or_path)
        config = model.config
        tokenizer = (
            model.model.generator_tokenizer
            if is_rag_model(hparams.model_type)
            else AutoTokenizer.from_pretrained(hparams.model_name_or_path)
        )

        # set extra_model_params for generator configs
        if hparams.model_type in ["rag_sequence", "rag_token"]:
            extra_model_params = (
                ("encoder_layerdrop", "decoder_layerdrop", "attention_dropout", "dropout")
                if isinstance(model.model.generator, BartForConditionalGeneration)
                else ("dropout_rate",)
            )
            for p in extra_model_params:
                if getattr(hparams, p, None):
                    assert hasattr(
                        model.model.generator.config, p
                    ), f"generator model config doesn't have a `{p}` attribute"
                    setattr(model.model.generator.config, p, getattr(hparams, p))
                    delattr(hparams, p)

        super().__init__(
            hparams, num_labels=None, mode=self.mode, config=config, tokenizer=tokenizer, model=model, **kwargs
        )
        self.model_type = model_type

        use_task_specific_params(self.model, "generative_qa")
        save_git_info(self.hparams.output_dir)
        self.metrics_save_path = Path(self.output_dir) / "metrics.json"
        self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        pickle_save(self.hparams, self.hparams_save_path)
        self.step_count = 0
        self.metrics = defaultdict(list)

        prefix = None
        if (
            is_rag_model(model.base_model_prefix) and isinstance(model.model.generator, T5ForConditionalGeneration)
        ) or isinstance(model, T5ForConditionalGeneration):
            prefix = "generate answer: "
            print("EOS token", self.tokenizer.eos_token)
            self.model.model.generator.config.prefix = prefix

        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            prefix=prefix or self.model.config.prefix or "",
        )
        n_observations_per_split = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

        self.target_lens = {
            "train": self.hparams.max_target_length,
            "val": self.hparams.val_max_target_length,
            "test": self.hparams.test_max_target_length,
        }
        assert self.target_lens["train"] <= self.target_lens["val"], f"target_lens: {self.target_lens}"
        assert self.target_lens["train"] <= self.target_lens["test"], f"target_lens: {self.target_lens}"

        if self.hparams.freeze_embeds:
            self.freeze_embeds()
        if self.hparams.freeze_encoder:
            freeze_params(self.model.get_encoder())
            assert_all_frozen(self.model.get_encoder())

        self.hparams.git_sha = get_git_info()["repo_sha"]
        self.num_workers = hparams.num_workers
        self.distributed_port = self.hparams.distributed_port

    def init_ddp_connection(self, global_rank: int, world_size: int, is_slurm_managing_tasks: bool = True):
        print("custom init_ddp_connection")
        os.environ["MASTER_PORT"] = str(self.distributed_port)
        super().init_ddp_connection(global_rank, world_size, is_slurm_managing_tasks)
        if self.model.base_model_prefix in ["rag_sequence", "rag_token"]:
            print("running init_retrieval")
            self.model.model.retriever.init_retrieval(self.distributed_port)

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        try:
            freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)
        except AttributeError:
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
        source_ids, source_mask, target_ids = batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"]

        kwargs = {}
        if isinstance(self.model, T5ForConditionalGeneration):
            decoder_input_ids = self.model._shift_right(target_ids)
            lm_labels = target_ids
        elif isinstance(self.model, BartForConditionalGeneration):
            decoder_input_ids = target_ids[
                :, :-1
            ].contiguous()  # Why this line? - to match the length of the lm_labels
            lm_labels = target_ids[:, 1:].clone()  # why clone? - shift left
        else:
            assert is_rag_model(self.model.base_model_prefix)
            generator = self.model.model.generator
            if isinstance(generator, T5ForConditionalGeneration):
                if isinstance(self.model, RagTokenModel):
                    raise NotImplementedError("Currently T5 support is not implemented for RAGToken.")
                decoder_start_token_id = generator.config.decoder_start_token_id
                decoder_input_ids = (
                    torch.cat(
                        [torch.Tensor([[decoder_start_token_id]] * target_ids.shape[0]).to(target_ids), target_ids,],
                        dim=1,
                    )
                    if target_ids.shape[0] < self.target_lens["train"]
                    else generator._shift_right(target_ids)
                )
                # decoder_input_ids = target_ids
            elif isinstance(generator, BartForConditionalGeneration):
                decoder_input_ids = target_ids
            lm_labels = None

        assert decoder_input_ids is not None

        source_strings = self.tokenizer.batch_decode(source_ids, skip_special_tokens=False)
        decoder_strings = self.tokenizer.batch_decode(decoder_input_ids, skip_special_tokens=False)

        print("target_ids", target_ids.shape, target_ids)
        print("source_ids", source_ids.shape, source_ids)
        print("decoder_input_ids", decoder_input_ids.shape, decoder_input_ids)

        print("source_strings", source_strings)
        print("decoder_strings", decoder_strings)

        if lm_labels is not None:
            outputs = self(
                source_ids,
                attention_mask=source_mask,
                decoder_input_ids=decoder_input_ids,
                use_cache=False,
                labels=lm_labels,
                return_dict=True,
            )
        else:  # RAG models
            outputs = self(
                source_ids,
                attention_mask=source_mask,
                decoder_input_ids=decoder_input_ids,
                use_cache=False,
                return_loss=True,
                reduce=True,
                label_smoothing=self.hparams.label_smoothing,
            )

        loss = outputs["loss"]
        return (loss,)

    @property
    def pad(self) -> int:
        return self.tokenizer.pad_token_id

    def training_step(self, batch, batch_idx) -> Dict:
        loss_tensors = self._step(batch)

        logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        # tokens per batch
        logs["tpb"] = batch["input_ids"].ne(self.pad).sum() + batch["decoder_input_ids"].ne(self.pad).sum()

        return {"loss": loss_tensors[0], "log": logs}

    def validation_step(self, batch, batch_idx) -> Dict:
        print("VALIDATION STEP")
        return self._generative_step(batch)

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        self.step_count += 1
        losses = {k: torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names}
        loss = losses["loss"]
        print("len(outputs)", len(outputs))
        print(outputs)
        gen_metrics = {
            k: np.array([x[k] for x in outputs]).mean() for k in self.metric_names + ["gen_time", "gen_len"]
        }
        metrics_tensor: torch.FloatTensor = torch.tensor(gen_metrics[self.val_metric]).type_as(loss)
        gen_metrics.update({k: v.item() for k, v in losses.items()})

        # fix for https://github.com/PyTorchLightning/pytorch-lightning/issues/2424
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        reduced_metrics = metrics_tensor / dist.get_world_size()
        print("gen_metrics reduced", len(gen_metrics), gen_metrics, reduced_metrics)
        gen_metrics.update({self.val_metric: reduced_metrics.item()})

        losses.update(gen_metrics)
        metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        metrics["step_count"] = self.step_count
        self.save_metrics(metrics, prefix)  # writes to self.metrics_save_path
        preds = flatten_list([x["preds"] for x in outputs])
        return {"log": metrics, "preds": preds, f"{prefix}_loss": loss, f"{prefix}_{self.val_metric}": reduced_metrics}

    def save_metrics(self, latest_metrics, type_path) -> None:
        self.metrics[type_path].append(latest_metrics)
        save_json(self.metrics, self.metrics_save_path)

    def calc_generative_metrics(self, preds, target) -> Dict:
        return calculate_exact_match(preds, target)

    def _generative_step(self, batch: dict) -> dict:
        t0 = time.time()
        generated_ids = self.model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=True,
            min_length=1,  # make sure short answers are allowed
            max_length=self.target_lens["val"],  # no need for crazy long answers in NQ
            dedup=False,  # rag specific parameter
        )

        print("input_ids {}, generated_ids {}".format(batch["input_ids"].shape, generated_ids.shape))
        gen_time = (time.time() - t0) / batch["input_ids"].shape[0]
        preds: List[str] = self.ids_to_clean_text(generated_ids)
        target: List[str] = self.ids_to_clean_text(batch["decoder_input_ids"])
        loss_tensors = self._step(batch)
        base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        gen_metrics: Dict = self.calc_generative_metrics(preds, target)

        summ_len = np.mean(lmap(len, generated_ids))
        base_metrics.update(gen_time=gen_time, gen_len=summ_len, preds=preds, target=target, **gen_metrics)
        return base_metrics

    def test_step(self, batch, batch_idx):
        return self._generative_step(batch)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, prefix="test")

    def get_dataset(self, type_path) -> Seq2SeqDataset:
        n_obs = self.n_obs[type_path]
        max_target_length = self.target_lens[type_path]
        dataset = Seq2SeqDataset(
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
            // self.hparams.accumulate_grad_batches
            * float(self.hparams.max_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        if max(scheduler.get_last_lr()) > 0:
            warnings.warn("All learning rates are 0")
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
            default=142,  # these defaults are optimized for CNNDM. For xsum, see README.md.
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
            type=str,
            required=True,
            help="The input data dir. Should contain train.source, train.target, val.source, val.target, test.source, test.target",
        )
        parser.add_argument("--freeze_encoder", action="store_true")
        parser.add_argument("--freeze_embeds", action="store_true")
        parser.add_argument("--sortish_sampler", action="store_true", default=False)
        parser.add_argument("--logger_name", type=str, choices=["default", "wandb", "wandb_shared"], default="default")
        parser.add_argument("--n_train", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_val", type=int, default=500, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_test", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument(
            "--task", type=str, default="generative_qa", required=False, help="# examples. -1 means use all."
        )
        parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)

        parser.add_argument("--src_lang", type=str, default="", required=False)
        parser.add_argument("--tgt_lang", type=str, default="", required=False)
        parser.add_argument(
            "--early_stopping_patience",
            type=int,
            default=-1,
            required=False,
            help="-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So val_check_interval will effect it.",
        )
        parser.add_argument(
            "--distributed-port", type=int, default=-1, required=False, help="# examples. -1 means use all."
        )
        parser.add_argument(
            "--model_type",
            choices=["rag_sequence", "rag_token", "bart"],
            type=str,
            help="RAG model type: sequence or token, if none specified, the type is inferred from the model_name_or_path",
        )
        return parser


def main(args, model=None) -> GenerativeQAModule:
    Path(args.output_dir).mkdir(exist_ok=True)
    # if len(os.listdir(args.output_dir)) > 3 and args.do_train:
    # raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if model is None:
        if args.task == "generative_qa":
            model: GenerativeQAModule = GenerativeQAModule(args)
        else:
            model: GenerativeQAModule = TranslationModule(args)

    dataset = Path(args.data_dir).name
    if (
        args.logger_name == "default"
        or args.fast_dev_run
        or str(args.output_dir).startswith("/tmp")
        or str(args.output_dir).startswith("/var")
    ):
        logger = True  # don't pollute wandb logs unnecessarily
    elif args.logger_name == "wandb":
        from pytorch_lightning.loggers import WandbLogger

        project = os.environ.get("WANDB_PROJECT", dataset)
        logger = WandbLogger(name=model.output_dir.name, project=project)

    elif args.logger_name == "wandb_shared":
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(name=model.output_dir.name, project=f"hf_{dataset}")

    if args.early_stopping_patience >= 0:
        es_callback = get_early_stopping_callback(model.val_metric, args.early_stopping_patience)
    else:
        es_callback = False
    trainer: pl.Trainer = generic_train(
        model,
        args,
        logging_callback=Seq2SeqLoggingCallback(),
        checkpoint_callback=get_checkpoint_callback(args.output_dir, model.val_metric),
        early_stopping_callback=es_callback,
        logger=logger,
        # TODO: early stopping callback seems messed up
    )
    pickle_save(model.hparams, model.output_dir / "hparams.pkl")

    if not args.do_predict:
        return model

    model.hparams.test_checkpoint = ""
    checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "*.ckpt"), recursive=True)))
    if checkpoints:
        model.hparams.test_checkpoint = checkpoints[-1]
        trainer.resume_from_checkpoint = checkpoints[-1]  # best checkpoint
    trainer.logger.log_hyperparams(model.hparams)

    # test() without a model tests using the best checkpoint automatically
    trainer.test()

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = GenerativeQAModule.add_model_specific_args(parser, os.getcwd())

    args = parser.parse_args()

    main(args)
