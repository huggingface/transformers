"""Finetuning script for RAG models. Adapted from examples.seq2seq.finetune.py"""

import argparse
import copy
import json
import logging
import multiprocessing
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from datasets import concatenate_datasets, load_from_disk
from torch.utils.data import DataLoader

from transformers import (
    AutoConfig,
    AutoTokenizer,
    BartForConditionalGeneration,
    BatchEncoding,
    DPRConfig,
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
    RagConfig,
    RagSequenceForGeneration,
    RagTokenForGeneration,
    RagTokenizer,
    T5ForConditionalGeneration,
)
from transformers import logging as transformers_logging
from transformers.integrations import is_ray_available


if is_ray_available():
    import ray
    from distributed_ray_retriever import RagRayDistributedRetriever, RayRetriever

from glob import glob

from callbacks_rag import Seq2SeqLoggingCallback, get_checkpoint_callback, get_early_stopping_callback
from kb_encode_utils import add_index, embed_update
from lightning_base import BaseTransformer, add_generic_args, generic_train
from pynvml import nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit
from utils_rag import (
    Seq2SeqDataset,
    calculate_exact_match,
    get_git_info,
    is_rag_model,
    lmap,
    pickle_save,
    save_git_info,
    save_json,
    set_extra_model_params,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

transformers_logging.set_verbosity_info()


sys.path.insert(2, str(Path(__file__).resolve().parents[1]))
isEmUpdateBusy = False
isAddIndexBusy = False
processes = []
threadHandle_index = None


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class GenerativeQAModule(BaseTransformer):
    mode = "generative_qa"
    loss_names = ["loss"]
    metric_names = ["em"]
    val_metric = "em"

    def __init__(self, hparams, **kwargs):
        # when loading from a pytorch lightning checkpoint, hparams are passed as dict
        if isinstance(hparams, dict):
            hparams = AttrDict(hparams)
        if hparams.model_type == "rag_sequence":
            self.model_class = RagSequenceForGeneration
        elif hparams.model_type == "rag_token":
            self.model_class = RagTokenForGeneration
        elif hparams.model_type == "bart":
            self.model_class = BartForConditionalGeneration
        else:
            self.model_class = T5ForConditionalGeneration
        self.is_rag_model = is_rag_model(hparams.model_type)

        config_class = RagConfig if self.is_rag_model else AutoConfig
        config = config_class.from_pretrained(hparams.model_name_or_path)

        # set retriever parameters
        config.index_name = hparams.index_name or config.index_name
        config.passages_path = hparams.passages_path or config.passages_path
        config.index_path = hparams.index_path or config.index_path
        config.use_dummy_dataset = hparams.use_dummy_dataset

        # set extra_model_params for generator configs and load_model
        extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "attention_dropout", "dropout")
        if self.is_rag_model:
            if hparams.prefix is not None:
                config.generator.prefix = hparams.prefix
            config.label_smoothing = hparams.label_smoothing
            hparams, config.generator = set_extra_model_params(extra_model_params, hparams, config.generator)
            if hparams.distributed_retriever == "ray":
                # The Ray retriever needs the handles to the retriever actors.
                retriever = RagRayDistributedRetriever.from_pretrained(
                    hparams.model_name_or_path, hparams.actor_handles, config=config
                )

                if hparams.end2end:
                    ctx_encoder_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(
                        "facebook/dpr-ctx_encoder-multiset-base"
                    )
                    retriever.set_ctx_encoder_tokenizer(ctx_encoder_tokenizer)
            else:
                logger.info("please use RAY as the distributed retrieval method")

            model = self.model_class.from_pretrained(hparams.model_name_or_path, config=config, retriever=retriever)
            if hparams.end2end:
                ctx_encoder = DPRContextEncoder.from_pretrained(hparams.context_encoder_name)
                model.set_context_encoder_for_training(ctx_encoder)
            prefix = config.question_encoder.prefix
        else:
            if hparams.prefix is not None:
                config.prefix = hparams.prefix
            hparams, config = set_extra_model_params(extra_model_params, hparams, config)
            model = self.model_class.from_pretrained(hparams.model_name_or_path, config=config)
            prefix = config.prefix

        tokenizer = (
            RagTokenizer.from_pretrained(hparams.model_name_or_path)
            if self.is_rag_model
            else AutoTokenizer.from_pretrained(hparams.model_name_or_path)
        )

        self.config_dpr = DPRConfig.from_pretrained(hparams.context_encoder_name)
        self.custom_config = hparams
        self.context_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(hparams.context_encoder_name)

        super().__init__(hparams, config=config, tokenizer=tokenizer, model=model)

        save_git_info(self.hparams.output_dir)
        self.output_dir = Path(self.hparams.output_dir)
        self.dpr_ctx_check_dir = str(Path(self.hparams.output_dir)) + "/dpr_ctx_checkpoint"
        self.metrics_save_path = Path(self.output_dir) / "metrics.json"
        self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        pickle_save(self.hparams, self.hparams_save_path)
        self.step_count = 0
        self.metrics = defaultdict(list)

        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            prefix=prefix or "",
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

        self.hparams.git_sha = get_git_info()["repo_sha"]
        self.num_workers = hparams.num_workers
        self.distributed_port = self.hparams.distributed_port

        # For single GPU training, init_ddp_connection is not called.
        # So we need to initialize the retrievers here.
        if hparams.gpus <= 1:
            if hparams.distributed_retriever == "ray":
                self.model.retriever.init_retrieval()
            else:
                logger.info("please use RAY as the distributed retrieval method")

        self.distributed_retriever = hparams.distributed_retriever

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

    def _step(self, batch: dict) -> Tuple:
        source_ids, source_mask, target_ids = batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"]

        rag_kwargs = {}
        if isinstance(self.model, T5ForConditionalGeneration):
            decoder_input_ids = self.model._shift_right(target_ids)
            lm_labels = target_ids
        elif isinstance(self.model, BartForConditionalGeneration):
            decoder_input_ids = target_ids[:, :-1].contiguous()
            lm_labels = target_ids[:, 1:].clone()
        else:
            assert self.is_rag_model
            generator = self.model.rag.generator
            if isinstance(generator, T5ForConditionalGeneration):
                decoder_start_token_id = generator.config.decoder_start_token_id
                decoder_input_ids = (
                    torch.cat(
                        [torch.tensor([[decoder_start_token_id]] * target_ids.shape[0]).to(target_ids), target_ids],
                        dim=1,
                    )
                    if target_ids.shape[0] < self.target_lens["train"]
                    else generator._shift_right(target_ids)
                )
            elif isinstance(generator, BartForConditionalGeneration):
                decoder_input_ids = target_ids
            lm_labels = decoder_input_ids
            rag_kwargs["reduce_loss"] = True

        assert decoder_input_ids is not None

        outputs = self(
            source_ids,
            attention_mask=source_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            labels=lm_labels,
            **rag_kwargs,
        )
        loss = outputs["loss"]
        return (loss,)

    @property
    def pad(self) -> int:
        raise NotImplementedError("pad not implemented")

    def training_step(self, batch, batch_idx) -> Dict:
        global isEmUpdateBusy  # use to check whether the entire embedding update process is finished or not
        global isAddIndexBusy  # use to check whether the entire indexing process  is finished or not
        global processes  # use to keep threads embedding update processes
        global threadHandle_index  # use to keep thread in embedding indexing processes

        if (self.trainer.global_rank == 0) and (self.custom_config.end2end):
            if (not batch_idx == 0) and (batch_idx % self.custom_config.indexing_freq == 0):
                free_gpu_list = []
                nvmlInit()
                deviceCount = nvmlDeviceGetCount()

                my_list = json.loads(self.custom_config.gpu_order)

                for i in range(deviceCount):
                    handle = nvmlDeviceGetHandleByIndex(i)
                    info = nvmlDeviceGetMemoryInfo(handle)

                    if info.used / 1e6 < 15:
                        position = my_list.index(i)
                        free_gpu_list.append("cuda:" + str(position))

                if len(free_gpu_list) >= self.custom_config.index_gpus:
                    has_free_gpus = True

                else:
                    has_free_gpus = False

                if (not isEmUpdateBusy) and has_free_gpus:
                    model_copy = type(self.model.rag.ctx_encoder)(
                        self.config_dpr
                    )  # get a new instance  #this will be load in the CPU
                    model_copy.load_state_dict(self.model.rag.ctx_encoder.state_dict())  # copy weights

                    processes = []

                    if len(free_gpu_list) > self.custom_config.index_gpus:
                        cuda_devices = random.sample(free_gpu_list, self.custom_config.index_gpus)
                    else:
                        cuda_devices = free_gpu_list

                    num_processes = len(cuda_devices)

                    for rank in range(num_processes):
                        logger.info("Iniitializing  embedding calculation process rank{}".format(rank))
                        device = cuda_devices[rank]
                        p = multiprocessing.Process(
                            target=embed_update,
                            args=(
                                copy.deepcopy(model_copy),
                                num_processes,
                                device,
                                rank,
                                self.custom_config.shard_dir,
                                self.custom_config.csv_path,
                            ),
                        )
                        processes.append(p)

                    for p in processes:
                        p.start()

                    isEmUpdateBusy = True

            if isEmUpdateBusy and (not isAddIndexBusy):
                index_process_list = [processes[k].is_alive() for k in range(self.custom_config.index_gpus)]
                if (
                    sum(index_process_list) == 0
                ):  # If entire list is false, we can say all embedding calculation process has finished
                    logger.info("Start adding the index")
                    threadHandle_index = multiprocessing.Process(
                        target=add_index,
                        args=(
                            self.custom_config.shard_dir,
                            self.config.index_path,
                        ),
                    )
                    threadHandle_index.start()
                    isAddIndexBusy = True

            # check when index building has started
            if isAddIndexBusy:
                # check still the index_building process is happening
                if not threadHandle_index.is_alive():
                    logger.info("Merging the dataset shards")
                    saved_dataset_shards = []

                    for address in glob(str(self.custom_config.shard_dir) + "/*/"):
                        saved_dataset_shards.append(load_from_disk(address))

                    concat = concatenate_datasets(saved_dataset_shards)
                    concat.save_to_disk(self.config.passages_path)  # here we update the main passage file on the disk
                    logger.info("done updating the dataset")

                    # To Do (@Aaron) : Useful in the future dynamic memory implementation.
                    # if you load the index from the disk make sure to update the index file here, otherwise it is ok to update the index file from the worker.
                    # logger.info("then updating the index")
                    # shutil.copy(self.custom_config.temp_index, self.config.idex_path)

                    logger.info("Loading new passages and iniitalzing new index")
                    self.trainer.model.module.module.model.rag.retriever.re_load()
                    self.trainer.model.module.module.model.rag.retriever.init_retrieval()

                    isEmUpdateBusy = False
                    isAddIndexBusy = False
        self.trainer.strategy.barrier("barrier")

        loss_tensors = self._step(batch)

        logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        # tokens per batch
        tgt_pad_token_id = (
            self.tokenizer.generator.pad_token_id
            if isinstance(self.tokenizer, RagTokenizer)
            else self.tokenizer.pad_token_id
        )
        src_pad_token_id = (
            self.tokenizer.question_encoder.pad_token_id
            if isinstance(self.tokenizer, RagTokenizer)
            else self.tokenizer.pad_token_id
        )
        logs["tpb"] = (
            batch["input_ids"].ne(src_pad_token_id).sum() + batch["decoder_input_ids"].ne(tgt_pad_token_id).sum()
        )
        self.log("loss", loss_tensors[0])
        return loss_tensors[0]

    def validation_step(self, batch, batch_idx) -> Dict:
        return self._generative_step(batch)

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        self.step_count += 1
        losses = {k: torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names}
        loss = losses["loss"]
        gen_metrics = {
            k: np.array([x[k] for x in outputs]).mean() for k in self.metric_names + ["gen_time", "gen_len"]
        }
        metrics_tensor: torch.FloatTensor = torch.tensor(gen_metrics[self.val_metric]).type_as(loss)
        gen_metrics.update({k: v.item() for k, v in losses.items()})

        # fix for https://github.com/PyTorchLightning/pytorch-lightning/issues/2424
        if dist.is_initialized():
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            metrics_tensor = metrics_tensor / dist.get_world_size()
            gen_metrics.update({self.val_metric: metrics_tensor.item()})

        losses.update(gen_metrics)
        metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        metrics["step_count"] = self.step_count
        self.save_metrics(metrics, prefix)  # writes to self.metrics_save_path

        log_dict = {
            f"{prefix}_avg_em": metrics[f"{prefix}_avg_em"],
            "step_count": metrics["step_count"],
            f"{prefix}_avg_loss": metrics[f"{prefix}_avg_loss"],
            f"{prefix}_loss": loss,
            f"{prefix}_em": metrics_tensor,
        }
        self.log_dict(log_dict)

    def save_metrics(self, latest_metrics, type_path) -> None:
        self.metrics[type_path].append(latest_metrics)
        save_json(self.metrics, self.metrics_save_path)

    def calc_generative_metrics(self, preds, target) -> Dict:
        return calculate_exact_match(preds, target)

    def _generative_step(self, batch: dict) -> dict:
        start_time = time.time()
        batch = BatchEncoding(batch).to(device=self.model.device)
        generated_ids = self.model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            do_deduplication=False,  # rag specific parameter
            use_cache=True,
            min_length=1,
            max_length=self.target_lens["val"],
        )
        gen_time = (time.time() - start_time) / batch["input_ids"].shape[0]
        preds: List[str] = self.ids_to_clean_text(generated_ids)
        target: List[str] = self.ids_to_clean_text(batch["decoder_input_ids"])
        # print(preds,target)
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

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size)

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = self.output_dir.joinpath("checkpoint{}".format(self.step_count))
        self.model.config.save_step = self.step_count
        # self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        if self.custom_config.end2end:
            modified_state_dict = self.model.state_dict()
            for key in self.model.state_dict().keys():
                if key.split(".")[1] == "ctx_encoder":
                    del modified_state_dict[key]
            self.model.save_pretrained(save_directory=save_path, state_dict=modified_state_dict)

            save_path_dpr = os.path.join(self.dpr_ctx_check_dir, "checkpoint{}".format(self.step_count))
            self.model.rag.ctx_encoder.save_pretrained(save_path_dpr)
            self.context_tokenizer.save_pretrained(save_path_dpr)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)
        add_generic_args(parser, root_dir)
        parser.add_argument(
            "--max_source_length",
            default=128,
            type=int,
            help=(
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            ),
        )
        parser.add_argument(
            "--max_target_length",
            default=25,
            type=int,
            help=(
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            ),
        )
        parser.add_argument(
            "--val_max_target_length",
            default=25,
            type=int,
            help=(
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            ),
        )
        parser.add_argument(
            "--test_max_target_length",
            default=25,
            type=int,
            help=(
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            ),
        )
        parser.add_argument("--logger_name", type=str, choices=["default", "wandb", "wandb_shared"], default="default")
        parser.add_argument("--n_train", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_val", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_test", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
        parser.add_argument(
            "--prefix",
            type=str,
            default=None,
            help="Prefix added at the beginning of each text, typically used with T5-based models.",
        )
        parser.add_argument(
            "--early_stopping_patience",
            type=int,
            default=-1,
            required=False,
            help=(
                "-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So"
                " val_check_interval will effect it."
            ),
        )
        parser.add_argument(
            "--distributed-port", type=int, default=-1, required=False, help="Port number for distributed training."
        )
        parser.add_argument(
            "--model_type",
            choices=["rag_sequence", "rag_token", "bart", "t5"],
            type=str,
            help=(
                "RAG model type: sequence or token, if none specified, the type is inferred from the"
                " model_name_or_path"
            ),
        )
        parser.add_argument(
            "--context_encoder_name",
            default="facebook/dpr-ctx_encoder-multiset-base",
            type=str,
            help="Name of the pre-trained context encoder checkpoint from the DPR",
        )
        parser.add_argument(
            "--csv_path",
            default=str(Path(__file__).parent / "test_run" / "dummy-kb" / "my_knowledge_dataset.csv"),
            type=str,
            help="path of the raw KB csv",
        )
        parser.add_argument("--end2end", action="store_true", help="whether to train the system end2end or not")
        parser.add_argument("--index_gpus", type=int, help="how many GPUs used in re-encoding process")
        parser.add_argument(
            "--shard_dir",
            type=str,
            default=str(Path(__file__).parent / "test_run" / "kb-shards"),
            help="directory used to keep temporary shards during the re-encode process",
        )

        parser.add_argument(
            "--gpu_order",
            type=str,
            help=(
                "order of the GPU used during the fine-tuning.  Used to finding free GPUs during the re-encode"
                " process. I do not have many GPUs :)"
            ),
        )

        parser.add_argument("--indexing_freq", type=int, help="frequency of re-encode process")
        return parser

    @staticmethod
    def add_retriever_specific_args(parser):
        parser.add_argument(
            "--index_name",
            type=str,
            default=None,
            help=(
                "Name of the index to use: 'hf' for a canonical dataset from the datasets library (default), 'custom'"
                " for a local index, or 'legacy' for the orignal one)"
            ),
        )
        parser.add_argument(
            "--passages_path",
            type=str,
            default=str(Path(__file__).parent / "test_run" / "dummy-kb" / "my_knowledge_dataset"),
            help=(
                "Path to the dataset of passages for custom index. More info about custom indexes in the RagRetriever"
                " documentation as well as in `examples/rag/use_own_knowledge_dataset.py`"
            ),
        )
        parser.add_argument(
            "--index_path",
            type=str,
            default=str(Path(__file__).parent / "test_run" / "dummy-kb" / "my_knowledge_dataset_hnsw_index.faiss"),
            help=(
                "Path to the faiss index for custom index. More info about custom indexes in the RagRetriever"
                " documentation as well as in `examples/rag/use_own_knowledge_dataset.py`"
            ),
        )
        parser.add_argument(
            "--distributed_retriever",
            choices=["ray", "pytorch"],
            type=str,
            default="ray",
            help=(
                "What implementation to use for distributed retriever? If "
                "pytorch is selected, the index is loaded on training "
                "worker 0, and torch.distributed is used to handle "
                "communication between training worker 0, and the other "
                "training workers. If ray is selected, the Ray library is "
                "used to create load the index on separate processes, "
                "and Ray handles the communication between the training "
                "workers and the retrieval actors."
            ),
        )
        parser.add_argument(
            "--use_dummy_dataset",
            type=bool,
            default=False,
            help=(
                "Whether to use the dummy version of the dataset index. More info about custom indexes in the"
                " RagRetriever documentation as well as in `examples/rag/use_own_knowledge_dataset.py`"
            ),
        )
        return parser

    @staticmethod
    def add_ray_specific_args(parser):
        # Ray cluster address.
        parser.add_argument(
            "--ray-address",
            default="auto",
            type=str,
            help=(
                "The address of the Ray cluster to connect to. If not "
                "specified, Ray will attempt to automatically detect the "
                "cluster. Has no effect if pytorch is used as the distributed "
                "retriever."
            ),
        )
        parser.add_argument(
            "--num_retrieval_workers",
            type=int,
            default=1,
            help=(
                "The number of retrieval actors to use when Ray is selected"
                "for the distributed retriever. Has no effect when "
                "distributed_retriever is set to pytorch."
            ),
        )
        return parser


def main(args=None, model=None) -> GenerativeQAModule:
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = GenerativeQAModule.add_model_specific_args(parser, os.getcwd())
    parser = GenerativeQAModule.add_retriever_specific_args(parser)
    args = args or parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True)
    Path(args.output_dir + "/dpr_ctx_checkpoint").mkdir(
        exist_ok=True
    )  # save dpr_context encoder seprately for the future use
    print(args.shard_dir)
    if os.path.exists(args.shard_dir):  # we do not need previous kb shards used in dataset re-conding and re-indexing
        shutil.rmtree(args.shard_dir)
    Path(args.shard_dir).mkdir(exist_ok=True)

    if os.path.exists(
        args.cache_dir
    ):  # we do not need previous cache files used in dataset re-conding and re-indexing
        shutil.rmtree(args.cache_dir)
    Path(args.cache_dir).mkdir(exist_ok=True)

    named_actors = []
    if args.distributed_retriever == "ray" and args.gpus > 1:
        if not is_ray_available():
            raise RuntimeError("Please install Ray to use the Ray distributed retriever.")
        # Connect to an existing Ray cluster.
        try:
            ray.init(address=args.ray_address, namespace="rag")
        except (ConnectionError, ValueError):
            logger.warning(
                "Connection to Ray cluster failed. Make sure a Ray"
                "cluster is running by either using Ray's cluster "
                "launcher (`ray up`) or by manually starting Ray on "
                "each node via `ray start --head` for the head node "
                "and `ray start --address='<ip address>:6379'` for "
                "additional nodes. See "
                "https://docs.ray.io/en/master/cluster/index.html "
                "for more info."
            )
            raise

        # Create Ray actors only for rank 0.
        if ("LOCAL_RANK" not in os.environ or os.environ["LOCAL_RANK"] == 0) and (
            "NODE_RANK" not in os.environ or os.environ["NODE_RANK"] == 0
        ):
            remote_cls = ray.remote(RayRetriever)
            named_actors = [
                remote_cls.options(name="retrieval_worker_{}".format(i)).remote()
                for i in range(args.num_retrieval_workers)
            ]
        else:
            logger.info(
                "Getting named actors for NODE_RANK {}, LOCAL_RANK {}".format(
                    os.environ["NODE_RANK"], os.environ["LOCAL_RANK"]
                )
            )
            named_actors = [ray.get_actor("retrieval_worker_{}".format(i)) for i in range(args.num_retrieval_workers)]
    args.actor_handles = named_actors
    assert args.actor_handles == named_actors

    if model is None:
        model: GenerativeQAModule = GenerativeQAModule(args)

    dataset = Path(args.data_dir).name
    if (
        args.logger_name == "default"
        or args.fast_dev_run
        or str(args.output_dir).startswith("/tmp")
        or str(args.output_dir).startswith("/var")
    ):
        training_logger = True  # don't pollute wandb logs unnecessarily
    elif args.logger_name == "wandb":
        from pytorch_lightning.loggers import WandbLogger

        project = os.environ.get("WANDB_PROJECT", dataset)
        training_logger = WandbLogger(name=model.output_dir.name, project=project)

    elif args.logger_name == "wandb_shared":
        from pytorch_lightning.loggers import WandbLogger

        training_logger = WandbLogger(name=model.output_dir.name, project=f"hf_{dataset}")

    es_callback = (
        get_early_stopping_callback(model.val_metric, args.early_stopping_patience)
        if args.early_stopping_patience >= 0
        else False
    )

    trainer: pl.Trainer = generic_train(
        model,
        args,
        logging_callback=Seq2SeqLoggingCallback(),
        checkpoint_callback=get_checkpoint_callback(args.output_dir, model.val_metric),
        early_stopping_callback=es_callback,
        logger=training_logger,
        profiler=pl.profiler.AdvancedProfiler() if args.profile else None,
    )

    pickle_save(model.hparams, model.output_dir / "hparams.pkl")
    if not args.do_predict:
        return model

    # test() without a model tests using the best checkpoint automatically
    trainer.test()
    return model


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = GenerativeQAModule.add_model_specific_args(parser, os.getcwd())
    parser = GenerativeQAModule.add_retriever_specific_args(parser)
    parser = GenerativeQAModule.add_ray_specific_args(parser)

    # Pytorch Lightning Profiler
    parser.add_argument(
        "--profile",
        action="store_true",
        help="If True, use pytorch_lightning.profiler.AdvancedProfiler to profile the Trainer.",
    )

    args = parser.parse_args()
    main(args)
