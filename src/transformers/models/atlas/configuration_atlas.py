# coding=utf-8
# Copyright 2020, The ATLAS Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" ATLAS model configuration"""

import copy

from ...configuration_utils import PretrainedConfig
from ...utils import add_start_docstrings


ATLAS_CONFIG_DOC = r"""
    [`AtlasConfig`] stores the configuration of a *AtlasModel*. Configuration objects inherit from [`PretrainedConfig`] and
    can be used to control the model outputs. Read the documentation from [`PretrainedConfig`] for more information.

    Args:
        title_sep (`str`, *optional*, defaults to  `" / "`):
            Separator inserted between the title and the text of the retrieved document when calling [`AtlasRetriever`].
        doc_sep (`str`, *optional*, defaults to  `" // "`):
            Separator inserted between the text of the retrieved document and the original input when calling
            [`AtlasRetriever`].
        n_docs (`int`, *optional*, defaults to 5):
            Number of documents to retrieve.
        max_combined_length (`int`, *optional*, defaults to 300):
            Max length of contextualized input returned by [`~AtlasRetriever.__call__`].
        retrieval_vector_size (`int`, *optional*, defaults to 768):
            Dimensionality of the document embeddings indexed by [`AtlasRetriever`].
        retrieval_batch_size (`int`, *optional*, defaults to 8):
            Retrieval batch size, defined as the number of queries issues concurrently to the faiss index encapsulated
            [`AtlasRetriever`].
        dataset (`str`, *optional*, defaults to `"wiki_dpr"`):
            A dataset identifier of the indexed dataset in HuggingFace Datasets (list all available datasets and ids
            using `datasets.list_datasets()`).
        dataset_split (`str`, *optional*, defaults to `"train"`)
            Which split of the `dataset` to load.
        index_name (`str`, *optional*, defaults to `"compressed"`)
            The index name of the index associated with the `dataset`. One can choose between `"legacy"`, `"exact"` and
            `"compressed"`.
        index_path (`str`, *optional*)
            The path to the serialized faiss index on disk.
        passages_path (`str`, *optional*):
            A path to text passages compatible with the faiss index. Required if using
            [`~models.atlas.retrieval_atlas.LegacyIndex`]
        use_dummy_dataset (`bool`, *optional*, defaults to `False`)
            Whether to load a "dummy" variant of the dataset specified by `dataset`.
        label_smoothing (`float`, *optional*, defaults to 0.0):
            Only relevant if `return_loss` is set to `True`. Controls the `epsilon` parameter value for label smoothing
            in the loss calculation. If set to 0, no label smoothing is performed.
        do_marginalize (`bool`, *optional*, defaults to `False`):
            If `True`, the logits are marginalized over all documents by making use of
            `torch.nn.functional.log_softmax`.
        reduce_loss (`bool`, *optional*, defaults to `False`):
            Whether or not to reduce the NLL loss using the `torch.Tensor.sum` operation.
        do_deduplication (`bool`, *optional*, defaults to `True`):
            Whether or not to deduplicate the generations from different context documents for a given input. Has to be
            set to `False` if used while training with distributed backend.
        exclude_bos_score (`bool`, *optional*, defaults to `False`):
            Whether or not to disregard the BOS token when computing the loss.
        output_retrieved(`bool`, *optional*, defaults to `False`):
            If set to `True`, `retrieved_doc_embeds`, `retrieved_doc_ids`, `context_input_ids` and
            `context_attention_mask` are returned. See returned tensors for more detail.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        forced_eos_token_id (`int`, *optional*):
            The id of the token to force as the last generated token when `max_length` is reached. Usually set to
            `eos_token_id`.
"""


@add_start_docstrings(ATLAS_CONFIG_DOC)
class AtlasConfig(PretrainedConfig):
    model_type = "atlas"
    is_composition = True

    def __init__(
        self,
        vocab_size=None,
        is_encoder_decoder=True,
        prefix=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_id=None,
        decoder_start_token_id=None,
        title_sep=" / ",
        doc_sep=" // ",
        n_docs=5,
        max_combined_length=300,
        retrieval_vector_size=768,
        retrieval_batch_size=8,
        dataset="wiki_dpr",
        dataset_split="train",
        index_name="compressed",
        index_path=None,
        passages_path=None,
        use_dummy_dataset=False,
        reduce_loss=False,
        label_smoothing=0.0,
        do_deduplication=True,
        exclude_bos_score=False,
        do_marginalize=False,
        output_retrieved=False,
        use_cache=True,
        forced_eos_token_id=None,
        **kwargs
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            prefix=prefix,
            vocab_size=vocab_size,
            **kwargs,
        )
        assert (
            "query_passage_encoder" in kwargs and "generator" in kwargs
        ), "Config has to be initialized with query_passage_encoder and generator config"
        query_passage_encoder_config = kwargs.pop("query_passage_encoder")
        query_passage_encoder_model_type = query_passage_encoder_config.pop("model_type")
        decoder_config = kwargs.pop("generator")
        decoder_model_type = decoder_config.pop("model_type")

        from ..auto.configuration_auto import AutoConfig

        self.query_passage_encoder = AutoConfig.for_model(query_passage_encoder_model_type, **query_passage_encoder_config)
        self.generator = AutoConfig.for_model(decoder_model_type, **decoder_config)

        self.reduce_loss = reduce_loss
        self.label_smoothing = label_smoothing
        self.exclude_bos_score = exclude_bos_score
        self.do_marginalize = do_marginalize

        self.title_sep = title_sep
        self.doc_sep = doc_sep
        self.n_docs = n_docs
        self.max_combined_length = max_combined_length

        self.dataset = dataset
        self.dataset_split = dataset_split
        self.index_name = index_name

        self.retrieval_vector_size = retrieval_vector_size
        self.retrieval_batch_size = retrieval_batch_size
        self.passages_path = passages_path
        self.index_path = index_path
        self.use_dummy_dataset = use_dummy_dataset

        self.output_retrieved = output_retrieved

        self.do_deduplication = do_deduplication

        self.use_cache = use_cache

        if self.forced_eos_token_id is None:
            self.forced_eos_token_id = getattr(self.generator, "forced_eos_token_id", None)

    @classmethod
    def from_query_passage_encoder_generator_configs(
        cls, query_passage_encoder_config: PretrainedConfig, generator_config: PretrainedConfig, **kwargs
    ) -> PretrainedConfig:
        r"""
        Instantiate a [`EncoderDecoderConfig`] (or a derived class) from a pre-trained encoder model configuration and
        decoder model configuration.

        Returns:
            [`EncoderDecoderConfig`]: An instance of a configuration object
        """
        return cls(query_passage_encoder=query_passage_encoder_config.to_dict(), generator=generator_config.to_dict(), **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["query_passage_encoder"] = self.query_passage_encoder.to_dict()
        output["generator"] = self.generator.to_dict()
        output["model_type"] = self.__class__.model_type
        return output





# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def initialize_parser(self):
        # basic parameters
        self.parser.add_argument(
            "--name", type=str, default="experiment_name", help="name of the experiment - also used as directory name "
        )
        self.parser.add_argument(
            "--checkpoint_dir",
            type=str,
            default="./checkpoint/",
            help="models are saved here",
        )
        self.parser.add_argument(
            "--model_path",
            type=str,
            default="none",
            help="Path to a pretrained model to initialize from (pass 'none' to init from t5 and contriever)",
        )
        self.parser.add_argument(
            "--per_gpu_batch_size",
            default=1,
            type=int,
            help="Batch size per GPU/CPU for training.",
        )

        self.parser.add_argument(
            "--per_gpu_embedder_batch_size",
            default=512,
            type=int,
            help="Embedder's batch size per GPU.",
        )

        self.parser.add_argument(
            "--local_rank",
            type=int,
            default=-1,
            help="For distributed training: local_rank",
        )
        self.parser.add_argument(
            "--main_port",
            type=int,
            default=-1,
            help="Main port (for multi-node jobs)",
        )
        self.parser.add_argument("--seed", type=int, default=0, help="random seed for initialization")
        self.parser.add_argument(
            "--log_freq",
            type=int,
            default=100,
            help="log train stats <log_freq> steps during training",
        )
        self.parser.add_argument(
            "--eval_freq",
            type=int,
            default=500,
            help="evaluate model every <eval_freq> steps during training",
        )
        self.parser.add_argument(
            "--save_freq",
            type=int,
            default=5000,
            help="save model every <save_freq> steps during training",
        )
        self.parser.add_argument(
            "--train_data", nargs="+", default=[], help="list of space-separated paths to jsonl-formatted train sets"
        )
        self.parser.add_argument(
            "--eval_data",
            nargs="+",
            default=[],
            help="list of space-separated paths to jsonl-formatted evaluation sets",
        )
        self.parser.add_argument("--write_results", action="store_true", help="save evaluation results to file")
        self.parser.add_argument(
            "--dont_write_passages",
            action="store_true",
            help="if writing results, passages can take up a lot of space, pass this flag not to write passages as part of dumped results",
        )

    def add_optim_options(self):
        self.parser.add_argument("--warmup_steps", type=int, default=1000, help="number of learning rate warmup steps")
        self.parser.add_argument("--total_steps", type=int, default=1000, help="total number of training steps")
        self.parser.add_argument(
            "--scheduler_steps",
            type=int,
            default=None,
            help="total number of step for the scheduler, if None then scheduler_total_step = total_step",
        )
        self.parser.add_argument("--accumulation_steps", type=int, default=1, help="gradient accumulation")
        self.parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
        self.parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
        self.parser.add_argument("--lr_retriever", type=float, default=1e-5, help="learning rate for retriever")
        self.parser.add_argument("--clip", type=float, default=1.0, help="gradient clipping")
        self.parser.add_argument(
            "--scheduler",
            type=str,
            default="cosine",
            choices=["linear", "cosine", "fixed"],
            help="learning rate schedule to use",
        )
        self.parser.add_argument(
            "--weight_decay", type=float, default=0.1, help="amount of weight decay to apply in training"
        )
        self.parser.add_argument(
            "--save_optimizer", action="store_true", help="Pass flag to save optimizer state in saved checkpoints"
        )
        self.parser.add_argument("--epsilon", type=float, default=1e-6, help="adamw epsilon value")
        self.parser.add_argument("--alpha", type=float, default=1.0, help="adamw alpha value")
        self.parser.add_argument("--beta2", type=float, default=0.999, help="adamw beta2 value")
        self.parser.add_argument(
            "--refresh_index",
            type=str,
            default="-1",
            help="index refresh schedule. format: startstep-endstep:refreshrate,startstep-endstep:refreshrate "
            "e.g. --refresh_index 0-100:10,100-1000000:500 will refresh the index every 10 steps for the first 100 steps, "
            "and then every 500 steps from step 100 to 1M."
            "Syntactic Sugar for a fixed schedule: can just pass in a single number e.g. --refresh_index 100 will refresh the index every 100 steps. "
            "-1 to never refresh.",
        )
        self.parser.add_argument("--shuffle", action="store_true", help="shuffle data for training")

        # memory optimizations:
        self.parser.add_argument(
            "--precision",
            type=str,
            default="fp32",
            choices=["fp16", "fp32", "bf16"],
            help="numerical precision - recommend bf16 if available, fp16 likely to be unstable for training",
        )
        self.parser.add_argument(
            "--shard_optim",
            action="store_true",
            help="train-time memory optimization: shards optimizer state over available GPUs using sharded data parallel, recommended for larger models",
        )
        self.parser.add_argument(
            "--shard_grads",
            action="store_true",
            help="train-time memory optimization: shards gradients over available GPUs using sharded data parallel, recommended for larger models",
        )
        self.parser.add_argument(
            "--use_gradient_checkpoint_reader",
            action="store_true",
            help="use gradient checkpointing in the reader",
        )
        self.parser.add_argument(
            "--use_gradient_checkpoint_retriever",
            action="store_true",
            help="use gradient checkpointing for retriever",
        )

    def add_modeling_options(self):
        self.parser.add_argument(
            "--reader_model_type",
            required=True,
            type=str,
            help="t5 Architecture for reader FID model, e.g. google/t5-xl-lm-adapt",
            choices=[
                "t5-small",
                "t5-base",
                "t5-large",
                "t5-3b",
                "t5-11b",
                "google/t5-v1_1-base",
                "google/t5-v1_1-large",
                "google/t5-v1_1-xl",
                "google/t5-v1_1-xxl",
                "google/t5-base-lm-adapt",
                "google/t5-large-lm-adapt",
                "google/t5-xl-lm-adapt",
                "google/t5-xxl-lm-adapt",
            ],
        )
        self.parser.add_argument(
            "--text_maxlength",
            type=int,
            default=200,
            help="maximum number of tokens in input text segments (concatenated question+passage). Inputs longer than this will be truncated.",
        )
        self.parser.add_argument(
            "--target_maxlength",
            type=int,
            default=None,
            help="Maximum length of target outputs in tokens when training the model. Targets longer than this will be truncated. No truncation if -1",
        )
        self.parser.add_argument("--n_context", type=int, default=1, help="number of top k passages to pass to reader")

        # Retriever modelling options
        self.parser.add_argument(
            "--passages",
            nargs="+",
            help="list of paths to jsonl files containing passages to index and retrieve from. Unused if loading a saved index using --load_index_path",
        )
        self.parser.add_argument(
            "--max_passages",
            type=int,
            default=-1,
            help="maximum number of passages to index. -1 to read all passages in passage files",
        )
        self.parser.add_argument(
            "--retriever_model_path",
            type=str,
            default="facebook/contriever",
            help="path to contriever model to init from (overridden if passing a value to --model_path ",
        )
        self.parser.add_argument(
            "--retrieve_only",
            action="store_true",
            help="Pass this to prevent loading a reader, and only run retrieval evaluation",
        )
        self.parser.add_argument(
            "--train_retriever", action="store_true", help="Pass to train retriever as well as reader"
        )
        self.parser.add_argument(
            "--use_file_passages",
            action="store_true",
            help='uses passages in "passages" field in train or eval jsonl files rather than retrieving passages',
        )
        self.parser.add_argument(
            "--retriever_n_context",
            type=int,
            default=5,
            help="number of top k passages to use to train the retriever with",
        )
        self.parser.add_argument(
            "--gold_score_mode",
            type=str,
            choices=["evalnormsum", "loop", "ppmean", "emdr", "pdist", "adist"],
            default="ppmean",
            help="retriever training method. `pdist` is the name used in the paper for `ppmean`. `adist` is the name used in the paper for `evalnormsum`",
        )
        self.parser.add_argument(
            "--closed_book",
            action="store_true",
            help="Dont use retrieval - reduces to T5. Overrides n_context, n_context_retriever and encoder_format if they are set",
        )
        self.parser.add_argument(
            "--temperature_score", type=float, default=0.01, help="softmax temperature for retriever"
        )
        self.parser.add_argument(
            "--temperature_gold",
            type=float,
            default=0.01,
            help="softmax temperature for target distribution for retriever distillation",
        )
        self.parser.add_argument("--compute_crossattention_stats", action="store_true")
        self.parser.add_argument(
            "--filtering_overretrieve_ratio",
            type=int,
            default=2,
            help="if filtering, over-retrieve the topK by this factor, and then filter out undesirable results. Useful, Set to 1 only if using a task that doesn't filter retrieved results",
        )
        self.parser.add_argument("--freeze_retriever_steps", type=int, default=-1, help="freezes retriever for n steps")
        self.parser.add_argument(
            "--query_side_retriever_training",
            action="store_true",
            help="pass to enable query-side finetuning of retriever (unties the parameters of the contriever encoder's passage and query encoders, and freezes the passage encoder. Useful to avoid index refreshes.",
        )
        self.parser.add_argument(
            "--retrieve_with_rerank",
            action="store_true",
            help="pass this to enable reranking with fresh passage encoder for retriever",
        )
        self.parser.add_argument(
            "--n_to_rerank_with_retrieve_with_rerank",
            type=int,
            default=128,
            help="n passages to rerank when passing --retrieve_with_rerank. Higher is slower but more accurate. Recommend 64-128",
        )

        # input and output formatting options:
        self.parser.add_argument(
            "--decoder_format",  # TODO: decide whether to remove functionality
            type=str,
            default=None,
            help="format for decoder, model will be train on the format and evaluation will be performed with the format contrary to the decoder_prompt_format option",
        )
        self.parser.add_argument(  # TODO: decide whether to remove functionality
            "--decoder_prompt_format",
            type=str,
            default=None,
            help='format for decoder prompting, for instance "what is the answer to {query}:"',
        )
        self.parser.add_argument(
            "--encoder_format",
            type=str,
            default="{query} title: {title} context: {text}",
            help="format string for reader's encoder preprocessing",
        )
        self.parser.add_argument(
            "--retriever_format",
            type=str,
            default="{title} {text}",
            help="format string for retriever's encoder preprocessing",
        )

        # Generation options
        self.parser.add_argument("--generation_max_length", type=int, default=128)
        self.parser.add_argument("--generation_min_length", type=int, default=None)
        self.parser.add_argument("--generation_length_penalty", type=float, default=1.0)
        self.parser.add_argument("--generation_num_beams", type=int, default=1)

        # Task-specific options:
        self.parser.add_argument(
            "--task",
            type=str,
            default=None,
            choices=["base", "mlm", "lm", "multiple_choice", "kilt", "section", "fever", "qa"],
            help="Task performed by the model. Used to setup preprocessing, retrieval filtering, evaluations, etc.",
        )

        # MLM task options:
        self.parser.add_argument(
            "--mlm_noise_density",
            type=float,
            default=0.15,
            help="how much of an input text should be masked by masking spans ",
        )
        self.parser.add_argument(
            "--mlm_mean_noise_span_length", type=float, default=3, help="average length of an MLM masking span"
        )
        self.parser.add_argument(
            "--min_words_per_lm_instance",
            type=int,
            default=None,
            help="Instances with fewer than min_words_per_lm_instance instances will be skipped for MLM/LM/Section Generation",
        )

        # LM task options:
        self.parser.add_argument(
            "--min_lm_context_ratio",
            type=float,
            default=0.5,
            help="Splits text into two segments for language modelling.'\
                'Left segment is conditioning context, right segment is for generating.'\
                'The left segment must be more than min_lm_context_ratio of the the right segment",
        )
        self.parser.add_argument(
            "--max_lm_context_ratio",
            type=float,
            default=0.5,
            help="Splits text into two segments for language modelling.'\
                'Left segment is conditioning context, right segment is for generating.'\
                'The left segment must be less than than max_lm_context_ratio of the the right segment",
        )

        # Open-domain task options:
        self.parser.add_argument(
            "--qa_prompt_format",
            type=str,
            default="question: {question} answer: <extra_id_0>",
            help="How to format question as input prompts when using --task qa",
        )

        # Multiple Choice task options:
        self.parser.add_argument(
            "--multiple_choice_num_options",
            type=int,
            default=4,
            help="How many choice options for multiple choice QA (MMLU is 4)",
        )
        self.parser.add_argument(
            "--multiple_choice_train_permutations",
            choices=["single", "cyclic", "all"],
            default="single",
            type=str,
            help="Whether to train with answer order permutations When training on multiple choice (e.g. MMLU)."
            " Can improve results by de-biasing models's preferences for arbitrary answer orderings. Recommend training with 'all'. "
            "single: no permutations. cyclic: cyclic permutations. all: all possible answer order permutations'",
        )
        self.parser.add_argument(
            "--multiple_choice_eval_permutations",
            choices=["single", "cyclic", "all"],
            default="single",
            type=str,
            help="Whether to evaluate with answer order permutations for multiple choice (e.g. MMLU)."
            " Can improve results by de-biasing models's preferences for arbitrary answer orderings. Best results with 'all' but very slow. 'cyclic' is a good compromise. "
            "single: no permutations. cyclic: cyclic permutations. all: all possible answer order permutations'",
        )

    def add_index_options(self):
        self.parser.add_argument(
            "--load_index_path",
            default=None,
            type=str,
            help="path for loading the index, passage embeddings and passages",
        )
        self.parser.add_argument(
            "--save_index_path",
            default=None,
            type=str,
            help="path for saving the index and/or embeddings",
        )
        self.parser.add_argument(
            "--save_index_n_shards",
            default=128,
            type=int,
            help="how many shards to save an index to file with. Must be an integer multiple of the number of workers.",
        )
        self.parser.add_argument(
            "--index_mode",
            type=str,
            default="flat",
            help="Use flat torch index or a faiss index for retrieving the k nearest neighbors",
            choices=["flat", "faiss"],
        )
        # faiss options:
        self.parser.add_argument(
            "--faiss_index_type",
            type=str,
            default="flat",
            help="IVFFlat, IndexFlatIP, IVFScalarQuantizer or IndexIVFPQ with faiss-gpu",
            choices=["ivfflat", "flat", "ivfsq", "ivfpq"],
        )
        self.parser.add_argument("--faiss_code_size", type=int, default=None, help="Parameter for PQ/SQ quantization")

    def print_options(self, opt):
        message = "\n"
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default_value = self.parser.get_default(k)
            if v != default_value:
                comment = f"\t(default: {default_value})"
            message += f"{k:>30}: {str(v):<40}{comment}\n"

        expr_dir = Path(opt.checkpoint_dir) / opt.name
        with open(expr_dir / "opt.log", "wt") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")

        logger.info(message)

    def parse(self, args=None):
        opt = self.parser.parse_args(args)
        if opt.closed_book:  # override flags to enable closed book mode
            opt.n_context = 1
            opt.retriever_n_context = 1
            opt.encoder_format = "{query}"
            opt.use_file_passages = True
        if opt.gold_score_mode == "pdist":  # allow paper name of retriever losses
            opt.gold_score_mode = "ppmean"
        if opt.gold_score_mode == "adist":  # allow paper name of retriever losses
            opt.gold_score_mode = "evalnormsum"
        if (
            opt.use_file_passages
        ):  # if passing use_file_passges, the following should be false (There is no retreiver loaded in this case)
            opt.train_retriever = False
            opt.query_side_retriever_training = False
            opt.use_gradient_checkpoint_retriever = False
        return opt


def get_options():
    options = Options()
    options.add_index_options()
    options.add_modeling_options()
    options.add_optim_options()
    return options
