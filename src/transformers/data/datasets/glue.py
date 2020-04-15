import logging
import os
import time
from typing import List, Optional

import torch
from torch.utils.data.dataset import Dataset

from ...tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_xlm_roberta import XLMRobertaTokenizer
from ...trainer import torch_distributed_zero_first
from ...training_args import DataTrainingArguments
from ..processors.glue import glue_convert_examples_to_features, glue_output_modes, glue_processors
from ..processors.utils import InputFeatures


logger = logging.getLogger(__name__)


class GlueDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    args: DataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
        self,
        args: DataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        evaluate=False,
        local_rank=-1,
    ):
        self.args = args
        processor = glue_processors[args.task_name]()
        self.output_mode = glue_output_modes[args.task_name]
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}_{}".format(
                "dev" if evaluate else "train",
                tokenizer.__class__.__name__,
                str(args.max_seq_length),
                str(args.task_name),
            ),
        )
        with torch_distributed_zero_first(local_rank):
            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")
                label_list = processor.get_labels()
                if args.task_name in ["mnli", "mnli-mm"] and tokenizer.__class__ in (
                    RobertaTokenizer,
                    RobertaTokenizerFast,
                    XLMRobertaTokenizer,
                ):
                    # HACK(label indices are swapped in RoBERTa pretrained model)
                    label_list[1], label_list[2] = label_list[2], label_list[1]
                examples = (
                    processor.get_dev_examples(args.data_dir)
                    if evaluate
                    else processor.get_train_examples(args.data_dir)
                )
                if limit_length is not None:
                    examples = examples[:limit_length]
                self.features = glue_convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    label_list=label_list,
                    output_mode=self.output_mode,
                )
                if local_rank in [-1, 0]:
                    start = time.time()
                    torch.save(self.features, cached_features_file)
                    # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                    logger.info(
                        f"Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                    )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]
