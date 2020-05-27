# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import logging
import math
import multiprocessing
import os
import tarfile
from itertools import chain
from pathlib import Path

from dataclasses import dataclass, field
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    ElectraForMaskedLM,
    ElectraForPreTraining,
    HfArgumentParser,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    set_seed,
)

from transformers.modeling_utils import PreTrainedModel
from transformers.training_args import TrainingArguments


logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    discriminator_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The discriminator checkpoint for weights initialization. Leave None if you want to train a model "
                    "from scratch."
        },
    )
    discriminator_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as the discriminator model_name"},
    )

    generator_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The generator checkpoint for weights initialization. Leave None if you want to train a model "
                    "from scratch."
        },
    )
    generator_config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as the generator model_name"}
    )

    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as discriminator model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )

    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )

    use_openwebtext: bool = field(
        default=False,
        metadata={"help": "Whether to use the OpenWebText dataset. If using this dataset, the user"
                          "should provide a path to the extracted zip directory. via the "
                          "--data_directory argument"}
    )

    data_directory: Optional[str] = field(
        default=False,
        metadata={"help": "The directory containing files that will be used for training and evaluation. Only used if"
                          "using OpenWebText."}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    num_dataset_building_processes: int = field(
        default=1, metadata={"help": "The number of workers that will be used to build the dataset."}
    )

    num_tensors_per_file: int = field(
        default=2048, metadata={"help": "The number of tensors that will be stored in each file after tokenization."
                                        "The smaller the amount, the smaller the filesize, but the larger the amount"
                                        "of files that will be created."}
    )

    mask_probability: float = field(
        default=0.15,
        metadata={"help": "Percentage of the input that will be masked or replaced."}
    )

    max_predictions_per_sequence: int = field(
        # Original implementation has a default of :(mask_probability + 0.005) * max_sequence_length
        default=int((0.15 + 0.005) * 512),
        metadata={"help": "Maximum tokens that will be masked in a sequence."}
    )


@dataclass
class EfficientTrainingArguments(TrainingArguments):
    max_steps: int = field(
        default=1_000_000,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )

    max_eval_steps: int = field(
        default=100,
        metadata={"help": "If > 0: set total number of eval steps to perform."},
    )
    warmup_steps: int = field(default=10_000, metadata={"help": "Linear warmup over warmup_steps."})

    weight_decay: float = field(default=0.1, metadata={"help": "Weight decay if we apply some."})

    generator_weight: float = field(default=1.0, metadata={"help": "Weight coefficient for the generator loss"})

    discriminator_weight: float = field(
        default=50.0,
        metadata={"help": "Weight coefficient for the discriminator loss"}
    )



def get_dataset(data_args: DataTrainingArguments, training_args: TrainingArguments, model_args: ModelArguments, tokenizer: Union[PreTrainedTokenizer, str], evaluate=False, local_rank=-1):
    if data_args.use_openwebtext:
        # Whether to overwrite the cache. We don't want to overwrite the cache when evaluating if we're both training
        # and evaluating, as the same dataset is used. We don't need to tokenize twice for training
        # and evaluation.
        should_overwrite_cache = False

        # If argument is specified and training, then respect the argument
        if data_args.overwrite_cache and not evaluate:
            should_overwrite_cache = True
        if data_args.overwrite_cache and training_args.do_eval and not training_args.do_train:
            should_overwrite_cache = True

        return OpenWebTextDataset(
            data_args,
            model_args,
            overwrite_cache=should_overwrite_cache
        )
    else:
        file_path = data_args.eval_data_file if evaluate else data_args.train_data_file
        return TextDataset(tokenizer=tokenizer, file_path=file_path, block_size=data_args.block_size, local_rank=local_rank,)


class OpenWebTextDataset(IterableDataset):
    def __init__(self, data_args, model_args, overwrite_cache=False):
        self.tokenizer_cache = model_args.cache_dir
        self.directory = Path(data_args.data_directory)
        self.archives = os.listdir(data_args.data_directory)
        self.tokenizer_identifier = model_args.tokenizer_name
        self.num_tensors_per_file = data_args.num_tensors_per_file
        self.feature_directory = self.directory / f"features_{self.tokenizer_identifier.replace('/', '_')}_{data_args.block_size if data_args.block_size is not None else 'no-max-seq'}_{self.num_tensors_per_file}"
        self.block_size = data_args.block_size

        # The dataset was already processed
        if os.path.exists(self.feature_directory) and not overwrite_cache:
            # TODO update to use logger
            logger.info(f"Re-using cache from {self.feature_directory}. Warning: we have no way of detecting an "
                        f"incomplete cache. If the tokenization was started but not finished, please use the "
                        f"`--ignore_cache=True` flag.")
            self.feature_set_paths = [
                self.feature_directory / feature_set_path for feature_set_path in os.listdir(self.feature_directory)
            ]
            return

        logger.info(f"Writing features at {self.feature_directory}")
        os.makedirs(self.feature_directory, exist_ok=overwrite_cache)

        n_archives_per_job = math.ceil(len(self.archives) / data_args.num_dataset_building_processes)
        self.job_archives = [
            self.archives[i * n_archives_per_job: (i + 1) * n_archives_per_job] for i in range(data_args.num_dataset_building_processes)
        ]
        # Sanity check: make sure we're not leaving any archive behind.
        assert sum([len(archive) for archive in self.job_archives]) == len(self.archives)

        if data_args.num_dataset_building_processes == 1:
            self.feature_set_paths = self._extract_open_web_text()
        else:
            pool = multiprocessing.Pool(processes=data_args.num_dataset_building_processes)
            self.feature_set_paths = pool.map(self._extract_open_web_text, range(data_args.num_dataset_building_processes))
            self.feature_set_paths = [file_path for feature_set in self.feature_set_paths for file_path in feature_set]

    def _extract_open_web_text(self, job_id=0):
        """
        OpenWebText is saved under the following format:

        openwebtext.zip
            |-> archive_xxx.zip
                |-> file_xxx.txt
                |-> file_xxz.txt
                ...
            |-> archive_xxz.zip
                |-> file_xxy.txt
                ...
            ...
        """
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_identifier, use_fast=True, cache_dir=self.tokenizer_cache)

        # Create openwebtext/tmp directory to store temporary files
        temporary_directory = self.directory / "tmp" / f"job_{job_id}"
        feature_index = 0
        feature_set_paths = []

        os.makedirs(temporary_directory, exist_ok=True)

        # Extract archives and tokenize in directory
        progress_bar = tqdm(
            self.job_archives[job_id], desc="Extracting archives", total=len(self.job_archives[0]), disable=job_id != 0
        )

        features = []
        for archive in progress_bar:
            if os.path.isdir(self.directory / archive):
                # TODO replace by logger
                print("Ignoring rogue directory.")
                continue
            with tarfile.open(self.directory / archive) as t:
                extracted_archive = temporary_directory / f"{archive}-extracted"
                t.extractall(extracted_archive)

            files = os.listdir(extracted_archive)
            for file in files:
                file_path = extracted_archive / file

                with open(file_path, "r") as f:
                    text = f.read()
                    block_size = tokenizer.model_max_length if self.block_size is None else self.block_size
                    encoding = tokenizer.encode_plus(
                        text, return_overflowing_tokens=True, max_length=block_size
                    )

                    features.append(torch.tensor(encoding["input_ids"]))

                    for overflowing_encoding in encoding.encodings[0].overflowing:
                        features.append(torch.tensor(overflowing_encoding.ids))

            while len(features) > self.num_tensors_per_file:
                feature_set_path = self.feature_directory / f"feature_set_{job_id}_{feature_index}.pt"
                torch.save(features[:self.num_tensors_per_file], feature_set_path)
                features = features[self.num_tensors_per_file:]
                feature_index += 1
                feature_set_paths.append(feature_set_path)

        if len(features) > 0:
            feature_set_path = self.feature_directory / f"feature_set_{job_id}_{feature_index}.pt"
            torch.save(features, feature_set_path)
            feature_set_paths.append(feature_set_path)

        return feature_set_paths

    @staticmethod
    def parse_file(file_index):
        try:
            features = torch.load(file_index)
            yield from features
        except RuntimeError:
            raise RuntimeError(f"Corrupted file {file_index}")

    def __len__(self):
        return len(self.feature_set_paths) * self.num_tensors_per_file

    def __iter__(self):
        return chain.from_iterable(map(self.parse_file, self.feature_set_paths))


class CombinedModel(nn.Module):
    def __init__(
            self,
            discriminator: PreTrainedModel,
            generator: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            training_args: EfficientTrainingArguments,
            data_args: DataTrainingArguments
    ):
        super().__init__()

        self.discriminator = discriminator
        self.generator = generator

        # Embeddings are shared
        self.discriminator.set_input_embeddings(self.generator.get_input_embeddings())

        self.tokenizer = tokenizer

        self.discriminator_weight = training_args.discriminator_weight
        self.generator_weight = training_args.generator_weight
        self.mask_probability = data_args.mask_probability
        self.max_predictions_per_sequence = data_args.max_predictions_per_sequence

        class Config:
            xla_device: bool = False
        self.config = Config()

    def mask_inputs(
        self,
        input_ids: torch.Tensor,
        mask_token_id,
        tokens_to_ignore,
        proposal_distribution=1.0,
    ):
        input_ids = input_ids.clone()
        inputs_which_can_be_masked = torch.ones_like(input_ids)
        for token in tokens_to_ignore:
            inputs_which_can_be_masked -= torch.eq(input_ids, token).long()

        total_number_of_tokens = input_ids.shape[-1]

        # Identify the number of tokens to be masked, which should be: 1 < num < max_predictions per seq.
        # It is set to be: n_tokens * mask_probability, but is truncated if it goes beyond bounds.
        number_of_tokens_to_be_masked = torch.max(
            torch.tensor(1),
            torch.min(
                torch.tensor(self.max_predictions_per_sequence),
                torch.tensor(total_number_of_tokens * self.mask_probability, dtype=torch.long),
            ),
        )

        # The probability of each token being masked
        sample_prob = proposal_distribution * inputs_which_can_be_masked
        sample_prob /= torch.sum(sample_prob)
        # Should be passed through a log function here

        # Weight of each position: 1 the position will be masked, 0 the position won't be masked
        masked_lm_weights = torch.tensor([0] * self.max_predictions_per_sequence, dtype=torch.bool, device=input_ids.device)
        masked_lm_weights[:number_of_tokens_to_be_masked] = True

        # Sample from the probabilities
        masked_lm_positions = sample_prob.multinomial(self.max_predictions_per_sequence)

        # Apply the weights to the positions
        masked_lm_positions *= masked_lm_weights.long()

        # Gather the IDs from the positions
        masked_lm_ids = input_ids.gather(-1, masked_lm_positions)

        # Apply weights to the IDs
        masked_lm_ids *= masked_lm_weights.long()

        replace_with_mask_positions = masked_lm_positions * (torch.rand(masked_lm_positions.shape, device=masked_lm_positions.device) < 0.85)

        # Replace the input IDs with masks on given positions
        masked_input_ids = input_ids.scatter(-1, replace_with_mask_positions, mask_token_id)

        # Updates to index 0 should be ignored
        masked_input_ids[..., 0] = input_ids[..., 0]

        return masked_input_ids, masked_lm_positions

    @staticmethod
    def gather_positions(sequence, positions):
        batch_size, sequence_length, dimension = sequence.shape
        position_shift = (sequence_length * torch.arange(batch_size, device=sequence.device)).unsqueeze(-1)
        flat_positions = torch.reshape(positions + position_shift, [-1]).long()
        flat_sequence = torch.reshape(sequence, [batch_size * sequence_length, dimension])
        gathered = flat_sequence.index_select(0, flat_positions)
        return torch.reshape(gathered, [batch_size, -1, dimension])

    @staticmethod
    def compute_metrics(
            input_ids: torch.Tensor,
            masked_lm_ids: torch.Tensor,
            masked_lm_preds: torch.Tensor,
            input_mask: torch.Tensor,
            discriminator_labels: torch.Tensor,
            discriminator_predictions: torch.Tensor,
            sampled_tokids: torch.Tensor
    ):

        input_ids
        masked_lm_accuracy = masked_lm_ids.eq(masked_lm_preds)
        sampled_masked_lm_accuracy = masked_lm_ids.eq(sampled_tokids)
        discriminator_accuracy = discriminator_labels.eq(discriminator_predictions)
        input_mask




    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        # get the masked positions as well as their original values
        masked_input_ids, masked_lm_positions = self.mask_inputs(
            input_ids,
            self.tokenizer.mask_token_id,
            [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.mask_token_id],
        )

        # only masked values should be counted in the loss; build a tensor containing the true values and -100 otherwise
        masked_lm_labels = torch.full_like(input_ids, -100)
        masked_lm_labels.scatter_(-1, masked_lm_positions, masked_input_ids)

        # mask the inputs with the mask token
        masked_tokens = torch.full_like(masked_input_ids, self.tokenizer.mask_token_id)
        masked_lm_inputs = input_ids.clone()
        masked_lm_inputs.scatter_(-1, masked_lm_positions, masked_tokens)
        masked_lm_inputs[..., 0] = 101

        generator_loss, generator_output = self.generator(
            masked_lm_inputs,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            position_ids,
            masked_lm_labels=masked_lm_labels,
        )[:2]

        # get the generator's predicted value on each masked position
        fake_logits = self.gather_positions(generator_output, masked_lm_positions)
        fake_argmaxes = fake_logits.argmax(-1)

        # create a tensor containing the predicted tokens
        fake_tokens = input_ids.scatter(-1, masked_lm_positions, fake_argmaxes)
        fake_tokens[:, 0] = input_ids[:, 0]
        discriminator_labels = torch.tensor(labels != fake_tokens, dtype=torch.uint8, device=input_ids.device)

        discriminator_loss, discriminator_output = self.discriminator(
            fake_tokens, attention_mask, token_type_ids, position_ids, head_mask, position_ids, labels=discriminator_labels
        )[:2]

        discriminator_predictions = torch.round((torch.sign(discriminator_output) + 1) / 2).int().tolist()

        total_loss = (self.discriminator_weight * discriminator_loss) + (self.generator_weight * generator_loss)

        return (total_loss, (generator_output, discriminator_output), (fake_tokens, discriminator_predictions))

    def save_pretrained(self, directory):
        if self.config.xla_device:
            self.discriminator.config.xla_device = True
            self.generator.config.xla_device = True
        else:
            self.discriminator.config.xla_device = False
            self.generator.config.xla_device = False

        generator_path = os.path.join(directory, "generator")
        discriminator_path = os.path.join(directory, "discriminator")

        if not os.path.exists(generator_path):
            os.makedirs(generator_path)

        if not os.path.exists(discriminator_path):
            os.makedirs(discriminator_path)

        self.generator.save_pretrained(generator_path)
        self.discriminator.save_pretrained(discriminator_path)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, EfficientTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if not data_args.use_openwebtext and data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.discriminator_config_name:
        discriminator_config = AutoConfig.from_pretrained(
            model_args.discriminator_config_name, cache_dir=model_args.cache_dir
        )
    elif model_args.discriminator_name_or_path:
        discriminator_config = AutoConfig.from_pretrained(
            model_args.discriminator_name_or_path, cache_dir=model_args.cache_dir
        )
    else:
        raise ValueError("Either --discriminator_config_name or --discriminator_name_or_path should be specified.")

    if model_args.generator_config_name:
        generator_config = AutoConfig.from_pretrained(model_args.generator_config_name, cache_dir=model_args.cache_dir)
    elif model_args.generator_name_or_path:
        generator_config = AutoConfig.from_pretrained(
            model_args.generator_name_or_path, cache_dir=model_args.cache_dir
        )
    else:
        raise ValueError("Either --generator_config_name or --generator_name_or_path should be specified.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.discriminator_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.discriminator_name_or_path, cache_dir=model_args.cache_dir
        )
        model_args.tokenizer_name = model_args.discriminator_name_or_path
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if model_args.discriminator_name_or_path:
        discriminator = ElectraForPreTraining.from_pretrained(
            model_args.discriminator_name_or_path,
            from_tf=bool(".ckpt" in model_args.discriminator_name_or_path),
            config=discriminator_config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        discriminator = ElectraForPreTraining(discriminator_config)

    if model_args.generator_name_or_path:
        generator = ElectraForMaskedLM.from_pretrained(
            model_args.generator_name_or_path,
            from_tf=bool(".ckpt" in model_args.generator_name_or_path),
            config=generator_config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        generator = ElectraForMaskedLM(generator_config)

    discriminator.resize_token_embeddings(len(tokenizer))
    generator.resize_token_embeddings(len(tokenizer))

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    try:
        import torch_xla.core.xla_model as xm
        if xm.is_master_ordinal(local=True):
            get_dataset(data_args, training_args, model_args, tokenizer=tokenizer, local_rank=training_args.local_rank)

        xm.rendezvous("dataset building")
    except ImportError:
        logger.info("Not running on TPU")

    # Get datasets
    train_dataset = (
        get_dataset(data_args, training_args, model_args, tokenizer=tokenizer, local_rank=training_args.local_rank)
        if training_args.do_train
        else None
    )
    eval_dataset = (
        get_dataset(data_args, training_args, model_args, tokenizer=tokenizer, local_rank=training_args.local_rank, evaluate=True)
        if training_args.do_eval
        else None
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, mlm_probability=0)

    model = CombinedModel(discriminator, generator, tokenizer, training_args, data_args)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
    )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.discriminator_name_or_path
            if model_args.discriminator_name_or_path is not None
            and os.path.isdir(model_args.discriminator_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        result = {"loss": eval_output["eval_loss"]}

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
