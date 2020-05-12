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
import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoTokenizer,
    ElectraForPreTraining,
    ElectraForMaskedLM,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
)

from transformers.trainer import is_apex_available
import torch
import torch.nn as nn
from typing import Optional
from transformers.modeling_utils import PreTrainedModel
from transformers.training_args import TrainingArguments

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    discriminator_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The discriminator checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    discriminator_config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as the discriminator model_name"}
    )

    generator_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The generator checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    generator_config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as the generator model_name"}
    )

    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as discriminator model_name"}
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


def get_dataset(args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, evaluate=False, local_rank=-1):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    return TextDataset(
        tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, local_rank=local_rank,
    )


class CombinedModel(nn.Module):
    def __init__(self, discriminator: PreTrainedModel, generator: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        super().__init__()

        self.discriminator = discriminator
        self.generator = generator

        # Embeddings are shared
        self.discriminator.set_input_embeddings(self.generator.get_input_embeddings())

        self.tokenizer = tokenizer

    @staticmethod
    def mask_inputs(
            input_ids: torch.Tensor,
            mask_token_id,
            mask_probability,
            tokens_to_ignore,
            max_predictions_per_seq,
            proposal_distribution=1.0
    ):
        inputs_which_can_be_masked = torch.ones_like(input_ids)
        for token in tokens_to_ignore:
            inputs_which_can_be_masked -= torch.eq(input_ids, token).long()

        total_number_of_tokens = input_ids.shape[-1]

        # Identify the number of tokens to be masked, which should be: 1 < num < max_predictions per seq.
        # It is set to be: n_tokens * mask_probability, but is truncated if it goes beyond bounds.
        number_of_tokens_to_be_masked = torch.max(
            torch.tensor(1),
            torch.min(
                torch.tensor(max_predictions_per_seq),
                torch.tensor(total_number_of_tokens * mask_probability, dtype=torch.long)
            )
        )

        # The probability of each token being masked
        sample_prob = proposal_distribution * inputs_which_can_be_masked
        sample_prob /= torch.sum(sample_prob)
        # Should be passed through a log function here

        # Weight of each position: 1 the position will be masked, 0 the position won't be masked
        masked_lm_weights = torch.tensor([0] * max_predictions_per_seq, dtype=torch.bool)
        masked_lm_weights[:number_of_tokens_to_be_masked] = True

        # Sample from the probabilities
        masked_lm_positions = sample_prob.multinomial(max_predictions_per_seq)

        # Apply the weights to the positions
        masked_lm_positions *= masked_lm_weights.long()

        # Gather the IDs from the positions
        masked_lm_ids = input_ids.gather(-1, masked_lm_positions)

        # Apply weights to the IDs
        masked_lm_ids *= masked_lm_weights.long()

        replace_with_mask_positions = masked_lm_positions * (torch.rand(masked_lm_positions.shape) < 0.85)

        # Replace the input IDs with masks on given positions
        masked_input_ids = input_ids.scatter(-1, replace_with_mask_positions, mask_token_id)

        # Updates to index 0 should be ignored
        masked_input_ids[..., 0] = input_ids[..., 0]

        return masked_input_ids, masked_lm_positions

    @staticmethod
    def gather_positions(
            sequence,
            positions
    ):
        batch_size, sequence_length, dimension = sequence.shape
        position_shift = (sequence_length * torch.arange(batch_size)).unsqueeze(-1)
        flat_positions = torch.reshape(positions + position_shift, [-1]).long()
        flat_sequence = torch.reshape(sequence, [batch_size * sequence_length, dimension])
        gathered = flat_sequence.index_select(0, flat_positions)
        return torch.reshape(gathered, [batch_size, -1, dimension])

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None
    ):
        masked_input_ids, masked_lm_positions = self.mask_inputs(
            input_ids,
            self.tokenizer.mask_token_id,
            0.2,
            [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.mask_token_id],
            30
        )

        generator_loss, generator_output = self.generator(
            masked_input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            position_ids,
            masked_lm_labels=labels
        )[:2]

        fake_logits = self.gather_positions(generator_output, masked_lm_positions)
        fake_argmaxes = fake_logits.argmax(-1)
        fake_tokens = masked_input_ids.scatter(-1, masked_lm_positions, fake_argmaxes)
        fake_tokens[:, 0] = input_ids[:, 0]

        # discriminator_output
        discriminator_loss, discriminator_output = self.discriminator(
            fake_tokens,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            position_ids,
            labels=labels
        )[:2]

        discriminator_predictions = torch.round((torch.sign(discriminator_output) + 1) / 2).int().tolist()

        total_loss = discriminator_loss + generator_loss

        return (
            total_loss,
            (discriminator_predictions, generator_output),
            (fake_tokens, masked_input_ids)
        )

    def save_pretrained(self, directory):
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

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.eval_data_file is None and training_args.do_eval:
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
        discriminator_config = AutoConfig.from_pretrained(model_args.discriminator_config_name, cache_dir=model_args.cache_dir)
    elif model_args.discriminator_name_or_path:
        discriminator_config = AutoConfig.from_pretrained(model_args.discriminator_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError("Either --discriminator_config_name or --discriminator_name_or_path should be specified.")

    if model_args.generator_config_name:
        generator_config = AutoConfig.from_pretrained(model_args.generator_config_name, cache_dir=model_args.cache_dir)
    elif model_args.generator_name_or_path:
        generator_config = AutoConfig.from_pretrained(model_args.generator_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError("Either --generator_config_name or --generator_name_or_path should be specified.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.discriminator_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.discriminator_name_or_path, cache_dir=model_args.cache_dir)
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
        discriminator = ElectraForPreTraining.from_config(discriminator_config)

    if model_args.generator_name_or_path:
        generator = ElectraForMaskedLM.from_pretrained(
            model_args.generator_name_or_path,
            from_tf=bool(".ckpt" in model_args.generator_name_or_path),
            config=generator_config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        generator = ElectraForMaskedLM.from_config(generator_config)

    discriminator.resize_token_embeddings(len(tokenizer))
    generator.resize_token_embeddings(len(tokenizer))

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    # Get datasets
    train_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, local_rank=training_args.local_rank)
        if training_args.do_train
        else None
    )
    eval_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, local_rank=training_args.local_rank, evaluate=True)
        if training_args.do_eval
        else None
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, mlm_probability=0
    )

    model = CombinedModel(discriminator, generator, tokenizer)

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
            if model_args.discriminator_name_or_path is not None and os.path.isdir(model_args.discriminator_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    return results


if __name__ == "__main__":
    main()
