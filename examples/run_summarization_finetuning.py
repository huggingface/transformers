# coding=utf-8
# Copyright 2019 The HuggingFace Inc. team.
# Copyright (c) 2019 The HuggingFace Inc.  All rights reserved.
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
""" Finetuning seq2seq models for sequence generation."""

import argparse
from collections import deque
import logging
import os
import pickle
import random
import sys

import numpy as np
from tqdm import tqdm, trange
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from transformers import AutoTokenizer, PreTrainedSeq2seq, Model2Model

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


# ------------
# Load dataset
# ------------


class TextDataset(Dataset):
    """ Abstracts the dataset used to train seq2seq models.

    CNN/Daily News:

    The CNN/Daily News raw datasets are downloaded from [1]. The stories are
    stored in different files; the summary appears at the end of the story as
    sentences that are prefixed by the special `@highlight` line. To process
    the data, untar both datasets in the same folder, and pass the path to this
    folder as the "data_dir argument. The formatting code was inspired by [2].

    [1] https://cs.nyu.edu/~kcho/
    [2] https://github.com/abisee/cnn-dailymail/
    """

    def __init__(self, tokenizer, prefix="train", data_dir="", block_size=512):
        assert os.path.isdir(data_dir)

        # Load the features that have already been computed, if any
        cached_features_file = os.path.join(
            data_dir, "cached_lm_{}_{}".format(block_size, prefix)
        )
        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as source:
                self.examples = pickle.load(source)
                return

        logger.info("Creating features from dataset at %s", data_dir)
        datasets = ["cnn", "dailymail"]

        self.examples = {"source": [], "target": []}
        for dataset in datasets:
            path_to_stories = os.path.join(data_dir, dataset, "stories")
            story_filenames_list = os.listdir(path_to_stories)
            for story_filename in story_filenames_list:
                path_to_story = os.path.join(path_to_stories, story_filename)
                if not os.path.isfile(path_to_story):
                    continue

                with open(path_to_story, encoding="utf-8") as source:
                    raw_story = source.read()
                    story_lines, summary_lines = process_story(raw_story)
                    if len(summary_lines) == 0 or len(story_lines) == 0:
                        continue

                story_token_ids, summary_token_ids = _encode_for_summarization(
                    story_lines, summary_lines, tokenizer
                )
                story_seq = _fit_to_block_size(story_token_ids, block_size)
                self.examples["source"].append(story_seq)

                summary_seq = _fit_to_block_size(summary_token_ids, block_size)
                self.examples["summary"].append(summary_seq)

        logger.info("Saving features into cache file %s", cached_features_file)
        with open(cached_features_file, "wb") as sink:
            pickle.dump(self.examples, sink, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, items):
        return (
            torch.tensor(self.examples["source"][items]),
            torch.tensor(self.examples["target"][items]),
        )


def process_story(raw_story):
    """ Extract the story and summary from a story file.

    Attributes:
        raw_story (str): content of the story file as an utf-8 encoded string.

    Raises:
        IndexError: If the stoy is empty or contains no highlights.
    """
    nonempty_lines = list(
        filter(lambda x: len(x) != 0, [line.strip() for line in raw_story.split("\n")])
    )

    # for some unknown reason some lines miss a period, add it
    nonempty_lines = [_add_missing_period(line) for line in nonempty_lines]

    # gather article lines
    story_lines = []
    lines = deque(nonempty_lines)
    while True:
        try:
            element = lines.popleft()
            if element.startswith("@highlight"):
                break
            story_lines.append(element)
        except IndexError:
            # if "@highlight" is absent from the file we pop
            # all elements until there is None.
            return story_lines, []

    # gather summary lines
    summary_lines = list(filter(lambda t: not t.startswith("@highlight"), lines))

    return story_lines, summary_lines


def _encode_for_summarization(story_lines, summary_lines, tokenizer):
    """ Encode the story and summary lines, and join them
    as specified in [1] by using `[SEP] [CLS]` tokens to separate
    sentences.
    """
    story_lines_token_ids = [
        tokenizer.add_special_tokens_single_sequence(tokenizer.encode(line))
        for line in story_lines
    ]
    summary_lines_token_ids = [
        tokenizer.add_special_tokens_single_sequence(tokenizer.encode(line))
        for line in summary_lines
    ]

    story_token_ids = [
        token for sentence in story_lines_token_ids for token in sentence
    ]
    summary_token_ids = [
        token for sentence in summary_lines_token_ids for token in sentence
    ]

    return story_token_ids, summary_token_ids


def _add_missing_period(line):
    END_TOKENS = [".", "!", "?", "...", "'", "`", '"', u"\u2019", u"\u2019", ")"]
    if line.startswith("@highlight"):
        return line
    if line[-1] in END_TOKENS:
        return line
    return line + "."


def _fit_to_block_size(sequence, block_size):
    """ Adapt the source and target sequences' lengths to the block size.
    If the sequence is shorter than the block size we pad it with -1 ids
    which correspond to padding tokens.
    """
    if len(sequence) > block_size:
        return sequence[:block_size]
    else:
        sequence.extend([0] * (block_size - len(sequence)))
        return sequence


def mask_padding_tokens(sequence):
    """ Padding token, encoded as 0, are represented by the value -1 in the
    masks """
    padded = sequence.clone()
    padded[padded == 0] = -1
    return padded


def load_and_cache_examples(args, tokenizer):
    dataset = TextDataset(tokenizer, data_dir=args.data_dir)
    return dataset


def compute_token_type_ids(batch, separator_token_id):
    """ Segment embeddings as described in [1]

    The values {0,1} were found in the repository [2].

    Attributes:
        batch: torch.Tensor, size [batch_size, block_size]
            Batch of input.
        separator_token_id: int
            The value of the token that separates the segments.

    [1] Liu, Yang, and Mirella Lapata. "Text summarization with pretrained encoders."
        arXiv preprint arXiv:1908.08345 (2019).
    [2] https://github.com/nlpyang/PreSumm (/src/prepro/data_builder.py, commit fac1217)
    """
    batch_embeddings = []
    sentence_num = 0
    for sequence in batch:
        embeddings = []
        for s in sequence:
            if s == separator_token_id:
                sentence_num += 1
            embeddings.append(sentence_num % 2)
        batch_embeddings.append(embeddings)
    return torch.tensor(batch_embeddings)


# ----------
# Optimizers
# ----------


class BertSumOptimizer(object):
    """ Specific optimizer for BertSum.

    As described in [1], the authors fine-tune BertSum for abstractive
    summarization using two Adam Optimizers with different warm-up steps and
    learning rate. They also use a custom learning rate scheduler.

    [1] Liu, Yang, and Mirella Lapata. "Text summarization with pretrained encoders."
        arXiv preprint arXiv:1908.08345 (2019).
    """

    def __init__(self, model, lr, warmup_steps, beta_1=0.99, beta_2=0.999, eps=1e-9):
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.lr = lr
        self.warmup_steps = warmup_steps

        self.optimizers = {
            "encoder": Adam(
                model.encoder.parameters(),
                lr=lr["encoder"],
                betas=(beta_1, beta_2),
                eps=eps,
            ),
            "decoder": Adam(
                model.decoder.parameters(),
                lr=lr["decoder"],
                betas=(beta_1, beta_2),
                eps=eps,
            ),
        }

        self._step = 0

    def _update_rate(self, stack):
        return self.lr[stack] * min(
            self._step ** (-0.5), self._step * self.warmup_steps[stack] ** (-0.5)
        )

    def zero_grad(self):
        self.optimizer_decoder.zero_grad()
        self.optimizer_encoder.zero_grad()

    def step(self):
        self._step += 1
        for stack, optimizer in self.optimizers.items():
            new_rate = self._update_rate(stack)
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_rate
            optimizer.step()


# ------------
# Train
# ------------


def train(args, model, tokenizer):
    """ Fine-tune the pretrained model on the corpus. """
    set_seed(args)

    # Load the data
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_dataset = load_and_cache_examples(args, tokenizer)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size
    )

    # Training schedule
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = t_total // (
            len(train_dataloader) // args.gradient_accumulation_steps + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    # Prepare the optimizer
    lr = {"encoder": 0.002, "decoder": 0.2}
    warmup_steps = {"encoder": 20000, "decoder": 10000}
    optimizer = BertSumOptimizer(model, lr, warmup_steps)

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size * args.gradient_accumulation_steps
        # * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    model.zero_grad()
    train_iterator = trange(args.num_train_epochs, desc="Epoch", disable=True)

    global_step = 0
    tr_loss = 0.0
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
        for step, batch in enumerate(epoch_iterator):
            source, target = batch
            token_type_ids = compute_token_type_ids(source, tokenizer.cls_token_id)
            labels_src = mask_padding_tokens(source)
            labels_tgt = mask_padding_tokens(target)

            source = source.to(args.device)
            target = target.to(args.device)
            token_type_ids = token_type_ids.to(args.device)
            labels_src = labels_src.to(args.device)
            labels_tgt = labels_tgt.to(args.device)

            model.train()
            outputs = model(
                source,
                target,
                token_type_ids=token_type_ids,
                decoder_encoder_attention_mask=labels_src,
                decoder_attention_mask=labels_tgt,
                decoder_lm_labels=labels_tgt,
                decoder_initialize_randomly=True,
            )

            loss = outputs[0]
            print(loss)
            if args.gradient_accumulation_steps > 1:
                loss /= args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                model.zero_grad()
                global_step += 1

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


# ------------
# Train
# ------------


def evaluate(args, model, tokenizer, prefix=""):
    set_seed(args)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
    )

    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        source, target = batch
        labels_src = mask_padding_tokens(source)
        labels_tgt = mask_padding_tokens(target)
        source.to(args.device)
        target.to(args.device)
        labels_src.to(args.device)
        labels_tgt.to(args.device)

        with torch.no_grad():
            outputs = model(
                source,
                target,
                decoder_encoder_attention_mask=labels_src,
                decoder_attention_mask=labels_tgt,
                decoder_lm_labels=labels_tgt,
            )
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity}

    # Save the evaluation's results
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input training data file (a text file).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Optional parameters
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--do_evaluate",
        type=bool,
        default=False,
        help="Run model evaluation on out-of-sample data.",
    )
    parser.add_argument("--do_train", type=bool, default=False, help="Run training.")
    parser.add_argument(
        "--do_overwrite_output_dir",
        type=bool,
        default=False,
        help="Whether to overwrite the output dir.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-cased",
        type=str,
        help="The model checkpoint to initialize the encoder and decoder's weights with.",
    )
    parser.add_argument(
        "--model_type",
        default="bert",
        type=str,
        help="The decoder architecture to be fine-tuned.",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--to_cpu", default=False, type=bool, help="Whether to force training on CPU."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=1,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.do_overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --do_overwrite_output_dir to overwrite.".format(
                args.output_dir
            )
        )

    # Set up training device
    if args.to_cpu or not torch.cuda.is_available():
        args.device = torch.device("cpu")
        args.n_gpu = 0
    else:
        args.device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()

    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = Model2Model.from_pretrained(args.model_name_or_path)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        0,
        args.device,
        args.n_gpu,
        False,
        False,
    )

    logger.info("Training/evaluation parameters %s", args)

    # Train the model
    model.to(args.device)
    if args.do_train:
        global_step, tr_loss = train(args, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, "training_arguments.bin"))

    # Evaluate the model
    results = {}
    if args.do_evaluate:
        checkpoints = []
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            encoder_checkpoint = os.path.join(checkpoint, "encoder")
            decoder_checkpoint = os.path.join(checkpoint, "decoder")
            model = PreTrainedSeq2seq.from_pretrained(
                encoder_checkpoint, decoder_checkpoint
            )
            model.to(args.device)
            results = "placeholder"

    return results


if __name__ == "__main__":
    main()
