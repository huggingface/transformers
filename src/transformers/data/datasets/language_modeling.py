import logging
import os
import pickle
import random
import time
from typing import Dict

import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset

from ...tokenization_utils import PreTrainedTokenizer


logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, overwrite_cache=False,
    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}".format(tokenizer.__class__.__name__, str(block_size), filename,),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()

                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

                for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
                    )
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)

class LineByLineWithNSPTextDataset(Dataset):
    """
    Dataset for sentence order prediction task, prepare sentence pairs for NSP task.

    Expected input file format has one sentence per line and blank lines between documents.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, random_seed: int):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            documents = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        self.examples = [self.create_examples_from_documents(i, documents, block_size, tokenizer) for i in range(len(documents))]

    def create_examples_from_documents(self, document_index, documents, block_size, tokenizer, short_seq_prob=0.1):
        """Creates examples for a single document."""
        # Account for special tokens
        max_num_tokens = block_size - tokenizer.num_special_tokens_to_add(pair=True)
        target_seq_length = max_num_tokens
        if random.random() < short_seq_prob:
            target_seq_length = random.randint(2, max_num_tokens)

        examples = [] 
        current_chunk = []
        current_length = 0
        line_index = 0

        document = documents[document_index]
        while line_index < len(document):
            # add a segment to current chunk
            segment = document[line_index]
            current_chunk.append(segment) 
            current_length += len(segment)
            # if current length goes to the target length or reaches the end of file, start building token a and b
            if line_index == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    # determine whether or not token_b is next to token_a
                    is_next = random.random() < 0.5

                    # number of segments from current_chunk that will go into the first sentence
                    a_end = 1
                    if len(current_chunk) >= 2:  
                        a_end = random.randint(1, len(current_chunk) - 1)
                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    # build token_b
                    tokens_b = []
                    if is_next:
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])
                    else:
                        # use a negative (random) sample of data
                        numbers = list(range(len(documents)))
                        numbers.remove(document_index)
                        neg_doc_index = random.choice(numbers)
                        neg_doc = documents[neg_doc_index]
                        # add a segment to the negative chunk
                        neg_segment  = neg_doc[random.randint(range(len(neg_doc)))]
                        tokens_b = [neg_segment]

                        # truncate if too long 
                    def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
                        """Truncates a pair of sequences to a maximum sequence length."""
                        while True:
                            total_length = len(tokens_a) + len(tokens_b)
                            if total_length > max_num_tokens:
                                trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
                                assert len(trunc_tokens) >= 1
                                # We want to sometimes truncate from the front and sometimes from the
                                # back to add more randomness and avoid biases.
                                if random.random() < 0.5:
                                    del trunc_tokens[0]
                                else:
                                    trunc_tokens.pop()

                    truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                    assert len(tokens_a) >= 1
                    assert len(tokens_b) >= 1

                    # add special tokens
                    input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
                    # add token type ids, 0 for sentence a, 1 for sentence b
                    token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)

                    example = {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                        "next_sentence_label": torch.tensor(0 if is_next else 1, dtype=torch.long)
                        }
                    examples.append(example)
                current_chunk = []  # clear current chunk
                current_length = 0  # reset current text length
            line_index += 1  # go to next line
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]