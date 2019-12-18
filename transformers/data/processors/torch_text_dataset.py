import os
import subprocess
import math
import itertools
import logging
from functools import partial
from multiprocessing import Pool


import tqdm
import pickle
import torch
from torch.utils.data import Dataset

logger = logging.getLogger()


def tokenize_line(tokenizer, line):
    tokens = tokenizer.tokenize(line)
    tokenized_text = tokenizer.convert_tokens_to_ids(tokens)
    return tokenized_text


def get_line_count(filename):
    try:
        out = subprocess.Popen(
            ["rg", "-cve" "^\s*$", filename],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ).communicate()[0]
        return int(out)
    except Exception:
        print("\n\n" +
              "COULDN'T FETCH LINE COUNT OF DATASET FILE, INSTALL RIPGREP TO DO SO EFFICIENTLY.\n" +
              "Install instructions: https://github.com/BurntSushi/ripgrep#installation" +
              "\n\n")
        return None


def line_buffer_generator(file_path, buffer_size):
    with open(file_path, encoding="utf-8") as f:
        buffer = []
        for line in f:
            if line == "\n":
                continue
            buffer.append(line)
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []
        yield buffer


class TextDataset(Dataset):
    def __init__(self, tokenizer, model_name_or_path, file_path='train', block_size=512, overwrite_cache=False):
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            model_name_or_path
            + "_cached_lm_parallel_"
            + str(block_size)
            + "_"
            + filename,
        )

        line_buffer_size = 500000

        part = partial(tokenize_line, tokenizer)

        if os.path.exists(cached_features_file) and not overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
                return
        else:
            logger.info("Creating features from dataset file at %s", directory)

            with Pool() as p:
                line_count = get_line_count(filename)
                total_line_chunks = math.ceil(line_count / line_buffer_size) if line_count else None
                tokenized_chunks = tqdm(
                    (
                        p.map(part, line_chunk, chunksize=(line_buffer_size // 10))
                        for line_chunk in line_buffer_generator(
                            file_path, line_buffer_size
                        )
                    ),
                    total=total_line_chunks,
                )
                tokenized_lines = list(
                    itertools.chain.from_iterable(tokenized_chunks)
                )  # flatten chunks
                tokenized_text = list(
                    itertools.chain.from_iterable(tokenized_lines)
                )  # flatten lines

                #  chunk tokens into block_size wide examples (truncate the last item if it doesn't fill the block_size)
                self.examples = [
                    tokenizer.build_inputs_with_special_tokens(
                        tokenized_text[i : i + block_size]
                    )
                    for i in range(0, len(tokenized_text) - block_size + 1, block_size)
                ]
            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])
