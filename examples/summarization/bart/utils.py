import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from durbango import lmap, pickle_load, pickle_save, tqdm_nice
from transformers.tokenization_utils import trim_batch


def encode_file(tokenizer, data_path, max_length, pad_to_max_length=True, return_tensors="pt", overwrite_cache=False):
    cache_path = f"{data_path}_{max_length}.pkl"
    if not overwrite_cache and Path(cache_path).exists():
        return torch.load(open(cache_path))


    examples = []
    with open(data_path, "r") as f:
        for text in tqdm_nice(f.readlines(), desc=f'Tokenizing {data_path}'):
            tokenized = tokenizer.batch_encode_plus(
                [text], max_length=max_length, pad_to_max_length=pad_to_max_length, return_tensors=return_tensors,
            )
            examples.append(tokenized)
    torch.save(lmap(dict, examples), cache_path)
    return examples


class SummarizationDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir="./cnn-dailymail/cnn_dm/",
        type_path="train",
        max_source_length=1024,
        max_target_length=56,
        overwrite_cache=False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.source = encode_file(
            tokenizer,
            os.path.join(data_dir, type_path + ".source"),
            max_source_length,
            overwrite_cache=overwrite_cache,
        )
        self.target = encode_file(
            tokenizer,
            os.path.join(data_dir, type_path + ".target"),
            max_target_length,
            overwrite_cache=overwrite_cache,
        )

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source_ids = self.source[index]["input_ids"].squeeze()
        target_ids = self.target[index]["input_ids"].squeeze()
        src_mask = self.source[index]["attention_mask"].squeeze()
        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids}

    @staticmethod
    def trim_seq2seq_batch(batch, pad_token_id):
        y = trim_batch(batch["decoder_input_ids"], pad_token_id)
        source_ids, source_mask = trim_batch(batch["input_ids"], pad_token_id, attention_mask=batch["attention_mask"])
        return source_ids, source_mask, y

    def collate_fn(self, batch) -> dict:
        input_ids = torch.stack([x["source_ids"] for x in batch])
        masks = torch.stack([x["source_mask"] for x in batch])
        target_ids = torch.stack([x["target_ids"] for x in batch])
        pad_token_id = self.tokenizer.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        batch = {"input_ids": source_ids, "attention_mask": source_mask, "decoder_input_ids": y}
        return batch
