import os
from pathlib import Path
from typing import List

import funcy
import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from durbango import DEFAULT_DEVICE, lmap, pickle_load, pickle_save, tqdm_nice
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers.tokenization_utils import trim_batch


try:
    from .grouped_batch_sampler import create_lengths_groups, GroupedBatchSampler
except ImportError:
    from grouped_batch_sampler import create_lengths_groups, GroupedBatchSampler


def load_pt(path):
    with open(path, "rb") as f:
        return torch.load(f)


def encode_file(tokenizer, data_path, max_length, pad_to_max_length=True, return_tensors="pt", overwrite_cache=False):
    cache_path = f"{data_path}_{max_length}.pkl"
    if not overwrite_cache and Path(cache_path).exists():
        return load_pt(cache_path)
    data_path = Path(data_path)
    examples = []
    lns = list(data_path.open().readlines())
    for text in tqdm_nice(lns, desc=f"Tokenizing {data_path.name}"):
        tokenized = tokenizer.batch_encode_plus(
            [text.strip()],  # DONT ADD SPACES
            max_length=max_length,
            pad_to_max_length=pad_to_max_length,
            add_prefix_space=True,
            return_tensors=return_tensors,
        )
        examples.append(tokenized)
    torch.save(lmap(dict, examples), cache_path)
    return examples


DATA_DIR = Path("/home/shleifer/transformers_fork/examples/summarization/bart/cnn_dm")


def summaries_for_file(
    model_name, type_path, data_dir=DATA_DIR, device=DEFAULT_DEVICE, num_workers=4, bs=48, **ds_kwargs
):
    model = BartForConditionalGeneration.from_pretrained(model_name)
    if torch.cuda.is_available():
        model = model.to(device).half()

    tokenizer = BartTokenizer.from_pretrained("bart-large")
    ds = SummarizationDataset(tokenizer, type_path=type_path, data_dir=data_dir, **ds_kwargs)
    dataloader = DataLoader(ds, batch_size=bs, collate_fn=ds.collate_fn, shuffle=False, num_workers=num_workers)
    save_path = data_dir / f"{type_path}_pseudo_ids.pkl"
    text_save_path = data_dir / f"{type_path}_pseudo_text.pkl"
    assert data_dir.exists()
    assert not save_path.exists()
    assert not text_save_path.exists()
    summary_ids = []
    summary_text = []
    i = 0
    for dct in tqdm_nice(dataloader):
        # dct = tokenizer.batch_encode_plus(batch, max_length=1024, return_tensors="pt", pad_to_max_length=True)
        summaries = model.generate(
            input_ids=dct["input_ids"].to(device), attention_mask=dct["attention_mask"].to(device),
        ).cpu()
        summaries = summaries[:, 1:]  # remove prefix eos
        text = tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        summary_text.append(text)
        summaries = trim_batch(summaries, 1)
        summary_ids.append(summaries)
        if i % 10 == 0:
            pickle_save(summary_ids, save_path)
            pickle_save(summary_text, text_save_path)
        i += 1
    pickle_save(summary_ids, save_path)
    pickle_save(summary_text, text_save_path)
    # Try to flatten
    flat_ids, flat_text = flatten_ids(summary_ids), flatten_list(summary_text)
    assert len(flat_ids) == sum(lmap(len, summary_ids))
    pickle_save(flat_ids, save_path)
    pickle_save(flat_text, text_save_path)
    return flat_ids, flat_text


def flatten_ids(ids):
    batches = []
    for id_batch in ids:
        batches.extend([trim_batch(x[None, :], 1).squeeze().tolist() for x in id_batch[:, 1:]])
    return batches


def flatten_list(summary_ids: List[List]):
    return [x for x in funcy.flatten(summary_ids)]


PSEUDO_ID_SUFFIX = "pseudo_target.pkl"


class SummarizationDataset(Dataset):
    pad_token_id = 1

    @classmethod
    def from_raw_data(
        cls,
        tokenizer,
        data_dir="./cnn-dailymail/cnn_dm/",
        type_path="train",
        max_source_length=1024,
        max_target_length=56,
        n_obs=None,
        overwrite_cache=False,
        tgt_suffix="",
    ):
        source = encode_file(
            tokenizer,
            os.path.join(data_dir, type_path + ".source"),
            max_source_length,
            overwrite_cache=overwrite_cache,
        )
        target = encode_file(
            tokenizer,
            os.path.join(data_dir, type_path + ".target" + tgt_suffix),
            max_target_length,
            overwrite_cache=overwrite_cache,
        )

        return cls(
            source, target, n_obs=n_obs, max_target_length=max_target_length, max_source_length=max_source_length
        )

    def __init__(self, source, target, max_source_length=1024, max_target_length=56, n_obs=None):
        if n_obs is not None:
            source = source[:n_obs]
            target = target[:n_obs]
        super().__init__()
        self.source = source
        self.target = target
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source_ids = self.source[index]["input_ids"].squeeze()
        target_ids = self.target[index]["input_ids"].squeeze()
        src_mask = self.source[index]["attention_mask"].squeeze()
        return {"input_ids": source_ids, "attention_mask": src_mask, "decoder_input_ids": target_ids}

    @staticmethod
    def trim_seq2seq_batch(batch, pad_token_id):
        y = trim_batch(batch["decoder_input_ids"], pad_token_id)
        source_ids, source_mask = trim_batch(batch["input_ids"], pad_token_id, attention_mask=batch["attention_mask"])
        return source_ids, source_mask, y

    def collate_fn(self, batch) -> dict:
        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        target_ids = torch.stack([x["decoder_input_ids"] for x in batch])
        pad_token_id = self.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        batch = {"input_ids": source_ids, "attention_mask": source_mask, "decoder_input_ids": y}
        return batch

    @property
    def src_lens(self):
        return lmap(len, self.source)

    @property
    def tgt_lens(self):
        return lmap(len, self.target)

    def make_sampler(self, params) -> GroupedBatchSampler:
        if params.n_gpu <= 1:
            sampler = RandomSampler(self)
        else:
            sampler = DistributedSampler(self)
        groups = create_lengths_groups(lengths=self.src_lens, k=self.max_source_length)
        sampler = GroupedBatchSampler(sampler=sampler, group_ids=groups, batch_size=params.train_batch_size)
        return sampler

    # TODO(SS): Unused Classmethods, can likely be deleted.
    @classmethod
    def from_pickle_paths(self, src_path, tgt_path, **kwargs):
        source = pickle_load(src_path)
        target = pickle_load(tgt_path)
        return self(source, target, **kwargs)

    @classmethod
    def from_predictions(
        cls,
        tokenizer,
        data_dir="./cnn-dailymail/cnn_dm/",
        type_path="train",
        max_source_length=1024,
        max_target_length=56,
        n_obs=None,
        overwrite_cache=False,
    ):
        source = encode_file(
            tokenizer,
            os.path.join(data_dir, type_path + ".source"),
            max_source_length,
            overwrite_cache=overwrite_cache,
        )
        tgt_path = Path(data_dir) / f"{type_path}_{PSEUDO_ID_SUFFIX}"
        assert tgt_path.exists()
        target = pickle_load(tgt_path)
        assert len(target) == len(source)

        return cls(
            source, target, n_obs=n_obs, max_target_length=max_target_length, max_source_length=max_source_length
        )
