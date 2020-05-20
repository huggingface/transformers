import os
from pathlib import Path
from typing import List

import funcy
import torch
from torch.utils.data import DataLoader, Dataset

from durbango import DEFAULT_DEVICE, lmap, pickle_load, pickle_save, tqdm_nice
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers.tokenization_utils import trim_batch


def load_pt(path):
    with open(path, "rb") as f:
        return torch.load(f)


def encode_file(tokenizer, data_path, max_length, pad_to_max_length=True, return_tensors="pt", overwrite_cache=False):
    cache_path = f"{data_path}_{max_length}.pkl"
    if not overwrite_cache and Path(cache_path).exists():
        return load_pt(cache_path)

    examples = []
    with open(data_path, "r") as f:
        for text in tqdm_nice(f.readlines(), desc=f"Tokenizing {data_path}"):
            tokenized = tokenizer.batch_encode_plus(
                [text], max_length=max_length, pad_to_max_length=pad_to_max_length, return_tensors=return_tensors,
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
        model = model.half().to(device)

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


class SummarizationDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir="./cnn-dailymail/cnn_dm/",
        type_path="train",
        max_source_length=1024,
        max_target_length=56,
        n_obs=None,
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
        if n_obs is not None:
            self.source = self.source[:n_obs]
            self.source = self.target[:n_obs]

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
