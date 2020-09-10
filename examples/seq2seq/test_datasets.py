import tempfile
from collections import defaultdict
from pathlib import Path

import pytest
from torch.utils.data import DataLoader

from durbango import pickle_save
from transformers import AutoTokenizer
from transformers.modeling_bart import shift_tokens_right

from .pack_dataset import pack_data_dir
from .test_seq2seq_examples import ARTICLES, BART_TINY, MARIAN_TINY, MBART_TINY, SUMMARIES, T5_TINY, make_test_data_dir
from .utils import LegacySeq2SeqDataset, Seq2SeqDataset


@pytest.mark.parametrize(
    ["tok_name"],
    [
        pytest.param(MBART_TINY),
        pytest.param(MARIAN_TINY),
        pytest.param(T5_TINY),
        pytest.param(BART_TINY),
        pytest.param("google/pegasus-xsum"),
    ],
)
def test_seq2seq_dataset_truncation(tok_name):
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    tmp_dir = make_test_data_dir()
    max_len_source = max(len(tokenizer.encode(a)) for a in ARTICLES)
    max_len_target = max(len(tokenizer.encode(a)) for a in SUMMARIES)
    max_src_len = 4
    max_tgt_len = 8
    assert max_len_target > max_src_len  # Will be truncated
    assert max_len_source > max_src_len  # Will be truncated
    src_lang, tgt_lang = "ro_RO", "de_DE"  # ignored for all but mbart, but never causes error.
    train_dataset = Seq2SeqDataset(
        tokenizer,
        data_dir=tmp_dir,
        type_path="train",
        max_source_length=max_src_len,
        max_target_length=max_tgt_len,  # ignored
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )
    dataloader = DataLoader(train_dataset, batch_size=2, collate_fn=train_dataset.collate_fn)
    for batch in dataloader:
        assert isinstance(batch, dict)
        assert batch["attention_mask"].shape == batch["input_ids"].shape
        # show that articles were trimmed.
        assert batch["input_ids"].shape[1] == max_src_len
        # show that targets are the same len
        assert batch["labels"].shape[1] == max_tgt_len
        if tok_name != MBART_TINY:
            continue
        # check language codes in correct place
        batch["decoder_input_ids"] = shift_tokens_right(batch["labels"], tokenizer.pad_token_id)
        assert batch["decoder_input_ids"][0, 0].item() == tokenizer.lang_code_to_id[tgt_lang]
        assert batch["decoder_input_ids"][0, -1].item() == tokenizer.eos_token_id
        assert batch["input_ids"][0, -2].item() == tokenizer.eos_token_id
        assert batch["input_ids"][0, -1].item() == tokenizer.lang_code_to_id[src_lang]

        break  # No need to test every batch


@pytest.mark.parametrize(["tok"], [pytest.param(BART_TINY), pytest.param("bert-base-cased")])
def test_legacy_dataset_truncation(tok):
    tokenizer = AutoTokenizer.from_pretrained(tok)
    tmp_dir = make_test_data_dir()
    max_len_source = max(len(tokenizer.encode(a)) for a in ARTICLES)
    max_len_target = max(len(tokenizer.encode(a)) for a in SUMMARIES)
    trunc_target = 4
    train_dataset = LegacySeq2SeqDataset(
        tokenizer,
        data_dir=tmp_dir,
        type_path="train",
        max_source_length=20,
        max_target_length=trunc_target,
    )
    dataloader = DataLoader(train_dataset, batch_size=2, collate_fn=train_dataset.collate_fn)
    for batch in dataloader:
        assert batch["attention_mask"].shape == batch["input_ids"].shape
        # show that articles were trimmed.
        assert batch["input_ids"].shape[1] == max_len_source
        assert 20 >= batch["input_ids"].shape[1]  # trimmed significantly
        # show that targets were truncated
        assert batch["labels"].shape[1] == trunc_target  # Truncated
        assert max_len_target > trunc_target  # Truncated
        break  # No need to test every batch


def test_pack_dataset():
    tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")

    tmp_dir = Path(make_test_data_dir())
    orig_examples = tmp_dir.joinpath("train.source").open().readlines()
    save_dir = Path(tempfile.mkdtemp(prefix="packed_"))
    pack_data_dir(tokenizer, tmp_dir, 128, save_dir)
    orig_paths = {x.name for x in tmp_dir.iterdir()}
    new_paths = {x.name for x in save_dir.iterdir()}
    packed_examples = save_dir.joinpath("train.source").open().readlines()
    # orig: [' Sam ate lunch today.\n', 'Sams lunch ingredients.']
    # desired_packed: [' Sam ate lunch today.\n Sams lunch ingredients.']
    assert len(packed_examples) < len(orig_examples)
    assert len(packed_examples) == 1
    assert len(packed_examples[0]) == sum(len(x) for x in orig_examples)
    assert orig_paths == new_paths


import os


def test_dynamic_batch_size():
    if os.getenv("USE_REAL_DATA", False):
        data_dir = "examples/seq2seq/wmt_en_ro"
        max_len = 128
        max_tokens = max_len * 2 * 64
    else:
        data_dir = "examples/seq2seq/test_data/wmt_en_ro"
        max_tokens = 320
        max_len = 64

    tokenizer = AutoTokenizer.from_pretrained(MARIAN_TINY)
    ds = Seq2SeqDataset(
        tokenizer, data_dir=data_dir, type_path="train", max_source_length=max_len, max_target_length=max_len, n_obs=1000
    )

    logs = defaultdict(list)
    mult = 64
    batch_sampler = ds.make_dynamic_sampler(max_tokens, required_batch_size_multiple=mult)
    from durbango import pickle_save
    pickle_save(batch_sampler, "batches/dynamic_sampler.pkl")
    batch_sizes = [len(x) for x in batch_sampler]
    assert len(set(batch_sizes)) > 1
    data_loader = DataLoader(
        ds,
        batch_sampler=batch_sampler,
        collate_fn=ds.collate_fn,
        # shuffle=True,
        num_workers=2,
        # batch_size=None,
    )

    failures = []
    pad = tokenizer.pad_token_id
    for batch in data_loader:
        logs["tpb"].append(batch["input_ids"].ne(pad).sum() + batch["labels"].ne(pad).sum())
        logs["bs"].append(batch["input_ids"].shape[0])
        logs["src_pad_tok"].append(batch["input_ids"].eq(pad).sum())
        logs["src_pad_frac"].append(batch["input_ids"].eq(pad).float().mean())
        shapes = {k: v.shape for k, v in batch.items()}
        bs = shapes["input_ids"][0]
        assert bs % mult == 0 or bs < mult
        num_src_tokens = shapes["input_ids"][0] * shapes["input_ids"][1]
        num_tgt_tokens = shapes["labels"][0] * shapes["labels"][1]
        if num_tgt_tokens + num_src_tokens > (max_tokens * 1.1):
            failures.append(num_tgt_tokens + num_src_tokens)
    from durbango import pickle_save

    pickle_save(logs, "batches/dynb_logs.pkl")
    # if failures:raise AssertionError(f"found {len(failures)} batches with more than {max_tokens}")


def test_sortish():
    if os.getenv("USE_REAL_DATA", False):
        data_dir = "examples/seq2seq/wmt_en_ro"
        max_len = 128
        max_tokens = max_len * 2 * 64
    else:
        data_dir = "examples/seq2seq/test_data/wmt_en_ro"
        max_tokens = 320
        max_len = 64

    tokenizer = AutoTokenizer.from_pretrained(MARIAN_TINY)
    ds = Seq2SeqDataset(
        tokenizer, data_dir=data_dir, type_path="train", max_source_length=max_len, max_target_length=max_len,
        n_obs=1000
    )
    sampler = ds.make_sortish_sampler(64)
    pickle_save(sampler, "batches/sortish.pkl")
    from collections import defaultdict

    logs = defaultdict(list)
    mult = 64
    # batch_sampler = ds.make_dynamic_sampler(max_tokens, required_batch_size_multiple=mult)
    #batch_sizes = [len(x) for x in batch_sampler]
    #assert len(set(batch_sizes)) > 1
    data_loader = DataLoader(
        ds,
        sampler=sampler,
        batch_size=64,
        # batch_sampler=batch_sampler,
        collate_fn=ds.collate_fn,
        # shuffle=True,
        num_workers=2,
        # batch_size=None,
    )

    failures = []
    pad = tokenizer.pad_token_id
    for batch in data_loader:
        logs["tpb"].append(batch["input_ids"].ne(pad).sum() + batch["labels"].ne(pad).sum())
        logs["bs"].append(batch["input_ids"].shape[0])
        logs["src_pad_tok"].append(batch["input_ids"].eq(pad).sum())
        logs["src_pad_frac"].append(batch["input_ids"].eq(pad).float().mean())
        shapes = {k: v.shape for k, v in batch.items()}
        bs = shapes["input_ids"][0]
        assert bs % mult == 0 or bs < mult
        num_src_tokens = shapes["input_ids"][0] * shapes["input_ids"][1]
        num_tgt_tokens = shapes["labels"][0] * shapes["labels"][1]
        if num_tgt_tokens + num_src_tokens > (max_tokens * 1.1):
            failures.append(num_tgt_tokens + num_src_tokens)

    pickle_save(logs, "batches/sortish_logs.pkl")
    # if failures:raise AssertionError(f"found {len(failures)} batches with more than {max_tokens}")
