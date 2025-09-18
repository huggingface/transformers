import json
import os
import tempfile

import pytest

from transformers.models.blueberry.tokenization_blueberry import (
    BlueberryTokenizer,
    HARMONY_CHAT_TEMPLATE,
    HARMONY_SPECIAL_TOKENS,
)
from transformers.models.blueberry.tokenization_blueberry_fast import BlueberryTokenizerFast


def _write_fake_gpt2_files(tmpdir):
    # minimal vocab and merges
    vocab = {
        "<|endoftext|>": 0,
        " !": 1,
        "Hello": 2,
        "World": 3,
        "[UNK]": 4,
    }
    merges = [
        "#version: 0.2",
        "H e",
        "l l",
        "l o",
    ]
    vocab_file = os.path.join(tmpdir, "vocab.json")
    merges_file = os.path.join(tmpdir, "merges.txt")
    with open(vocab_file, "w", encoding="utf-8") as vf:
        json.dump(vocab, vf)
    with open(merges_file, "w", encoding="utf-8") as mf:
        mf.write("\n".join(merges))
    return vocab_file, merges_file


def test_blueberry_tokenizer_basic_encode_decode():
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_file, merges_file = _write_fake_gpt2_files(tmpdir)
        tok = BlueberryTokenizer(vocab_file=vocab_file, merges_file=merges_file)
        text = "Hello World!"
        ids = tok.encode(text)
        assert isinstance(ids, list), f"Expected list of ids, got {type(ids)}"
        # ensure some non-zero / non-UNK id for known tokens
        assert any((i != tok.unk_token_id for i in ids)), "All ids are unknown"
        decoded = tok.decode(ids, skip_special_tokens=True)  # skip specials for easier matching
        assert isinstance(decoded, str), f"Expected decoded to be str, got {type(decoded)}"
        # normalized decoded should contain core words
        assert "Hello" in decoded, f"'Hello' not in decoded: {decoded}"
        assert "World" in decoded, f"'World' not in decoded: {decoded}"


def test_blueberry_chat_template_present_and_special_tokens_added():
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_file, merges_file = _write_fake_gpt2_files(tmpdir)
        tok = BlueberryTokenizer(vocab_file=vocab_file, merges_file=merges_file)
        # chat template
        assert hasattr(tok, "chat_template"), "Tokenizer missing chat_template attribute"
        assert tok.chat_template == HARMONY_CHAT_TEMPLATE, f"chat_template mismatch: {tok.chat_template}"
        # special tokens in vocabulary or special attributes
        # Using tokenizer properties to check
        for special in HARMONY_SPECIAL_TOKENS:
            # maybe special tokens are in tok.all_special_tokens, or in vocab, etc.
            # Check either
            assert (special in tok.get_vocab()) or (special in tok.all_special_tokens), (
                f"Special token {special} not found in vocab or special tokens"
            )

        # format simple chat
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "channel": "default", "content": "a"},
        ]
        rendered = tok.apply_chat_template(messages, tokenize=False)
        assert isinstance(rendered, str), f"Rendered template not str, got {type(rendered)}"
        # Expect assistant marker
        assert "<|assistant|>" in rendered, f"Assistant token missing in rendered: {rendered}"


def test_blueberry_fast_tokenizer_load_from_slow():
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_file, merges_file = _write_fake_gpt2_files(tmpdir)
        slow = BlueberryTokenizer(vocab_file=vocab_file, merges_file=merges_file)
        # Save slow tokenizer’s files in tmpdir so fast can load them from directory
        # If your slow tokenizer has `save_pretrained`, use that; else manually write necessary files
        slow.save_pretrained(tmpdir)

        fast = BlueberryTokenizerFast.from_pretrained(tmpdir, from_slow=True, __slow_tokenizer=slow)
        assert isinstance(fast, BlueberryTokenizerFast), f"Fast tokenizer not correct type: {type(fast)}"
        # same chat_template
        assert hasattr(fast, "chat_template"), "Fast tokenizer missing chat_template"
        assert fast.chat_template == HARMONY_CHAT_TEMPLATE, f"Fast chat_template mismatch: {fast.chat_template}"
        # test encode / decode similarly
        text = "Hello World!"
        ids = fast.encode(text)
        assert isinstance(ids, list), "Fast encode did not return list"
        decoded = fast.decode(ids, skip_special_tokens=True)
        assert isinstance(decoded, str), "Fast decode did not return str"
        assert "Hello" in decoded and "World" in decoded, f"Fast decoded content wrong: {decoded}"
