import json
import os
import tempfile

from transformers.models.blueberry.tokenization_blueberry import BlueberryTokenizer, HARMONY_CHAT_TEMPLATE, HARMONY_SPECIAL_TOKENS
from transformers.models.blueberry.tokenization_blueberry_fast import BlueberryTokenizerFast


def _write_fake_gpt2_files(tmpdir):
    # minimal vocab and merges
    vocab = {
        "<|endoftext|>": 0,
        " !": 1,
        "Hello": 2,
        "World": 3,
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
        assert isinstance(ids, list)
        decoded = tok.decode(ids)
        assert isinstance(decoded, str)


def test_blueberry_chat_template_present_and_special_tokens_added():
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_file, merges_file = _write_fake_gpt2_files(tmpdir)
        tok = BlueberryTokenizer(vocab_file=vocab_file, merges_file=merges_file)
        # chat template
        assert tok.chat_template == HARMONY_CHAT_TEMPLATE
        # special tokens
        for t in HARMONY_SPECIAL_TOKENS:
            assert t in tok.get_vocab()
        # format simple chat
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "channel": "default", "content": "a"},
        ]
        rendered = tok.apply_chat_template(messages, tokenize=False)
        assert "<|assistant|>" in rendered


def test_blueberry_fast_tokenizer_load_from_slow():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_file, merges_file = _write_fake_gpt2_files(tmpdir)
        slow = BlueberryTokenizer(vocab_file=vocab_file, merges_file=merges_file)
        fast = BlueberryTokenizerFast.from_pretrained(tmpdir, from_slow=True, __slow_tokenizer=slow)
        text = "Hello World!"
        ids = fast.encode(text)
        assert isinstance(ids, list)
        assert fast.chat_template == HARMONY_CHAT_TEMPLATE

