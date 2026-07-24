# Copyright 2026 Mistral AI and The HuggingFace Inc. team. All rights reserved.
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

"""Tests for Mistral tekken tokenizer detection, conversion, and save utilities."""

import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from huggingface_hub import hf_hub_download
from parameterized import parameterized

from tests.integrations.mistral.tekken_fixtures import (
    FAKE_TEKKEN_PATTERN,
    FAKE_TEKKEN_SPECIAL_TOKENS,
    FULL_BYTE_VOCAB,
    NUM_SPECIAL_TOKENS,
    write_fake_tekken_json,
)
from transformers import AutoTokenizer
from transformers.integrations.mistral import (
    MistralConverter,
    convert_tekken_tokenizer,
    resolve_mistral_format,
)
from transformers.integrations.mistral.tokenizer import _resolve_chat_template
from transformers.testing_utils import require_mistral_common, slow
from transformers.utils.import_utils import is_mistral_common_available


if is_mistral_common_available():
    from transformers.tokenization_mistral_common import MistralCommonBackend

# Diverse test strings used across all test classes.
_TEST_STRINGS = [
    "Hello, world!",
    "Bonjour le monde!",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "The quick brown fox jumps over the lazy dog.",
    "🎉 Unicode: café, naïve, résumé",
    "   Multiple   spaces   and\ttabs\nand\nnewlines",
    "12345 + 67890 = 80235",
    "!@#$%^&*()",
    "  leading and trailing  ",
    "MiXeD CaSe TeXt",
    "a",
]

# Real repos spanning different tekken versions, used by the slow parity tests.
_INTEGRATION_REPOS = [
    "mistralai/Ministral-3-3B-Instruct-2512",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "mistralai/Pixtral-12B-2409",
    "mistralai/Mistral-Small-4-119B-2603",
]


class TestResolveMistralFormat(unittest.TestCase):
    def test_false_returns_false_none(self):
        result = resolve_mistral_format("fake/path", mistral_format=False)
        self.assertEqual(result, (False, None))

    def test_none_without_tekken_file_returns_false(self):
        # Use a real local directory that has no tekken.json
        with tempfile.TemporaryDirectory() as tmp_dir:
            use, path = resolve_mistral_format(tmp_dir, mistral_format=None)
            self.assertFalse(use)

    @patch("transformers.integrations.mistral.tokenizer.is_mistral_common_available", return_value=True)
    def test_true_without_tekken_file_raises_helpful_error(self, _mock):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(OSError) as ctx:
                resolve_mistral_format(tmp_dir, mistral_format=True)
            self.assertIn("mistral_format=False", str(ctx.exception))

    @patch("transformers.integrations.mistral.tokenizer.is_mistral_common_available", return_value=False)
    def test_true_without_mistral_common_raises(self, _mock):
        with self.assertRaises(ImportError):
            resolve_mistral_format("fake/path", mistral_format=True)

    def test_none_tolerates_forced_cached_file_kwargs(self):
        """Callers (e.g. AutoProcessor) may pass _raise_exceptions_for_* kwargs that
        resolve_mistral_format forces internally; no TypeError should be raised."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # No tekken.json present → should return (False, None) without TypeError.
            result = resolve_mistral_format(
                tmp_dir,
                None,
                _raise_exceptions_for_missing_entries=True,
                _raise_exceptions_for_connection_errors=True,
                _raise_exceptions_for_gated_repo=True,
            )
            self.assertEqual(result, (False, None))

    @require_mistral_common
    def test_auto_native_even_with_hf_files(self):
        """Auto mode returns (True, path) when tekken.json AND an HF marker coexist (tekken-first)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            write_fake_tekken_json(tmp_path)
            (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")

            use_mistral, tekken_path = resolve_mistral_format(tmp_dir, None)
            self.assertTrue(use_mistral)
            self.assertIsNotNone(tekken_path)
            self.assertTrue(tekken_path.endswith("tekken.json"))

    @require_mistral_common
    def test_auto_goes_native_when_hf_absent(self):
        """Auto mode returns (True, path) when only tekken.json is present (+ params.json OK)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            write_fake_tekken_json(tmp_path)
            # params.json must NOT suppress native detection
            (tmp_path / "params.json").write_text("{}", encoding="utf-8")

            use_mistral, tekken_path = resolve_mistral_format(tmp_dir, None)
            self.assertTrue(use_mistral)
            self.assertIsNotNone(tekken_path)
            self.assertTrue(tekken_path.endswith("tekken.json"))

    @require_mistral_common
    def test_explicit_true_ignores_hf_markers(self):
        """mistral_format=True forces native even when config.json is present."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            write_fake_tekken_json(tmp_path)
            (tmp_path / "config.json").write_text("{}", encoding="utf-8")

            use_mistral, tekken_path = resolve_mistral_format(tmp_dir, True)
            self.assertTrue(use_mistral)
            self.assertIsNotNone(tekken_path)
            self.assertTrue(tekken_path.endswith("tekken.json"))

    def test_explicit_false_ignores_tekken(self):
        """mistral_format=False always returns (False, None) regardless of tekken.json."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            write_fake_tekken_json(tmp_path)

            result = resolve_mistral_format(tmp_dir, False)
            self.assertEqual(result, (False, None))

    @require_mistral_common
    def test_preprocessor_config_alone_does_not_suppress_native(self):
        """preprocessor_config.json must NOT suppress native detection in auto mode."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            write_fake_tekken_json(tmp_path)
            (tmp_path / "preprocessor_config.json").write_text("{}", encoding="utf-8")

            use_mistral, tekken_path = resolve_mistral_format(tmp_dir, None)
            self.assertTrue(use_mistral)
            self.assertIsNotNone(tekken_path)


class TestMistralConverter(unittest.TestCase):
    """Unit tests for MistralConverter using a synthetic tekken.json."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp_dir = tempfile.TemporaryDirectory()
        cls._tekken_path = write_fake_tekken_json(Path(cls._tmp_dir.name))
        cls._converter = MistralConverter(str(cls._tekken_path))
        cls._tokenizer = cls._converter.converted()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp_dir.cleanup()

    def test_init_sets_precomputed_fields(self):
        self.assertIsNotNone(self._converter._precomputed_vocab)
        self.assertIsNotNone(self._converter._precomputed_merges)

    def test_converted_produces_working_tokenizer(self):
        ids = self._tokenizer.encode("a b c").ids
        self.assertIsInstance(ids, list)
        self.assertGreater(len(ids), 0)

    def test_roundtrip_encode_decode(self):
        for text in ["hello world", "abc 123", "test"]:
            encoded = self._tokenizer.encode(text)
            decoded = self._tokenizer.decode(encoded.ids)
            self.assertEqual(decoded, text, f"Roundtrip failed for {text!r}")

    def test_special_tokens_in_vocab(self):
        vocab = self._tokenizer.get_vocab()
        for entry in FAKE_TEKKEN_SPECIAL_TOKENS:
            self.assertIn(entry["token_str"], vocab, f"Special token {entry['token_str']!r} missing")

    def test_vocab_size(self):
        self.assertEqual(self._tokenizer.get_vocab_size(), FULL_BYTE_VOCAB)

    def test_special_tokens_assigned_by_rank_not_list_order(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shuffled_specials = list(reversed(FAKE_TEKKEN_SPECIAL_TOKENS))
            num_bpe = FULL_BYTE_VOCAB - NUM_SPECIAL_TOKENS
            vocab_list = [
                {
                    "rank": rank,
                    "token_bytes": base64.b64encode(bytes([rank % 256])).decode("ascii"),
                    "token_str": None,
                }
                for rank in range(num_bpe)
            ]
            tekken_data = {
                "vocab": vocab_list,
                "special_tokens": shuffled_specials,
                "config": {"pattern": FAKE_TEKKEN_PATTERN},
                "version": 1,
                "type": "tekken",
            }
            tekken_path = tmp_path / "tekken.json"
            with open(tekken_path, "w", encoding="utf-8") as f:
                json.dump(tekken_data, f, ensure_ascii=False)

            converter = MistralConverter(str(tekken_path))

            for entry in FAKE_TEKKEN_SPECIAL_TOKENS:
                self.assertEqual(
                    converter._precomputed_vocab[entry["token_str"]],
                    entry["rank"],
                    f"Special token {entry['token_str']!r} got wrong id",
                )


class TestConvertTekkenTokenizer(unittest.TestCase):
    def test_basic_conversion(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tekken_path = write_fake_tekken_json(tmp_path)

            tokenizer = convert_tekken_tokenizer(str(tekken_path))
            self.assertIsNotNone(tokenizer)
            self.assertEqual(tokenizer.vocab_size, FULL_BYTE_VOCAB)

    def test_special_tokens_set(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tekken_path = write_fake_tekken_json(tmp_path)

            tokenizer = convert_tekken_tokenizer(str(tekken_path))
            self.assertEqual(tokenizer.bos_token, "<s>")
            self.assertEqual(tokenizer.eos_token, "</s>")
            self.assertEqual(tokenizer.pad_token, "<pad>")
            self.assertEqual(tokenizer.unk_token, "<unk>")

    def test_explicit_template_attached(self):
        """Explicit chat_template arg is attached to the returned tokenizer unchanged."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tekken_path = write_fake_tekken_json(tmp_path)

            tokenizer = convert_tekken_tokenizer(str(tekken_path), chat_template="EXPLICIT")

            self.assertEqual(tokenizer.chat_template, "EXPLICIT")

    def test_sibling_jinja_used(self):
        """Sibling chat_template.jinja is attached when no explicit arg given."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tekken_path = write_fake_tekken_json(tmp_path)
            (tmp_path / "chat_template.jinja").write_text("JINJA", encoding="utf-8")

            tokenizer = convert_tekken_tokenizer(str(tekken_path))

            self.assertEqual(tokenizer.chat_template, "JINJA")

    @patch("transformers.integrations.mistral.tokenizer.is_mistral_common_available", return_value=False)
    def test_none_when_mistral_common_off(self, _mock):
        """No siblings, no explicit arg, mistral-common unavailable → chat_template is None."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tekken_path = write_fake_tekken_json(tmp_path)

            tokenizer = convert_tekken_tokenizer(str(tekken_path))

            self.assertIsNone(tokenizer.chat_template)

    @patch("transformers.integrations.mistral.tokenizer.is_mistral_common_available", return_value=False)
    def test_core_unchanged(self, _mock):
        """Core behavior (special tokens, tokenization) unaffected by new param."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tekken_path = write_fake_tekken_json(tmp_path)

            tokenizer = convert_tekken_tokenizer(str(tekken_path), chat_template="T")

            # Special tokens
            self.assertEqual(tokenizer.bos_token, "<s>")
            self.assertEqual(tokenizer.eos_token, "</s>")
            self.assertEqual(tokenizer.pad_token, "<pad>")
            self.assertEqual(tokenizer.unk_token, "<unk>")

            # Tokenization still works
            ids = tokenizer.encode("hello world", add_special_tokens=False)
            self.assertIsInstance(ids, list)
            self.assertGreater(len(ids), 0)
            self.assertEqual(tokenizer.decode(ids), "hello world")


class TestSaveMistralFormat(unittest.TestCase):
    """Tests for the copy-based save_pretrained(save_format='mistral') behavior."""

    def test_in_session_copy_is_byte_identical(self):
        """Saving immediately after conversion copies tekken.json byte-for-byte."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_dir = tmp_path / "src"
            src_dir.mkdir()
            tekken_path = write_fake_tekken_json(src_dir)
            out_dir = str(tmp_path / "out")

            tok = convert_tekken_tokenizer(str(tekken_path))
            tok.save_pretrained(out_dir, save_format="mistral")

            saved = Path(out_dir) / "tekken.json"
            self.assertTrue(saved.exists())
            with open(tekken_path, encoding="utf-8") as f:
                original = json.load(f)
            with open(saved, encoding="utf-8") as f:
                copied = json.load(f)
            self.assertEqual(original, copied)

    def test_missing_source_raises_clear_error(self):
        """save_pretrained(save_format='mistral') raises OSError when source tekken.json is gone."""
        with tempfile.TemporaryDirectory() as src_dir:
            src_path = Path(src_dir)
            tekken_path = write_fake_tekken_json(src_path)
            tok = convert_tekken_tokenizer(str(tekken_path))
            # Delete the source file so the path is no longer valid.
            tekken_path.unlink()

        with tempfile.TemporaryDirectory() as out_dir:
            with self.assertRaises(OSError) as ctx:
                tok.save_pretrained(out_dir, save_format="mistral")
            self.assertIn("tekken.json", str(ctx.exception))


@require_mistral_common
class TestSaveHFFormat(unittest.TestCase):
    """Tests for MistralCommonBackend.save_pretrained(save_format="hf") native->HF conversion."""

    def test_save_format_hf_produces_loadable_hf_tokenizer(self):
        """Saving in HF format writes tokenizer.json + tokenizer_config.json, and the reloaded
        tokenizer encodes identically to a direct convert_tekken_tokenizer call."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_dir = tmp_path / "src"
            src_dir.mkdir()
            tekken_path = write_fake_tekken_json(src_dir)
            out_dir = str(tmp_path / "out")

            backend = MistralCommonBackend(tokenizer_path=str(tekken_path))
            backend.save_pretrained(out_dir, save_format="hf")

            out_path = Path(out_dir)
            self.assertTrue((out_path / "tokenizer.json").exists())
            self.assertTrue((out_path / "tokenizer_config.json").exists())

            text = "hello world"
            reloaded = AutoTokenizer.from_pretrained(out_dir, mistral_format=False)
            expected = convert_tekken_tokenizer(str(tekken_path)).encode(text, add_special_tokens=False)
            actual = reloaded.encode(text, add_special_tokens=False)
            self.assertEqual(actual, expected)

    def test_save_format_hf_missing_source_raises(self):
        """save_pretrained(save_format='hf') raises OSError when the source tekken.json is gone."""
        with tempfile.TemporaryDirectory() as src_dir:
            src_path = Path(src_dir)
            tekken_path = write_fake_tekken_json(src_path)
            backend = MistralCommonBackend(tokenizer_path=str(tekken_path))
            # Delete the source file so the path is no longer valid.
            tekken_path.unlink()

        with tempfile.TemporaryDirectory() as out_dir:
            with self.assertRaises(OSError):
                backend.save_pretrained(out_dir, save_format="hf")


@require_mistral_common
class TestMistralConverterVsCommonBackend(unittest.TestCase):
    """Compare MistralConverter raw encoding/decoding with MistralCommonBackend on a synthetic tekken.json.

    MistralConverter.converted() does NOT add BOS/EOS — that is the wrapper's job.
    All comparisons use add_special_tokens=False on MistralCommonBackend.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp_dir = tempfile.TemporaryDirectory()
        tekken_path = write_fake_tekken_json(Path(cls._tmp_dir.name))

        converter = MistralConverter(str(tekken_path))
        cls.hf_tokenizer = converter.converted()
        cls.mc_tokenizer = MistralCommonBackend(tokenizer_path=str(tekken_path))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp_dir.cleanup()

    def test_encode_matches(self) -> None:
        for text in _TEST_STRINGS:
            hf_ids = self.hf_tokenizer.encode(text).ids
            mc_ids = self.mc_tokenizer.encode(text, add_special_tokens=False)
            self.assertEqual(hf_ids, mc_ids, f"Encoding mismatch for {text!r}")

    def test_decode_matches(self) -> None:
        for text in _TEST_STRINGS:
            ids = self.mc_tokenizer.encode(text, add_special_tokens=False)
            hf_decoded = self.hf_tokenizer.decode(ids)
            mc_decoded = self.mc_tokenizer.decode(ids, skip_special_tokens=True)
            self.assertEqual(hf_decoded, mc_decoded, f"Decode mismatch for {text!r}")

    def test_vocab_size(self) -> None:
        self.assertEqual(self.hf_tokenizer.get_vocab_size(), self.mc_tokenizer.vocab_size)


@require_mistral_common
@slow
class TestMistralConverterIntegration(unittest.TestCase):
    """Integration tests with real tekken.json files spanning multiple tekken versions.

    Each parity check runs over the repos in `_INTEGRATION_REPOS`. MistralConverter.converted()
    returns a raw tokenizers.Tokenizer without BOS/EOS injection. All encoding comparisons use
    add_special_tokens=False on MistralCommonBackend to compare at the same abstraction level.
    """

    _tokenizers: dict = {}

    @classmethod
    def setUpClass(cls) -> None:
        cls._tokenizers = {}

    @classmethod
    def _get_tokenizers(cls, repo: str):
        """Download and build (hf_tokenizer, mc_tokenizer) for a repo, caching per repo."""
        if repo not in cls._tokenizers:
            tekken_path = hf_hub_download(repo, "tekken.json")
            converter = MistralConverter(tekken_path)
            cls._tokenizers[repo] = (
                converter.converted(),
                MistralCommonBackend(tokenizer_path=tekken_path),
            )
        return cls._tokenizers[repo]

    # ── Vocabulary ──────────────────────────────────────────────────────

    @parameterized.expand(_INTEGRATION_REPOS)
    def test_vocab_size(self, repo: str) -> None:
        hf_tokenizer, mc_tokenizer = self._get_tokenizers(repo)
        self.assertEqual(hf_tokenizer.get_vocab_size(), mc_tokenizer.vocab_size)

    @parameterized.expand(_INTEGRATION_REPOS)
    def test_full_vocab_decode_single_token_matches(self, repo: str) -> None:
        """Decoding every single token ID (skip_special_tokens=True) produces the same string."""
        hf_tokenizer, mc_tokenizer = self._get_tokenizers(repo)
        mismatches = []
        for token_id in range(mc_tokenizer.vocab_size):
            hf_decoded = hf_tokenizer.decode([token_id], skip_special_tokens=True)
            mc_decoded = mc_tokenizer.decode([token_id], skip_special_tokens=True)
            if hf_decoded != mc_decoded:
                mismatches.append((token_id, hf_decoded, mc_decoded))
        self.assertEqual(mismatches, [], f"Found {len(mismatches)} decode mismatches (first 10): {mismatches[:10]}")

    @parameterized.expand(_INTEGRATION_REPOS)
    def test_special_tokens_ids(self, repo: str) -> None:
        hf_tokenizer, mc_tokenizer = self._get_tokenizers(repo)
        for token_str, attr in {"<s>": "bos", "</s>": "eos", "<unk>": "unk", "<pad>": "pad"}.items():
            hf_id = hf_tokenizer.token_to_id(token_str)
            mc_id = getattr(mc_tokenizer, f"{attr}_token_id")
            self.assertIsNotNone(hf_id, f"HF tokenizer missing {token_str}")
            self.assertIsNotNone(mc_id, f"MC tokenizer missing {attr}_token_id")
            self.assertEqual(hf_id, mc_id, f"{token_str} ID mismatch: HF={hf_id} MC={mc_id}")

    # ── Encode ──────────────────────────────────────────────────────────

    @parameterized.expand(_INTEGRATION_REPOS)
    def test_encode(self, repo: str) -> None:
        hf_tokenizer, mc_tokenizer = self._get_tokenizers(repo)
        for text in _TEST_STRINGS:
            hf_ids = hf_tokenizer.encode(text).ids
            mc_ids = mc_tokenizer.encode(text, add_special_tokens=False)
            self.assertEqual(hf_ids, mc_ids, f"Encoding mismatch for {text!r}")

    @parameterized.expand(_INTEGRATION_REPOS)
    def test_encode_long_text(self, repo: str) -> None:
        hf_tokenizer, mc_tokenizer = self._get_tokenizers(repo)
        long_text = "The quick brown fox jumps over the lazy dog. " * 100
        hf_ids = hf_tokenizer.encode(long_text).ids
        mc_ids = mc_tokenizer.encode(long_text, add_special_tokens=False)
        self.assertEqual(hf_ids, mc_ids)
        self.assertGreater(len(hf_ids), 100, "Long text should produce many tokens")

    @parameterized.expand(_INTEGRATION_REPOS)
    def test_encode_multilingual(self, repo: str) -> None:
        hf_tokenizer, mc_tokenizer = self._get_tokenizers(repo)
        texts = [
            "日本語のテスト",  # Japanese
            "Привет мир",  # Russian
            "مرحبا بالعالم",  # Arabic
            "你好世界",  # Chinese
            "한국어 테스트",  # Korean
            "Ñoño español",  # Spanish with diacritics
            "Ελληνικά",  # Greek
        ]
        for text in texts:
            hf_ids = hf_tokenizer.encode(text).ids
            mc_ids = mc_tokenizer.encode(text, add_special_tokens=False)
            self.assertEqual(hf_ids, mc_ids, f"Multilingual encoding mismatch for {text!r}")

    @parameterized.expand(_INTEGRATION_REPOS)
    def test_encode_code_snippets(self, repo: str) -> None:
        hf_tokenizer, mc_tokenizer = self._get_tokenizers(repo)
        snippets = [
            "import torch\nmodel = torch.nn.Linear(10, 20)",
            "for i in range(100):\n    print(f'{i=}')",
            "class Foo:\n    def __init__(self):\n        self.x = 42",
            "// C++ comment\nint main() { return 0; }",
            "SELECT * FROM users WHERE id = 1;",
            '{"key": "value", "nested": {"a": [1, 2, 3]}}',
        ]
        for text in snippets:
            hf_ids = hf_tokenizer.encode(text).ids
            mc_ids = mc_tokenizer.encode(text, add_special_tokens=False)
            self.assertEqual(hf_ids, mc_ids, f"Code encoding mismatch for {text!r}")

    # ── Decode ──────────────────────────────────────────────────────────

    @parameterized.expand(_INTEGRATION_REPOS)
    def test_decode(self, repo: str) -> None:
        """Decode token IDs (no special tokens) — both backends produce the same string."""
        hf_tokenizer, mc_tokenizer = self._get_tokenizers(repo)
        for text in _TEST_STRINGS:
            ids = mc_tokenizer.encode(text, add_special_tokens=False)
            hf_decoded = hf_tokenizer.decode(ids)
            mc_decoded = mc_tokenizer.decode(ids, skip_special_tokens=True)
            self.assertEqual(hf_decoded, mc_decoded, f"Decode mismatch for {text!r}")

    @parameterized.expand(_INTEGRATION_REPOS)
    def test_decode_with_special_token_ids(self, repo: str) -> None:
        """Decode sequences that contain BOS/EOS IDs — skip_special_tokens strips them equally."""
        hf_tokenizer, mc_tokenizer = self._get_tokenizers(repo)
        bos_id = hf_tokenizer.token_to_id("<s>")
        eos_id = hf_tokenizer.token_to_id("</s>")
        for text in _TEST_STRINGS:
            ids = mc_tokenizer.encode(text, add_special_tokens=False)
            ids_with_special = [bos_id] + ids + [eos_id]

            hf_decoded = hf_tokenizer.decode(ids_with_special, skip_special_tokens=True)
            mc_decoded = mc_tokenizer.decode(ids_with_special, skip_special_tokens=True)
            self.assertEqual(hf_decoded, mc_decoded, f"Decode skip BOS+EOS mismatch for {text!r}")

    @parameterized.expand(_INTEGRATION_REPOS)
    def test_encode_decode_roundtrip(self, repo: str) -> None:
        """Encode then decode should recover the original text in both backends."""
        hf_tokenizer, mc_tokenizer = self._get_tokenizers(repo)
        for text in _TEST_STRINGS:
            if not text:
                continue
            hf_ids = hf_tokenizer.encode(text).ids
            hf_roundtrip = hf_tokenizer.decode(hf_ids)
            mc_roundtrip = mc_tokenizer.decode(hf_ids, skip_special_tokens=True)
            self.assertEqual(hf_roundtrip, text, f"HF roundtrip failed for {text!r}")
            self.assertEqual(mc_roundtrip, text, f"MC roundtrip failed for {text!r}")

    # ── Token-level ─────────────────────────────────────────────────────

    @parameterized.expand(_INTEGRATION_REPOS)
    def test_per_token_decode_matches(self, repo: str) -> None:
        """Decoding each token individually should produce the same string in both backends."""
        hf_tokenizer, mc_tokenizer = self._get_tokenizers(repo)
        for text in _TEST_STRINGS:
            ids = mc_tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue
            for token_id in ids:
                hf_decoded = hf_tokenizer.decode([token_id], skip_special_tokens=True)
                mc_decoded = mc_tokenizer.decode([token_id], skip_special_tokens=True)
                self.assertEqual(hf_decoded, mc_decoded, f"Per-token decode mismatch for id={token_id} in {text!r}")


class TestResolveChatTemplate(unittest.TestCase):
    """Unit tests for _resolve_chat_template precedence helper."""

    def test_explicit_arg_wins_over_jinja_sibling(self):
        """Explicit chat_template arg takes precedence over any sibling file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tekken_path = write_fake_tekken_json(tmp_path)
            (tmp_path / "chat_template.jinja").write_text("JINJA", encoding="utf-8")

            result = _resolve_chat_template(tekken_path, "EXPLICIT")

            self.assertEqual(result, "EXPLICIT")

    def test_empty_string_explicit_arg_returned_as_is(self):
        """Empty string explicit arg is returned as-is, even if a sibling .jinja exists."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tekken_path = write_fake_tekken_json(tmp_path)
            (tmp_path / "chat_template.jinja").write_text("JINJA", encoding="utf-8")

            result = _resolve_chat_template(tekken_path, "")

            self.assertEqual(result, "")

    def test_jinja_sibling_returned_when_no_arg(self):
        """chat_template.jinja sibling is returned when no explicit arg given."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tekken_path = write_fake_tekken_json(tmp_path)
            (tmp_path / "chat_template.jinja").write_text("JINJA", encoding="utf-8")

            result = _resolve_chat_template(tekken_path, None)

            self.assertEqual(result, "JINJA")

    def test_json_sibling_returned_when_no_jinja(self):
        """chat_template.json sibling is used when no .jinja sibling and no explicit arg."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tekken_path = write_fake_tekken_json(tmp_path)
            (tmp_path / "chat_template.json").write_text(json.dumps({"chat_template": "JSON"}), encoding="utf-8")

            result = _resolve_chat_template(tekken_path, None)

            self.assertEqual(result, "JSON")

    def test_missing_key_in_chat_template_json_raises_key_error(self):
        """chat_template.json without 'chat_template' key raises KeyError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tekken_path = write_fake_tekken_json(tmp_path)
            (tmp_path / "chat_template.json").write_text("{}", encoding="utf-8")

            with self.assertRaises(KeyError):
                _resolve_chat_template(tekken_path, None)

    def test_jinja_beats_json_when_both_present(self):
        """chat_template.jinja takes precedence over chat_template.json."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tekken_path = write_fake_tekken_json(tmp_path)
            (tmp_path / "chat_template.jinja").write_text("JINJA", encoding="utf-8")
            (tmp_path / "chat_template.json").write_text(json.dumps({"chat_template": "JSON"}), encoding="utf-8")

            result = _resolve_chat_template(tekken_path, None)

            self.assertEqual(result, "JINJA")

    @require_mistral_common
    @patch("transformers.integrations.mistral.tokenizer.is_mistral_common_available", return_value=True)
    def test_generate_called_when_no_siblings(self, _mock_avail):
        """When no sibling files, generator is called and its return value is used."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tekken_path = write_fake_tekken_json(tmp_path)

            with patch(
                "mistral_common.integrations.chat_templates.chat_templates.convert_tokenizer_to_chat_template",
                return_value="GEN",
            ) as mock_gen:
                result = _resolve_chat_template(tekken_path, None)

            self.assertEqual(result, "GEN")
            mock_gen.assert_called_once_with(tekken_path)

    @require_mistral_common
    @patch("transformers.integrations.mistral.tokenizer.is_mistral_common_available", return_value=False)
    def test_returns_none_when_mistral_common_unavailable(self, _mock_avail):
        """Returns None without calling generator when mistral-common is not available.

        @require_mistral_common is present only so the patched import target resolves,
        even though availability is patched to False inside the test.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tekken_path = write_fake_tekken_json(tmp_path)

            with patch(
                "mistral_common.integrations.chat_templates.chat_templates.convert_tokenizer_to_chat_template",
            ) as mock_gen:
                result = _resolve_chat_template(tekken_path, None)

            self.assertIsNone(result)
            mock_gen.assert_not_called()

    @require_mistral_common
    @patch("transformers.integrations.mistral.tokenizer.is_mistral_common_available", return_value=True)
    def test_generation_failure_returns_none_with_warning(self, _mock_avail):
        """When generator raises Exception, returns None and logs a warning."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tekken_path = write_fake_tekken_json(tmp_path)

            with patch(
                "mistral_common.integrations.chat_templates.chat_templates.convert_tokenizer_to_chat_template",
                side_effect=RuntimeError("generation error"),
            ):
                with patch("transformers.integrations.mistral.tokenizer.logger") as mock_logger:
                    result = _resolve_chat_template(tekken_path, None)

            self.assertIsNone(result)
            mock_logger.warning_once.assert_called_once()
            call_args = mock_logger.warning_once.call_args[0]
            warning_text = " ".join(str(a) for a in call_args)
            self.assertIn(str(tekken_path), warning_text)


if __name__ == "__main__":
    unittest.main()
