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
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from huggingface_hub import hf_hub_download
from parameterized import parameterized
from tokenizers import Tokenizer

from tests.integrations.mistral.tekken_fixtures import (
    FAKE_TEKKEN_PATTERN,
    FAKE_TEKKEN_SPECIAL_TOKENS,
    FULL_BYTE_VOCAB,
    NUM_SPECIAL_TOKENS,
    build_fake_tekken_dict,
    write_fake_tekken_json,
)
from transformers import AutoTokenizer, TokenizersBackend
from transformers.convert_slow_tokenizer import convert_slow_tokenizer
from transformers.integrations.mistral import (
    MistralConverter,
    convert_tekken_tokenizer,
    resolve_mistral_format,
)
from transformers.integrations.mistral.constants import is_tekken_vocab_filename
from transformers.integrations.mistral.tokenizer import (
    _check_tekken_vocab_unchanged,
    _derive_tekken_specials,
    _resolve_chat_template,
)
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.testing_utils import require_mistral_common, slow
from transformers.utils.import_utils import BACKENDS_MAPPING, is_mistral_common_available


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


@contextmanager
def _converted_tokenizer(**write_kwargs: object) -> Iterator[tuple[TokenizersBackend, Path]]:
    """Yield `(tokenizer, tekken_path)` for a tokenizer freshly converted from a temp tekken.json.

    Bundles the `TemporaryDirectory` + `write_fake_tekken_json` + `convert_tekken_tokenizer`
    triple that recurs across the divergence-guard tests. `**write_kwargs` are forwarded to
    `write_fake_tekken_json`.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tekken_path = write_fake_tekken_json(Path(tmp_dir), **write_kwargs)
        yield convert_tekken_tokenizer(str(tekken_path)), tekken_path


class TestIsTekkenVocabFilename(unittest.TestCase):
    """Unit tests for `is_tekken_vocab_filename`, the shared tekken classification predicate."""

    @parameterized.expand(
        [
            ("canonical", "tekken.json", True),
            ("versioned", "tekken_240718.json", True),
            ("prefixed", "my_tekken.json", True),
            ("canonical_full_path", "/some/dir/tekken.json", True),
            ("versioned_full_path", "/some/dir/tekken_240911.json", True),
            ("prefixed_full_path", "/some/dir/my_tekken.json", True),
            ("no_tekken_substring", "tokenizer.json", False),
            ("wrong_suffix", "tekken.txt", False),
            ("no_suffix", "tekken", False),
        ]
    )
    def test_classification(self, _name, path, expected):
        self.assertEqual(is_tekken_vocab_filename(path), expected)


class _FakeSlowTokenizer:
    """Minimal stand-in for a slow tokenizer instance, exposing only what
    `convert_slow_tokenizer`'s Mistral branch reads: `vocab_file` and a class name absent
    from `SLOW_TO_FAST_CONVERTERS`."""

    def __init__(self, vocab_file: str) -> None:
        self.vocab_file = vocab_file
        self.extra_special_tokens = {}


class TestIsTekkenVocabFilenameCallSites(unittest.TestCase):
    """Tests that genuinely drive each call site of `is_tekken_vocab_filename`, rather than
    only the pure predicate. Both are reachable only via non-default entry points:
    `convert_to_native_format` requires an explicit `vocab_file=` kwarg (the class's
    `VOCAB_FILES_NAMES` maps `vocab_file` to `tokenizer.model`, not a tekken name), and
    `convert_slow_tokenizer`'s Mistral branch requires a tokenizer class absent from
    `SLOW_TO_FAST_CONVERTERS`.
    """

    def test_convert_to_native_format_loads_versioned_tekken_filename(self):
        """`TokenizersBackend.convert_to_native_format(vocab_file=...)` classifies a
        versioned tekken filename via `is_tekken_vocab_filename`, not `.endswith("tekken.json")`."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tekken_path = write_fake_tekken_json(Path(tmp_dir), filename="tekken_240911.json")

            local_kwargs = TokenizersBackend.convert_to_native_format(vocab_file=str(tekken_path))

            self.assertIn("tokenizer_object", local_kwargs)

    def test_convert_slow_tokenizer_mistral_branch_loads_versioned_tekken_filename(self):
        """`convert_slow_tokenizer`'s Mistral branch classifies a versioned tekken filename
        via `is_tekken_vocab_filename`, not `.endswith("tekken.json")`."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tekken_path = write_fake_tekken_json(Path(tmp_dir), filename="tekken_240911.json")

            converted = convert_slow_tokenizer(_FakeSlowTokenizer(str(tekken_path)))

            self.assertIsInstance(converted, Tokenizer)

    def test_convert_slow_tokenizer_none_vocab_file_does_not_crash_on_classification(self):
        """A slow tokenizer with `vocab_file=None` must not reach `is_tekken_vocab_filename`
        or `MistralConverter`'s raw `open()`: it should fall through to the Tiktoken path
        and fail there with a clear `ValueError`, not a `TypeError` from
        `os.path.basename(None)`."""
        with self.assertRaises(ValueError):
            convert_slow_tokenizer(_FakeSlowTokenizer(None))


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
        cls._converter = MistralConverter(vocab_file=str(cls._tekken_path))
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

            converter = MistralConverter(vocab_file=str(tekken_path))

            for entry in FAKE_TEKKEN_SPECIAL_TOKENS:
                self.assertEqual(
                    converter._precomputed_vocab[entry["token_str"]],
                    entry["rank"],
                    f"Special token {entry['token_str']!r} got wrong id",
                )

    def test_non_contiguous_special_token_ranks_raises_value_error(self):
        """special_tokens ranks that skip a value (not contiguous from 0) must raise, not
        silently produce a tokenizer with wrong ids."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            non_contiguous_specials = [dict(entry) for entry in FAKE_TEKKEN_SPECIAL_TOKENS]
            non_contiguous_specials[-1]["rank"] = len(non_contiguous_specials)  # skips one rank
            tekken_path = write_fake_tekken_json(tmp_path, special_tokens=non_contiguous_specials)

            with self.assertRaises(ValueError) as ctx:
                MistralConverter(vocab_file=str(tekken_path))
            self.assertIn("contiguous", str(ctx.exception))


class TestConvertTekkenTokenizer(unittest.TestCase):
    def test_basic_conversion(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tekken_path = write_fake_tekken_json(tmp_path)

            tokenizer = convert_tekken_tokenizer(str(tekken_path))
            self.assertIs(type(tokenizer), TokenizersBackend)
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
    """Tests for `save_pretrained` on a tekken-derived `TokenizersBackend`, across all `save_format` values."""

    def test_save_pretrained_mistral_format_copy_is_byte_identical(self):
        """Saving immediately after conversion copies tekken.json byte-for-byte."""
        with _converted_tokenizer() as (tok, tekken_path):
            out_dir = tekken_path.parent / "out"

            tok.save_pretrained(str(out_dir), save_format="mistral")

            saved = out_dir / "tekken.json"
            self.assertTrue(saved.exists())
            self.assertEqual(saved.read_bytes(), tekken_path.read_bytes())

    def test_save_pretrained_mistral_format_in_place_resave_succeeds(self):
        """Resaving into the same directory a tokenizer was converted from is idempotent:
        no SameFileError, and the source tekken.json is left byte-for-byte unchanged."""
        with _converted_tokenizer() as (tok, tekken_path):
            original_bytes = tekken_path.read_bytes()

            result = tok.save_pretrained(str(tekken_path.parent), save_format="mistral")

            self.assertEqual(result, (str(tekken_path),))
            self.assertEqual(tekken_path.read_bytes(), original_bytes)

    def test_save_pretrained_mistral_format_in_place_resave_honors_guard(self):
        """An in-place resave of a mutated tokenizer is still rejected by the divergence
        guard, and the source tekken.json is left untouched."""
        with _converted_tokenizer() as (tok, tekken_path):
            original_bytes = tekken_path.read_bytes()
            tok.add_tokens(["<x>"])

            with self.assertRaises(ValueError):
                tok.save_pretrained(str(tekken_path.parent), save_format="mistral")

            self.assertEqual(tekken_path.read_bytes(), original_bytes)

    def test_save_pretrained_mistral_format_missing_source_raises_clear_error(self):
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

    @parameterized.expand(
        [
            ("omitted_kwarg", None, False),
            ("explicit_none", None, True),
            ("explicit_hf", "hf", True),
        ]
    )
    def test_save_pretrained_hf_or_default_format_produces_loadable_hf_tokenizer(self, _name, save_format, pass_kwarg):
        """Saving in HF format (kwarg omitted, explicit `None`, or explicit `"hf"`) writes
        tokenizer.json + tokenizer_config.json, and the reloaded tokenizer encodes/decodes
        identically to the original tokenizer."""
        with _converted_tokenizer() as (tok, tekken_path):
            out_dir = str(tekken_path.parent / "out")
            if pass_kwarg:
                tok.save_pretrained(out_dir, save_format=save_format)
            else:
                tok.save_pretrained(out_dir)

            out_path = Path(out_dir)
            self.assertTrue((out_path / "tokenizer.json").exists())
            self.assertTrue((out_path / "tokenizer_config.json").exists())

            reloaded = AutoTokenizer.from_pretrained(out_dir, mistral_format=False)
            self.assertIsInstance(reloaded, TokenizersBackend)
            text = "hello world"
            expected_ids = tok.encode(text, add_special_tokens=False)
            actual_ids = reloaded.encode(text, add_special_tokens=False)
            self.assertEqual(actual_ids, expected_ids)
            self.assertEqual(reloaded.decode(actual_ids), tok.decode(expected_ids))

    def test_save_pretrained_unknown_format_raises_value_error(self):
        """save_pretrained rejects any save_format outside 'hf'/'mistral'/None."""
        with _converted_tokenizer() as (tok, tekken_path):
            out_dir = str(tekken_path.parent / "out")
            with self.assertRaises(ValueError) as ctx:
                tok.save_pretrained(out_dir, save_format="bogus")
            self.assertIn("Unknown save_format", str(ctx.exception))

    @staticmethod
    def _build_tokenizer_without_vocab_file(tekken_path: Path) -> TokenizersBackend:
        """Build a `TokenizersBackend` locally with no `vocab_file`, so it has no tekken.json source."""
        converter = MistralConverter(vocab_file=str(tekken_path))
        return TokenizersBackend(
            tokenizer_object=converter.converted(),
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            unk_token="<unk>",
        )

    def test_save_pretrained_mistral_format_rejected_for_ordinary_tokenizer(self):
        """save_pretrained(save_format='mistral') raises OSError for a tokenizer with no tekken.json source."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tekken_path = write_fake_tekken_json(Path(tmp_dir))
            tok = self._build_tokenizer_without_vocab_file(tekken_path)

            with tempfile.TemporaryDirectory() as out_dir:
                with self.assertRaises(OSError) as ctx:
                    tok.save_pretrained(out_dir, save_format="mistral")
                self.assertIn("tekken.json", str(ctx.exception))

    def test_save_pretrained_mistral_format_failure_leaves_no_directory(self):
        """A failed mistral save must not create the output directory as a side effect."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tekken_path = write_fake_tekken_json(Path(tmp_dir))
            tok = self._build_tokenizer_without_vocab_file(tekken_path)

            out_dir = Path(tmp_dir) / "does-not-exist-yet"
            with self.assertRaises(OSError):
                tok.save_pretrained(str(out_dir), save_format="mistral")
            self.assertFalse(out_dir.exists())

    def test_save_pretrained_mistral_format_rejects_added_token(self):
        """save_pretrained(save_format='mistral') raises ValueError after add_tokens, naming the culprit and 'hf'."""
        with _converted_tokenizer() as (tok, _tekken_path):
            tok.add_tokens(["<my_new_token>"])

            with tempfile.TemporaryDirectory() as out_dir:
                with self.assertRaises(ValueError) as ctx:
                    tok.save_pretrained(out_dir, save_format="mistral")
                message = str(ctx.exception)
                self.assertIn("<my_new_token>", message)
                self.assertIn("save_format='hf'", message)

    def test_save_pretrained_mistral_format_rejects_added_special_token(self):
        """save_pretrained(save_format='mistral') raises ValueError after add_special_tokens, naming the culprit."""
        with _converted_tokenizer() as (tok, _tekken_path):
            tok.add_special_tokens({"additional_special_tokens": ["<my_special>"]})

            with tempfile.TemporaryDirectory() as out_dir:
                with self.assertRaises(ValueError) as ctx:
                    tok.save_pretrained(out_dir, save_format="mistral")
                self.assertIn("<my_special>", str(ctx.exception))

    def test_save_pretrained_mistral_format_rejection_leaves_no_directory(self):
        """A guard-rejected mistral save (mutated tokenizer) must not create the output directory."""
        with _converted_tokenizer() as (tok, tekken_path):
            tok.add_tokens(["<my_new_token>"])

            out_dir = tekken_path.parent / "does-not-exist-yet"
            with self.assertRaises(ValueError):
                tok.save_pretrained(str(out_dir), save_format="mistral")
            self.assertFalse(out_dir.exists())

    def test_save_pretrained_hf_format_ignores_guard_and_keeps_added_token(self):
        """The divergence guard only applies to save_format='mistral'; 'hf' saves mutations untouched."""
        with _converted_tokenizer() as (tok, tekken_path):
            out_dir = str(tekken_path.parent / "out")

            tok.add_tokens(["<my_new_token>"])
            tok.save_pretrained(out_dir, save_format="hf")

            reloaded = TokenizersBackend.from_pretrained(out_dir)
            self.assertIn("<my_new_token>", reloaded.get_vocab())

    def test_save_pretrained_mistral_format_push_to_hub_snapshots_before_write(self):
        """The files-timestamps snapshot for push_to_hub must be taken before tekken.json is written,
        otherwise `_upload_modified_files` sees it as already-present and never uploads it."""
        with _converted_tokenizer() as (tok, tekken_path):
            out_dir = tekken_path.parent / "does-not-exist-yet"

            with (
                patch("transformers.tokenization_utils_tokenizers.hf_api") as mock_hf_api,
                patch.object(TokenizersBackend, "_upload_modified_files") as mock_upload,
            ):
                mock_hf_api.return_value.create_repo.return_value.repo_id = "fake-repo"
                tok.save_pretrained(str(out_dir), save_format="mistral", push_to_hub=True)

            mock_upload.assert_called_once()
            files_timestamps = mock_upload.call_args.kwargs["files_timestamps"]
            self.assertNotIn("tekken.json", files_timestamps)

    def test_save_pretrained_mistral_format_rejection_creates_no_hub_repo(self):
        """A guard-rejected mistral save (mutated tokenizer) must not create a Hub repo."""
        with _converted_tokenizer() as (tok, tekken_path):
            tok.add_tokens(["<my_new_token>"])
            out_dir = tekken_path.parent / "does-not-exist-yet"

            with patch("transformers.tokenization_utils_tokenizers.hf_api") as mock_hf_api:
                with self.assertRaises(ValueError):
                    tok.save_pretrained(str(out_dir), save_format="mistral", push_to_hub=True)

            mock_hf_api.return_value.create_repo.assert_not_called()

    @parameterized.expand(
        [
            ("string_template", "KNOWN TEMPLATE"),
            ("dict_template", {"default": "KNOWN TEMPLATE"}),
        ]
    )
    def test_save_pretrained_mistral_format_never_writes_chat_template(self, _name, chat_template):
        """save_format='mistral' writes only tekken.json regardless of chat_template's value or
        type: no chat_template.jinja sidecar is written, and the returned tuple is exactly
        (tekken.json path,). A dict-valued chat_template must not raise TypeError."""
        with _converted_tokenizer() as (tok, tekken_path):
            tok.chat_template = chat_template
            out_dir = tekken_path.parent / "out"

            saved_files = tok.save_pretrained(str(out_dir), save_format="mistral")

            self.assertFalse((out_dir / "chat_template.jinja").exists())
            self.assertEqual(saved_files, (str(out_dir / "tekken.json"),))

    def test_save_pretrained_mistral_format_writes_only_tekken_json(self):
        """A mistral-format save writes only tekken.json into the output directory: any other
        pre-existing file (e.g. a chat_template.jinja from an earlier save) is left untouched."""
        with _converted_tokenizer() as (tok, tekken_path):
            tok.chat_template = "NEW TEMPLATE"
            out_dir = tekken_path.parent / "out"
            out_dir.mkdir()
            template_path = out_dir / "chat_template.jinja"
            template_path.write_text("SENTINEL", encoding="utf-8")

            saved_files = tok.save_pretrained(str(out_dir), save_format="mistral")

            self.assertEqual(template_path.read_text(encoding="utf-8"), "SENTINEL")
            self.assertEqual(saved_files, (str(out_dir / "tekken.json"),))

    def test_save_pretrained_hf_format_writes_chat_template(self):
        """save_format='hf' still writes the chat template through the base class's standard
        chat_template.jinja machinery, unlike save_format='mistral'."""
        with _converted_tokenizer() as (tok, tekken_path):
            tok.chat_template = "HF TEMPLATE"
            out_dir = tekken_path.parent / "out"

            tok.save_pretrained(str(out_dir), save_format="hf")

            template_path = out_dir / "chat_template.jinja"
            self.assertTrue(template_path.exists())
            self.assertEqual(template_path.read_text(encoding="utf-8"), "HF TEMPLATE")


class TestNonCanonicalTekkenFilenames(unittest.TestCase):
    """Tests for tekken vocabulary files whose basename is not exactly `tekken.json`."""

    def test_versioned_filename_loads(self):
        """A directory containing tekken_240911.json converts into a working tokenizer."""
        with _converted_tokenizer(filename="tekken_240911.json") as (tok, _tekken_path):
            ids = tok.encode("hello world", add_special_tokens=False)
            self.assertGreater(len(ids), 0)
            self.assertEqual(tok.decode(ids), "hello world")

    def test_versioned_filename_save_normalizes_name(self):
        """Saving a tokenizer sourced from tekken_240911.json normalizes the output to
        tekken.json, byte-identical to the source, and does not preserve the source's own
        basename in the destination."""
        with _converted_tokenizer(filename="tekken_240911.json") as (tok, tekken_path):
            out_dir = tekken_path.parent / "out"

            result = tok.save_pretrained(str(out_dir), save_format="mistral")

            saved = out_dir / "tekken.json"
            self.assertTrue(saved.exists())
            self.assertEqual(saved.read_bytes(), tekken_path.read_bytes())
            self.assertEqual(result, (str(saved),))
            self.assertFalse((out_dir / "tekken_240911.json").exists())

    def test_non_canonical_filename_load_and_save(self):
        """A tekken vocabulary named my_tekken.json loads, and is normalized to tekken.json
        on save."""
        with _converted_tokenizer(filename="my_tekken.json") as (tok, tekken_path):
            ids = tok.encode("hello world", add_special_tokens=False)
            self.assertGreater(len(ids), 0)

            out_dir = tekken_path.parent / "out"
            result = tok.save_pretrained(str(out_dir), save_format="mistral")

            saved = out_dir / "tekken.json"
            self.assertTrue(saved.exists())
            self.assertEqual(saved.read_bytes(), tekken_path.read_bytes())
            self.assertEqual(result, (str(saved),))
            self.assertFalse((out_dir / "my_tekken.json").exists())

    def test_versioned_filename_save_round_trips_through_auto_tokenizer(self):
        """The point of normalizing the save destination: a versioned-name source still
        produces an output directory that AutoTokenizer can discover and load, even though
        the source directory (with only the versioned name) could not be."""
        with _converted_tokenizer(filename="tekken_240911.json") as (tok, tekken_path):
            out_dir = tekken_path.parent / "out"
            tok.save_pretrained(str(out_dir), save_format="mistral")

            reloaded = AutoTokenizer.from_pretrained(str(out_dir), config=MistralConfig(), mistral_format=False)

            self.assertIsInstance(reloaded, TokenizersBackend)
            text = "hello world"
            expected_ids = tok.encode(text, add_special_tokens=False)
            actual_ids = reloaded.encode(text, add_special_tokens=False)
            self.assertEqual(actual_ids, expected_ids)

    def test_non_tekken_json_source_rejected(self):
        """A .json vocab file without 'tekken' in its name cannot be saved as mistral format."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            vocab_path = write_fake_tekken_json(Path(tmp_dir), filename="vocab.json")
            tok = convert_tekken_tokenizer(str(vocab_path))

            with tempfile.TemporaryDirectory() as out_dir:
                with self.assertRaisesRegex(OSError, "the original tekken\\.json is not available"):
                    tok.save_pretrained(out_dir, save_format="mistral")

    def test_tekken_txt_extension_rejected(self):
        """A file named tekken.txt fails the `.json`-suffix half of the classification rule."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tekken_path = write_fake_tekken_json(Path(tmp_dir), filename="tekken.txt")
            tok = convert_tekken_tokenizer(str(tekken_path))

            with tempfile.TemporaryDirectory() as out_dir:
                with self.assertRaisesRegex(OSError, "the original tekken\\.json is not available"):
                    tok.save_pretrained(out_dir, save_format="mistral")

    def test_versioned_filename_with_diverged_vocab_rejected_for_divergence_not_classification(self):
        """A versioned tekken source (e.g. tekken_240911.json) is accepted by the loose
        classification rule and reaches the divergence guard; a mutated tokenizer is then
        rejected with ValueError, not OSError."""
        with _converted_tokenizer(filename="tekken_240911.json") as (tok, _tekken_path):
            tok.add_tokens(["<x>"])

            with tempfile.TemporaryDirectory() as out_dir:
                with self.assertRaisesRegex(ValueError, "diverged from its source"):
                    tok.save_pretrained(out_dir, save_format="mistral")

    def test_in_place_resave_writes_canonical_alongside_versioned(self):
        """Resaving a versioned-filename tokenizer into its own source directory is NOT a
        short-circuit: source and destination basenames differ (tekken_240911.json vs.
        tekken.json), so a real copy is made and the directory ends up holding both files.
        This is correct and discoverable: mistral-common's discovery only ever admits a file
        named exactly `tekken.json` as a tekken candidate (see
        `_filter_valid_tokenizer_files`), so the versioned source is never itself a discovery
        candidate — the canonical file written beside it is the only one found."""
        with _converted_tokenizer(filename="tekken_240911.json") as (tok, tekken_path):
            original_bytes = tekken_path.read_bytes()

            result = tok.save_pretrained(str(tekken_path.parent), save_format="mistral")

            canonical = tekken_path.parent / "tekken.json"
            self.assertEqual(result, (str(canonical),))
            self.assertTrue(canonical.exists())
            self.assertEqual(canonical.read_bytes(), original_bytes)
            self.assertTrue(tekken_path.exists())
            self.assertEqual(tekken_path.read_bytes(), original_bytes)


class TestDeriveTekkenSpecials(unittest.TestCase):
    """Tests for `_derive_tekken_specials`'s old-tekken-format fallback (no top-level
    `special_tokens` key), which requires `mistral-common` to supply its deprecated
    defaults."""

    def test_old_format_without_mistral_common_raises_import_error(self):
        """Regression: when `mistral-common` is unavailable, the old-format fallback must
        raise a clear `ImportError` naming `MistralConverter`, not a `TypeError` from
        iterating a `None` special-tokens list."""
        raw = build_fake_tekken_dict(keys_to_drop=(("top", "special_tokens"),))
        unavailable_mapping = dict(BACKENDS_MAPPING)
        unavailable_mapping["mistral-common"] = (lambda: False, BACKENDS_MAPPING["mistral-common"][1])

        with patch.dict(BACKENDS_MAPPING, unavailable_mapping):
            with self.assertRaises(ImportError) as ctx:
                _derive_tekken_specials(raw)
            self.assertIn("MistralConverter", str(ctx.exception))
            self.assertIn("mistral-common", str(ctx.exception))

    @require_mistral_common
    def test_old_format_with_mistral_common_uses_deprecated_defaults(self):
        """With `mistral-common` installed, the old-format fallback produces its
        `DEPRECATED_SPECIAL_TOKENS`, plus `<SPECIAL_i>` fillers when the config asks for
        more special tokens than are declared."""
        from mistral_common.tokens.tokenizers.tekken import Tekkenizer

        expected_deprecated = [
            str(getattr(entry["token_str"], "value", entry["token_str"]))
            for entry in sorted(Tekkenizer.DEPRECATED_SPECIAL_TOKENS, key=lambda entry: entry["rank"])
        ]

        raw_exact = build_fake_tekken_dict(
            keys_to_drop=(("top", "special_tokens"),), num_special_tokens=len(expected_deprecated)
        )
        special_strings, num_special_tokens = _derive_tekken_specials(raw_exact)
        self.assertEqual(special_strings, expected_deprecated)
        self.assertEqual(num_special_tokens, len(expected_deprecated))

        raw_with_fillers = build_fake_tekken_dict(
            keys_to_drop=(("top", "special_tokens"),), num_special_tokens=len(expected_deprecated) + 3
        )
        special_strings, num_special_tokens = _derive_tekken_specials(raw_with_fillers)
        self.assertEqual(special_strings[: len(expected_deprecated)], expected_deprecated)
        self.assertEqual(special_strings[len(expected_deprecated) :], ["<SPECIAL_20>", "<SPECIAL_21>", "<SPECIAL_22>"])
        self.assertEqual(num_special_tokens, len(expected_deprecated) + 3)


class TestCheckTekkenVocabUnchanged(unittest.TestCase):
    """Tests for `_check_tekken_vocab_unchanged`, the mistral-format divergence guard."""

    def test_unmutated_tokenizer_passes(self):
        """A freshly converted tokenizer always matches its own source tekken.json."""
        with _converted_tokenizer() as (tok, tekken_path):
            _check_tekken_vocab_unchanged(tok, str(tekken_path))

    def test_unmutated_tokenizer_with_filler_special_tokens_passes(self):
        """A tokenizer converted from a fixture with filler <SPECIAL_i> tokens still passes unmutated."""
        with _converted_tokenizer(vocab_size=FULL_BYTE_VOCAB + 4, num_special_tokens=NUM_SPECIAL_TOKENS + 4) as (
            tok,
            tekken_path,
        ):
            _check_tekken_vocab_unchanged(tok, str(tekken_path))

    @parameterized.expand(
        [
            ("no_default_vocab_size_exact_vocab", {"keys_to_drop": (("config", "default_vocab_size"),)}),
            (
                "no_default_vocab_size_padded_vocab",
                {
                    "keys_to_drop": (("config", "default_vocab_size"),),
                    "vocab_size": FULL_BYTE_VOCAB + 4,
                    "num_special_tokens": NUM_SPECIAL_TOKENS + 4,
                },
            ),
            ("no_default_num_special_tokens", {"keys_to_drop": (("config", "default_num_special_tokens"),)}),
            (
                "neither_default_key",
                {"keys_to_drop": (("config", "default_vocab_size"), ("config", "default_num_special_tokens"))},
            ),
        ]
    )
    def test_unmutated_tokenizer_passes_without_default_keys(self, _name, write_kwargs):
        """An unmutated tokenizer passes the guard even if its source tekken.json omits
        `default_vocab_size` and/or `default_num_special_tokens` from its config."""
        with _converted_tokenizer(**write_kwargs) as (tok, tekken_path):
            _check_tekken_vocab_unchanged(tok, str(tekken_path))

    @require_mistral_common
    def test_unmutated_tokenizer_passes_old_format_without_special_tokens_key(self):
        """An unmutated tokenizer passes the guard for the old tekken format, which has no
        top-level `special_tokens` key and falls back to mistral-common's deprecated defaults."""
        with _converted_tokenizer(keys_to_drop=(("top", "special_tokens"),)) as (tok, tekken_path):
            _check_tekken_vocab_unchanged(tok, str(tekken_path))

    def test_size_only_divergence_reports_both_possible_causes(self):
        """A size-only divergence cannot tell an in-session resize apart from a tekken.json
        whose `default_num_special_tokens` is smaller than its declared special tokens, so the
        error reports the observed sizes and names both causes."""
        with _converted_tokenizer(vocab_size=30, num_special_tokens=10) as (tok, tekken_path):
            with self.assertRaises(ValueError) as ctx:
                _check_tekken_vocab_unchanged(tok, str(tekken_path))
            message = str(ctx.exception)
            self.assertIn("resized", message)
            self.assertIn("save_format='hf'", message)
            self.assertIn("regenerated", message)
            # The added-token branch must not have fired.
            self.assertNotIn("Unexpected added tokens", message)
            self.assertNotIn("Missing expected special tokens", message)

    def test_chat_template_change_does_not_trigger_guard(self):
        with _converted_tokenizer() as (tok, tekken_path):
            tok.chat_template = "SOME OTHER TEMPLATE"
            _check_tekken_vocab_unchanged(tok, str(tekken_path))


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

        converter = MistralConverter(vocab_file=str(tekken_path))
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
            converter = MistralConverter(vocab_file=tekken_path)
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
