"""Tests for Wav2Vec2PhonemeCTCTokenizer stale phonemizer_lang bug (#46614)."""
import unittest
from unittest.mock import MagicMock, patch

from transformers.models.wav2vec2_phoneme.tokenization_wav2vec2_phoneme import (
    Wav2Vec2PhonemeCTCTokenizer,
)


class Wav2Vec2PhonemeTokenizerTest(unittest.TestCase):
    """The phonemize() method must update self.phonemizer_lang when switching backends.

    Regression test for #46614: phonemize() re-creates the espeak backend when the
    requested language differs from self.phonemizer_lang, but never updates
    self.phonemizer_lang. A subsequent call with the ORIGINAL language skips the
    backend re-init and uses the wrong (cached) backend.
    """

    def test_phonemize_updates_phonemizer_lang(self):
        """After phonemize() switches to a different language, self.phonemizer_lang
        must reflect the new language so the next switch-back is detected."""
        tokenizer = Wav2Vec2PhonemeCTCTokenizer.__new__(Wav2Vec2PhonemeCTCTokenizer)
        # Set minimal required attrs that phonemize reads
        tokenizer.phonemizer_lang = "en-us"
        tokenizer._phonemizer_backend = MagicMock()
        tokenizer._phonemizer_backend.phonemize.return_value = ["dummy"]
        tokenizer.word_delimiter_token = None
        tokenizer.phone_delimiter_token = ""
        tokenizer.do_phonemize = True

        # Mock init_backend so it doesn't actually import phonemizer
        tokenizer.init_backend = MagicMock()

        # Call phonemize with a different language
        tokenizer.phonemize("hello", phonemizer_lang="es")

        # self.phonemizer_lang must be updated
        self.assertEqual(
            tokenizer.phonemizer_lang, "es",
            "phonemizer_lang must be updated after switching backends",
        )

    def test_phonemize_same_language_skips_backend_init(self):
        """Calling phonemize with the same language must NOT re-init the backend."""
        tokenizer = Wav2Vec2PhonemeCTCTokenizer.__new__(Wav2Vec2PhonemeCTCTokenizer)
        tokenizer.phonemizer_lang = "en-us"
        tokenizer._phonemizer_backend = MagicMock()
        tokenizer._phonemizer_backend.phonemize.return_value = ["dummy"]
        tokenizer.word_delimiter_token = None
        tokenizer.phone_delimiter_token = ""
        tokenizer.do_phonemize = True

        tokenizer.init_backend = MagicMock()

        # Call with same language
        tokenizer.phonemize("hello", phonemizer_lang="en-us")

        tokenizer.init_backend.assert_not_called()

    def test_phonemize_different_language_inits_backend(self):
        """Calling phonemize with a different language MUST re-init the backend."""
        tokenizer = Wav2Vec2PhonemeCTCTokenizer.__new__(Wav2Vec2PhonemeCTCTokenizer)
        tokenizer.phonemizer_lang = "en-us"
        tokenizer._phonemizer_backend = MagicMock()
        tokenizer._phonemizer_backend.phonemize.return_value = ["dummy"]
        tokenizer.word_delimiter_token = None
        tokenizer.phone_delimiter_token = ""
        tokenizer.do_phonemize = True

        tokenizer.init_backend = MagicMock()

        tokenizer.phonemize("hello", phonemizer_lang="es")

        tokenizer.init_backend.assert_called_once_with("es")

    def test_phonemize_switch_back_and_forth(self):
        """Switching languages repeatedly must work correctly (#46614 regression)."""
        tokenizer = Wav2Vec2PhonemeCTCTokenizer.__new__(Wav2Vec2PhonemeCTCTokenizer)
        tokenizer.phonemizer_lang = "en-us"
        tokenizer._phonemizer_backend = MagicMock()
        tokenizer._phonemizer_backend.phonemize.return_value = ["dummy"]
        tokenizer.word_delimiter_token = None
        tokenizer.phone_delimiter_token = ""
        tokenizer.do_phonemize = True

        tokenizer.init_backend = MagicMock()

        # Switch to Spanish
        tokenizer.phonemize("hola", phonemizer_lang="es")
        self.assertEqual(tokenizer.phonemizer_lang, "es")

        # Switch back to English - would fail without the fix
        tokenizer.init_backend.reset_mock()
        tokenizer.phonemize("hello", phonemizer_lang="en-us")
        self.assertEqual(tokenizer.phonemizer_lang, "en-us")
        tokenizer.init_backend.assert_called_once_with("en-us")
