import json
import warnings
from typing import Dict, List, Optional, Union

import sentencepiece

from .file_utils import S3_BUCKET_PREFIX
from .tokenization_utils import BatchEncoding, PreTrainedTokenizer


vocab_files_names = {
    "source_spm": "source.spm",
    "target_spm": "target.spm",
    "vocab": "vocab.json",
    "tokenizer_config_file": "tokenizer_config.json",
}
MODEL_NAMES = ("opus-mt-en-de",)
PRETRAINED_VOCAB_FILES_MAP = {
    k: {m: f"{S3_BUCKET_PREFIX}/Helsinki-NLP/{m}/{fname}" for m in MODEL_NAMES}
    for k, fname in vocab_files_names.items()
}
# Example URL https://s3.amazonaws.com/models.huggingface.co/bert/Helsinki-NLP/opus-mt-en-de/vocab.json


class MarianSentencePieceTokenizer(PreTrainedTokenizer):
    vocab_files_names = vocab_files_names
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = {m: 512 for m in MODEL_NAMES}
    model_input_names = ["attention_mask"]  # actually attention_mask, decoder_attention_mask

    def __init__(
        self,
        vocab=None,
        source_spm=None,
        target_spm=None,
        source_lang=None,
        target_lang=None,
        unk_token="<unk>",
        eos_token="</s>",
        pad_token="<pad>",
        max_len=512,
    ):

        super().__init__(
            # bos_token=bos_token,
            max_len=max_len,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
        )
        self.encoder = load_json(vocab)
        assert self.pad_token in self.encoder
        self.decoder = {v: k for k, v in self.encoder.items()}

        self.source_lang = source_lang
        self.target_lang = target_lang

        # load SentencePiece model for pre-processing
        self.paths = {}

        self.spm_source = sentencepiece.SentencePieceProcessor()
        self.spm_source.Load(source_spm)

        self.spm_target = sentencepiece.SentencePieceProcessor()
        self.spm_target.Load(target_spm)

        # Note(SS): splitter would require lots of book-keeping.
        # self.sentence_splitter = MosesSentenceSplitter(source_lang)
        try:
            from mosestokenizer import MosesPunctuationNormalizer

            self.punc_normalizer = MosesPunctuationNormalizer(source_lang)
        except ImportError:
            warnings.warn("Recommended: pip install mosestokenizer")
            self.punc_normalizer = lambda x: x

    def _convert_token_to_id(self, token):
        return self.encoder[token]

    def _tokenize(self, text: str, src=True) -> List[str]:
        spm = self.spm_source if src else self.spm_target
        return spm.EncodeAsPieces(text)

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the encoder."""
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Uses target language sentencepiece model"""
        return self.spm_target.DecodePieces(tokens)

    def _append_special_tokens_and_truncate(self, tokens: str, max_length: int,) -> List[int]:
        ids: list = self.convert_tokens_to_ids(tokens)[:max_length]
        return ids + [self.eos_token_id]

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        # We don't expect to process pairs, but leave the pair logic for API consistency
        return token_ids_0 + token_ids_1 + [self.eos_token_id]

    def decode_batch(self, token_ids, **kwargs) -> List[str]:
        return [self.decode(ids, **kwargs) for ids in token_ids]

    def prepare_translation_batch(
        self,
        src_texts: List[str],
        tgt_texts: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        pad_to_max_length: bool = True,
        return_tensors: str = "pt",
    ) -> BatchEncoding:
        """
        Arguments:
            src_texts: list of src language texts
            src_lang: default en_XX (english)
            tgt_texts: list of tgt language texts
            tgt_lang: default ro_RO (romanian)
            max_length: (None) defer to config (1024 for mbart-large-en-ro)
            pad_to_max_length: (bool)

        Returns:
            BatchEncoding: with keys [input_ids, attention_mask, decoder_input_ids,  decoder_attention_mask]
            all shaped bs, seq_len. (BatchEncoding is a dict of string -> tensor or lists)

        Examples:
            from transformers import MarianS
        """
        model_inputs: BatchEncoding = self.batch_encode_plus(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            pad_to_max_length=pad_to_max_length,
            src=True,
        )
        if tgt_texts is None:
            return model_inputs

        decoder_inputs: BatchEncoding = self.batch_encode_plus(
            tgt_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            pad_to_max_length=pad_to_max_length,
            src=False,
        )
        for k, v in decoder_inputs.items():
            model_inputs[f"decoder_{k}"] = v
        return model_inputs

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)


def load_json(path: str) -> Union[Dict, List]:
    with open(path, "r") as f:
        return json.load(f)
