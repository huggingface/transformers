import json
import re
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
MODEL_NAMES = ("opus-mt-en-de",)  # TODO(SS): the only required constant is vocab_files_names
PRETRAINED_VOCAB_FILES_MAP = {
    k: {m: f"{S3_BUCKET_PREFIX}/Helsinki-NLP/{m}/{fname}" for m in MODEL_NAMES}
    for k, fname in vocab_files_names.items()
}
# Example URL https://s3.amazonaws.com/models.huggingface.co/bert/Helsinki-NLP/opus-mt-en-de/vocab.json


class MarianTokenizer(PreTrainedTokenizer):
    """Sentencepiece tokenizer for marian. Source and target languages have different SPM models.
    The logic is use the relevant source_spm or target_spm to encode txt as pieces, then look up each piece in a vocab dictionary.

    Examples::

        from transformers import MarianTokenizer
        tok = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
        src_texts = [ "I am a small frog.", "Tom asked his teacher for advice."]
        tgt_texts = ["Ich bin ein kleiner Frosch.", "Tom bat seinen Lehrer um Rat."]  # optional
        batch_enc: BatchEncoding = tok.prepare_translation_batch(src_texts, tgt_texts=tgt_texts)
        # keys  [input_ids, attention_mask, decoder_input_ids,  decoder_attention_mask].
        # model(**batch) should work
    """

    vocab_files_names = vocab_files_names
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = {m: 512 for m in MODEL_NAMES}
    model_input_names = ["attention_mask"]  # actually attention_mask, decoder_attention_mask
    language_code_re: re.Pattern = re.compile(">>.+<<")

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
        if self.unk_token not in self.encoder:
            raise KeyError("<unk> token must be in vocab")
        assert self.pad_token in self.encoder
        self.decoder = {v: k for k, v in self.encoder.items()}

        self.source_lang = source_lang
        self.target_lang = target_lang

        # load SentencePiece model for pre-processing
        self.spm_source = sentencepiece.SentencePieceProcessor()
        self.spm_source.Load(source_spm)

        self.spm_target = sentencepiece.SentencePieceProcessor()
        self.spm_target.Load(target_spm)

        # Multilingual target side: default to using first supported language code.
        self.supported_language_codes: list = [k for k in self.encoder if k.startswith(">>") and k.endswith("<<")]

        # Note(SS): sentence_splitter would require lots of book-keeping.
        try:
            from mosestokenizer import MosesPunctuationNormalizer

            self.punc_normalizer = MosesPunctuationNormalizer(source_lang)
        except ImportError:
            warnings.warn("Recommended: pip install mosestokenizer")
            self.punc_normalizer = lambda x: x

    def _convert_token_to_id(self, token):
        return self.encoder.get(token, self.encoder[self.unk_token])

    def remove_language_code(self, text: str):
        """Remove language codes like <<fr>> before sentencepiece"""
        match = self.language_code_re.match(text)
        code: list = [match.group(0)] if match else []
        return code, self.language_code_re.sub("", text)

    def _tokenize(self, text: str) -> List[str]:
        code, text = self.remove_language_code(text)
        pieces = self.current_spm.EncodeAsPieces(text)
        return code + pieces

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the encoder."""
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Uses target language sentencepiece model"""
        return self.spm_target.DecodePieces(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        # We don't expect to process pairs, but leave the pair logic for API consistency
        return token_ids_0 + token_ids_1 + [self.eos_token_id]

    def batch_decode(self, token_ids, **kwargs) -> List[str]:
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
            tgt_texts: list of tgt language texts
            max_length: (None) defer to config (1024 for mbart-large-en-ro)
            pad_to_max_length: (bool)
            return_tensors: (str) default "pt" returns pytorch tensors, pass None to return lists.

        Returns:
            BatchEncoding: with keys [input_ids, attention_mask, decoder_input_ids,  decoder_attention_mask]
            all shaped bs, seq_len. (BatchEncoding is a dict of string -> tensor or lists).
            If no tgt_text is specified, the only keys will be input_ids and attention_mask.
        """
        self.current_spm = self.spm_source
        src_texts = [self.punc_normalizer(t) for t in src_texts]
        model_inputs: BatchEncoding = self.batch_encode_plus(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            pad_to_max_length=pad_to_max_length,
        )
        if tgt_texts is None:
            return model_inputs

        self.current_spm = self.spm_target
        decoder_inputs: BatchEncoding = self.batch_encode_plus(
            tgt_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            pad_to_max_length=pad_to_max_length,
        )
        for k, v in decoder_inputs.items():
            model_inputs[f"decoder_{k}"] = v
        self.current_spm = self.spm_source
        return model_inputs

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)


def load_json(path: str) -> Union[Dict, List]:
    with open(path, "r") as f:
        return json.load(f)
