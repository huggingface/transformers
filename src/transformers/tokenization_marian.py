import json
from typing import Dict, List, Optional

import sentencepiece
import yaml
from mosestokenizer import MosesPunctuationNormalizer
from torch import Tensor

from .tokenization_utils import PreTrainedTokenizer


PRETRAINED_VOCAB_FILES_MAP = {
    "vocab": {
        # "marian/en-de": "https://s3.amazonaws.com/models.huggingface.co/bert/marian/en-de/source.spm"
    }
}


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_yaml(path):
    with open(path) as f:
        return yaml.load(f, Loader=yaml.BaseLoader)


class MarianSPTokenizer(PreTrainedTokenizer):
    vocab_files_names = {
        "source_spm": "source.spm",
        "target_spm": "target.spm",
        "vocab": "vocab.json",
        "tokenizer_config_file": "tokenizer_config.json",
    }

    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = {m: 512 for m in PRETRAINED_VOCAB_FILES_MAP}
    model_input_names = ["attention_mask"]  # really attention_mask, decoder_attention_mask

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
        **kwargs
    ):

        super().__init__(
            # bos_token=bos_token,
            max_len=max_len,
            eos_token=eos_token,
            unk_token=unk_token,
            # sep_token=sep_token,
            # cls_token=cls_token,
            pad_token=pad_token,
            # mask_token=mask_token,
            # **kwargs,
        )
        self.encoder = load_json(vocab)  # {k: int(v) for k, v in load_json(vocab).items()}
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
        self.punc_normalizer = MosesPunctuationNormalizer(source_lang)

    def _convert_token_to_id(self, token):
        return self.encoder[token]

    def _tokenize(self, text: str, src=True) -> list:
        spm = self.spm_source if src else self.spm_target
        return spm.EncodeAsPieces(text)

    def _convert_id_to_token(self, index: int):
        """Converts an index (integer) in a token (str) using the encoder."""
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]):
        return self.spm_target.DecodePieces(tokens)

    #
    # def _tokenize(self, text):
    #     return self.spm_source.EncodeAsPieces(text)

    def _append_special_tokens_and_truncate(self, tokens: str, max_length: int,) -> List[int]:
        ids: list = self.convert_tokens_to_ids(tokens)[:max_length]
        return ids + [self.eos_token_id]

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A RoBERTa sequence has the following format:
            single sequence: <s> X </s>
            pair of sequences: <s> A </s></s> B </s>
        """
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        return token_ids_0 + token_ids_1 + [self.eos_token_id]

    def decode_batch(self, token_ids, **kwargs) -> List[str]:
        return [self.decode(ids) for ids in token_ids]

    def prepare_translation_batch(
        self,
        src_texts: List[str],
        tgt_texts: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        pad_to_max_length: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, Tensor]:
        """
        Arguments:
            src_texts: list of src language texts
            src_lang: default en_XX (english)
            tgt_texts: list of tgt language texts
            tgt_lang: default ro_RO (romanian)
            max_length: (None) defer to config (1024 for mbart-large-en-ro)
            pad_to_max_length: (bool)

        Returns:
            dict with keys  [input_ids, attention_mask, decoder_input_ids,  decoder_attention_mask]
            all shaped bs, seq_len.
        """
        model_inputs = self.batch_encode_plus(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            pad_to_max_length=pad_to_max_length,
            src=True,
        )

        if tgt_texts is None:
            return model_inputs
        decoder_inputs = self.batch_encode_plus(
            tgt_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            pad_to_max_length=pad_to_max_length,
            src=False,
        )
        for k, v in decoder_inputs.items():
            model_inputs[f"decoder_{k}"] = v

        # model_inputs["decoder_attention_mask"] = decoder_inputs["decoder_attention_mask"]
        return model_inputs

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)
