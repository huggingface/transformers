"""Tokenization Fast class for Sophon adapted for Transformers 5.0.0."""
import json
from typing import Any, Dict, Optional, Tuple, List

from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE

from transformers.tokenization_utils_tokenizers import TokenizersBackend
from transformers.utils import logging

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"tokenizer_file": "tokenizer.json"}

PRETOKENIZE_REGEX = r"'(?i:[sdmt]|ll|ve|re)| (?=\p{Han}|[＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·．！？｡。])|[＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·．！？｡。]+[\r\n]*|[^\r\n\p{L}\p{N}]?+[\p{L}\p{M}]+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"

class OpenPanguV2Tokenizer(TokenizersBackend):
    vocab_files_names = VOCAB_FILES_NAMES
    padding_side = "left"
    model_input_names = ["input_ids", "attention_mask"]
    model = BPE

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        merges: Optional[List[str]] = None,
        tokenizer_file: Optional[str] = None,
        bos_token: str = "<|pangu_text_start|>",
        eos_token: str = "<|pangu_text_end|>",
        unk_token: str = "<unk>",
        add_bos_token: bool = True,
        add_eos_token: bool = False,
        add_prefix_space: bool = False,
        **kwargs,
    ):
        print("[YCHDEBUG] init OpenPanguV2Tokenizer", flush=True)
        
        self._vocab = vocab or {}
        self._merges = merges or []
        self._add_bos_token = add_bos_token
        self._add_eos_token = add_eos_token

        self._tokenizer = Tokenizer(
            BPE(
                vocab=self._vocab,
                merges=self._merges,
                dropout=None,
                unk_token=unk_token,
                byte_fallback=True,
            )
        )


        self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(
                Regex(PRETOKENIZE_REGEX),
                behavior="isolated",
            ),
            pre_tokenizers.ByteLevel(
                add_prefix_space=add_prefix_space,
                use_regex=False,
            ),
        ])

        self._tokenizer.decoder = decoders.ByteLevel()

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            **kwargs,
        )

        self.update_post_processor()

    def update_post_processor(self):
        bos = self.bos_token
        bos_id = self.bos_token_id
        eos = self.eos_token
        eos_id = self.eos_token_id

        single = f"{bos}:0 $A:0" if self._add_bos_token else "$A:0"
        if self._add_eos_token:
            single += f" {eos}:0"

        special_tokens = []
        if self._add_bos_token:
            special_tokens.append((bos, bos_id))
        if self._add_eos_token:
            special_tokens.append((eos, eos_id))

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=single,
            pair=f"{single} {single.replace('$A', '$B')}",
            special_tokens=special_tokens,
        )

    @property
    def add_bos_token(self):
        return self._add_bos_token

    @add_bos_token.setter
    def add_bos_token(self, value):
        self._add_bos_token = value
        self.update_post_processor()

    @property
    def add_eos_token(self):
        return self._add_eos_token

    @add_eos_token.setter
    def add_eos_token(self, value):
        self._add_eos_token = value
        self.update_post_processor()

__all__ = ["OpenPanguV2Tokenizer"]