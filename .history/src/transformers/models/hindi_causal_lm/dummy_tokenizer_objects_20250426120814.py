# coding=utf-8
# Dummy placeholder for SentencePiece-dependent tokenizer

from ...utils import DummyObject

class HindiCausalLMTokenizer(metaclass=DummyObject):
    _backends = ["sentencepiece"]
