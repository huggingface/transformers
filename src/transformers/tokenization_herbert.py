from typing import Optional

from tokenizers import CharBPETokenizer
from tokenizers.normalizers import BertNormalizer
from tokenizers.processors import BertProcessing

from .tokenization_bert import BasicTokenizer
from .tokenization_utils_fast import PreTrainedTokenizerFast
from .tokenization_xlm import XLMTokenizer
from .utils import logging

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "allegro/herbert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/allegro/herbert-base-cased/vocab.json",
        "allegro/herbert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/allegro/herbert-large-cased/vocab.json",
    },
    "merges_file": {
        "allegro/herbert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/allegro/herbert-base-cased/merges.txt",
        "allegro/herbert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/allegro/herbert-large-cased/merges.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "allegro/herbert-base-cased": 512,
    "allegro/herbert-large-cased": 512,
}


class HerbertTokenizer(XLMTokenizer):
    """
    Construct a BPE tokenizer for HerBERT.

    Peculiarities:

    - uses BERT's pre-tokenizer: BaseTokenizer splits tokens on spaces, and also on punctuation.
    Each occurence of a punctuation character will be treated separately.

    - Such pretokenized input is BPE subtokenized

    This tokenizer inherits from :class:`~transformers.XLMTokenizer` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cls_token = "<s>"
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        self.mask_token = "<mask>"
        self.sep_token = "</s>"
        self.do_lowercase_and_remove_accent = False

        self.bert_pre_tokenizer = BasicTokenizer(
            do_lower_case=False, never_split=self.all_special_tokens, tokenize_chinese_chars=False, strip_accents=False
        )

    def _tokenize(self, text):

        pre_tokens = self.bert_pre_tokenizer.tokenize(text)

        split_tokens = []
        for token in pre_tokens:
            if token:
                split_tokens.extend([t for t in self.bpe(token).split(" ")])

        return split_tokens


class HerbertTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "Fast" BPE tokenizer for HerBERT (backed by HuggingFace's `tokenizers` library).

    Peculiarities:

    - uses BERT's pre-tokenizer: BertPreTokenizer splits tokens on spaces, and also on punctuation.
    Each occurence of a punctuation character will be treated separately.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.

    Args:
        vocab_file (:obj:`str`):
            Path to the vocabulary file.
        merges_file (:obj:`str`):
            Path to the merges file.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = HerbertTokenizer

    def __init__(self, vocab_file, merges_file, dropout: Optional[float] = None, **kwargs):
        kwargs["cls_token"] = "<s>"
        kwargs["unk_token"] = "<unk>"
        kwargs["pad_token"] = "<pad>"
        kwargs["mask_token"] = "<mask>"
        kwargs["sep_token"] = "</s>"

        super().__init__(
            vocab_file,
            merges_file,
            **kwargs,
        )
