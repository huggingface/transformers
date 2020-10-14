from typing import Optional, List

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

        kwargs["cls_token"] = "<s>"
        kwargs["unk_token"] = "<unk>"
        kwargs["pad_token"] = "<pad>"
        kwargs["mask_token"] = "<mask>"
        kwargs["sep_token"] = "</s>"
        kwargs["do_lowercase_and_remove_accent"] = False

        super().__init__(**kwargs)
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

    def __init__(self, vocab_file, merges_file, **kwargs):

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

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        An CamemBERT sequence has the following format:

        - single sequence: ``<s> X </s>``
        - pair of sequences: ``<s> A </s></s> B </s>``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """

        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` method.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task.
        CamemBERT, like RoBERTa, does not make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of zeros.
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + token_ids_1 + sep) * [0]
