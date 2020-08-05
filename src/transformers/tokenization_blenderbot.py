# this code is almost the same as tokenization_gpt2.
# the original blenderbot is also using byte-level bpe tokenization based on subword-nmt
import json
import logging
from typing import List
import os

import regex as re

from .tokenization_roberta import RobertaTokenizer
from .tokenization_utils import AddedToken, PreTrainedTokenizer


logger = logging.getLogger(__name__)


VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

# will update paths once uploded files on S3
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/blenderbot-3B": "https://cdn.huggingface.co/sshleifer/blenderbot-3B/vocab.json",
        "facebook/blenderbot-9B": "https://cdn.huggingface.co/sshleifer/blenderbot-3B/vocab.json",  # uses the same vocab and merges files as the 3B model
    },
    "merges_file": {
        "facebook/blenderbot-3B": "https://cdn.huggingface.co/sshleifer/blenderbot-3B/merges.txt",
        "facebook/blenderbot-9B": "https://cdn.huggingface.co/sshleifer/blenderbot-3B/merges.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/blenderbot-3B": 128,
    "facebook/blenderbot-9B": 128,
}
logger = logging.getLogger(__name__)


class BlenderbotTokenizer(RobertaTokenizer):

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="__end__",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        add_prefix_space=True,
        **kwargs
    ):
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        # Mask token behave like a normal word, i.e. include the space before it
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

    
    
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: List[int] = None
    ):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A RoBERTa sequence has the following format:

        - single sequence: `` X </s>``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added

        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        return token_ids_0 + [self.sep_token_id]
    
    
        


BLENDERBOT_90M_PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {"facebook/blenderbot-90M": "https://cdn.huggingface.co/sshleifer/blenderbot-90M/vocab.json"},
    "merges_file": {"facebook/blenderbot-90M": "https://cdn.huggingface.co/sshleifer/blenderbot-90M/merges.txt"},
}

BLENDERBOT_90M_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/blenderbot-90M": 512,
}


def get_pairs(word):
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char

    pairs = set(pairs)
    return pairs


class BlenderbotSmallTokenizer(PreTrainedTokenizer):
    """
    Constructs a Blenderbot-90M tokenizer. Peculiarities:

    - Byte-Pair-Encoding

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.

    Args:
        vocab_file (:obj:`str`):
            Path to the vocabulary file.
        merges_file (:obj:`str`):
            Path to the merges file.
         bos_token (:obj:`string`, `optional`, defaults to "__start__"):
            The beginning of sentence token. 
         eos_token (:obj:`string`, `optional`, defaults to "__end__"):
            The end of sentence token. 
        unk_token (:obj:`string`, `optional`, defaults to "<unk>"):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = BLENDERBOT_90M_PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = BLENDERBOT_90M_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        merges_file,
        bos_token="__start__",
        eos_token="__end__",
        unk_token="__unk__",
        pad_token="__null",
        **kwargs
    ):
        super().__init__(unk_token=unk_token, bos_token=bos_token, eos_token=eos_token, pad_token=pad_token, **kwargs)

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        token = re.sub('([.,!?()])', r' \1', token)
        token = re.sub('(\')', r' \1 ', token)
        token = re.sub('\s{2,}', ' ', token)
        if "\n" in token:
            token = token.replace("\n", " __newln__")
        
        tokens = token.split(' ')
        words = []
        for token in tokens:
            token = token.lower()
            word = tuple(token)
            word = tuple(list(word[:-1]) + [word[-1] + "</w>"])
            pairs = get_pairs(word)

            if not pairs:
                words.append(token)
                continue
            

            while True:
                bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
                if bigram not in self.bpe_ranks:
                    break
                first, second = bigram
                new_word = []
                i = 0
                
                while i < len(word):
                    try:
                        j = word.index(first, i)
                        new_word.extend(word[i:j])
                        i = j
                    except ValueError:
                        new_word.extend(word[i:])
                        break
                    
                    if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                        new_word.append(first + second)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word = tuple(new_word)
                word = new_word
                if len(word) == 1:
                    break
                else:
                    pairs = get_pairs(word)
            word = "@@ ".join(word)
            word = word[:-4]
            
            self.cache[token] = word
            words.append(word)
        return ' '.join(words)

    def _tokenize(self, text):
        """ Tokenize a string.
        """
        split_tokens = []

        words = re.findall(r"\S+\n?", text)

        for token in words:
            split_tokens.extend([t for t in self.bpe(token).split(" ")])
        return split_tokens

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        token = token.lower()
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = " ".join(tokens).replace("@@ ", "").strip()
        return out_string

    def save_vocabulary(self, save_directory):
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (:obj:`str`):
                The directory in which to save the vocabulary.

        Returns:
            :obj:`Tuple(str)`: Paths to the files saved.
        """
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return
        vocab_file = os.path.join(save_directory, VOCAB_FILES_NAMES["vocab_file"])
        merge_file = os.path.join(save_directory, VOCAB_FILES_NAMES["merges_file"])

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, ensure_ascii=False))

        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!".format(merge_file)
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        return vocab_file, merge_file
