# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors and Thomas Wolf
"""
Prepare input data for Google's BERT Model using WordPiece tokenization
and build arrays of word, position and sentence embeddings.

The WordPiece tokenization classes and functions are taken from the tensor2tensor library:
https://github.com/tensorflow/tensor2tensor

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import NamedTuple, List, Union, Tuple
import re
import os
import six
import sys
import time
import glob
import logging
import collections
import unicodedata
from itertools import chain
from six.moves import range  # pylint: disable=redefined-builtin
import numpy as np

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# Reserved tokens for things like padding and EOS symbols.
PAD = "<pad>"
EOS = "<EOS>"
RESERVED_TOKENS = [PAD, EOS]
NUM_RESERVED_TOKENS = len(RESERVED_TOKENS)
PAD_ID = RESERVED_TOKENS.index(PAD)  # Normally 0
EOS_ID = RESERVED_TOKENS.index(EOS)  # Normally 1

# Regular expression for unescaping token strings.
# '\u' is converted to '_'
# '\\' is converted to '\'
# '\213;' is converted to unichr(213)
_UNESCAPE_REGEX = re.compile(r"\\u|\\\\|\\([0-9]+);")
_ESCAPE_CHARS = set(u"\\_u;0123456789")

# This set contains all letter and number characters.
_ALPHANUMERIC_CHAR_SET = set(
    six.unichr(i) for i in range(sys.maxunicode)
    if (unicodedata.category(six.unichr(i)).startswith("L") or
            unicodedata.category(six.unichr(i)).startswith("N")))


# Unicode utility functions that work with Python 2 and 3
def native_to_unicode(s):
    if is_unicode(s):
        return s
    try:
        return to_unicode(s)
    except UnicodeDecodeError:
        res = to_unicode(s, ignore_errors=True)
        logger.info("Ignoring Unicode error, outputting: {}".format(res))
        return res


def unicode_to_native(s):
    if six.PY2:
        return s.encode("utf-8") if is_unicode(s) else s
    else:
        return s


def is_unicode(s):
    if six.PY2:
        if isinstance(s, unicode):
            return True
    else:
        if isinstance(s, str):
            return True
    return False


def to_unicode(s, ignore_errors=False):
    if is_unicode(s):
        return s
    error_mode = "ignore" if ignore_errors else "strict"
    return s.decode("utf-8", errors=error_mode)


def to_unicode_ignore_errors(s):
    return to_unicode(s, ignore_errors=True)


def strip_ids(ids, ids_to_strip):
    """Strip ids_to_strip from the end ids."""
    ids = list(ids)
    while ids and ids[-1] in ids_to_strip:
        ids.pop()
    return ids


def tokenizer_encode(text):
    """A simple invertible tokenizer (in words).

    Converts from a unicode string to a list of tokens
    (represented as Unicode strings).

    This tokenizer has the following desirable properties:
    - It is invertible.
    - Alphanumeric characters are broken away from non-alphanumeric characters.
    - A single space between words does not produce an extra token.
    - The full Unicode punctuation and separator set is recognized.

    The tokenization algorithm is as follows:

    1.  Split the text into a list of tokens, splitting at every boundary of an
            alphanumeric character and a non-alphanumeric character.  This produces
            a list which alternates between "alphanumeric tokens"
            (strings of alphanumeric characters) and "non-alphanumeric tokens"
            (strings of non-alphanumeric characters).

    2.  Remove every token consisting of a single space, unless it is
            the very first or very last token in the list.  These tokens are now
            implied by the fact that there are two adjacent alphanumeric tokens.

    e.g.  u"Dude - that's so cool."
                    -> [u"Dude", u" - ", u"that", u"'", u"s", u"so", u"cool", u"."]

    Args:
        text: a unicode string
    Returns:
        a list of tokens as Unicode strings
    """
    if not text:
        return []
    ret = []
    token_start = 0
    # Classify each character in the input string
    is_alnum = [c in _ALPHANUMERIC_CHAR_SET for c in text]
    for pos in range(1, len(text)):
        if is_alnum[pos] != is_alnum[pos - 1]:
            token = text[token_start:pos]
            if token != u" " or token_start == 0:
                ret.append(token)
            token_start = pos
    final_token = text[token_start:]
    ret.append(final_token)
    return ret


def tokenizer_decode(tokens):
    """Decode a list of tokens to a unicode string.

    Args:
        tokens: a list of Unicode strings
    Returns:
        a unicode string
    """
    token_is_alnum = [t[0] in _ALPHANUMERIC_CHAR_SET for t in tokens]
    ret = []
    for i, token in enumerate(tokens):
        if i > 0 and token_is_alnum[i - 1] and token_is_alnum[i]:
            ret.append(u" ")
        ret.append(token)
    return "".join(ret)


class TextEncoder(object):
    """Base class for converting from ints to/from human readable strings."""

    def __init__(self, num_reserved_ids=NUM_RESERVED_TOKENS):
        self._num_reserved_ids = num_reserved_ids

    @property
    def num_reserved_ids(self):
        return self._num_reserved_ids

    def encode(self, s):
        """Transform a human-readable string into a sequence of int ids.

        The ids should be in the range [num_reserved_ids, vocab_size). Ids [0,
        num_reserved_ids) are reserved.

        EOS is not appended.

        Args:
            s: human-readable string to be converted.

        Returns:
            ids: list of integers
        """
        return [int(w) + self._num_reserved_ids for w in s.split()]

    def decode(self, ids, strip_extraneous=False):
        """Transform a sequence of int ids into a human-readable string.

        EOS is not expected in ids.

        Args:
            ids: list of integers to be converted.
            strip_extraneous: bool, whether to strip off extraneous tokens
                (EOS and PAD).

        Returns:
            s: human-readable string.
        """
        if strip_extraneous:
            ids = strip_ids(ids, list(range(self._num_reserved_ids or 0)))
        return " ".join(self.decode_list(ids))

    def decode_list(self, ids):
        """Transform a sequence of int ids into a their string versions.

        This method supports transforming individual input/output ids to their
        string versions so that sequence to/from text conversions can be visualized
        in a human readable format.

        Args:
            ids: list of integers to be converted.

        Returns:
            strs: list of human-readable string.
        """
        decoded_ids = []
        for id_ in ids:
            if 0 <= id_ < self._num_reserved_ids:
                decoded_ids.append(RESERVED_TOKENS[int(id_)])
            else:
                decoded_ids.append(id_ - self._num_reserved_ids)
        return [str(d) for d in decoded_ids]

    @property
    def vocab_size(self):
        raise NotImplementedError()


def _escape_token(token, alphabet):
    """Escape away underscores and OOV characters and append '_'.

    This allows the token to be expressed as the concatenation of a list
    of subtokens from the vocabulary. The underscore acts as a sentinel
    which allows us to invertibly concatenate multiple such lists.

    Args:
        token: A unicode string to be escaped.
        alphabet: A set of all characters in the vocabulary's alphabet.

    Returns:
        escaped_token: An escaped unicode string.

    Raises:
        ValueError: If the provided token is not unicode.
    """
    if not isinstance(token, six.text_type):
        raise ValueError("Expected string type for token, got %s" % type(token))

    token = token.replace(u"\\", u"\\\\").replace(u"_", u"\\u")
    ret = [c if c in alphabet and c != u"\n" else r"\%d;" % ord(c) for c in token]
    return u"".join(ret) + "_"


def _unescape_token(escaped_token):
    """Inverse of _escape_token().

    Args:
        escaped_token: a unicode string

    Returns:
        token: a unicode string
    """

    def match(m):
        if m.group(1) is None:
            return u"_" if m.group(0) == u"\\u" else u"\\"

        try:
            return six.unichr(int(m.group(1)))
        except (ValueError, OverflowError) as _:
            return u"\u3013"  # Unicode for undefined character.

    trimmed = escaped_token[:-1] if escaped_token.endswith("_") else escaped_token
    return _UNESCAPE_REGEX.sub(match, trimmed)


class SubwordTextEncoder(TextEncoder):
    """Class for invertibly encoding text using a limited vocabulary.

    Invertibly encodes a native string as a sequence of subtokens from a limited
    vocabulary.

    A SubwordTextEncoder is built from a corpus (so it is tailored to the text in
    the corpus), and stored to a file. See text_encoder_build_subword.py.

    It can then be loaded and used to encode/decode any text.

    Encoding has four phases:

    1. Tokenize into a list of tokens.  Each token is a unicode string of either
         all alphanumeric characters or all non-alphanumeric characters.  We drop
         tokens consisting of a single space that are between two alphanumeric
         tokens.

    2. Escape each token.  This escapes away special and out-of-vocabulary
         characters, and makes sure that each token ends with an underscore, and
         has no other underscores.

    3. Represent each escaped token as a the concatenation of a list of subtokens
         from the limited vocabulary.  Subtoken selection is done greedily from
         beginning to end.  That is, we construct the list in order, always picking
         the longest subtoken in our vocabulary that matches a prefix of the
         remaining portion of the encoded token.

    4. Concatenate these lists.  This concatenation is invertible due to the
         fact that the trailing underscores indicate when one list is finished.

    """

    def __init__(self, filename=None):
        """Initialize and read from a file, if provided.

        Args:
            filename: filename from which to read vocab. If None, do not load a
                vocab
        """
        self._alphabet = set()
        self.filename = filename
        if filename is not None:
            self._load_from_file(filename)
        super(SubwordTextEncoder, self).__init__()

    def encode(self, s):
        """Converts a native string to a list of subtoken ids.

        Args:
            s: a native string.
        Returns:
            a list of integers in the range [0, vocab_size)
        """
        return self._tokens_to_subtoken_ids(
                tokenizer_encode(native_to_unicode(s)))

    def encode_without_tokenizing(self, token_text):
        """Converts string to list of subtoken ids without calling tokenizer.

        This treats `token_text` as a single token and directly converts it
        to subtoken ids. This may be useful when the default tokenizer doesn't
        do what we want (e.g., when encoding text with tokens composed of lots of
        nonalphanumeric characters). It is then up to the caller to make sure that
        raw text is consistently converted into tokens. Only use this if you are
        sure that `encode` doesn't suit your needs.

        Args:
            token_text: A native string representation of a single token.
        Returns:
            A list of subword token ids; i.e., integers in the range [0, vocab_size).
        """
        return self._tokens_to_subtoken_ids([native_to_unicode(token_text)])

    def decode(self, ids, strip_extraneous=False):
        """Converts a sequence of subtoken ids to a native string.

        Args:
            ids: a list of integers in the range [0, vocab_size)
            strip_extraneous: bool, whether to strip off extraneous tokens
                (EOS and PAD).

        Returns:
            a native string
        """
        if strip_extraneous:
            ids = strip_ids(ids, list(range(self._num_reserved_ids or 0)))
        return unicode_to_native(
                tokenizer_decode(self._subtoken_ids_to_tokens(ids)))

    def decode_list(self, ids):
        return [self._subtoken_id_to_subtoken_string(s) for s in ids]

    @property
    def vocab_size(self):
        """The subtoken vocabulary size."""
        return len(self._all_subtoken_strings)

    def _tokens_to_subtoken_ids(self, tokens):
        """Converts a list of tokens to a list of subtoken ids.

        Args:
            tokens: a list of strings.
        Returns:
            a list of integers in the range [0, vocab_size)
        """
        ret = []
        for token in tokens:
            ret.extend(self._token_to_subtoken_ids(token))
        return ret

    def _token_to_subtoken_ids(self, token):
        """Converts token to a list of subtoken ids.

        Args:
            token: a string.
        Returns:
            a list of integers in the range [0, vocab_size)
        """
        cache_location = hash(token) % self._cache_size
        cache_key, cache_value = self._cache[cache_location]
        if cache_key == token:
            return cache_value
        ret = self._escaped_token_to_subtoken_ids(
                _escape_token(token, self._alphabet))
        self._cache[cache_location] = (token, ret)
        return ret

    def _subtoken_ids_to_tokens(self, subtokens):
        """Converts a list of subtoken ids to a list of tokens.

        Args:
            subtokens: a list of integers in the range [0, vocab_size)
        Returns:
            a list of strings.
        """
        concatenated = "".join(
                [self._subtoken_id_to_subtoken_string(s) for s in subtokens])
        split = concatenated.split("_")
        ret = []
        for t in split:
            if t:
                unescaped = _unescape_token(t + "_")
                if unescaped:
                    ret.append(unescaped)
        return ret

    def _subtoken_id_to_subtoken_string(self, subtoken):
        """Converts a subtoken integer ID to a subtoken string."""
        if 0 <= subtoken < self.vocab_size:
            return self._all_subtoken_strings[subtoken]
        return u""

    def _escaped_token_to_subtoken_strings(self, escaped_token):
        """Converts an escaped token string to a list of subtoken strings.

        Args:
            escaped_token: An escaped token as a unicode string.
        Returns:
            A list of subtokens as unicode strings.
        """
        # NOTE: This algorithm is greedy; it won't necessarily produce the "best"
        # list of subtokens.
        ret = []
        start = 0
        token_len = len(escaped_token)
        while start < token_len:
            for end in range(
                    min(token_len, start + self._max_subtoken_len), start, -1):
                subtoken = escaped_token[start:end]
                if subtoken in self._subtoken_string_to_id:
                    ret.append(subtoken)
                    start = end
                    break

            else:  # Did not break
                # If there is no possible encoding of the escaped token then one of the
                # characters in the token is not in the alphabet. This should be
                # impossible and would be indicative of a bug.
                assert False, "Token substring not found in subtoken vocabulary."

        return ret

    def _escaped_token_to_subtoken_ids(self, escaped_token):
        """Converts an escaped token string to a list of subtoken IDs.

        Args:
            escaped_token: An escaped token as a unicode string.
        Returns:
            A list of subtoken IDs as integers.
        """
        return [
                self._subtoken_string_to_id[subtoken]
                for subtoken in self._escaped_token_to_subtoken_strings(escaped_token)
        ]

    @classmethod
    def build_from_generator(cls,
                                                     generator,
                                                     target_size,
                                                     max_subtoken_length=None,
                                                     reserved_tokens=None):
        """Builds a SubwordTextEncoder from the generated text.

        Args:
            generator: yields text.
            target_size: int, approximate vocabulary size to create.
            max_subtoken_length: Maximum length of a subtoken. If this is not set,
                then the runtime and memory use of creating the vocab is quadratic in
                the length of the longest token. If this is set, then it is instead
                O(max_subtoken_length * length of longest token).
            reserved_tokens: List of reserved tokens. The global variable
                `RESERVED_TOKENS` must be a prefix of `reserved_tokens`. If this
                argument is `None`, it will use `RESERVED_TOKENS`.

        Returns:
            SubwordTextEncoder with `vocab_size` approximately `target_size`.
        """
        token_counts = collections.defaultdict(int)
        for item in generator:
            for tok in tokenizer_encode(native_to_unicode(item)):
                token_counts[tok] += 1
        encoder = cls.build_to_target_size(
                target_size, token_counts, 1, 1e3,
                max_subtoken_length=max_subtoken_length,
                reserved_tokens=reserved_tokens)
        return encoder

    @classmethod
    def build_to_target_size(cls,
                                                     target_size,
                                                     token_counts,
                                                     min_val,
                                                     max_val,
                                                     max_subtoken_length=None,
                                                     reserved_tokens=None,
                                                     num_iterations=4):
        """Builds a SubwordTextEncoder that has `vocab_size` near `target_size`.

        Uses simple recursive binary search to find a minimum token count that most
        closely matches the `target_size`.

        Args:
            target_size: Desired vocab_size to approximate.
            token_counts: A dictionary of token counts, mapping string to int.
            min_val: An integer; lower bound for the minimum token count.
            max_val: An integer; upper bound for the minimum token count.
            max_subtoken_length: Maximum length of a subtoken. If this is not set,
                then the runtime and memory use of creating the vocab is quadratic in
                the length of the longest token. If this is set, then it is instead
                O(max_subtoken_length * length of longest token).
            reserved_tokens: List of reserved tokens. The global variable
                `RESERVED_TOKENS` must be a prefix of `reserved_tokens`. If this
                argument is `None`, it will use `RESERVED_TOKENS`.
            num_iterations: An integer; how many iterations of refinement.

        Returns:
            A SubwordTextEncoder instance.

        Raises:
            ValueError: If `min_val` is greater than `max_val`.
        """
        if min_val > max_val:
            raise ValueError("Lower bound for the minimum token count "
                                             "is greater than the upper bound.")
        if target_size < 1:
            raise ValueError("Target size must be positive.")

        if reserved_tokens is None:
            reserved_tokens = RESERVED_TOKENS

        def bisect(min_val, max_val):
            """Bisection to find the right size."""
            present_count = (max_val + min_val) // 2
            logger.info("Trying min_count %d" % present_count)
            subtokenizer = cls()
            subtokenizer.build_from_token_counts(
                    token_counts, present_count, num_iterations,
                    max_subtoken_length=max_subtoken_length,
                    reserved_tokens=reserved_tokens)

            # Being within 1% of the target size is ok.
            is_ok = abs(subtokenizer.vocab_size - target_size) * 100 < target_size
            # If min_val == max_val, we can't do any better than this.
            if is_ok or min_val >= max_val or present_count < 2:
                return subtokenizer

            if subtokenizer.vocab_size > target_size:
                other_subtokenizer = bisect(present_count + 1, max_val)
            else:
                other_subtokenizer = bisect(min_val, present_count - 1)

            if other_subtokenizer is None:
                return subtokenizer

            if (abs(other_subtokenizer.vocab_size - target_size) <
                    abs(subtokenizer.vocab_size - target_size)):
                return other_subtokenizer
            return subtokenizer

        return bisect(min_val, max_val)

    def build_from_token_counts(self,
                                                            token_counts,
                                                            min_count,
                                                            num_iterations=4,
                                                            reserved_tokens=None,
                                                            max_subtoken_length=None):
        """Train a SubwordTextEncoder based on a dictionary of word counts.

        Args:
            token_counts: a dictionary of Unicode strings to int.
            min_count: an integer - discard subtokens with lower counts.
            num_iterations: an integer.  how many iterations of refinement.
            reserved_tokens: List of reserved tokens. The global variable
                `RESERVED_TOKENS` must be a prefix of `reserved_tokens`. If this
                argument is `None`, it will use `RESERVED_TOKENS`.
            max_subtoken_length: Maximum length of a subtoken. If this is not set,
                then the runtime and memory use of creating the vocab is quadratic in
                the length of the longest token. If this is set, then it is instead
                O(max_subtoken_length * length of longest token).

        Raises:
            ValueError: if reserved is not 0 or len(RESERVED_TOKENS). In this case, it
                is not clear what the space is being reserved for, or when it will be
                filled in.
        """
        if reserved_tokens is None:
            reserved_tokens = RESERVED_TOKENS
        else:
            # There is not complete freedom in replacing RESERVED_TOKENS.
            for default, proposed in zip(RESERVED_TOKENS, reserved_tokens):
                if default != proposed:
                    raise ValueError("RESERVED_TOKENS must be a prefix of "
                                                     "reserved_tokens.")

        # Initialize the alphabet. Note, this must include reserved tokens or it can
        # result in encoding failures.
        alphabet_tokens = chain(six.iterkeys(token_counts),
                                                        [native_to_unicode(t) for t in reserved_tokens])

        self._init_alphabet_from_tokens(alphabet_tokens)

        # Bootstrap the initial list of subtokens with the characters from the
        # alphabet plus the escaping characters.
        self._init_subtokens_from_list(list(self._alphabet),
                                                                     reserved_tokens=reserved_tokens)

        # We build iteratively.  On each iteration, we segment all the words,
        # then count the resulting potential subtokens, keeping the ones
        # with high enough counts for our new vocabulary.
        if min_count < 1:
            min_count = 1
        for i in range(num_iterations):
            logger.info("Iteration {0}".format(i))

            # Collect all substrings of the encoded token that break along current
            # subtoken boundaries.
            subtoken_counts = collections.defaultdict(int)
            for token, count in six.iteritems(token_counts):
                iter_start_time = time.time()
                escaped_token = _escape_token(token, self._alphabet)
                subtokens = self._escaped_token_to_subtoken_strings(escaped_token)
                start = 0
                for subtoken in subtokens:
                    last_position = len(escaped_token) + 1
                    if max_subtoken_length is not None:
                        last_position = min(last_position, start + max_subtoken_length)

                    for end in range(start + 1, last_position):
                        new_subtoken = escaped_token[start:end]
                        subtoken_counts[new_subtoken] += count
                    start += len(subtoken)
                iter_time_secs = time.time() - iter_start_time
                if iter_time_secs > 0.1:
                    logger.info(u"Processing token [{0}] took {1} seconds, consider "
                                                    "setting Text2TextProblem.max_subtoken_length to a "
                                                    "smaller value.".format(token, iter_time_secs))

            # Array of sets of candidate subtoken strings, by length.
            len_to_subtoken_strings = []
            for subtoken_string, count in six.iteritems(subtoken_counts):
                lsub = len(subtoken_string)
                if count >= min_count:
                    while len(len_to_subtoken_strings) <= lsub:
                        len_to_subtoken_strings.append(set())
                    len_to_subtoken_strings[lsub].add(subtoken_string)

            # Consider the candidates longest to shortest, so that if we accept
            # a longer subtoken string, we can decrement the counts of its prefixes.
            new_subtoken_strings = []
            for lsub in range(len(len_to_subtoken_strings) - 1, 0, -1):
                subtoken_strings = len_to_subtoken_strings[lsub]
                for subtoken_string in subtoken_strings:
                    count = subtoken_counts[subtoken_string]
                    if count >= min_count:
                        # Exclude alphabet tokens here, as they must be included later,
                        # explicitly, regardless of count.
                        if subtoken_string not in self._alphabet:
                            new_subtoken_strings.append((count, subtoken_string))
                        for l in range(1, lsub):
                            subtoken_counts[subtoken_string[:l]] -= count

            # Include the alphabet explicitly to guarantee all strings are encodable.
            new_subtoken_strings.extend((subtoken_counts.get(a, 0), a)
                                                                    for a in self._alphabet)
            new_subtoken_strings.sort(reverse=True)

            # Reinitialize to the candidate vocabulary.
            new_subtoken_strings = [subtoken for _, subtoken in new_subtoken_strings]
            if reserved_tokens:
                escaped_reserved_tokens = [
                        _escape_token(native_to_unicode(t), self._alphabet)
                        for t in reserved_tokens
                ]
                new_subtoken_strings = escaped_reserved_tokens + new_subtoken_strings

            self._init_subtokens_from_list(new_subtoken_strings)
            logger.info("vocab_size = %d" % self.vocab_size)

    @property
    def all_subtoken_strings(self):
        return tuple(self._all_subtoken_strings)

    def dump(self):
        """Debugging dump of the current subtoken vocabulary."""
        subtoken_strings = [(i, s)
                                                for s, i in six.iteritems(self._subtoken_string_to_id)]
        print(u", ".join(u"{0} : '{1}'".format(i, s)
                                         for i, s in sorted(subtoken_strings)))

    def _init_subtokens_from_list(self, subtoken_strings, reserved_tokens=None):
        """Initialize token information from a list of subtoken strings.

        Args:
            subtoken_strings: a list of subtokens
            reserved_tokens: List of reserved tokens. We must have `reserved_tokens`
                as None or the empty list, or else the global variable `RESERVED_TOKENS`
                must be a prefix of `reserved_tokens`.

        Raises:
            ValueError: if reserved is not 0 or len(RESERVED_TOKENS). In this case, it
                is not clear what the space is being reserved for, or when it will be
                filled in.
        """
        if reserved_tokens is None:
            reserved_tokens = []

        if reserved_tokens:
            self._all_subtoken_strings = reserved_tokens + subtoken_strings
        else:
            self._all_subtoken_strings = subtoken_strings

        # we remember the maximum length of any subtoken to avoid having to
        # check arbitrarily long strings.
        self._max_subtoken_len = max([len(s) for s in subtoken_strings])
        self._subtoken_string_to_id = {
                s: i + len(reserved_tokens)
                for i, s in enumerate(subtoken_strings) if s
        }
        # Initialize the cache to empty.
        self._cache_size = 2 ** 20
        self._cache = [(None, None)] * self._cache_size

    def _init_alphabet_from_tokens(self, tokens):
        """Initialize alphabet from an iterable of token or subtoken strings."""
        # Include all characters from all tokens in the alphabet to guarantee that
        # any token can be encoded. Additionally, include all escaping characters.
        self._alphabet = {c for token in tokens for c in token}
        self._alphabet |= _ESCAPE_CHARS

    def _load_from_file_object(self, f):
        """Load from a file object.

        Args:
            f: File object to load vocabulary from
        """
        subtoken_strings = []
        for line in f:
            s = line.strip()
            # Some vocab files wrap words in single quotes, but others don't
            if ((s.startswith("'") and s.endswith("'")) or
                    (s.startswith("\"") and s.endswith("\""))):
                s = s[1:-1]
            subtoken_strings.append(native_to_unicode(s))
        self._init_subtokens_from_list(subtoken_strings)
        self._init_alphabet_from_tokens(subtoken_strings)

    def _load_from_file(self, filename):
        """Load from a vocab file."""
        if not os.path.isfile(filename):
            raise ValueError("File %s not found" % filename)
        with open(filename) as f:
            self._load_from_file_object(f)

    def store_to_file(self, filename, add_single_quotes=True):
        with open(filename, "w") as f:
            for subtoken_string in self._all_subtoken_strings:
                if add_single_quotes:
                    f.write("'" + unicode_to_native(subtoken_string) + "'\n")
                else:
                    f.write(unicode_to_native(subtoken_string) + "\n")


class DataProcessor():
    def __init__(self, bert_vocab_path, n_ctx=512):
        self.text_encoder = SubwordTextEncoder(bert_vocab_path)
        self.clf_token = self.text_encoder.decode('Clf')
        self.sep_token = self.text_encoder.decode('Sep')
        self.A_token = self.text_encoder.decode('A')
        self.B_token = self.text_encoder.decode('B')
        self.n_ctx = 512

    def encode_single_sentences(self, input_sentences: List[str]) -> np.array:
        """ Prepare a torch.Tensor of inputs for BERT model from a string.

        Args:
            input_sentences: list of single sentences (always considered as a sentence_A type)

        Return:
            Numpy array of formated inputs for BERT model
        """
        batch_size = len(input_sentences)
        input_array = np.zeros((batch_size, self.n_ctx, 3), dtype=np.int32)
        input_mask = np.zeros((batch_size, self.n_ctx), dtype=np.float32)
        i = 0
        for sentence in input_sentences:
            tokenized_sentence = self.text_encoder.encode(sentence)
            x1j = [self.clf_token] + tokenized_sentence
            lxj = len(x1j)
            input_array[i, :lxj, 0] = x1j
            input_array[i, :lxj, 0] = [self.A_token] * lxj
            input_array[i, :, 1] = np.arange(self.n_vocab+self.n_special, self.n_vocab+self.n_special+self.n_ctx)
            input_mask[i, :lxj] = 1
            i += 1
        return input_array, input_mask