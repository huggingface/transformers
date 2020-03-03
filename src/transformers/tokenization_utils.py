# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes for OpenAI GPT."""

import copy
import itertools
import json
import logging
import os
import re
from collections import defaultdict
from contextlib import contextmanager

from tokenizers.implementations import BaseTokenizer

from .file_utils import cached_path, hf_bucket_url, is_remote_url, is_tf_available, is_torch_available


if is_tf_available():
    import tensorflow as tf
if is_torch_available():
    import torch

logger = logging.getLogger(__name__)

SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
ADDED_TOKENS_FILE = "added_tokens.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"


@contextmanager
def truncate_and_pad(
    tokenizer: BaseTokenizer,
    max_length: int,
    stride: int,
    strategy: str,
    pad_to_max_length: bool,
    padding_side: str,
    pad_token_id: int,
    pad_token_type_id: int,
    pad_token: str,
):
    """
    This contextmanager is in charge of defining the truncation and the padding strategies and then
    restore the tokenizer settings afterwards.

    This contextmanager assumes the provider tokenizer has no padding / truncation strategy
    before the managed section. If your tokenizer set a padding / truncation strategy before,
    then it will be reset to no padding/truncation when exiting the managed section.

    :param tokenizer:
    :param max_length:
    :param stride:
    :param strategy:
    :param pad_to_max_length:
    :param padding_side:
    :param pad_token_id:
    :param pad_token_type_id:
    :param pad_token:
    :return:
    """

    # Handle all the truncation and padding stuff
    if max_length is not None:
        tokenizer.enable_truncation(max_length, stride=stride, strategy=strategy)

    if pad_to_max_length and (pad_token and pad_token_id >= 0):
        tokenizer.enable_padding(
            max_length=max_length,
            direction=padding_side,
            pad_id=pad_token_id,
            pad_type_id=pad_token_type_id,
            pad_token=pad_token,
        )
    elif pad_to_max_length:
        logger.warning(
            "Disabled padding because no padding token set (pad_token: {}, pad_token_id: {}).\n"
            "To remove this error, you can add a new pad token and then resize model embedding:\n"
            "\ttokenizer.pad_token = '<PAD>'\n\tmodel.resize_token_embeddings(len(tokenizer))".format(
                pad_token, pad_token_id
            )
        )

    yield

    if max_length is not None:
        tokenizer.no_truncation()

    if pad_to_max_length and (pad_token and pad_token_id >= 0):
        tokenizer.no_padding()


class PreTrainedTokenizer(object):
    """ Base class for all tokenizers.
    Handle all the shared methods for tokenization and special tokens as well as methods downloading/caching/loading pretrained tokenizers as well as adding tokens to the vocabulary.

    This class also contain the added tokens in a unified way on top of all tokenizers so we don't have to handle the specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece...).

    Class attributes (overridden by derived classes):

        - ``vocab_files_names``: a python ``dict`` with, as keys, the ``__init__`` keyword name of each vocabulary file required by the model, and as associated values, the filename for saving the associated file (string).
        - ``pretrained_vocab_files_map``: a python ``dict of dict`` the high-level keys being the ``__init__`` keyword name of each vocabulary file required by the model, the low-level being the `short-cut-names` (string) of the pretrained models with, as associated values, the `url` (string) to the associated pretrained vocabulary file.
        - ``max_model_input_sizes``: a python ``dict`` with, as keys, the `short-cut-names` (string) of the pretrained models, and as associated values, the maximum length of the sequence inputs of this model, or None if the model has no maximum input size.
        - ``pretrained_init_configuration``: a python ``dict`` with, as keys, the `short-cut-names` (string) of the pretrained models, and as associated values, a dictionnary of specific arguments to pass to the ``__init__``method of the tokenizer class for this pretrained model when loading the tokenizer with the ``from_pretrained()`` method.

    Parameters:

        - ``bos_token``: (`Optional`) string: a beginning of sentence token. Will be associated to ``self.bos_token`` and ``self.bos_token_id``

        - ``eos_token``: (`Optional`) string: an end of sentence token. Will be associated to ``self.eos_token`` and ``self.eos_token_id``

        - ``unk_token``: (`Optional`) string: an unknown token. Will be associated to ``self.unk_token`` and ``self.unk_token_id``

        - ``sep_token``: (`Optional`) string: a separation token (e.g. to separate context and query in an input sequence). Will be associated to ``self.sep_token`` and ``self.sep_token_id``

        - ``pad_token``: (`Optional`) string: a padding token. Will be associated to ``self.pad_token`` and ``self.pad_token_id``

        - ``cls_token``: (`Optional`) string: a classification token (e.g. to extract a summary of an input sequence leveraging self-attention along the full depth of the model). Will be associated to ``self.cls_token`` and ``self.cls_token_id``

        - ``mask_token``: (`Optional`) string: a masking token (e.g. when training a model with masked-language modeling). Will be associated to ``self.mask_token`` and ``self.mask_token_id``

        - ``additional_special_tokens``: (`Optional`) list: a list of additional special tokens. Adding all special tokens here ensure they won't be split by the tokenization process. Will be associated to ``self.additional_special_tokens`` and ``self.additional_special_tokens_ids``
    """

    vocab_files_names = {}
    pretrained_vocab_files_map = {}
    pretrained_init_configuration = {}
    max_model_input_sizes = {}

    SPECIAL_TOKENS_ATTRIBUTES = [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens",
    ]

    padding_side = "right"

    NO_PAD_TOKEN_FOR_BATCH_MSG = (
        "No padding token is set for this model, therefore no batch can be made with uneven "
        "sequences. Set a padding token or adjust the lengths of the sequences building the "
        "batch so that every sequence is of the same length."
    )

    UNEVEN_SEQUENCES_FOR_BATCH_MSG = (
        "The sequences building the batch are not of the same size, no tensor "
        "can be built. Set `pad_to_max_length=True` to pad the smaller sequences"
        "up to the larger sequence's length."
    )

    @property
    def bos_token(self):
        """ Beginning of sentence token (string). Log an error if used while not having been set. """
        if self._bos_token is None:
            logger.error("Using bos_token, but it is not set yet.")
        return self._bos_token

    @property
    def eos_token(self):
        """ End of sentence token (string). Log an error if used while not having been set. """
        if self._eos_token is None:
            logger.error("Using eos_token, but it is not set yet.")
        return self._eos_token

    @property
    def unk_token(self):
        """ Unknown token (string). Log an error if used while not having been set. """
        if self._unk_token is None:
            logger.error("Using unk_token, but it is not set yet.")
        return self._unk_token

    @property
    def sep_token(self):
        """ Separation token (string). E.g. separate context and query in an input sequence. Log an error if used while not having been set. """
        if self._sep_token is None:
            logger.error("Using sep_token, but it is not set yet.")
        return self._sep_token

    @property
    def pad_token(self):
        """ Padding token (string). Log an error if used while not having been set. """
        if self._pad_token is None:
            logger.error("Using pad_token, but it is not set yet.")
        return self._pad_token

    @property
    def cls_token(self):
        """ Classification token (string). E.g. to extract a summary of an input sequence leveraging self-attention along the full depth of the model. Log an error if used while not having been set. """
        if self._cls_token is None:
            logger.error("Using cls_token, but it is not set yet.")
        return self._cls_token

    @property
    def mask_token(self):
        """ Mask token (string). E.g. when training a model with masked-language modeling. Log an error if used while not having been set. """
        if self._mask_token is None:
            logger.error("Using mask_token, but it is not set yet.")
        return self._mask_token

    @property
    def additional_special_tokens(self):
        """ All the additional special tokens you may want to use (list of strings). Log an error if used while not having been set. """
        if self._additional_special_tokens is None:
            logger.error("Using additional_special_tokens, but it is not set yet.")
        return self._additional_special_tokens

    @bos_token.setter
    def bos_token(self, value):
        self._bos_token = value

    @eos_token.setter
    def eos_token(self, value):
        self._eos_token = value

    @unk_token.setter
    def unk_token(self, value):
        self._unk_token = value

    @sep_token.setter
    def sep_token(self, value):
        self._sep_token = value

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value

    @cls_token.setter
    def cls_token(self, value):
        self._cls_token = value

    @mask_token.setter
    def mask_token(self, value):
        self._mask_token = value

    @additional_special_tokens.setter
    def additional_special_tokens(self, value):
        self._additional_special_tokens = value

    @property
    def bos_token_id(self):
        """ Id of the beginning of sentence token in the vocabulary. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.bos_token)

    @property
    def eos_token_id(self):
        """ Id of the end of sentence token in the vocabulary. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.eos_token)

    @property
    def unk_token_id(self):
        """ Id of the unknown token in the vocabulary. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.unk_token)

    @property
    def sep_token_id(self):
        """ Id of the separation token in the vocabulary. E.g. separate context and query in an input sequence. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.sep_token)

    @property
    def pad_token_id(self):
        """ Id of the padding token in the vocabulary. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.pad_token)

    @property
    def pad_token_type_id(self):
        """ Id of the padding token type in the vocabulary."""
        return self._pad_token_type_id

    @property
    def cls_token_id(self):
        """ Id of the classification token in the vocabulary. E.g. to extract a summary of an input sequence leveraging self-attention along the full depth of the model. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.cls_token)

    @property
    def mask_token_id(self):
        """ Id of the mask token in the vocabulary. E.g. when training a model with masked-language modeling. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.mask_token)

    @property
    def additional_special_tokens_ids(self):
        """ Ids of all the additional special tokens in the vocabulary (list of integers). Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.additional_special_tokens)

    def get_vocab(self):
        """ Returns the vocabulary as a dict of {token: index} pairs. `tokenizer.get_vocab()[token]` is equivalent to `tokenizer.convert_tokens_to_ids(token)` when `token` is in the vocab. """
        raise NotImplementedError()

    def __init__(self, max_len=None, **kwargs):
        self._bos_token = None
        self._eos_token = None
        self._unk_token = None
        self._sep_token = None
        self._pad_token = None
        self._cls_token = None
        self._mask_token = None
        self._pad_token_type_id = 0
        self._additional_special_tokens = []

        self.max_len = max_len if max_len is not None else int(1e12)

        # Padding side is right by default and over-riden in subclasses. If specified in the kwargs, it is changed.
        self.padding_side = kwargs.pop("padding_side", self.padding_side)

        # Added tokens
        self.added_tokens_encoder = {}
        self.unique_added_tokens_encoder = set()
        self.added_tokens_decoder = {}

        # inputs and kwargs for saving and re-loading (see ``from_pretrained`` and ``save_pretrained``)
        self.init_inputs = ()
        self.init_kwargs = {}

        for key, value in kwargs.items():
            if key in self.SPECIAL_TOKENS_ATTRIBUTES:
                if key == "additional_special_tokens":
                    assert isinstance(value, (list, tuple)) and all(isinstance(t, str) for t in value)
                else:
                    assert isinstance(value, str)
                setattr(self, key, value)

    @classmethod
    def from_pretrained(cls, *inputs, **kwargs):
        r"""
        Instantiate a :class:`~transformers.PreTrainedTokenizer` (or a derived class) from a predefined tokenizer.

        Args:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a predefined tokenizer to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a predefined tokenizer that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing vocabulary files required by the tokenizer, for instance saved using the :func:`~transformers.PreTrainedTokenizer.save_pretrained` method, e.g.: ``./my_model_directory/``.
                - (not applicable to all derived classes, deprecated) a path or url to a single saved vocabulary file if and only if the tokenizer only requires a single vocabulary file (e.g. Bert, XLNet), e.g.: ``./my_model_directory/vocab.txt``.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded predefined tokenizer vocabulary files should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the vocabulary files and override the cached versions if they exists.

            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            inputs: (`optional`) positional arguments: will be passed to the Tokenizer ``__init__`` method.

            kwargs: (`optional`) keyword arguments: will be passed to the Tokenizer ``__init__`` method. Can be used to set special tokens like ``bos_token``, ``eos_token``, ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``, ``mask_token``, ``additional_special_tokens``. See parameters in the doc string of :class:`~transformers.PreTrainedTokenizer` for details.

        Examples::

            # We can't instantiate directly the base class `PreTrainedTokenizer` so let's show our examples on a derived class: BertTokenizer

            # Download vocabulary from S3 and cache.
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

            # Download vocabulary from S3 (user-uploaded) and cache.
            tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-german-cased')

            # If vocabulary files are in a directory (e.g. tokenizer was saved using `save_pretrained('./test/saved_model/')`)
            tokenizer = BertTokenizer.from_pretrained('./test/saved_model/')

            # If the tokenizer uses a single vocabulary file, you can point directly to this file
            tokenizer = BertTokenizer.from_pretrained('./test/saved_model/my_vocab.txt')

            # You can link tokens to special vocabulary when instantiating
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', unk_token='<unk>')
            # You should be sure '<unk>' is in the vocabulary when doing that.
            # Otherwise use tokenizer.add_special_tokens({'unk_token': '<unk>'}) instead)
            assert tokenizer.unk_token == '<unk>'

        """
        return cls._from_pretrained(*inputs, **kwargs)

    @classmethod
    def _from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs):
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)

        s3_models = list(cls.max_model_input_sizes.keys())
        vocab_files = {}
        init_configuration = {}
        if pretrained_model_name_or_path in s3_models:
            # Get the vocabulary from AWS S3 bucket
            for file_id, map_list in cls.pretrained_vocab_files_map.items():
                vocab_files[file_id] = map_list[pretrained_model_name_or_path]
            if (
                cls.pretrained_init_configuration
                and pretrained_model_name_or_path in cls.pretrained_init_configuration
            ):
                init_configuration = cls.pretrained_init_configuration[pretrained_model_name_or_path].copy()
        else:
            # Get the vocabulary from local files
            logger.info(
                "Model name '{}' not found in model shortcut name list ({}). "
                "Assuming '{}' is a path, a model identifier, or url to a directory containing tokenizer files.".format(
                    pretrained_model_name_or_path, ", ".join(s3_models), pretrained_model_name_or_path
                )
            )

            if os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
                if len(cls.vocab_files_names) > 1:
                    raise ValueError(
                        "Calling {}.from_pretrained() with the path to a single file or url is not supported."
                        "Use a model identifier or the path to a directory instead.".format(cls.__name__)
                    )
                logger.warning(
                    "Calling {}.from_pretrained() with the path to a single file or url is deprecated".format(
                        cls.__name__
                    )
                )
                file_id = list(cls.vocab_files_names.keys())[0]
                vocab_files[file_id] = pretrained_model_name_or_path
            else:
                # At this point pretrained_model_name_or_path is either a directory or a model identifier name
                additional_files_names = {
                    "added_tokens_file": ADDED_TOKENS_FILE,
                    "special_tokens_map_file": SPECIAL_TOKENS_MAP_FILE,
                    "tokenizer_config_file": TOKENIZER_CONFIG_FILE,
                }
                # Look for the tokenizer main vocabulary files + the additional tokens files
                for file_id, file_name in {**cls.vocab_files_names, **additional_files_names}.items():
                    if os.path.isdir(pretrained_model_name_or_path):
                        full_file_name = os.path.join(pretrained_model_name_or_path, file_name)
                        if not os.path.exists(full_file_name):
                            logger.info("Didn't find file {}. We won't load it.".format(full_file_name))
                            full_file_name = None
                    else:
                        full_file_name = hf_bucket_url(pretrained_model_name_or_path, postfix=file_name)

                    vocab_files[file_id] = full_file_name

        # Get files from url, cache, or disk depending on the case
        try:
            resolved_vocab_files = {}
            for file_id, file_path in vocab_files.items():
                if file_path is None:
                    resolved_vocab_files[file_id] = None
                else:
                    resolved_vocab_files[file_id] = cached_path(
                        file_path,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        resume_download=resume_download,
                        local_files_only=local_files_only,
                    )
        except EnvironmentError:
            if pretrained_model_name_or_path in s3_models:
                msg = "Couldn't reach server at '{}' to download vocabulary files."
            else:
                msg = (
                    "Model name '{}' was not found in tokenizers model name list ({}). "
                    "We assumed '{}' was a path or url to a directory containing vocabulary files "
                    "named {}, but couldn't find such vocabulary files at this path or url.".format(
                        pretrained_model_name_or_path,
                        ", ".join(s3_models),
                        pretrained_model_name_or_path,
                        list(cls.vocab_files_names.values()),
                    )
                )

            raise EnvironmentError(msg)

        if all(full_file_name is None for full_file_name in resolved_vocab_files.values()):
            raise EnvironmentError(
                "Model name '{}' was not found in tokenizers model name list ({}). "
                "We assumed '{}' was a path, a model identifier, or url to a directory containing vocabulary files "
                "named {} but couldn't find such vocabulary files at this path or url.".format(
                    pretrained_model_name_or_path,
                    ", ".join(s3_models),
                    pretrained_model_name_or_path,
                    list(cls.vocab_files_names.values()),
                )
            )

        for file_id, file_path in vocab_files.items():
            if file_path == resolved_vocab_files[file_id]:
                logger.info("loading file {}".format(file_path))
            else:
                logger.info("loading file {} from cache at {}".format(file_path, resolved_vocab_files[file_id]))

        # Prepare tokenizer initialization kwargs
        # Did we saved some inputs and kwargs to reload ?
        tokenizer_config_file = resolved_vocab_files.pop("tokenizer_config_file", None)
        if tokenizer_config_file is not None:
            with open(tokenizer_config_file, encoding="utf-8") as tokenizer_config_handle:
                init_kwargs = json.load(tokenizer_config_handle)
            saved_init_inputs = init_kwargs.pop("init_inputs", ())
            if not init_inputs:
                init_inputs = saved_init_inputs
        else:
            init_kwargs = init_configuration

        # Update with newly provided kwargs
        init_kwargs.update(kwargs)

        # Set max length if needed
        if pretrained_model_name_or_path in cls.max_model_input_sizes:
            # if we're using a pretrained model, ensure the tokenizer
            # wont index sequences longer than the number of positional embeddings
            max_len = cls.max_model_input_sizes[pretrained_model_name_or_path]
            if max_len is not None and isinstance(max_len, (int, float)):
                init_kwargs["max_len"] = min(init_kwargs.get("max_len", int(1e12)), max_len)

        # Merge resolved_vocab_files arguments in init_kwargs.
        added_tokens_file = resolved_vocab_files.pop("added_tokens_file", None)
        special_tokens_map_file = resolved_vocab_files.pop("special_tokens_map_file", None)
        for args_name, file_path in resolved_vocab_files.items():
            if args_name not in init_kwargs:
                init_kwargs[args_name] = file_path
        if special_tokens_map_file is not None:
            with open(special_tokens_map_file, encoding="utf-8") as special_tokens_map_handle:
                special_tokens_map = json.load(special_tokens_map_handle)
            for key, value in special_tokens_map.items():
                if key not in init_kwargs:
                    init_kwargs[key] = value

        # Instantiate tokenizer.
        try:
            tokenizer = cls(*init_inputs, **init_kwargs)
        except OSError:
            raise OSError(
                "Unable to load vocabulary from file. "
                "Please check that the provided vocabulary is accessible and not corrupted."
            )

        # Save inputs and kwargs for saving and re-loading with ``save_pretrained``
        tokenizer.init_inputs = init_inputs
        tokenizer.init_kwargs = init_kwargs

        # update unique_added_tokens_encoder with special tokens for correct tokenization
        tokenizer.unique_added_tokens_encoder.update(set(tokenizer.all_special_tokens))

        # Add supplementary tokens.
        if added_tokens_file is not None:
            with open(added_tokens_file, encoding="utf-8") as added_tokens_handle:
                added_tok_encoder = json.load(added_tokens_handle)
            added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
            tokenizer.added_tokens_encoder.update(added_tok_encoder)
            tokenizer.added_tokens_decoder.update(added_tok_decoder)
            tokenizer.unique_added_tokens_encoder.update(set(tokenizer.added_tokens_encoder.keys()))

        return tokenizer

    def save_pretrained(self, save_directory):
        """ Save the tokenizer vocabulary files together with:
                - added tokens,
                - special-tokens-to-class-attributes-mapping,
                - tokenizer instantiation positional and keywords inputs (e.g. do_lower_case for Bert).

            This won't save modifications other than (added tokens and special token mapping) you may have
            applied to the tokenizer after the instantiation (e.g. modifying tokenizer.do_lower_case after creation).

            This method make sure the full tokenizer can then be re-loaded using the :func:`~transformers.PreTrainedTokenizer.from_pretrained` class method.
        """
        if not os.path.isdir(save_directory):
            logger.error("Saving directory ({}) should be a directory".format(save_directory))
            return

        special_tokens_map_file = os.path.join(save_directory, SPECIAL_TOKENS_MAP_FILE)
        added_tokens_file = os.path.join(save_directory, ADDED_TOKENS_FILE)
        tokenizer_config_file = os.path.join(save_directory, TOKENIZER_CONFIG_FILE)

        tokenizer_config = copy.deepcopy(self.init_kwargs)
        if len(self.init_inputs) > 0:
            tokenizer_config["init_inputs"] = copy.deepcopy(self.init_inputs)
        for file_id in self.vocab_files_names.keys():
            tokenizer_config.pop(file_id, None)

        with open(tokenizer_config_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(tokenizer_config, ensure_ascii=False))

        with open(special_tokens_map_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.special_tokens_map, ensure_ascii=False))

        if len(self.added_tokens_encoder) > 0:
            with open(added_tokens_file, "w", encoding="utf-8") as f:
                out_str = json.dumps(self.added_tokens_encoder, ensure_ascii=False)
                f.write(out_str)

        vocab_files = self.save_vocabulary(save_directory)

        return vocab_files + (special_tokens_map_file, added_tokens_file)

    def save_vocabulary(self, save_directory):
        """ Save the tokenizer vocabulary to a directory. This method does *NOT* save added tokens
            and special token mappings.

            Please use :func:`~transformers.PreTrainedTokenizer.save_pretrained` `()` to save the full Tokenizer state if you want to reload it using the :func:`~transformers.PreTrainedTokenizer.from_pretrained` class method.
        """
        raise NotImplementedError

    def vocab_size(self):
        """ Size of the base vocabulary (without the added tokens) """
        raise NotImplementedError

    def __len__(self):
        """ Size of the full vocabulary with the added tokens """
        return self.vocab_size + len(self.added_tokens_encoder)

    def add_tokens(self, new_tokens):
        """
        Add a list of new tokens to the tokenizer class. If the new tokens are not in the
        vocabulary, they are added to it with indices starting from length of the current vocabulary.

        Args:
            new_tokens: string or list of string. Each string is a token to add. Tokens are only added if they are not already in the vocabulary (tested by checking if the tokenizer assign the index of the ``unk_token`` to them).

        Returns:
            Number of tokens added to the vocabulary.

        Examples::

            # Let's see how to increase the vocabulary of Bert model and tokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased')

            num_added_toks = tokenizer.add_tokens(['new_tok1', 'my_new-tok2'])
            print('We have added', num_added_toks, 'tokens')
            model.resize_token_embeddings(len(tokenizer))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
        """
        if not new_tokens:
            return 0

        if not isinstance(new_tokens, list):
            new_tokens = [new_tokens]

        to_add_tokens = []
        for token in new_tokens:
            assert isinstance(token, str)
            if self.init_kwargs.get("do_lower_case", False) and token not in self.all_special_tokens:
                token = token.lower()
            if (
                token != self.unk_token
                and self.convert_tokens_to_ids(token) == self.convert_tokens_to_ids(self.unk_token)
                and token not in to_add_tokens
            ):
                to_add_tokens.append(token)
                logger.info("Adding %s to the vocabulary", token)

        added_tok_encoder = dict((tok, len(self) + i) for i, tok in enumerate(to_add_tokens))
        added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
        self.added_tokens_encoder.update(added_tok_encoder)
        self.unique_added_tokens_encoder = set(self.added_tokens_encoder.keys()).union(set(self.all_special_tokens))
        self.added_tokens_decoder.update(added_tok_decoder)

        return len(to_add_tokens)

    def num_added_tokens(self, pair=False):
        """
        Returns the number of added tokens when encoding a sequence with special tokens.

        Note:
            This encodes inputs and checks the number of added tokens, and is therefore not efficient. Do not put this
            inside your training loop.

        Args:
            pair: Returns the number of added tokens in the case of a sequence pair if set to True, returns the
                number of added tokens in the case of a single sequence if set to False.

        Returns:
            Number of tokens added to sequences
        """
        token_ids_0 = []
        token_ids_1 = []
        return len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1 if pair else None))

    def add_special_tokens(self, special_tokens_dict):
        """
        Add a dictionary of special tokens (eos, pad, cls...) to the encoder and link them
        to class attributes. If special tokens are NOT in the vocabulary, they are added
        to it (indexed starting from the last index of the current vocabulary).

        Using `add_special_tokens` will ensure your special tokens can be used in several ways:

        - special tokens are carefully handled by the tokenizer (they are never split)
        - you can easily refer to special tokens using tokenizer class attributes like `tokenizer.cls_token`. This makes it easy to develop model-agnostic training and fine-tuning scripts.

        When possible, special tokens are already registered for provided pretrained models (ex: BertTokenizer cls_token is already registered to be '[CLS]' and XLM's one is also registered to be '</s>')

        Args:
            special_tokens_dict: dict of string. Keys should be in the list of predefined special attributes:
                [``bos_token``, ``eos_token``, ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``, ``mask_token``,
                ``additional_special_tokens``].

                Tokens are only added if they are not already in the vocabulary (tested by checking if the tokenizer assign the index of the ``unk_token`` to them).

        Returns:
            Number of tokens added to the vocabulary.

        Examples::

            # Let's see how to add a new classification token to GPT-2
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT2Model.from_pretrained('gpt2')

            special_tokens_dict = {'cls_token': '<CLS>'}

            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            print('We have added', num_added_toks, 'tokens')
            model.resize_token_embeddings(len(tokenizer))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.

            assert tokenizer.cls_token == '<CLS>'
        """
        if not special_tokens_dict:
            return 0

        added_tokens = 0
        for key, value in special_tokens_dict.items():
            assert key in self.SPECIAL_TOKENS_ATTRIBUTES
            if key == "additional_special_tokens":
                assert isinstance(value, (list, tuple)) and all(isinstance(t, str) for t in value)
                added_tokens += self.add_tokens(value)
            else:
                assert isinstance(value, str)
                added_tokens += self.add_tokens([value])
            logger.info("Assigning %s to the %s key of the tokenizer", value, key)
            setattr(self, key, value)

        return added_tokens

    def tokenize(self, text, **kwargs):
        """ Converts a string in a sequence of tokens (string), using the tokenizer.
            Split in words for word-based vocabulary or sub-words for sub-word-based
            vocabularies (BPE/SentencePieces/WordPieces).

            Take care of added tokens.

            text: The sequence to be encoded.
            add_prefix_space: Only applies to GPT-2 and RoBERTa tokenizers. When `True`, this ensures that the sequence
                begins with an empty space. False by default except for when using RoBERTa with `add_special_tokens=True`.
            **kwargs: passed to the `prepare_for_tokenization` preprocessing method.
        """
        all_special_tokens = self.all_special_tokens
        text = self.prepare_for_tokenization(text, **kwargs)

        def lowercase_text(t):
            # convert non-special tokens to lowercase
            escaped_special_toks = [re.escape(s_tok) for s_tok in all_special_tokens]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            return re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), t)

        if self.init_kwargs.get("do_lower_case", False):
            text = lowercase_text(text)

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                sub_text = sub_text.rstrip()
                if i == 0 and not sub_text:
                    result += [tok]
                elif i == len(split_text) - 1:
                    if sub_text:
                        result += [sub_text]
                    else:
                        pass
                else:
                    if sub_text:
                        result += [sub_text]
                    result += [tok]
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []
            if not tok_list:
                return self._tokenize(text)

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_added_tokens_encoder:
                        tokenized_text += split_on_token(tok, sub_text)
                    else:
                        tokenized_text += [sub_text]
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token) if token not in self.unique_added_tokens_encoder else [token]
                        for token in tokenized_text
                    )
                )
            )

        added_tokens = self.unique_added_tokens_encoder
        tokenized_text = split_on_tokens(added_tokens, text)
        return tokenized_text

    def _tokenize(self, text, **kwargs):
        """ Converts a string in a sequence of tokens (string), using the tokenizer.
            Split in words for word-based vocabulary or sub-words for sub-word-based
            vocabularies (BPE/SentencePieces/WordPieces).

            Do NOT take care of added tokens.
        """
        raise NotImplementedError

    def convert_tokens_to_ids(self, tokens):
        """ Converts a single token, or a sequence of tokens, (str) in a single integer id
            (resp. a sequence of ids), using the vocabulary.
        """
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        return ids

    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None

        if token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]
        return self._convert_token_to_id(token)

    def _convert_token_to_id(self, token):
        raise NotImplementedError

    def encode(
        self,
        text,
        text_pair=None,
        add_special_tokens=True,
        max_length=None,
        stride=0,
        truncation_strategy="longest_first",
        pad_to_max_length=False,
        return_tensors=None,
        **kwargs
    ):
        """
        Converts a string in a sequence of ids (integer), using the tokenizer and vocabulary.

        Same as doing ``self.convert_tokens_to_ids(self.tokenize(text))``.

        Args:
            text: The first sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method)
            text_pair: Optional second sequence to be encoded. This can be a string, a list of strings (tokenized
                string using the `tokenize` method) or a list of integers (tokenized string ids using the
                `convert_tokens_to_ids` method)
            add_special_tokens: if set to ``True``, the sequences will be encoded with the special tokens relative
                to their model.
            max_length: if set to a number, will limit the total sequence returned so that it has a maximum length.
                If there are overflowing tokens, those will be added to the returned dictionary
            stride: if set to a number along with max_length, the overflowing tokens returned will contain some tokens
                from the main sequence returned. The value of this argument defines the number of additional tokens.
            truncation_strategy: string selected in the following options:
                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
                    starting from the longest one at each token (when there is a pair of input sequences)
                - 'only_first': Only truncate the first sequence
                - 'only_second': Only truncate the second sequence
                - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_length)
            pad_to_max_length: if set to True, the returned sequences will be padded according to the model's padding side and
                padding index, up to their max length. If no max length is specified, the padding is done up to the model's max length.
                The tokenizer padding sides are handled by the class attribute `padding_side` which can be set to the following strings:
                - 'left': pads on the left of the sequences
                - 'right': pads on the right of the sequences
                Defaults to False: no padding.
            return_tensors: (optional) can be set to 'tf' or 'pt' to return respectively TensorFlow tf.constant
                or PyTorch torch.Tensor instead of a list of python integers.
            add_prefix_space: Only applies to GPT-2 and RoBERTa tokenizers. When `True`, this ensures that the sequence
                begins with an empty space. False by default except for when using RoBERTa with `add_special_tokens=True`.
            **kwargs: passed to the `self.tokenize()` method
        """
        encoded_inputs = self.encode_plus(
            text,
            text_pair=text_pair,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            stride=stride,
            truncation_strategy=truncation_strategy,
            pad_to_max_length=pad_to_max_length,
            return_tensors=return_tensors,
            **kwargs,
        )

        return encoded_inputs["input_ids"]

    def encode_plus(
        self,
        text,
        text_pair=None,
        add_special_tokens=True,
        max_length=None,
        stride=0,
        truncation_strategy="longest_first",
        pad_to_max_length=False,
        return_tensors=None,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        return_offsets_mapping=False,
        **kwargs
    ):
        """
        Returns a dictionary containing the encoded sequence or sequence pair and additional informations:
        the mask for sequence classification and the overflowing elements if a ``max_length`` is specified.

        Args:
            text: The first sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method)
            text_pair: Optional second sequence to be encoded. This can be a string, a list of strings (tokenized
                string using the `tokenize` method) or a list of integers (tokenized string ids using the
                `convert_tokens_to_ids` method)
            add_special_tokens: if set to ``True``, the sequences will be encoded with the special tokens relative
                to their model.
            max_length: if set to a number, will limit the total sequence returned so that it has a maximum length.
                If there are overflowing tokens, those will be added to the returned dictionary
            stride: if set to a number along with max_length, the overflowing tokens returned will contain some tokens
                from the main sequence returned. The value of this argument defines the number of additional tokens.
            truncation_strategy: string selected in the following options:
                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
                    starting from the longest one at each token (when there is a pair of input sequences)
                - 'only_first': Only truncate the first sequence
                - 'only_second': Only truncate the second sequence
                - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_length)
            pad_to_max_length: if set to True, the returned sequences will be padded according to the model's padding side and
                padding index, up to their max length. If no max length is specified, the padding is done up to the model's max length.
                The tokenizer padding sides are handled by the class attribute `padding_side` which can be set to the following strings:
                - 'left': pads on the left of the sequences
                - 'right': pads on the right of the sequences
                Defaults to False: no padding.
            return_tensors: (optional) can be set to 'tf' or 'pt' to return respectively TensorFlow tf.constant
                or PyTorch torch.Tensor instead of a list of python integers.
            add_prefix_space: Only applies to GPT-2 and RoBERTa tokenizers. When `True`, this ensures that the sequence
                begins with an empty space. False by default except for when using RoBERTa with `add_special_tokens=True`.
            return_token_type_ids: (optional) Set to False to avoid returning token_type_ids (default True).
            return_attention_mask: (optional) Set to False to avoid returning attention mask (default True)
            return_overflowing_tokens: (optional) Set to True to return overflowing token information (default False).
            return_special_tokens_mask: (optional) Set to True to return special tokens mask information (default False).
            return_offsets_mapping: (optional) Set to True to return (char_start, char_end) for each token (default False).
                If using Python's tokenizer, this method will raise NotImplementedError. This one is only available on
                Rust-based tokenizers inheriting from PreTrainedTokenizerFast.
            **kwargs: passed to the `self.tokenize()` method

        Return:
            A Dictionary of shape::

                {
                    input_ids: list[int],
                    token_type_ids: list[int] if return_token_type_ids is True (default)
                    attention_mask: list[int] if return_attention_mask is True (default)
                    overflowing_tokens: list[int] if a ``max_length`` is specified and return_overflowing_tokens is True
                    num_truncated_tokens: int if a ``max_length`` is specified and return_overflowing_tokens is True
                    special_tokens_mask: list[int] if ``add_special_tokens`` if set to ``True`` and return_special_tokens_mask is True
                }

            With the fields:
                ``input_ids``: list of token ids to be fed to a model
                ``token_type_ids``: list of token type ids to be fed to a model
                ``attention_mask``: list of indices specifying which tokens should be attended to by the model
                ``overflowing_tokens``: list of overflowing tokens if a max length is specified.
                ``num_truncated_tokens``: number of overflowing tokens a ``max_length`` is specified
                ``special_tokens_mask``: if adding special tokens, this is a list of [0, 1], with 0 specifying special added
                tokens and 1 specifying sequence tokens.
        """

        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text, add_special_tokens=add_special_tokens, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers."
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        # Throw an error if we can pad because there is no padding token
        if pad_to_max_length and self.pad_token_id is None:
            raise ValueError(
                "Unable to set proper padding strategy as the tokenizer does not have a padding token. In this case please set the `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` or add a new pad token via the function add_special_tokens if you want to use a padding strategy"
            )

        first_ids = get_input_ids(text)
        second_ids = get_input_ids(text_pair) if text_pair is not None else None

        return self.prepare_for_model(
            first_ids,
            pair_ids=second_ids,
            max_length=max_length,
            pad_to_max_length=pad_to_max_length,
            add_special_tokens=add_special_tokens,
            stride=stride,
            truncation_strategy=truncation_strategy,
            return_tensors=return_tensors,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
        )

    def batch_encode_plus(
        self,
        batch_text_or_text_pairs=None,
        add_special_tokens=True,
        max_length=None,
        stride=0,
        truncation_strategy="longest_first",
        pad_to_max_length=False,
        return_tensors=None,
        return_token_type_ids=True,
        return_attention_masks=True,
        return_overflowing_tokens=False,
        return_special_tokens_masks=False,
        return_offsets_mapping=False,
        return_input_lengths=False,
        **kwargs
    ):
        """
        Returns a dictionary containing the encoded sequence or sequence pair and additional information:
        the mask for sequence classification and the overflowing elements if a ``max_length`` is specified.

        Args:
            batch_text_or_text_pairs: Batch of sequences or pair of sequences to be encoded.
                This can be a list of string/string-sequences/int-sequences or a list of pair of
                string/string-sequences/int-sequence (see details in encode_plus)
            add_special_tokens: if set to ``True``, the sequences will be encoded with the special tokens relative
                to their model.
            max_length: if set to a number, will limit the total sequence returned so that it has a maximum length.
                If there are overflowing tokens, those will be added to the returned dictionary`
            stride: if set to a number along with max_length, the overflowing tokens returned will contain some tokens
                from the main sequence returned. The value of this argument defines the number of additional tokens.
            truncation_strategy: string selected in the following options:
                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
                    starting from the longest one at each token (when there is a pair of input sequences)
                - 'only_first': Only truncate the first sequence
                - 'only_second': Only truncate the second sequence
                - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_length)
            pad_to_max_length: if set to True, the returned sequences will be padded according to the model's padding side and
                padding index, up to their max length. If no max length is specified, the padding is done up to the model's max length.
                The tokenizer padding sides are handled by the class attribute `padding_side` which can be set to the following strings:
                - 'left': pads on the left of the sequences
                - 'right': pads on the right of the sequences
                Defaults to False: no padding.
            return_tensors: (optional) can be set to 'tf' or 'pt' to return respectively TensorFlow tf.constant
                or PyTorch torch.Tensor instead of a list of python integers.
            return_input_lengths: (optional) If set the resulting dictionary will include the length of each sample
            return_attention_masks: (optional) Set to True to return the attention mask (default False)
            return_offsets_mapping: (optional) Not available, should be set to False or it will throw NotImplementError
            **kwargs: passed to the `self.tokenize()` method

        Return:
            A Dictionary of shape::

                {
                    input_ids: list[List[int]],
                    token_type_ids: list[List[int]] if return_token_type_ids is True (default)
                    attention_mask: list[List[int]] if return_attention_mask is True (default)
                    overflowing_tokens: list[List[int]] if a ``max_length`` is specified and return_overflowing_tokens is True
                    num_truncated_tokens: List[int] if a ``max_length`` is specified and return_overflowing_tokens is True
                    special_tokens_mask: list[List[int]] if ``add_special_tokens`` if set to ``True`` and return_special_tokens_mask is True
                }

            With the fields:
                ``input_ids``: list of token ids to be fed to a model
                ``token_type_ids``: list of token type ids to be fed to a model
                ``attention_mask``: list of indices specifying which tokens should be attended to by the model
                ``overflowing_tokens``: list of overflowing tokens if a max length is specified.
                ``num_truncated_tokens``: number of overflowing tokens a ``max_length`` is specified
                ``special_tokens_mask``: if adding special tokens, this is a list of [0, 1], with 0 specifying special added
                tokens and 1 specifying sequence tokens.
        """

        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text, add_special_tokens=add_special_tokens, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        # Throw an error if we can pad because there is no padding token
        if pad_to_max_length and self.pad_token_id is None:
            raise ValueError(
                "Unable to set proper padding strategy as the tokenizer does not have a padding token. In this case please set the `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` or add a new pad token via the function add_special_tokens if you want to use a padding strategy"
            )

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers."
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        input_ids = []
        for ids_or_pair_ids in batch_text_or_text_pairs:
            if isinstance(ids_or_pair_ids, (list, tuple)) and len(ids_or_pair_ids) == 2:
                ids, pair_ids = ids_or_pair_ids
            else:
                ids, pair_ids = ids_or_pair_ids, None

            first_ids = get_input_ids(ids)
            second_ids = get_input_ids(pair_ids) if pair_ids is not None else None
            input_ids.append((first_ids, second_ids))

        if max_length is None and pad_to_max_length:

            def total_sequence_length(input_pairs):
                first_ids, second_ids = input_pairs
                return len(first_ids) + (
                    self.num_added_tokens()
                    if second_ids is None
                    else (len(second_ids) + self.num_added_tokens(pair=True))
                )

            max_length = max([total_sequence_length(ids) for ids in input_ids])

        batch_outputs = {}
        for first_ids, second_ids in input_ids:
            # Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by
            # the model. It adds special tokens, truncates sequences if overflowing while taking into account
            # the special tokens and manages a window stride for overflowing tokens
            outputs = self.prepare_for_model(
                first_ids,
                pair_ids=second_ids,
                max_length=max_length,
                pad_to_max_length=pad_to_max_length,
                add_special_tokens=add_special_tokens,
                stride=stride,
                truncation_strategy=truncation_strategy,
                return_attention_mask=return_attention_masks,
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_masks,
            )

            # Append the non-padded length to the output
            if return_input_lengths:
                outputs["input_len"] = len(outputs["input_ids"])

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        if return_tensors is not None:

            # Do the tensor conversion in batch
            for key, value in batch_outputs.items():
                if return_tensors == "tf" and is_tf_available():
                    try:
                        batch_outputs[key] = tf.constant(value)
                    except ValueError:
                        if None in [item for sequence in value for item in sequence]:
                            raise ValueError(self.NO_PAD_TOKEN_FOR_BATCH_MSG)
                        else:
                            raise ValueError(self.UNEVEN_SEQUENCES_FOR_BATCH_MSG)
                elif return_tensors == "pt" and is_torch_available():
                    try:
                        batch_outputs[key] = torch.tensor(value)
                    except ValueError:
                        raise ValueError(self.UNEVEN_SEQUENCES_FOR_BATCH_MSG)
                    except RuntimeError:
                        if None in [item for sequence in value for item in sequence]:
                            raise ValueError(self.NO_PAD_TOKEN_FOR_BATCH_MSG)
                        else:
                            raise
                elif return_tensors is not None:
                    logger.warning(
                        "Unable to convert output to tensors format {}, PyTorch or TensorFlow is not available.".format(
                            return_tensors
                        )
                    )

        return batch_outputs

    def prepare_for_model(
        self,
        ids,
        pair_ids=None,
        max_length=None,
        add_special_tokens=True,
        stride=0,
        truncation_strategy="longest_first",
        pad_to_max_length=False,
        return_tensors=None,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
    ):
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model.
        It adds special tokens, truncates
        sequences if overflowing while taking into account the special tokens and manages a window stride for
        overflowing tokens

        Args:
            ids: list of tokenized input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            pair_ids: Optional second list of input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            max_length: maximum length of the returned list. Will truncate by taking into account the special tokens.
            add_special_tokens: if set to ``True``, the sequences will be encoded with the special tokens relative
                to their model.
            stride: window stride for overflowing tokens. Can be useful for edge effect removal when using sequential
                list of inputs.
            truncation_strategy: string selected in the following options:
                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
                    starting from the longest one at each token (when there is a pair of input sequences)
                - 'only_first': Only truncate the first sequence
                - 'only_second': Only truncate the second sequence
                - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_length)
            pad_to_max_length: if set to True, the returned sequences will be padded according to the model's padding side and
                padding index, up to their max length. If no max length is specified, the padding is done up to the model's max length.
                The tokenizer padding sides are handled by the following strings:
                - 'left': pads on the left of the sequences
                - 'right': pads on the right of the sequences
                Defaults to False: no padding.
            return_tensors: (optional) can be set to 'tf' or 'pt' to return respectively TensorFlow tf.constant
                or PyTorch torch.Tensor instead of a list of python integers.
            return_token_type_ids: (optional) Set to False to avoid returning token_type_ids (default True).
            return_attention_mask: (optional) Set to False to avoid returning attention mask (default True)
            return_overflowing_tokens: (optional) Set to True to return overflowing token information (default False).
            return_special_tokens_mask: (optional) Set to True to return special tokens mask information (default False).

        Return:
            A Dictionary of shape::

                {
                    input_ids: list[int],
                    token_type_ids: list[int] if return_token_type_ids is True (default)
                    overflowing_tokens: list[int] if a ``max_length`` is specified and return_overflowing_tokens is True
                    num_truncated_tokens: int if a ``max_length`` is specified and return_overflowing_tokens is True
                    special_tokens_mask: list[int] if ``add_special_tokens`` if set to ``True`` and return_special_tokens_mask is True
                }

            With the fields:
                ``input_ids``: list of token ids to be fed to a model
                ``token_type_ids``: list of token type ids to be fed to a model

                ``overflowing_tokens``: list of overflowing tokens if a max length is specified.
                ``num_truncated_tokens``: number of overflowing tokens a ``max_length`` is specified
                ``special_tokens_mask``: if adding special tokens, this is a list of [0, 1], with 0 specifying special added
                tokens and 1 specifying sequence tokens.
        """
        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        encoded_inputs = {}

        # Handle max sequence length
        total_len = len_ids + len_pair_ids + (self.num_added_tokens(pair=pair) if add_special_tokens else 0)
        if max_length and total_len > max_length:
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )
            if return_overflowing_tokens:
                encoded_inputs["overflowing_tokens"] = overflowing_tokens
                encoded_inputs["num_truncated_tokens"] = total_len - max_length

        # Handle special_tokens
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([1] * len(pair_ids) if pair else [])

        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)
            else:
                encoded_inputs["special_tokens_mask"] = [0] * len(sequence)

        encoded_inputs["input_ids"] = sequence
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids

        if max_length and len(encoded_inputs["input_ids"]) > max_length:
            encoded_inputs["input_ids"] = encoded_inputs["input_ids"][:max_length]
            if return_token_type_ids:
                encoded_inputs["token_type_ids"] = encoded_inputs["token_type_ids"][:max_length]
            if return_special_tokens_mask:
                encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"][:max_length]

        if max_length is None and len(encoded_inputs["input_ids"]) > self.max_len:
            logger.warning(
                "Token indices sequence length is longer than the specified maximum sequence length "
                "for this model ({} > {}). Running this sequence through the model will result in "
                "indexing errors".format(len(ids), self.max_len)
            )

        needs_to_be_padded = pad_to_max_length and (
            max_length
            and len(encoded_inputs["input_ids"]) < max_length
            or max_length is None
            and len(encoded_inputs["input_ids"]) < self.max_len
            and self.max_len <= 10000
        )

        if pad_to_max_length and max_length is None and self.max_len > 10000:
            logger.warning(
                "Sequence can't be padded as no maximum length is specified and the model maximum length is too high."
            )

        if needs_to_be_padded:
            difference = (max_length if max_length is not None else self.max_len) - len(encoded_inputs["input_ids"])

            if self.padding_side == "right":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"]) + [0] * difference
                if return_token_type_ids:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference
                    )
                if return_special_tokens_mask:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
                encoded_inputs["input_ids"] = encoded_inputs["input_ids"] + [self.pad_token_id] * difference
            elif self.padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + [1] * len(encoded_inputs["input_ids"])
                if return_token_type_ids:
                    encoded_inputs["token_type_ids"] = [self.pad_token_type_id] * difference + encoded_inputs[
                        "token_type_ids"
                    ]
                if return_special_tokens_mask:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs["input_ids"] = [self.pad_token_id] * difference + encoded_inputs["input_ids"]

            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))

        elif return_attention_mask:
            encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"])

        # Prepare inputs as tensors if asked
        if return_tensors == "tf" and is_tf_available():
            encoded_inputs["input_ids"] = tf.constant([encoded_inputs["input_ids"]])

            if "token_type_ids" in encoded_inputs:
                encoded_inputs["token_type_ids"] = tf.constant([encoded_inputs["token_type_ids"]])

            if "attention_mask" in encoded_inputs:
                encoded_inputs["attention_mask"] = tf.constant([encoded_inputs["attention_mask"]])

        elif return_tensors == "pt" and is_torch_available():
            encoded_inputs["input_ids"] = torch.tensor([encoded_inputs["input_ids"]])

            if "token_type_ids" in encoded_inputs:
                encoded_inputs["token_type_ids"] = torch.tensor([encoded_inputs["token_type_ids"]])

            if "attention_mask" in encoded_inputs:
                encoded_inputs["attention_mask"] = torch.tensor([encoded_inputs["attention_mask"]])
        elif return_tensors is not None:
            logger.warning(
                "Unable to convert output to tensors format {}, PyTorch or TensorFlow is not available.".format(
                    return_tensors
                )
            )

        return encoded_inputs

    def prepare_for_tokenization(self, text, **kwargs):
        """ Performs any necessary transformations before tokenization """
        return text

    def truncate_sequences(
        self, ids, pair_ids=None, num_tokens_to_remove=0, truncation_strategy="longest_first", stride=0
    ):
        """Truncates a sequence pair in place to the maximum length.
            truncation_strategy: string selected in the following options:
                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
                    starting from the longest one at each token (when there is a pair of input sequences).
                    Overflowing tokens only contains overflow from the first sequence.
                - 'only_first': Only truncate the first sequence. raise an error if the first sequence is shorter or equal to than num_tokens_to_remove.
                - 'only_second': Only truncate the second sequence
                - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_length)
        """
        if num_tokens_to_remove <= 0:
            return ids, pair_ids, []

        if truncation_strategy == "longest_first":
            overflowing_tokens = []
            for _ in range(num_tokens_to_remove):
                if pair_ids is None or len(ids) > len(pair_ids):
                    overflowing_tokens = [ids[-1]] + overflowing_tokens
                    ids = ids[:-1]
                else:
                    pair_ids = pair_ids[:-1]
            window_len = min(len(ids), stride)
            if window_len > 0:
                overflowing_tokens = ids[-window_len:] + overflowing_tokens
        elif truncation_strategy == "only_first":
            assert len(ids) > num_tokens_to_remove
            window_len = min(len(ids), stride + num_tokens_to_remove)
            overflowing_tokens = ids[-window_len:]
            ids = ids[:-num_tokens_to_remove]
        elif truncation_strategy == "only_second":
            assert pair_ids is not None and len(pair_ids) > num_tokens_to_remove
            window_len = min(len(pair_ids), stride + num_tokens_to_remove)
            overflowing_tokens = pair_ids[-window_len:]
            pair_ids = pair_ids[:-num_tokens_to_remove]
        elif truncation_strategy == "do_not_truncate":
            raise ValueError("Input sequence are too long for max_length. Please select a truncation strategy.")
        else:
            raise ValueError(
                "Truncation_strategy should be selected in ['longest_first', 'only_first', 'only_second', 'do_not_truncate']"
            )
        return (ids, pair_ids, overflowing_tokens)

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return len(token_ids_0) * [0]
        return [0] * len(token_ids_0) + [1] * len(token_ids_1)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A RoBERTa sequence has the following format:
            single sequence: <s> X </s>
            pair of sequences: <s> A </s></s> B </s>
        """
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        Args:
            token_ids_0: list of ids (must not contain special tokens)
            token_ids_1: Optional list of ids (must not contain special tokens), necessary when fetching sequence ids
                for sequence pairs
            already_has_special_tokens: (default False) Set to True if the token list is already formated with
                special tokens for the model

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        return [0] * ((len(token_ids_1) if token_ids_1 else 0) + len(token_ids_0))

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        """ Converts a single index or a sequence of indices (integers) in a token "
            (resp.) a sequence of tokens (str), using the vocabulary and added tokens.

            Args:
                skip_special_tokens: Don't decode special tokens (self.all_special_tokens). Default: False
        """
        if isinstance(ids, int):
            if ids in self.added_tokens_decoder:
                return self.added_tokens_decoder[ids]
            else:
                return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            if index in self.added_tokens_decoder:
                tokens.append(self.added_tokens_decoder[index])
            else:
                tokens.append(self._convert_id_to_token(index))
        return tokens

    def _convert_id_to_token(self, index):
        raise NotImplementedError

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string.
            The most simple way to do it is ' '.join(self.convert_ids_to_tokens(token_ids))
            but we often want to remove sub-word tokenization artifacts at the same time.
        """
        return " ".join(self.convert_ids_to_tokens(tokens))

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True):
        """
        Converts a sequence of ids (integer) in a string, using the tokenizer and vocabulary
        with options to remove special tokens and clean up tokenization spaces.
        Similar to doing ``self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))``.

        Args:
            token_ids: list of tokenized input ids. Can be obtained using the `encode` or `encode_plus` methods.
            skip_special_tokens: if set to True, will replace special tokens.
            clean_up_tokenization_spaces: if set to True, will clean up the tokenization spaces.
        """
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        # To avoid mixing byte-level and unicode for byte-level BPT
        # we need to build string separatly for added tokens and byte-level tokens
        # cf. https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            if token in self.added_tokens_encoder:
                if current_sub_text:
                    sub_texts.append(self.convert_tokens_to_string(current_sub_text))
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))
        text = " ".join(sub_texts)

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    @property
    def special_tokens_map(self):
        """ A dictionary mapping special token class attribute (cls_token, unk_token...) to their
            values ('<unk>', '<cls>'...)
        """
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, "_" + attr)
            if attr_value:
                set_attr[attr] = attr_value
        return set_attr

    @property
    def all_special_tokens(self):
        """ List all the special tokens ('<unk>', '<cls>'...) mapped to class attributes
            (cls_token, unk_token...).
        """
        all_toks = []
        set_attr = self.special_tokens_map
        for attr_value in set_attr.values():
            all_toks = all_toks + (list(attr_value) if isinstance(attr_value, (list, tuple)) else [attr_value])
        all_toks = list(set(all_toks))
        return all_toks

    @property
    def all_special_ids(self):
        """ List the vocabulary indices of the special tokens ('<unk>', '<cls>'...) mapped to
            class attributes (cls_token, unk_token...).
        """
        all_toks = self.all_special_tokens
        all_ids = self.convert_tokens_to_ids(all_toks)
        return all_ids

    @staticmethod
    def clean_up_tokenization(out_string):
        """ Clean up a list of simple English tokenization artifacts like spaces before punctuations and abreviated forms.
        """
        out_string = (
            out_string.replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm", "'m")
            .replace(" do not", " don't")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
        )
        return out_string


class PreTrainedTokenizerFast(PreTrainedTokenizer):
    def __init__(self, tokenizer: BaseTokenizer, **kwargs):
        if tokenizer is None:
            raise ValueError("Provided tokenizer cannot be None")
        self._tokenizer = tokenizer

        super().__init__(**kwargs)
        self.max_len_single_sentence = self.max_len - self.num_added_tokens(False)  # take into account special tokens
        self.max_len_sentences_pair = self.max_len - self.num_added_tokens(True)  # take into account special tokens

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def decoder(self):
        return self._tokenizer._tokenizer.decoder

    @property
    def vocab_size(self):
        return self._tokenizer.get_vocab_size(with_added_tokens=False)

    def __len__(self):
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    @PreTrainedTokenizer.bos_token.setter
    def bos_token(self, value):
        self._bos_token = value
        self._update_special_tokens()

    @PreTrainedTokenizer.eos_token.setter
    def eos_token(self, value):
        self._eos_token = value
        self._update_special_tokens()

    @PreTrainedTokenizer.unk_token.setter
    def unk_token(self, value):
        self._unk_token = value
        self._update_special_tokens()

    @PreTrainedTokenizer.sep_token.setter
    def sep_token(self, value):
        self._sep_token = value
        self._update_special_tokens()

    @PreTrainedTokenizer.pad_token.setter
    def pad_token(self, value):
        self._pad_token = value
        self._update_special_tokens()

    @PreTrainedTokenizer.cls_token.setter
    def cls_token(self, value):
        self._cls_token = value
        self._update_special_tokens()

    @PreTrainedTokenizer.mask_token.setter
    def mask_token(self, value):
        self._mask_token = value
        self._update_special_tokens()

    @PreTrainedTokenizer.additional_special_tokens.setter
    def additional_special_tokens(self, value):
        self._additional_special_tokens = value
        self._update_special_tokens()

    def _update_special_tokens(self):
        if self._tokenizer is not None:
            self._tokenizer.add_special_tokens(self.all_special_tokens)

    @staticmethod
    def _convert_encoding(
        encoding,
        return_tensors=None,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        return_offsets_mapping=False,
    ):
        if return_overflowing_tokens and encoding.overflowing is not None:
            encodings = [encoding] + encoding.overflowing
        else:
            encodings = [encoding]

        encoding_dict = defaultdict(list)
        for e in encodings:
            encoding_dict["input_ids"].append(e.ids)

            if return_token_type_ids:
                encoding_dict["token_type_ids"].append(e.type_ids)
            if return_attention_mask:
                encoding_dict["attention_mask"].append(e.attention_mask)
            if return_special_tokens_mask:
                encoding_dict["special_tokens_mask"].append(e.special_tokens_mask)
            if return_offsets_mapping:
                encoding_dict["offset_mapping"].append([e.original_str.offsets(o) for o in e.offsets])

        # Prepare inputs as tensors if asked
        if return_tensors == "tf" and is_tf_available():
            encoding_dict["input_ids"] = tf.constant(encoding_dict["input_ids"])
            if "token_type_ids" in encoding_dict:
                encoding_dict["token_type_ids"] = tf.constant(encoding_dict["token_type_ids"])

            if "attention_mask" in encoding_dict:
                encoding_dict["attention_mask"] = tf.constant(encoding_dict["attention_mask"])

        elif return_tensors == "pt" and is_torch_available():
            encoding_dict["input_ids"] = torch.tensor(encoding_dict["input_ids"])
            if "token_type_ids" in encoding_dict:
                encoding_dict["token_type_ids"] = torch.tensor(encoding_dict["token_type_ids"])

            if "attention_mask" in encoding_dict:
                encoding_dict["attention_mask"] = torch.tensor(encoding_dict["attention_mask"])
        elif return_tensors is not None:
            logger.warning(
                "Unable to convert output to tensors format {}, PyTorch or TensorFlow is not available.".format(
                    return_tensors
                )
            )

        return encoding_dict

    def _convert_token_to_id_with_added_voc(self, token):
        id = self._tokenizer.token_to_id(token)
        if id is None:
            return self.unk_token_id
        return id

    def _convert_id_to_token(self, index):
        return self._tokenizer.id_to_token(int(index))

    def convert_tokens_to_string(self, tokens):
        return self._tokenizer.decode(tokens)

    def add_tokens(self, new_tokens):
        if isinstance(new_tokens, str):
            new_tokens = [new_tokens]
        return self._tokenizer.add_tokens(new_tokens)

    def add_special_tokens(self, special_tokens_dict):
        added = super().add_special_tokens(special_tokens_dict)
        self._update_special_tokens()
        return added

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return token_ids_0
        else:
            return token_ids_0 + token_ids_1

    def num_added_tokens(self, pair=False):
        return self.tokenizer.num_special_tokens_to_add(pair)

    def tokenize(self, text, **kwargs):
        return self.tokenizer.encode(text).tokens

    def batch_encode_plus(
        self,
        batch_text_or_text_pairs=None,
        add_special_tokens=True,
        max_length=None,
        stride=0,
        truncation_strategy="longest_first",
        pad_to_max_length=False,
        return_tensors=None,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        return_offsets_mapping=False,
        **kwargs
    ):
        if not add_special_tokens:
            logger.warning(
                "Fast tokenizers add special tokens by default. To remove special tokens, please specify"
                "`add_special_tokens=False` during the initialisation rather than when calling `encode`,"
                "`encode_plus` or `batch_encode_plus`."
            )

        # Needed if we have to return a tensor
        pad_to_max_length = pad_to_max_length or (return_tensors is not None)

        # Throw an error if we can pad because there is no padding token
        if pad_to_max_length and self.pad_token_id is None:
            raise ValueError("Unable to set proper padding strategy as the tokenizer does not have a padding token")

        # Set the truncation and padding strategy and restore the initial configuration
        with truncate_and_pad(
            tokenizer=self._tokenizer,
            max_length=max_length,
            stride=stride,
            strategy=truncation_strategy,
            pad_to_max_length=pad_to_max_length,
            padding_side=self.padding_side,
            pad_token_id=self.pad_token_id,
            pad_token_type_id=self.pad_token_type_id,
            pad_token=self._pad_token,
        ):

            if not isinstance(batch_text_or_text_pairs, list):
                raise TypeError(
                    "batch_text_or_text_pairs has to be a list (got {})".format(type(batch_text_or_text_pairs))
                )

            # Avoid thread overhead if only one example.
            if len(batch_text_or_text_pairs) == 1:
                if isinstance(batch_text_or_text_pairs[0], (tuple, list)):
                    tokens = self._tokenizer.encode(*batch_text_or_text_pairs[0])
                else:
                    tokens = self._tokenizer.encode(batch_text_or_text_pairs[0])
                tokens = [tokens]
            else:
                tokens = self._tokenizer.encode_batch(batch_text_or_text_pairs)

        # Convert encoding to dict
        tokens = [
            self._convert_encoding(
                encoding=encoding,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
            )
            for encoding in tokens
        ]

        # Sanitize the output to have dict[list] from list[dict]
        sanitized = {}
        for key in tokens[0].keys():
            stack = [e for item in tokens for e in item[key]]
            if return_tensors == "tf":
                stack = tf.stack(stack, axis=0)
            elif return_tensors == "pt":
                stack = torch.stack(stack, dim=0)
            elif not return_tensors and len(stack) == 1:
                stack = stack[0]

            sanitized[key] = stack

        # If returning overflowing tokens, we need to return a mapping
        # from the batch idx to the original sample
        if return_overflowing_tokens:
            overflow_to_sample_mapping = [
                i if len(item["input_ids"]) == 1 else [i] * len(item["input_ids"]) for i, item in enumerate(tokens)
            ]
            sanitized["overflow_to_sample_mapping"] = overflow_to_sample_mapping
        return sanitized

    def encode_plus(
        self,
        text,
        text_pair=None,
        add_special_tokens=False,
        max_length=None,
        pad_to_max_length=False,
        stride=0,
        truncation_strategy="longest_first",
        return_tensors=None,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        return_offsets_mapping=False,
        **kwargs
    ):
        batched_input = [(text, text_pair)] if text_pair else [text]
        batched_output = self.batch_encode_plus(
            batched_input,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            stride=stride,
            truncation_strategy=truncation_strategy,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            pad_to_max_length=pad_to_max_length,
            **kwargs,
        )

        # Return tensor is None, then we can remove the leading batch axis
        if not return_tensors:
            return {key: value[0] if isinstance(value[0], list) else value for key, value in batched_output.items()}
        else:
            return batched_output

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True):
        text = self.tokenizer.decode(token_ids, skip_special_tokens)

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def save_vocabulary(self, save_directory):
        if os.path.isdir(save_directory):
            files = self._tokenizer.save(save_directory)
        else:
            folder, file = os.path.split(os.path.abspath(save_directory))
            files = self._tokenizer.save(folder, name=file)

        return tuple(files)
