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
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging
import os
import json
import six
import copy
from io import open

from .file_utils import cached_path, is_tf_available, is_torch_available

if is_tf_available():
    import tensorflow as tf
if is_torch_available():
    import torch

logger = logging.getLogger(__name__)

SPECIAL_TOKENS_MAP_FILE = 'special_tokens_map.json'
ADDED_TOKENS_FILE = 'added_tokens.json'
TOKENIZER_CONFIG_FILE = 'tokenizer_config.json'

class PreTrainedTokenizer(object):
    """ Base class for all tokenizers.
    Handle all the shared methods for tokenization and special tokens as well as methods dowloading/caching/loading pretrained tokenizers as well as adding tokens to the vocabulary.

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

    SPECIAL_TOKENS_ATTRIBUTES = ["bos_token", "eos_token", "unk_token", "sep_token",
                                 "pad_token", "cls_token", "mask_token",
                                 "additional_special_tokens"]

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

    def __init__(self, max_len=None, **kwargs):
        self._bos_token = None
        self._eos_token = None
        self._unk_token = None
        self._sep_token = None
        self._pad_token = None
        self._cls_token = None
        self._mask_token = None
        self._additional_special_tokens = []

        self.max_len = max_len if max_len is not None else int(1e12)

        # Added tokens
        self.added_tokens_encoder = {}
        self.added_tokens_decoder = {}

        # inputs and kwargs for saving and re-loading (see ``from_pretrained`` and ``save_pretrained``)
        self.init_inputs = ()
        self.init_kwargs = {}

        for key, value in kwargs.items():
            if key in self.SPECIAL_TOKENS_ATTRIBUTES:
                if key == 'additional_special_tokens':
                    assert isinstance(value, (list, tuple)) and all(isinstance(t, str) or (six.PY2 and isinstance(t, unicode)) for t in value)
                else:
                    assert isinstance(value, str) or (six.PY2 and isinstance(value, unicode))
                setattr(self, key, value)


    @classmethod
    def from_pretrained(cls, *inputs, **kwargs):
        r"""
        Instantiate a :class:`~transformers.PreTrainedTokenizer` (or a derived class) from a predefined tokenizer.

        Args:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a predefined tokenizer to load from cache or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing vocabulary files required by the tokenizer, for instance saved using the :func:`~transformers.PreTrainedTokenizer.save_pretrained` method, e.g.: ``./my_model_directory/``.
                - (not applicable to all derived classes) a path or url to a single saved vocabulary file if and only if the tokenizer only requires a single vocabulary file (e.g. Bert, XLNet), e.g.: ``./my_model_directory/vocab.txt``.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded predefined tokenizer vocabulary files should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the vocabulary files and override the cached versions if they exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            inputs: (`optional`) positional arguments: will be passed to the Tokenizer ``__init__`` method.

            kwargs: (`optional`) keyword arguments: will be passed to the Tokenizer ``__init__`` method. Can be used to set special tokens like ``bos_token``, ``eos_token``, ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``, ``mask_token``, ``additional_special_tokens``. See parameters in the doc string of :class:`~transformers.PreTrainedTokenizer` for details.

        Examples::

            # We can't instantiate directly the base class `PreTrainedTokenizer` so let's show our examples on a derived class: BertTokenizer

            # Download vocabulary from S3 and cache.
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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
        cache_dir = kwargs.pop('cache_dir', None)
        force_download = kwargs.pop('force_download', False)
        proxies = kwargs.pop('proxies', None)

        s3_models = list(cls.max_model_input_sizes.keys())
        vocab_files = {}
        init_configuration = {}
        if pretrained_model_name_or_path in s3_models:
            # Get the vocabulary from AWS S3 bucket
            for file_id, map_list in cls.pretrained_vocab_files_map.items():
                vocab_files[file_id] = map_list[pretrained_model_name_or_path]
            if cls.pretrained_init_configuration and pretrained_model_name_or_path in cls.pretrained_init_configuration:
                init_configuration = cls.pretrained_init_configuration[pretrained_model_name_or_path]
        else:
            # Get the vocabulary from local files
            logger.info(
                "Model name '{}' not found in model shortcut name list ({}). "
                "Assuming '{}' is a path or url to a directory containing tokenizer files.".format(
                    pretrained_model_name_or_path, ', '.join(s3_models),
                    pretrained_model_name_or_path))

            # Look for the tokenizer main vocabulary files
            for file_id, file_name in cls.vocab_files_names.items():
                if os.path.isdir(pretrained_model_name_or_path):
                    # If a directory is provided we look for the standard filenames
                    full_file_name = os.path.join(pretrained_model_name_or_path, file_name)
                else:
                    # If a path to a file is provided we use it (will only work for non-BPE tokenizer using a single vocabulary file)
                    full_file_name = pretrained_model_name_or_path
                if not os.path.exists(full_file_name):
                    logger.info("Didn't find file {}. We won't load it.".format(full_file_name))
                    full_file_name = None
                vocab_files[file_id] = full_file_name

            # Look for the additional tokens files
            additional_files_names = {'added_tokens_file': ADDED_TOKENS_FILE,
                                      'special_tokens_map_file': SPECIAL_TOKENS_MAP_FILE,
                                      'tokenizer_config_file': TOKENIZER_CONFIG_FILE,
                                      }

            # If a path to a file was provided, get the parent directory
            saved_directory = pretrained_model_name_or_path
            if os.path.exists(saved_directory) and not os.path.isdir(saved_directory):
                saved_directory = os.path.dirname(saved_directory)

            for file_id, file_name in additional_files_names.items():
                full_file_name = os.path.join(saved_directory, file_name)
                if not os.path.exists(full_file_name):
                    logger.info("Didn't find file {}. We won't load it.".format(full_file_name))
                    full_file_name = None
                vocab_files[file_id] = full_file_name

            if all(full_file_name is None for full_file_name in vocab_files.values()):
                raise EnvironmentError(
                    "Model name '{}' was not found in tokenizers model name list ({}). "
                    "We assumed '{}' was a path or url to a directory containing vocabulary files "
                    "named {} but couldn't find such vocabulary files at this path or url.".format(
                        pretrained_model_name_or_path, ', '.join(s3_models),
                        pretrained_model_name_or_path, 
                        list(cls.vocab_files_names.values())))

        # Get files from url, cache, or disk depending on the case
        try:
            resolved_vocab_files = {}
            for file_id, file_path in vocab_files.items():
                if file_path is None:
                    resolved_vocab_files[file_id] = None
                else:
                    resolved_vocab_files[file_id] = cached_path(file_path, cache_dir=cache_dir, force_download=force_download, proxies=proxies)
        except EnvironmentError:
            if pretrained_model_name_or_path in s3_models:
                msg = "Couldn't reach server at '{}' to download vocabulary files."
            else:
                msg = "Model name '{}' was not found in tokenizers model name list ({}). " \
                    "We assumed '{}' was a path or url to a directory containing vocabulary files " \
                    "named {}, but couldn't find such vocabulary files at this path or url.".format(
                        pretrained_model_name_or_path, ', '.join(s3_models),
                        pretrained_model_name_or_path,
                        list(cls.vocab_files_names.values()))

            raise EnvironmentError(msg)

        for file_id, file_path in vocab_files.items():
            if file_path == resolved_vocab_files[file_id]:
                logger.info("loading file {}".format(file_path))
            else:
                logger.info("loading file {} from cache at {}".format(
                    file_path, resolved_vocab_files[file_id]))

        # Prepare tokenizer initialization kwargs
        # Did we saved some inputs and kwargs to reload ?
        tokenizer_config_file = resolved_vocab_files.pop('tokenizer_config_file', None)
        if tokenizer_config_file is not None:
            init_kwargs = json.load(open(tokenizer_config_file, encoding="utf-8"))
            saved_init_inputs = init_kwargs.pop('init_inputs', ())
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
                init_kwargs['max_len'] = min(init_kwargs.get('max_len', int(1e12)), max_len)

        # Merge resolved_vocab_files arguments in init_kwargs.
        added_tokens_file = resolved_vocab_files.pop('added_tokens_file', None)
        special_tokens_map_file = resolved_vocab_files.pop('special_tokens_map_file', None)
        for args_name, file_path in resolved_vocab_files.items():
            if args_name not in init_kwargs:
                init_kwargs[args_name] = file_path
        if special_tokens_map_file is not None:
            special_tokens_map = json.load(open(special_tokens_map_file, encoding="utf-8"))
            for key, value in special_tokens_map.items():
                if key not in init_kwargs:
                    init_kwargs[key] = value

        # Instantiate tokenizer.
        tokenizer = cls(*init_inputs, **init_kwargs)

        # Save inputs and kwargs for saving and re-loading with ``save_pretrained``
        tokenizer.init_inputs = init_inputs
        tokenizer.init_kwargs = init_kwargs

        # Add supplementary tokens.
        if added_tokens_file is not None:
            added_tok_encoder = json.load(open(added_tokens_file, encoding="utf-8"))
            added_tok_decoder = {v:k for k, v in added_tok_encoder.items()}
            tokenizer.added_tokens_encoder.update(added_tok_encoder)
            tokenizer.added_tokens_decoder.update(added_tok_decoder)

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
        tokenizer_config['init_inputs'] = copy.deepcopy(self.init_inputs)
        for file_id in self.vocab_files_names.keys():
            tokenizer_config.pop(file_id, None)

        with open(tokenizer_config_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_config, ensure_ascii=False))

        with open(special_tokens_map_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.special_tokens_map, ensure_ascii=False))

        with open(added_tokens_file, 'w', encoding='utf-8') as f:
            if self.added_tokens_encoder:
                out_str = json.dumps(self.added_tokens_encoder, ensure_ascii=False)
            else:
                out_str = u"{}"
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
            new_tokens: list of string. Each string is a token to add. Tokens are only added if they are not already in the vocabulary (tested by checking if the tokenizer assign the index of the ``unk_token`` to them).

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

        to_add_tokens = []
        for token in new_tokens:
            assert isinstance(token, str) or (six.PY2 and isinstance(token, unicode))
            if token != self.unk_token and \
                    self.convert_tokens_to_ids(token) == self.convert_tokens_to_ids(self.unk_token) and \
                    token not in to_add_tokens:
                to_add_tokens.append(token)
                logger.info("Adding %s to the vocabulary", token)

        added_tok_encoder = dict((tok, len(self) + i) for i, tok in enumerate(to_add_tokens))
        added_tok_decoder = {v:k for k, v in added_tok_encoder.items()}
        self.added_tokens_encoder.update(added_tok_encoder)
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
            if key == 'additional_special_tokens':
                assert isinstance(value, (list, tuple)) and all(isinstance(t, str) or (six.PY2 and isinstance(t, unicode)) for t in value)
                added_tokens += self.add_tokens(value)
            else:
                assert isinstance(value, str) or (six.PY2 and isinstance(value, unicode))
                added_tokens += self.add_tokens([value])
            logger.info("Assigning %s to the %s key of the tokenizer", value, key)
            setattr(self, key, value)

        return added_tokens

    def tokenize(self, text, **kwargs):
        """ Converts a string in a sequence of tokens (string), using the tokenizer.
            Split in words for word-based vocabulary or sub-words for sub-word-based
            vocabularies (BPE/SentencePieces/WordPieces).

            Take care of added tokens.
        """
        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                sub_text = sub_text.strip()
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
            if not text:
                return []
            if not tok_list:
                return self._tokenize(text, **kwargs)

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.added_tokens_encoder \
                            and sub_text not in self.all_special_tokens:
                        tokenized_text += split_on_token(tok, sub_text)
                    else:
                        tokenized_text += [sub_text]
                text_list = tokenized_text

            return sum((self._tokenize(token, **kwargs) if token not \
                    in self.added_tokens_encoder and token not in self.all_special_tokens \
                    else [token] for token in tokenized_text), [])

        added_tokens = list(self.added_tokens_encoder.keys()) + self.all_special_tokens
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
        """ Converts a single token, or a sequence of tokens, (str/unicode) in a single integer id
            (resp. a sequence of ids), using the vocabulary.
        """
        if tokens is None:
            return None

        if isinstance(tokens, str) or (six.PY2 and isinstance(tokens, unicode)):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        if len(ids) > self.max_len:
            logger.warning("Token indices sequence length is longer than the specified maximum sequence length "
                           "for this model ({} > {}). Running this sequence through the model will result in "
                           "indexing errors".format(len(ids), self.max_len))
        return ids

    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None

        if token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]
        return self._convert_token_to_id(token)

    def _convert_token_to_id(self, token):
        raise NotImplementedError

    def encode(self,
                text,
                text_pair=None,
                add_special_tokens=False,
                max_length=None,
                stride=0,
                truncation_strategy='longest_first',
                return_tensors=None,
                **kwargs):
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
            return_tensors: (optional) can be set to 'tf' or 'pt' to return respectively TensorFlow tf.constant
                or PyTorch torch.Tensor instead of a list of python integers.
            **kwargs: passed to the `self.tokenize()` method
        """
        encoded_inputs = self.encode_plus(text,
                                          text_pair=text_pair,
                                          max_length=max_length,
                                          add_special_tokens=add_special_tokens,
                                          stride=stride,
                                          truncation_strategy=truncation_strategy,
                                          return_tensors=return_tensors,
                                          **kwargs)

        return encoded_inputs["input_ids"]

    def encode_plus(self,
                    text,
                    text_pair=None,
                    add_special_tokens=False,
                    max_length=None,
                    stride=0,
                    truncation_strategy='longest_first',
                    return_tensors=None,
                    **kwargs):
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
            return_tensors: (optional) can be set to 'tf' or 'pt' to return respectively TensorFlow tf.constant
                or PyTorch torch.Tensor instead of a list of python integers.
            **kwargs: passed to the `self.tokenize()` method
        """

        def get_input_ids(text):
            if isinstance(text, six.string_types):
                return self.convert_tokens_to_ids(self.tokenize(text, **kwargs))
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], six.string_types):
                return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError("Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers.")

        first_ids = get_input_ids(text)
        second_ids = get_input_ids(text_pair) if text_pair is not None else None

        return self.prepare_for_model(first_ids,
                                      pair_ids=second_ids,
                                      max_length=max_length,
                                      add_special_tokens=add_special_tokens,
                                      stride=stride,
                                      truncation_strategy=truncation_strategy,
                                      return_tensors=return_tensors)

    def prepare_for_model(self, ids, pair_ids=None, max_length=None, add_special_tokens=False, stride=0,
                          truncation_strategy='longest_first', return_tensors=None):
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
            return_tensors: (optional) can be set to 'tf' or 'pt' to return respectively TensorFlow tf.constant
                or PyTorch torch.Tensor instead of a list of python integers.

        Return:
            A Dictionary of shape::

                {
                    input_ids: list[int],
                    overflowing_tokens: list[int] if a ``max_length`` is specified, else None
                    special_tokens_mask: list[int] if ``add_special_tokens`` if set to ``True``
                }

            With the fields:
                ``input_ids``: list of tokens to be fed to a model

                ``overflowing_tokens``: list of overflowing tokens if a max length is specified.

                ``special_tokens_mask``: if adding special tokens, this is a list of [0, 1], with 0 specifying special added
                tokens and 1 specifying sequence tokens.
        """
        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        encoded_inputs = {}
        total_len = len_ids + len_pair_ids + (self.num_added_tokens(pair=pair) if add_special_tokens else 0)
        if max_length and total_len > max_length:
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(ids, pair_ids=pair_ids,
                                                                        num_tokens_to_remove=total_len-max_length,
                                                                        truncation_strategy=truncation_strategy,
                                                                        stride=stride)
            encoded_inputs["overflowing_tokens"] = overflowing_tokens
            encoded_inputs["num_truncated_tokens"] = total_len - max_length

        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
            encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([1] * len(pair_ids) if pair else [])

        if return_tensors == 'tf' and is_tf_available():
            sequence = tf.constant([sequence])
            token_type_ids = tf.constant([token_type_ids])
        elif return_tensors == 'pt' and is_torch_available():
            sequence = torch.tensor([sequence])
            token_type_ids = torch.tensor([token_type_ids])
        elif return_tensors is not None:
            logger.warning("Unable to convert output to tensors format {}, PyTorch or TensorFlow is not available.".format(return_tensors))

        encoded_inputs["input_ids"] = sequence
        encoded_inputs["token_type_ids"] = token_type_ids

        if max_length and len(encoded_inputs["input_ids"]) > max_length:
            encoded_inputs["input_ids"] = encoded_inputs["input_ids"][:max_length]
            encoded_inputs["token_type_ids"] = encoded_inputs["token_type_ids"][:max_length]
            encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"][:max_length]

        return encoded_inputs

    def truncate_sequences(self, ids, pair_ids=None, num_tokens_to_remove=0, truncation_strategy='longest_first', stride=0):
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

        if truncation_strategy == 'longest_first':
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
        elif truncation_strategy == 'only_first':
            assert len(ids) > num_tokens_to_remove
            window_len = min(len(ids), stride + num_tokens_to_remove)
            overflowing_tokens = ids[-window_len:]
            ids = ids[:-num_tokens_to_remove]
        elif truncation_strategy == 'only_second':
            assert pair_ids is not None and len(pair_ids) > num_tokens_to_remove
            window_len = min(len(pair_ids), stride + num_tokens_to_remove)
            overflowing_tokens = pair_ids[-window_len:]
            pair_ids = pair_ids[:-num_tokens_to_remove]
        elif truncation_strategy == 'do_not_truncate':
            raise ValueError("Input sequence are too long for max_length. Please select a truncation strategy.")
        else:
            raise ValueError("Truncation_strategy should be selected in ['longest_first', 'only_first', 'only_second', 'do_not_truncate']")
        return (ids, pair_ids, overflowing_tokens)

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        logger.warning("This tokenizer does not make use of special tokens.")
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
        logger.warning("This tokenizer does not make use of special tokens. Input is returned with no modification.")
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
            A list of integers in the range [0, 1]: 0 for a special token, 1 for a sequence token.
        """
        return [0] * ((len(token_ids_1) if token_ids_1 else 0) + len(token_ids_0))

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        """ Converts a single index or a sequence of indices (integers) in a token "
            (resp.) a sequence of tokens (str/unicode), using the vocabulary and added tokens.

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
        return ' '.join(self.convert_ids_to_tokens(tokens))

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
                sub_texts.append(" " + token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))
        text = ''.join(sub_texts)

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
        all_ids = list(self._convert_token_to_id(t) for t in all_toks)
        return all_ids

    @staticmethod
    def clean_up_tokenization(out_string):
        """ Clean up a list of simple English tokenization artifacts like spaces before punctuations and abreviated forms.
        """
        out_string = out_string.replace(' .', '.').replace(' ?', '?').replace(' !', '!').replace(' ,', ','
                        ).replace(" ' ", "'").replace(" n't", "n't").replace(" 'm", "'m").replace(" do not", " don't"
                        ).replace(" 's", "'s").replace(" 've", "'ve").replace(" 're", "'re")
        return out_string
