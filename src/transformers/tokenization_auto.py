# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
""" Auto Model class. """


import logging
from collections import OrderedDict

from .configuration_auto import (
    AlbertConfig,
    AutoConfig,
    BartConfig,
    BertConfig,
    CamembertConfig,
    CTRLConfig,
    DistilBertConfig,
    FlaubertConfig,
    GPT2Config,
    OpenAIGPTConfig,
    RobertaConfig,
    T5Config,
    TransfoXLConfig,
    XLMConfig,
    XLMRobertaConfig,
    XLNetConfig,
)
from .configuration_utils import PretrainedConfig
from .tokenization_albert import AlbertTokenizer
from .tokenization_bart import BartTokenizer
from .tokenization_bert import BertTokenizer, BertTokenizerFast
from .tokenization_bert_japanese import BertJapaneseTokenizer
from .tokenization_camembert import CamembertTokenizer
from .tokenization_ctrl import CTRLTokenizer
from .tokenization_distilbert import DistilBertTokenizer, DistilBertTokenizerFast
from .tokenization_flaubert import FlaubertTokenizer
from .tokenization_gpt2 import GPT2Tokenizer, GPT2TokenizerFast
from .tokenization_openai import OpenAIGPTTokenizer, OpenAIGPTTokenizerFast
from .tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast
from .tokenization_t5 import T5Tokenizer
from .tokenization_transfo_xl import TransfoXLTokenizer, TransfoXLTokenizerFast
from .tokenization_xlm import XLMTokenizer
from .tokenization_xlm_roberta import XLMRobertaTokenizer
from .tokenization_xlnet import XLNetTokenizer


logger = logging.getLogger(__name__)


TOKENIZER_MAPPING = OrderedDict(
    [
        (T5Config, (T5Tokenizer, None)),
        (DistilBertConfig, (DistilBertTokenizer, DistilBertTokenizerFast)),
        (AlbertConfig, (AlbertTokenizer, None)),
        (CamembertConfig, (CamembertTokenizer, None)),
        (XLMRobertaConfig, (XLMRobertaTokenizer, None)),
        (BartConfig, (BartTokenizer, None)),
        (RobertaConfig, (RobertaTokenizer, RobertaTokenizerFast)),
        (BertConfig, (BertTokenizer, BertTokenizerFast)),
        (OpenAIGPTConfig, (OpenAIGPTTokenizer, OpenAIGPTTokenizerFast)),
        (GPT2Config, (GPT2Tokenizer, GPT2TokenizerFast)),
        (TransfoXLConfig, (TransfoXLTokenizer, TransfoXLTokenizerFast)),
        (XLNetConfig, (XLNetTokenizer, None)),
        (FlaubertConfig, (FlaubertTokenizer, None)),
        (XLMConfig, (XLMTokenizer, None)),
        (CTRLConfig, (CTRLTokenizer, None)),
    ]
)


class AutoTokenizer:
    r""":class:`~transformers.AutoTokenizer` is a generic tokenizer class
        that will be instantiated as one of the tokenizer classes of the library
        when created with the `AutoTokenizer.from_pretrained(pretrained_model_name_or_path)`
        class method.

        The `from_pretrained()` method take care of returning the correct tokenizer class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string.

        The tokenizer class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `t5`: T5Tokenizer (T5 model)
            - contains `distilbert`: DistilBertTokenizer (DistilBert model)
            - contains `albert`: AlbertTokenizer (ALBERT model)
            - contains `camembert`: CamembertTokenizer (CamemBERT model)
            - contains `xlm-roberta`: XLMRobertaTokenizer (XLM-RoBERTa model)
            - contains `roberta`: RobertaTokenizer (RoBERTa model)
            - contains `bert`: BertTokenizer (Bert model)
            - contains `openai-gpt`: OpenAIGPTTokenizer (OpenAI GPT model)
            - contains `gpt2`: GPT2Tokenizer (OpenAI GPT-2 model)
            - contains `transfo-xl`: TransfoXLTokenizer (Transformer-XL model)
            - contains `xlnet`: XLNetTokenizer (XLNet model)
            - contains `xlm`: XLMTokenizer (XLM model)
            - contains `ctrl`: CTRLTokenizer (Salesforce CTRL model)

        This class cannot be instantiated using `__init__()` (throw an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoTokenizer is designed to be instantiated "
            "using the `AutoTokenizer.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        r""" Instantiate one of the tokenizer classes of the library
        from a pre-trained model vocabulary.

        The tokenizer class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `t5`: T5Tokenizer (T5 model)
            - contains `distilbert`: DistilBertTokenizer (DistilBert model)
            - contains `albert`: AlbertTokenizer (ALBERT model)
            - contains `camembert`: CamembertTokenizer (CamemBERT model)
            - contains `xlm-roberta`: XLMRobertaTokenizer (XLM-RoBERTa model)
            - contains `roberta`: RobertaTokenizer (RoBERTa model)
            - contains `bert-base-japanese`: BertJapaneseTokenizer (Bert model)
            - contains `bert`: BertTokenizer (Bert model)
            - contains `openai-gpt`: OpenAIGPTTokenizer (OpenAI GPT model)
            - contains `gpt2`: GPT2Tokenizer (OpenAI GPT-2 model)
            - contains `transfo-xl`: TransfoXLTokenizer (Transformer-XL model)
            - contains `xlnet`: XLNetTokenizer (XLNet model)
            - contains `xlm`: XLMTokenizer (XLM model)
            - contains `ctrl`: CTRLTokenizer (Salesforce CTRL model)

        Params:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a predefined tokenizer to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a predefined tokenizer that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing vocabulary files required by the tokenizer, for instance saved using the :func:`~transformers.PreTrainedTokenizer.save_pretrained` method, e.g.: ``./my_model_directory/``.
                - (not applicable to all derived classes) a path or url to a single saved vocabulary file if and only if the tokenizer only requires a single vocabulary file (e.g. Bert, XLNet), e.g.: ``./my_model_directory/vocab.txt``.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded predefined tokenizer vocabulary files should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the vocabulary files and override the cached versions if they exists.

            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            use_fast: (`optional`) boolean, default False:
                Indicate if transformers should try to load the fast version of the tokenizer (True) or use the Python one (False).

            inputs: (`optional`) positional arguments: will be passed to the Tokenizer ``__init__`` method.

            kwargs: (`optional`) keyword arguments: will be passed to the Tokenizer ``__init__`` method. Can be used to set special tokens like ``bos_token``, ``eos_token``, ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``, ``mask_token``, ``additional_special_tokens``. See parameters in the doc string of :class:`~transformers.PreTrainedTokenizer` for details.

        Examples::

            # Download vocabulary from S3 and cache.
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

            # Download vocabulary from S3 (user-uploaded) and cache.
            tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-german-cased')

            # If vocabulary files are in a directory (e.g. tokenizer was saved using `save_pretrained('./test/saved_model/')`)
            tokenizer = AutoTokenizer.from_pretrained('./test/bert_saved_model/')

        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        if "bert-base-japanese" in pretrained_model_name_or_path:
            return BertJapaneseTokenizer.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

        use_fast = kwargs.pop("use_fast", False)
        for config_class, (tokenizer_class_py, tokenizer_class_fast) in TOKENIZER_MAPPING.items():
            if isinstance(config, config_class):
                if tokenizer_class_fast and use_fast:
                    return tokenizer_class_fast.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
                else:
                    return tokenizer_class_py.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

        raise ValueError(
            "Unrecognized configuration class {} to build an AutoTokenizer.\n"
            "Model type should be one of {}.".format(
                config.__class__, ", ".join(c.__name__ for c in TOKENIZER_MAPPING.keys())
            )
        )
