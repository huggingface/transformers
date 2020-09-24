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
""" Auto Tokenizer class. """


from collections import OrderedDict

from .configuration_auto import (
    AlbertConfig,
    AutoConfig,
    BartConfig,
    BertConfig,
    BertGenerationConfig,
    CamembertConfig,
    CTRLConfig,
    DistilBertConfig,
    DPRConfig,
    ElectraConfig,
    EncoderDecoderConfig,
    FlaubertConfig,
    FSMTConfig,
    FunnelConfig,
    GPT2Config,
    LayoutLMConfig,
    LongformerConfig,
    LxmertConfig,
    MarianConfig,
    MBartConfig,
    MobileBertConfig,
    OpenAIGPTConfig,
    PegasusConfig,
    RagConfig,
    ReformerConfig,
    RetriBertConfig,
    RobertaConfig,
    T5Config,
    TransfoXLConfig,
    XLMConfig,
    XLMRobertaConfig,
    XLNetConfig,
    replace_list_option_in_docstrings,
)
from .configuration_utils import PretrainedConfig
from .tokenization_albert import AlbertTokenizer
from .tokenization_bart import BartTokenizer, BartTokenizerFast
from .tokenization_bert import BertTokenizer, BertTokenizerFast
from .tokenization_bert_generation import BertGenerationTokenizer
from .tokenization_bert_japanese import BertJapaneseTokenizer
from .tokenization_bertweet import BertweetTokenizer
from .tokenization_camembert import CamembertTokenizer
from .tokenization_ctrl import CTRLTokenizer
from .tokenization_distilbert import DistilBertTokenizer, DistilBertTokenizerFast
from .tokenization_dpr import DPRQuestionEncoderTokenizer, DPRQuestionEncoderTokenizerFast
from .tokenization_electra import ElectraTokenizer, ElectraTokenizerFast
from .tokenization_flaubert import FlaubertTokenizer
from .tokenization_fsmt import FSMTTokenizer
from .tokenization_funnel import FunnelTokenizer, FunnelTokenizerFast
from .tokenization_gpt2 import GPT2Tokenizer, GPT2TokenizerFast
from .tokenization_layoutlm import LayoutLMTokenizer, LayoutLMTokenizerFast
from .tokenization_longformer import LongformerTokenizer, LongformerTokenizerFast
from .tokenization_lxmert import LxmertTokenizer, LxmertTokenizerFast
from .tokenization_marian import MarianTokenizer
from .tokenization_mbart import MBartTokenizer
from .tokenization_mobilebert import MobileBertTokenizer, MobileBertTokenizerFast
from .tokenization_openai import OpenAIGPTTokenizer, OpenAIGPTTokenizerFast
from .tokenization_pegasus import PegasusTokenizer
from .tokenization_phobert import PhobertTokenizer
from .tokenization_rag import RagTokenizer
from .tokenization_reformer import ReformerTokenizer
from .tokenization_retribert import RetriBertTokenizer, RetriBertTokenizerFast
from .tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast
from .tokenization_t5 import T5Tokenizer
from .tokenization_transfo_xl import TransfoXLTokenizer, TransfoXLTokenizerFast
from .tokenization_xlm import XLMTokenizer
from .tokenization_xlm_roberta import XLMRobertaTokenizer
from .tokenization_xlnet import XLNetTokenizer
from .utils import logging


logger = logging.get_logger(__name__)


TOKENIZER_MAPPING = OrderedDict(
    [
        (RetriBertConfig, (RetriBertTokenizer, RetriBertTokenizerFast)),
        (T5Config, (T5Tokenizer, None)),
        (MobileBertConfig, (MobileBertTokenizer, MobileBertTokenizerFast)),
        (DistilBertConfig, (DistilBertTokenizer, DistilBertTokenizerFast)),
        (AlbertConfig, (AlbertTokenizer, None)),
        (CamembertConfig, (CamembertTokenizer, None)),
        (PegasusConfig, (PegasusTokenizer, None)),
        (MBartConfig, (MBartTokenizer, None)),
        (XLMRobertaConfig, (XLMRobertaTokenizer, None)),
        (MarianConfig, (MarianTokenizer, None)),
        (BartConfig, (BartTokenizer, BartTokenizerFast)),
        (LongformerConfig, (LongformerTokenizer, LongformerTokenizerFast)),
        (RobertaConfig, (BertweetTokenizer, None)),
        (RobertaConfig, (PhobertTokenizer, None)),
        (RobertaConfig, (RobertaTokenizer, RobertaTokenizerFast)),
        (ReformerConfig, (ReformerTokenizer, None)),
        (ElectraConfig, (ElectraTokenizer, ElectraTokenizerFast)),
        (FunnelConfig, (FunnelTokenizer, FunnelTokenizerFast)),
        (LxmertConfig, (LxmertTokenizer, LxmertTokenizerFast)),
        (LayoutLMConfig, (LayoutLMTokenizer, LayoutLMTokenizerFast)),
        (DPRConfig, (DPRQuestionEncoderTokenizer, DPRQuestionEncoderTokenizerFast)),
        (BertConfig, (BertTokenizer, BertTokenizerFast)),
        (OpenAIGPTConfig, (OpenAIGPTTokenizer, OpenAIGPTTokenizerFast)),
        (GPT2Config, (GPT2Tokenizer, GPT2TokenizerFast)),
        (TransfoXLConfig, (TransfoXLTokenizer, TransfoXLTokenizerFast)),
        (XLNetConfig, (XLNetTokenizer, None)),
        (FlaubertConfig, (FlaubertTokenizer, None)),
        (XLMConfig, (XLMTokenizer, None)),
        (CTRLConfig, (CTRLTokenizer, None)),
        (FSMTConfig, (FSMTTokenizer, None)),
        (BertGenerationConfig, (BertGenerationTokenizer, None)),
        (LayoutLMConfig, (LayoutLMTokenizer, None)),
        (RagConfig, (RagTokenizer, None)),
    ]
)

SLOW_TOKENIZER_MAPPING = {k: v[0] for k, v in TOKENIZER_MAPPING.items()}


class AutoTokenizer:
    r"""
    This is a generic tokenizer class that will be instantiated as one of the tokenizer classes of the library
    when created with the :meth:`AutoTokenizer.from_pretrained` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoTokenizer is designed to be instantiated "
            "using the `AutoTokenizer.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    @replace_list_option_in_docstrings(SLOW_TOKENIZER_MAPPING)
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        r"""
        Instantiate one of the tokenizer classes of the library from a pretrained model vocabulary.

        The tokenizer class to instantiate is selected based on the :obj:`model_type` property of the config object
        (either passed as an argument or loaded from :obj:`pretrained_model_name_or_path` if possible), or when it's
        missing, by falling back to using pattern matching on :obj:`pretrained_model_name_or_path`:

        List options

        Params:
            pretrained_model_name_or_path (:obj:`str`):
                Can be either:

                    - A string with the `shortcut name` of a predefined tokenizer to load from cache or download, e.g.,
                      ``bert-base-uncased``.
                    - A string with the `identifier name` of a predefined tokenizer that was user-uploaded to our S3,
                      e.g., ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing vocabulary files required by the tokenizer, for instance saved
                      using the :func:`~transformers.PreTrainedTokenizer.save_pretrained` method, e.g.,
                      ``./my_model_directory/``.
                    - A path or url to a single saved vocabulary file if and only if the tokenizer only requires a
                      single vocabulary file (like Bert or XLNet), e.g.: ``./my_model_directory/vocab.txt``.
                      (Not applicable to all derived classes)
            inputs (additional positional arguments, `optional`):
                Will be passed along to the Tokenizer ``__init__()`` method.
            config (:class:`~transformers.PreTrainedConfig`, `optional`)
                The configuration object used to dertermine the tokenizer class to instantiate.
            cache_dir (:obj:`str`, `optional`):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force the (re-)download the model weights and configuration files and override the
                cached versions if they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (:obj:`Dict[str, str]`, `optional`):
                A dictionary of proxy servers to use by protocol or endpoint, e.g.,
                :obj:`{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each
                request.
            use_fast (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to try to load the fast version of the tokenizer.
            kwargs (additional keyword arguments, `optional`):
                Will be passed to the Tokenizer ``__init__()`` method. Can be used to set special tokens like
                ``bos_token``, ``eos_token``, ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``,
                ``mask_token``, ``additional_special_tokens``. See parameters in the ``__init__()`` for more details.

        Examples::

            >>> from transformers import AutoTokenizer

            >>> # Download vocabulary from S3 and cache.
            >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

            >>> # Download vocabulary from S3 (user-uploaded) and cache.
            >>> tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-german-cased')

            >>> # If vocabulary files are in a directory (e.g. tokenizer was saved using `save_pretrained('./test/saved_model/')`)
            >>> tokenizer = AutoTokenizer.from_pretrained('./test/bert_saved_model/')

        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        if "bert-base-japanese" in str(pretrained_model_name_or_path):
            return BertJapaneseTokenizer.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

        use_fast = kwargs.pop("use_fast", False)

        if config.tokenizer_class is not None:
            if use_fast and not config.tokenizer_class.endswith("Fast"):
                tokenizer_class_candidate = f"{config.tokenizer_class}Fast"
            else:
                tokenizer_class_candidate = config.tokenizer_class
            tokenizer_class = globals().get(tokenizer_class_candidate)
            if tokenizer_class is None:
                raise ValueError(
                    "Tokenizer class {} does not exist or is not currently imported.".format(tokenizer_class_candidate)
                )
            return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

        # if model is an encoder decoder, the encoder tokenizer class is used by default
        if isinstance(config, EncoderDecoderConfig):
            if type(config.decoder) is not type(config.encoder):  # noqa: E721
                logger.warn(
                    f"The encoder model config class: {config.encoder.__class__} is different from the decoder model "
                    f"config class: {config.decoder.__class}. It is not recommended to use the "
                    "`AutoTokenizer.from_pretrained()` method in this case. Please use the encoder and decoder "
                    "specific tokenizer classes."
                )
            config = config.encoder

        if type(config) in TOKENIZER_MAPPING.keys():
            tokenizer_class_py, tokenizer_class_fast = TOKENIZER_MAPPING[type(config)]
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
