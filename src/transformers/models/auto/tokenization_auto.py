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

from ...configuration_utils import PretrainedConfig
from ...file_utils import is_sentencepiece_available, is_tokenizers_available
from ...utils import logging
from ..bart.tokenization_bart import BartTokenizer
from ..bert.tokenization_bert import BertTokenizer
from ..bert_japanese.tokenization_bert_japanese import BertJapaneseTokenizer
from ..bertweet.tokenization_bertweet import BertweetTokenizer
from ..blenderbot.tokenization_blenderbot import BlenderbotSmallTokenizer
from ..ctrl.tokenization_ctrl import CTRLTokenizer
from ..deberta.tokenization_deberta import DebertaTokenizer
from ..distilbert.tokenization_distilbert import DistilBertTokenizer
from ..dpr.tokenization_dpr import DPRQuestionEncoderTokenizer
from ..electra.tokenization_electra import ElectraTokenizer
from ..flaubert.tokenization_flaubert import FlaubertTokenizer
from ..fsmt.tokenization_fsmt import FSMTTokenizer
from ..funnel.tokenization_funnel import FunnelTokenizer
from ..gpt2.tokenization_gpt2 import GPT2Tokenizer
from ..herbert.tokenization_herbert import HerbertTokenizer
from ..layoutlm.tokenization_layoutlm import LayoutLMTokenizer
from ..longformer.tokenization_longformer import LongformerTokenizer
from ..lxmert.tokenization_lxmert import LxmertTokenizer
from ..mobilebert.tokenization_mobilebert import MobileBertTokenizer
from ..openai.tokenization_openai import OpenAIGPTTokenizer
from ..phobert.tokenization_phobert import PhobertTokenizer
from ..prophetnet.tokenization_prophetnet import ProphetNetTokenizer
from ..rag.tokenization_rag import RagTokenizer
from ..retribert.tokenization_retribert import RetriBertTokenizer
from ..roberta.tokenization_roberta import RobertaTokenizer
from ..squeezebert.tokenization_squeezebert import SqueezeBertTokenizer
from ..transfo_xl.tokenization_transfo_xl import TransfoXLTokenizer
from ..xlm.tokenization_xlm import XLMTokenizer
from .configuration_auto import (
    AlbertConfig,
    AutoConfig,
    BartConfig,
    BertConfig,
    BertGenerationConfig,
    BlenderbotConfig,
    CamembertConfig,
    CTRLConfig,
    DebertaConfig,
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
    MT5Config,
    OpenAIGPTConfig,
    PegasusConfig,
    ProphetNetConfig,
    RagConfig,
    ReformerConfig,
    RetriBertConfig,
    RobertaConfig,
    SqueezeBertConfig,
    T5Config,
    TransfoXLConfig,
    XLMConfig,
    XLMProphetNetConfig,
    XLMRobertaConfig,
    XLNetConfig,
    replace_list_option_in_docstrings,
)


if is_sentencepiece_available():
    from ..albert.tokenization_albert import AlbertTokenizer
    from ..bert_generation.tokenization_bert_generation import BertGenerationTokenizer
    from ..camembert.tokenization_camembert import CamembertTokenizer
    from ..marian.tokenization_marian import MarianTokenizer
    from ..mbart.tokenization_mbart import MBartTokenizer
    from ..pegasus.tokenization_pegasus import PegasusTokenizer
    from ..reformer.tokenization_reformer import ReformerTokenizer
    from ..t5.tokenization_t5 import T5Tokenizer
    from ..xlm_prophetnet.tokenization_xlm_prophetnet import XLMProphetNetTokenizer
    from ..xlm_roberta.tokenization_xlm_roberta import XLMRobertaTokenizer
    from ..xlnet.tokenization_xlnet import XLNetTokenizer
else:
    AlbertTokenizer = None
    BertGenerationTokenizer = None
    CamembertTokenizer = None
    MarianTokenizer = None
    MBartTokenizer = None
    PegasusTokenizer = None
    ReformerTokenizer = None
    T5Tokenizer = None
    XLMRobertaTokenizer = None
    XLNetTokenizer = None
    XLMProphetNetTokenizer = None

if is_tokenizers_available():
    from ..albert.tokenization_albert_fast import AlbertTokenizerFast
    from ..bart.tokenization_bart_fast import BartTokenizerFast
    from ..bert.tokenization_bert_fast import BertTokenizerFast
    from ..camembert.tokenization_camembert_fast import CamembertTokenizerFast
    from ..distilbert.tokenization_distilbert_fast import DistilBertTokenizerFast
    from ..dpr.tokenization_dpr_fast import DPRQuestionEncoderTokenizerFast
    from ..electra.tokenization_electra_fast import ElectraTokenizerFast
    from ..funnel.tokenization_funnel_fast import FunnelTokenizerFast
    from ..gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
    from ..herbert.tokenization_herbert_fast import HerbertTokenizerFast
    from ..layoutlm.tokenization_layoutlm_fast import LayoutLMTokenizerFast
    from ..longformer.tokenization_longformer_fast import LongformerTokenizerFast
    from ..lxmert.tokenization_lxmert_fast import LxmertTokenizerFast
    from ..mbart.tokenization_mbart_fast import MBartTokenizerFast
    from ..mobilebert.tokenization_mobilebert_fast import MobileBertTokenizerFast
    from ..openai.tokenization_openai_fast import OpenAIGPTTokenizerFast
    from ..pegasus.tokenization_pegasus_fast import PegasusTokenizerFast
    from ..reformer.tokenization_reformer_fast import ReformerTokenizerFast
    from ..retribert.tokenization_retribert_fast import RetriBertTokenizerFast
    from ..roberta.tokenization_roberta_fast import RobertaTokenizerFast
    from ..squeezebert.tokenization_squeezebert_fast import SqueezeBertTokenizerFast
    from ..t5.tokenization_t5_fast import T5TokenizerFast
    from ..xlm_roberta.tokenization_xlm_roberta_fast import XLMRobertaTokenizerFast
    from ..xlnet.tokenization_xlnet_fast import XLNetTokenizerFast
else:
    AlbertTokenizerFast = None
    BartTokenizerFast = None
    BertTokenizerFast = None
    CamembertTokenizerFast = None
    DistilBertTokenizerFast = None
    DPRQuestionEncoderTokenizerFast = None
    ElectraTokenizerFast = None
    FunnelTokenizerFast = None
    GPT2TokenizerFast = None
    HerbertTokenizerFast = None
    LayoutLMTokenizerFast = None
    LongformerTokenizerFast = None
    LxmertTokenizerFast = None
    MBartTokenizerFast = None
    MobileBertTokenizerFast = None
    OpenAIGPTTokenizerFast = None
    PegasusTokenizerFast = None
    ReformerTokenizerFast = None
    RetriBertTokenizerFast = None
    RobertaTokenizerFast = None
    SqueezeBertTokenizerFast = None
    T5TokenizerFast = None
    XLMRobertaTokenizerFast = None
    XLNetTokenizerFast = None

logger = logging.get_logger(__name__)


TOKENIZER_MAPPING = OrderedDict(
    [
        (RetriBertConfig, (RetriBertTokenizer, RetriBertTokenizerFast)),
        (T5Config, (T5Tokenizer, T5TokenizerFast)),
        (MT5Config, (T5Tokenizer, T5TokenizerFast)),
        (MobileBertConfig, (MobileBertTokenizer, MobileBertTokenizerFast)),
        (DistilBertConfig, (DistilBertTokenizer, DistilBertTokenizerFast)),
        (AlbertConfig, (AlbertTokenizer, AlbertTokenizerFast)),
        (CamembertConfig, (CamembertTokenizer, CamembertTokenizerFast)),
        (PegasusConfig, (PegasusTokenizer, PegasusTokenizerFast)),
        (MBartConfig, (MBartTokenizer, MBartTokenizerFast)),
        (XLMRobertaConfig, (XLMRobertaTokenizer, XLMRobertaTokenizerFast)),
        (MarianConfig, (MarianTokenizer, None)),
        (BlenderbotConfig, (BlenderbotSmallTokenizer, None)),
        (LongformerConfig, (LongformerTokenizer, LongformerTokenizerFast)),
        (BartConfig, (BartTokenizer, BartTokenizerFast)),
        (LongformerConfig, (LongformerTokenizer, LongformerTokenizerFast)),
        (RobertaConfig, (RobertaTokenizer, RobertaTokenizerFast)),
        (ReformerConfig, (ReformerTokenizer, ReformerTokenizerFast)),
        (ElectraConfig, (ElectraTokenizer, ElectraTokenizerFast)),
        (FunnelConfig, (FunnelTokenizer, FunnelTokenizerFast)),
        (LxmertConfig, (LxmertTokenizer, LxmertTokenizerFast)),
        (LayoutLMConfig, (LayoutLMTokenizer, LayoutLMTokenizerFast)),
        (DPRConfig, (DPRQuestionEncoderTokenizer, DPRQuestionEncoderTokenizerFast)),
        (SqueezeBertConfig, (SqueezeBertTokenizer, SqueezeBertTokenizerFast)),
        (BertConfig, (BertTokenizer, BertTokenizerFast)),
        (OpenAIGPTConfig, (OpenAIGPTTokenizer, OpenAIGPTTokenizerFast)),
        (GPT2Config, (GPT2Tokenizer, GPT2TokenizerFast)),
        (TransfoXLConfig, (TransfoXLTokenizer, None)),
        (XLNetConfig, (XLNetTokenizer, XLNetTokenizerFast)),
        (FlaubertConfig, (FlaubertTokenizer, None)),
        (XLMConfig, (XLMTokenizer, None)),
        (CTRLConfig, (CTRLTokenizer, None)),
        (FSMTConfig, (FSMTTokenizer, None)),
        (BertGenerationConfig, (BertGenerationTokenizer, None)),
        (DebertaConfig, (DebertaTokenizer, None)),
        (RagConfig, (RagTokenizer, None)),
        (XLMProphetNetConfig, (XLMProphetNetTokenizer, None)),
        (ProphetNetConfig, (ProphetNetTokenizer, None)),
    ]
)

# For tokenizers which are not directly mapped from a config
NO_CONFIG_TOKENIZER = [
    BertJapaneseTokenizer,
    BertweetTokenizer,
    HerbertTokenizer,
    HerbertTokenizerFast,
    PhobertTokenizer,
]


SLOW_TOKENIZER_MAPPING = {
    k: (v[0] if v[0] is not None else v[1])
    for k, v in TOKENIZER_MAPPING.items()
    if (v[0] is not None or v[1] is not None)
}


def tokenizer_class_from_name(class_name: str):
    all_tokenizer_classes = (
        [v[0] for v in TOKENIZER_MAPPING.values() if v[0] is not None]
        + [v[1] for v in TOKENIZER_MAPPING.values() if v[1] is not None]
        + NO_CONFIG_TOKENIZER
    )
    for c in all_tokenizer_classes:
        if c.__name__ == class_name:
            return c


class AutoTokenizer:
    r"""
    This is a generic tokenizer class that will be instantiated as one of the tokenizer classes of the library when
    created with the :meth:`AutoTokenizer.from_pretrained` class method.

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

                    - A string, the `model id` of a predefined tokenizer hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                      a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing vocabulary files required by the tokenizer, for instance saved
                      using the :func:`~transformers.PreTrainedTokenizer.save_pretrained` method, e.g.,
                      ``./my_model_directory/``.
                    - A path or url to a single saved vocabulary file if and only if the tokenizer only requires a
                      single vocabulary file (like Bert or XLNet), e.g.: ``./my_model_directory/vocab.txt``. (Not
                      applicable to all derived classes)
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
                A dictionary of proxy servers to use by protocol or endpoint, e.g., :obj:`{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
            subfolder (:obj:`str`, `optional`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co (e.g. for
                facebook/rag-token-base), specify it here.
            use_fast (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to try to load the fast version of the tokenizer.
            kwargs (additional keyword arguments, `optional`):
                Will be passed to the Tokenizer ``__init__()`` method. Can be used to set special tokens like
                ``bos_token``, ``eos_token``, ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``,
                ``mask_token``, ``additional_special_tokens``. See parameters in the ``__init__()`` for more details.

        Examples::

            >>> from transformers import AutoTokenizer

            >>> # Download vocabulary from huggingface.co and cache.
            >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

            >>> # Download vocabulary from huggingface.co (user-uploaded) and cache.
            >>> tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-german-cased')

            >>> # If vocabulary files are in a directory (e.g. tokenizer was saved using `save_pretrained('./test/saved_model/')`)
            >>> tokenizer = AutoTokenizer.from_pretrained('./test/bert_saved_model/')

        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        use_fast = kwargs.pop("use_fast", True)

        if config.tokenizer_class is not None:
            tokenizer_class = None
            if use_fast and not config.tokenizer_class.endswith("Fast"):
                tokenizer_class_candidate = f"{config.tokenizer_class}Fast"
                tokenizer_class = tokenizer_class_from_name(tokenizer_class_candidate)
            if tokenizer_class is None:
                tokenizer_class_candidate = config.tokenizer_class
                tokenizer_class = tokenizer_class_from_name(tokenizer_class_candidate)

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
            if tokenizer_class_fast and (use_fast or tokenizer_class_py is None):
                return tokenizer_class_fast.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
            else:
                return tokenizer_class_py.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

        raise ValueError(
            "Unrecognized configuration class {} to build an AutoTokenizer.\n"
            "Model type should be one of {}.".format(
                config.__class__, ", ".join(c.__name__ for c in TOKENIZER_MAPPING.keys())
            )
        )
