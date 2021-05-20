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

from ... import GPTNeoConfig
from ...configuration_utils import PretrainedConfig
from ...file_utils import is_sentencepiece_available, is_tokenizers_available
from ...utils import logging
from ..bart.tokenization_bart import BartTokenizer
from ..bert.tokenization_bert import BertTokenizer
from ..bert_japanese.tokenization_bert_japanese import BertJapaneseTokenizer
from ..bertweet.tokenization_bertweet import BertweetTokenizer
from ..blenderbot.tokenization_blenderbot import BlenderbotTokenizer
from ..blenderbot_small.tokenization_blenderbot_small import BlenderbotSmallTokenizer
from ..convbert.tokenization_convbert import ConvBertTokenizer
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
from ..led.tokenization_led import LEDTokenizer
from ..longformer.tokenization_longformer import LongformerTokenizer
from ..luke.tokenization_luke import LukeTokenizer
from ..lxmert.tokenization_lxmert import LxmertTokenizer
from ..mobilebert.tokenization_mobilebert import MobileBertTokenizer
from ..mpnet.tokenization_mpnet import MPNetTokenizer
from ..openai.tokenization_openai import OpenAIGPTTokenizer
from ..phobert.tokenization_phobert import PhobertTokenizer
from ..prophetnet.tokenization_prophetnet import ProphetNetTokenizer
from ..rag.tokenization_rag import RagTokenizer
from ..retribert.tokenization_retribert import RetriBertTokenizer
from ..roberta.tokenization_roberta import RobertaTokenizer
from ..roformer.tokenization_roformer import RoFormerTokenizer
from ..squeezebert.tokenization_squeezebert import SqueezeBertTokenizer
from ..tapas.tokenization_tapas import TapasTokenizer
from ..transfo_xl.tokenization_transfo_xl import TransfoXLTokenizer
from ..wav2vec2.tokenization_wav2vec2 import Wav2Vec2CTCTokenizer
from ..xlm.tokenization_xlm import XLMTokenizer
from .configuration_auto import (
    AlbertConfig,
    AutoConfig,
    BartConfig,
    BertConfig,
    BertGenerationConfig,
    BigBirdConfig,
    BigBirdPegasusConfig,
    BlenderbotConfig,
    BlenderbotSmallConfig,
    CamembertConfig,
    ConvBertConfig,
    CTRLConfig,
    DebertaConfig,
    DebertaV2Config,
    DistilBertConfig,
    DPRConfig,
    ElectraConfig,
    EncoderDecoderConfig,
    FlaubertConfig,
    FSMTConfig,
    FunnelConfig,
    GPT2Config,
    IBertConfig,
    LayoutLMConfig,
    LEDConfig,
    LongformerConfig,
    LukeConfig,
    LxmertConfig,
    M2M100Config,
    MarianConfig,
    MBartConfig,
    MobileBertConfig,
    MPNetConfig,
    MT5Config,
    OpenAIGPTConfig,
    PegasusConfig,
    ProphetNetConfig,
    RagConfig,
    ReformerConfig,
    RetriBertConfig,
    RobertaConfig,
    RoFormerConfig,
    Speech2TextConfig,
    SqueezeBertConfig,
    T5Config,
    TapasConfig,
    TransfoXLConfig,
    Wav2Vec2Config,
    XLMConfig,
    XLMProphetNetConfig,
    XLMRobertaConfig,
    XLNetConfig,
    replace_list_option_in_docstrings,
)


if is_sentencepiece_available():
    from ..albert.tokenization_albert import AlbertTokenizer
    from ..barthez.tokenization_barthez import BarthezTokenizer
    from ..bert_generation.tokenization_bert_generation import BertGenerationTokenizer
    from ..big_bird.tokenization_big_bird import BigBirdTokenizer
    from ..camembert.tokenization_camembert import CamembertTokenizer
    from ..cpm.tokenization_cpm import CpmTokenizer
    from ..deberta_v2.tokenization_deberta_v2 import DebertaV2Tokenizer
    from ..m2m_100 import M2M100Tokenizer
    from ..marian.tokenization_marian import MarianTokenizer
    from ..mbart.tokenization_mbart import MBartTokenizer
    from ..mbart.tokenization_mbart50 import MBart50Tokenizer
    from ..mt5 import MT5Tokenizer
    from ..pegasus.tokenization_pegasus import PegasusTokenizer
    from ..reformer.tokenization_reformer import ReformerTokenizer
    from ..speech_to_text import Speech2TextTokenizer
    from ..t5.tokenization_t5 import T5Tokenizer
    from ..xlm_prophetnet.tokenization_xlm_prophetnet import XLMProphetNetTokenizer
    from ..xlm_roberta.tokenization_xlm_roberta import XLMRobertaTokenizer
    from ..xlnet.tokenization_xlnet import XLNetTokenizer
else:
    AlbertTokenizer = None
    BarthezTokenizer = None
    BertGenerationTokenizer = None
    BigBirdTokenizer = None
    CamembertTokenizer = None
    CpmTokenizer = None
    DebertaV2Tokenizer = None
    MarianTokenizer = None
    MBartTokenizer = None
    MBart50Tokenizer = None
    MT5Tokenizer = None
    PegasusTokenizer = None
    ReformerTokenizer = None
    T5Tokenizer = None
    XLMRobertaTokenizer = None
    XLNetTokenizer = None
    XLMProphetNetTokenizer = None
    M2M100Tokenizer = None
    Speech2TextTokenizer = None

if is_tokenizers_available():
    from ..albert.tokenization_albert_fast import AlbertTokenizerFast
    from ..bart.tokenization_bart_fast import BartTokenizerFast
    from ..barthez.tokenization_barthez_fast import BarthezTokenizerFast
    from ..bert.tokenization_bert_fast import BertTokenizerFast
    from ..big_bird.tokenization_big_bird_fast import BigBirdTokenizerFast
    from ..camembert.tokenization_camembert_fast import CamembertTokenizerFast
    from ..convbert.tokenization_convbert_fast import ConvBertTokenizerFast
    from ..deberta.tokenization_deberta_fast import DebertaTokenizerFast
    from ..distilbert.tokenization_distilbert_fast import DistilBertTokenizerFast
    from ..dpr.tokenization_dpr_fast import DPRQuestionEncoderTokenizerFast
    from ..electra.tokenization_electra_fast import ElectraTokenizerFast
    from ..funnel.tokenization_funnel_fast import FunnelTokenizerFast
    from ..gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
    from ..herbert.tokenization_herbert_fast import HerbertTokenizerFast
    from ..layoutlm.tokenization_layoutlm_fast import LayoutLMTokenizerFast
    from ..led.tokenization_led_fast import LEDTokenizerFast
    from ..longformer.tokenization_longformer_fast import LongformerTokenizerFast
    from ..lxmert.tokenization_lxmert_fast import LxmertTokenizerFast
    from ..mbart.tokenization_mbart50_fast import MBart50TokenizerFast
    from ..mbart.tokenization_mbart_fast import MBartTokenizerFast
    from ..mobilebert.tokenization_mobilebert_fast import MobileBertTokenizerFast
    from ..mpnet.tokenization_mpnet_fast import MPNetTokenizerFast
    from ..mt5 import MT5TokenizerFast
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
    BarthezTokenizerFast = None
    BertTokenizerFast = None
    BigBirdTokenizerFast = None
    CamembertTokenizerFast = None
    ConvBertTokenizerFast = None
    DebertaTokenizerFast = None
    DistilBertTokenizerFast = None
    DPRQuestionEncoderTokenizerFast = None
    ElectraTokenizerFast = None
    FunnelTokenizerFast = None
    GPT2TokenizerFast = None
    HerbertTokenizerFast = None
    LayoutLMTokenizerFast = None
    LEDTokenizerFast = None
    LongformerTokenizerFast = None
    LxmertTokenizerFast = None
    MBartTokenizerFast = None
    MBart50TokenizerFast = None
    MobileBertTokenizerFast = None
    MPNetTokenizerFast = None
    MT5TokenizerFast = None
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
        (RoFormerConfig, (RoFormerTokenizer, None)),
        (T5Config, (T5Tokenizer, T5TokenizerFast)),
        (MT5Config, (MT5Tokenizer, MT5TokenizerFast)),
        (MobileBertConfig, (MobileBertTokenizer, MobileBertTokenizerFast)),
        (DistilBertConfig, (DistilBertTokenizer, DistilBertTokenizerFast)),
        (AlbertConfig, (AlbertTokenizer, AlbertTokenizerFast)),
        (CamembertConfig, (CamembertTokenizer, CamembertTokenizerFast)),
        (PegasusConfig, (PegasusTokenizer, PegasusTokenizerFast)),
        (MBartConfig, (MBartTokenizer, MBartTokenizerFast)),
        (XLMRobertaConfig, (XLMRobertaTokenizer, XLMRobertaTokenizerFast)),
        (MarianConfig, (MarianTokenizer, None)),
        (BlenderbotSmallConfig, (BlenderbotSmallTokenizer, None)),
        (BlenderbotConfig, (BlenderbotTokenizer, None)),
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
        (DebertaConfig, (DebertaTokenizer, DebertaTokenizerFast)),
        (DebertaV2Config, (DebertaV2Tokenizer, None)),
        (RagConfig, (RagTokenizer, None)),
        (XLMProphetNetConfig, (XLMProphetNetTokenizer, None)),
        (Speech2TextConfig, (Speech2TextTokenizer, None)),
        (M2M100Config, (M2M100Tokenizer, None)),
        (ProphetNetConfig, (ProphetNetTokenizer, None)),
        (MPNetConfig, (MPNetTokenizer, MPNetTokenizerFast)),
        (TapasConfig, (TapasTokenizer, None)),
        (LEDConfig, (LEDTokenizer, LEDTokenizerFast)),
        (ConvBertConfig, (ConvBertTokenizer, ConvBertTokenizerFast)),
        (BigBirdConfig, (BigBirdTokenizer, BigBirdTokenizerFast)),
        (IBertConfig, (RobertaTokenizer, RobertaTokenizerFast)),
        (Wav2Vec2Config, (Wav2Vec2CTCTokenizer, None)),
        (GPTNeoConfig, (GPT2Tokenizer, GPT2TokenizerFast)),
        (LukeConfig, (LukeTokenizer, None)),
        (BigBirdPegasusConfig, (PegasusTokenizer, PegasusTokenizerFast)),
    ]
)

# For tokenizers which are not directly mapped from a config
NO_CONFIG_TOKENIZER = [
    BertJapaneseTokenizer,
    BertweetTokenizer,
    CpmTokenizer,
    HerbertTokenizer,
    HerbertTokenizerFast,
    PhobertTokenizer,
    BarthezTokenizer,
    BarthezTokenizerFast,
    MBart50Tokenizer,
    MBart50TokenizerFast,
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
        + [v for v in NO_CONFIG_TOKENIZER if v is not None]
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
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
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
            cache_dir (:obj:`str` or :obj:`os.PathLike`, `optional`):
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
        kwargs["_from_auto"] = True
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
                    f"Tokenizer class {tokenizer_class_candidate} does not exist or is not currently imported."
                )
            return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

        # if model is an encoder decoder, the encoder tokenizer class is used by default
        if isinstance(config, EncoderDecoderConfig):
            if type(config.decoder) is not type(config.encoder):  # noqa: E721
                logger.warning(
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
                if tokenizer_class_py is not None:
                    return tokenizer_class_py.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
                else:
                    raise ValueError(
                        "This tokenizer cannot be instantiated. Please make sure you have `sentencepiece` installed "
                        "in order to use this tokenizer."
                    )

        raise ValueError(
            f"Unrecognized configuration class {config.__class__} to build an AutoTokenizer.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in TOKENIZER_MAPPING.keys())}."
        )
