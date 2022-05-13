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
""" Auto Tokenizer class."""

import importlib
import json
import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

from ...configuration_utils import PretrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import TOKENIZER_CONFIG_FILE
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import get_file_from_repo, is_sentencepiece_available, is_tokenizers_available, logging
from ..encoder_decoder import EncoderDecoderConfig
from .auto_factory import _LazyAutoMapping
from .configuration_auto import (
    CONFIG_MAPPING_NAMES,
    AutoConfig,
    config_class_to_model_type,
    model_type_to_module_name,
    replace_list_option_in_docstrings,
)


logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    # This significantly improves completion suggestion performance when
    # the transformers package is used with Microsoft's Pylance language server.
    TOKENIZER_MAPPING_NAMES: OrderedDict[str, Tuple[Optional[str], Optional[str]]] = OrderedDict()
else:
    TOKENIZER_MAPPING_NAMES = OrderedDict(
        [
            ("plbart", ("PLBartTokenizer" if is_sentencepiece_available() else None, None)),
            ("realm", ("RealmTokenizer", "RealmTokenizerFast" if is_tokenizers_available() else None)),
            ("fnet", ("FNetTokenizer", "FNetTokenizerFast" if is_tokenizers_available() else None)),
            ("retribert", ("RetriBertTokenizer", "RetriBertTokenizerFast" if is_tokenizers_available() else None)),
            ("roformer", ("RoFormerTokenizer", "RoFormerTokenizerFast" if is_tokenizers_available() else None)),
            (
                "t5",
                (
                    "T5Tokenizer" if is_sentencepiece_available() else None,
                    "T5TokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "mt5",
                (
                    "MT5Tokenizer" if is_sentencepiece_available() else None,
                    "MT5TokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("mobilebert", ("MobileBertTokenizer", "MobileBertTokenizerFast" if is_tokenizers_available() else None)),
            ("distilbert", ("DistilBertTokenizer", "DistilBertTokenizerFast" if is_tokenizers_available() else None)),
            (
                "albert",
                (
                    "AlbertTokenizer" if is_sentencepiece_available() else None,
                    "AlbertTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "camembert",
                (
                    "CamembertTokenizer" if is_sentencepiece_available() else None,
                    "CamembertTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "pegasus",
                (
                    "PegasusTokenizer" if is_sentencepiece_available() else None,
                    "PegasusTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "mbart",
                (
                    "MBartTokenizer" if is_sentencepiece_available() else None,
                    "MBartTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "xlm-roberta",
                (
                    "XLMRobertaTokenizer" if is_sentencepiece_available() else None,
                    "XLMRobertaTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("marian", ("MarianTokenizer" if is_sentencepiece_available() else None, None)),
            ("blenderbot-small", ("BlenderbotSmallTokenizer", None)),
            ("blenderbot", ("BlenderbotTokenizer", "BlenderbotTokenizerFast")),
            ("tapex", ("TapexTokenizer", None)),
            ("bart", ("BartTokenizer", "BartTokenizerFast")),
            ("longformer", ("LongformerTokenizer", "LongformerTokenizerFast" if is_tokenizers_available() else None)),
            ("roberta", ("RobertaTokenizer", "RobertaTokenizerFast" if is_tokenizers_available() else None)),
            (
                "reformer",
                (
                    "ReformerTokenizer" if is_sentencepiece_available() else None,
                    "ReformerTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("electra", ("ElectraTokenizer", "ElectraTokenizerFast" if is_tokenizers_available() else None)),
            ("funnel", ("FunnelTokenizer", "FunnelTokenizerFast" if is_tokenizers_available() else None)),
            ("lxmert", ("LxmertTokenizer", "LxmertTokenizerFast" if is_tokenizers_available() else None)),
            ("layoutlm", ("LayoutLMTokenizer", "LayoutLMTokenizerFast" if is_tokenizers_available() else None)),
            ("layoutlmv2", ("LayoutLMv2Tokenizer", "LayoutLMv2TokenizerFast" if is_tokenizers_available() else None)),
            ("layoutxlm", ("LayoutXLMTokenizer", "LayoutXLMTokenizerFast" if is_tokenizers_available() else None)),
            (
                "dpr",
                (
                    "DPRQuestionEncoderTokenizer",
                    "DPRQuestionEncoderTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "squeezebert",
                ("SqueezeBertTokenizer", "SqueezeBertTokenizerFast" if is_tokenizers_available() else None),
            ),
            ("bert", ("BertTokenizer", "BertTokenizerFast" if is_tokenizers_available() else None)),
            ("openai-gpt", ("OpenAIGPTTokenizer", "OpenAIGPTTokenizerFast" if is_tokenizers_available() else None)),
            ("gpt2", ("GPT2Tokenizer", "GPT2TokenizerFast" if is_tokenizers_available() else None)),
            ("gptj", ("GPT2Tokenizer", "GPT2TokenizerFast" if is_tokenizers_available() else None)),
            ("opt", ("GPT2Tokenizer", None)),
            ("transfo-xl", ("TransfoXLTokenizer", None)),
            (
                "xlnet",
                (
                    "XLNetTokenizer" if is_sentencepiece_available() else None,
                    "XLNetTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("flaubert", ("FlaubertTokenizer", None)),
            ("xlm", ("XLMTokenizer", None)),
            ("ctrl", ("CTRLTokenizer", None)),
            ("fsmt", ("FSMTTokenizer", None)),
            ("bert-generation", ("BertGenerationTokenizer" if is_sentencepiece_available() else None, None)),
            ("deberta", ("DebertaTokenizer", "DebertaTokenizerFast" if is_tokenizers_available() else None)),
            (
                "deberta-v2",
                (
                    "DebertaV2Tokenizer" if is_sentencepiece_available() else None,
                    "DebertaV2TokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("rag", ("RagTokenizer", None)),
            ("xlm-prophetnet", ("XLMProphetNetTokenizer" if is_sentencepiece_available() else None, None)),
            ("speech_to_text", ("Speech2TextTokenizer" if is_sentencepiece_available() else None, None)),
            ("speech_to_text_2", ("Speech2Text2Tokenizer", None)),
            ("m2m_100", ("M2M100Tokenizer" if is_sentencepiece_available() else None, None)),
            ("prophetnet", ("ProphetNetTokenizer", None)),
            ("mpnet", ("MPNetTokenizer", "MPNetTokenizerFast" if is_tokenizers_available() else None)),
            ("tapas", ("TapasTokenizer", None)),
            ("led", ("LEDTokenizer", "LEDTokenizerFast" if is_tokenizers_available() else None)),
            ("convbert", ("ConvBertTokenizer", "ConvBertTokenizerFast" if is_tokenizers_available() else None)),
            (
                "big_bird",
                (
                    "BigBirdTokenizer" if is_sentencepiece_available() else None,
                    "BigBirdTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("ibert", ("RobertaTokenizer", "RobertaTokenizerFast" if is_tokenizers_available() else None)),
            ("qdqbert", ("BertTokenizer", "BertTokenizerFast" if is_tokenizers_available() else None)),
            ("wav2vec2", ("Wav2Vec2CTCTokenizer", None)),
            ("hubert", ("Wav2Vec2CTCTokenizer", None)),
            ("gpt_neo", ("GPT2Tokenizer", "GPT2TokenizerFast" if is_tokenizers_available() else None)),
            ("luke", ("LukeTokenizer", None)),
            ("mluke", ("MLukeTokenizer" if is_sentencepiece_available() else None, None)),
            ("bigbird_pegasus", ("PegasusTokenizer", "PegasusTokenizerFast" if is_tokenizers_available() else None)),
            ("canine", ("CanineTokenizer", None)),
            ("bertweet", ("BertweetTokenizer", None)),
            ("bert-japanese", ("BertJapaneseTokenizer", None)),
            ("splinter", ("SplinterTokenizer", "SplinterTokenizerFast")),
            ("byt5", ("ByT5Tokenizer", None)),
            (
                "cpm",
                (
                    "CpmTokenizer" if is_sentencepiece_available() else None,
                    "CpmTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("herbert", ("HerbertTokenizer", "HerbertTokenizerFast" if is_tokenizers_available() else None)),
            ("phobert", ("PhobertTokenizer", None)),
            ("bartpho", ("BartphoTokenizer", None)),
            (
                "barthez",
                (
                    "BarthezTokenizer" if is_sentencepiece_available() else None,
                    "BarthezTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "mbart50",
                (
                    "MBart50Tokenizer" if is_sentencepiece_available() else None,
                    "MBart50TokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "rembert",
                (
                    "RemBertTokenizer" if is_sentencepiece_available() else None,
                    "RemBertTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "clip",
                (
                    "CLIPTokenizer",
                    "CLIPTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("wav2vec2_phoneme", ("Wav2Vec2PhonemeCTCTokenizer", None)),
            (
                "perceiver",
                (
                    "PerceiverTokenizer",
                    None,
                ),
            ),
            (
                "xglm",
                (
                    "XGLMTokenizer" if is_sentencepiece_available() else None,
                    "XGLMTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("visual_bert", ("BertTokenizer", "BertTokenizerFast" if is_tokenizers_available() else None)),
            ("megatron-bert", ("BertTokenizer", "BertTokenizerFast" if is_tokenizers_available() else None)),
            (
                "nystromformer",
                (
                    "AlbertTokenizer" if is_sentencepiece_available() else None,
                    "AlbertTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("xlm-roberta-xl", ("RobertaTokenizer", "RobertaTokenizerFast" if is_tokenizers_available() else None)),
            (
                "yoso",
                (
                    "AlbertTokenizer" if is_sentencepiece_available() else None,
                    "AlbertTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("data2vec-text", ("RobertaTokenizer", "RobertaTokenizerFast" if is_tokenizers_available() else None)),
        ]
    )

TOKENIZER_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TOKENIZER_MAPPING_NAMES)

CONFIG_TO_TYPE = {v: k for k, v in CONFIG_MAPPING_NAMES.items()}


def tokenizer_class_from_name(class_name: str):
    if class_name == "PreTrainedTokenizerFast":
        return PreTrainedTokenizerFast

    for module_name, tokenizers in TOKENIZER_MAPPING_NAMES.items():
        if class_name in tokenizers:
            module_name = model_type_to_module_name(module_name)

            module = importlib.import_module(f".{module_name}", "transformers.models")
            return getattr(module, class_name)

    for config, tokenizers in TOKENIZER_MAPPING._extra_content.items():
        for tokenizer in tokenizers:
            if getattr(tokenizer, "__name__", None) == class_name:
                return tokenizer

    return None


def get_tokenizer_config(
    pretrained_model_name_or_path: Union[str, os.PathLike],
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: bool = False,
    proxies: Optional[Dict[str, str]] = None,
    use_auth_token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    **kwargs,
):
    """
    Loads the tokenizer configuration from a pretrained model tokenizer configuration.

    Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
              huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced
              under a user or organization name, like `dbmdz/bert-base-german-cased`.
            - a path to a *directory* containing a configuration file saved using the
              [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.

        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download (`bool`, *optional*, defaults to `False`):
            Whether or not to delete incompletely received file. Attempts to resume the download if such a file exists.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        use_auth_token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `transformers-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.

    <Tip>

    Passing `use_auth_token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Dict`: The configuration of the tokenizer.

    Examples:

    ```python
    # Download configuration from huggingface.co and cache.
    tokenizer_config = get_tokenizer_config("bert-base-uncased")
    # This model does not have a tokenizer config so the result will be an empty dict.
    tokenizer_config = get_tokenizer_config("xlm-roberta-base")

    # Save a pretrained tokenizer locally and you can reload its config
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    tokenizer.save_pretrained("tokenizer-test")
    tokenizer_config = get_tokenizer_config("tokenizer-test")
    ```"""
    resolved_config_file = get_file_from_repo(
        pretrained_model_name_or_path,
        TOKENIZER_CONFIG_FILE,
        cache_dir=cache_dir,
        force_download=force_download,
        resume_download=resume_download,
        proxies=proxies,
        use_auth_token=use_auth_token,
        revision=revision,
        local_files_only=local_files_only,
    )
    if resolved_config_file is None:
        logger.info("Could not locate the tokenizer configuration file, will try to use the model config instead.")
        return {}

    with open(resolved_config_file, encoding="utf-8") as reader:
        return json.load(reader)


class AutoTokenizer:
    r"""
    This is a generic tokenizer class that will be instantiated as one of the tokenizer classes of the library when
    created with the [`AutoTokenizer.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoTokenizer is designed to be instantiated "
            "using the `AutoTokenizer.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    @replace_list_option_in_docstrings(TOKENIZER_MAPPING_NAMES)
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        r"""
        Instantiate one of the tokenizer classes of the library from a pretrained model vocabulary.

        The tokenizer class to instantiate is selected based on the `model_type` property of the config object (either
        passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it's missing, by
        falling back to using pattern matching on `pretrained_model_name_or_path`:

        List options

        Params:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:

                    - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                      using the [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.
                    - A path or url to a single saved vocabulary file if and only if the tokenizer only requires a
                      single vocabulary file (like Bert or XLNet), e.g.: `./my_model_directory/vocab.txt`. (Not
                      applicable to all derived classes)
            inputs (additional positional arguments, *optional*):
                Will be passed along to the Tokenizer `__init__()` method.
            config ([`PretrainedConfig`], *optional*)
                The configuration object used to dertermine the tokenizer class to instantiate.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download the model weights and configuration files and override the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            subfolder (`str`, *optional*):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co (e.g. for
                facebook/rag-token-base), specify it here.
            use_fast (`bool`, *optional*, defaults to `True`):
                Whether or not to try to load the fast version of the tokenizer.
            tokenizer_type (`str`, *optional*):
                Tokenizer type to be loaded.
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                should only be set to `True` for repositories you trust and in which you have read the code, as it will
                execute code present on the Hub on your local machine.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the Tokenizer `__init__()` method. Can be used to set special tokens like
                `bos_token`, `eos_token`, `unk_token`, `sep_token`, `pad_token`, `cls_token`, `mask_token`,
                `additional_special_tokens`. See parameters in the `__init__()` for more details.

        Examples:

        ```python
        >>> from transformers import AutoTokenizer

        >>> # Download vocabulary from huggingface.co and cache.
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        >>> # Download vocabulary from huggingface.co (user-uploaded) and cache.
        >>> tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")

        >>> # If vocabulary files are in a directory (e.g. tokenizer was saved using *save_pretrained('./test/saved_model/')*)
        >>> tokenizer = AutoTokenizer.from_pretrained("./test/bert_saved_model/")
        ```"""
        config = kwargs.pop("config", None)
        kwargs["_from_auto"] = True

        use_fast = kwargs.pop("use_fast", True)
        tokenizer_type = kwargs.pop("tokenizer_type", None)
        trust_remote_code = kwargs.pop("trust_remote_code", False)

        # First, let's see whether the tokenizer_type is passed so that we can leverage it
        if tokenizer_type is not None:
            tokenizer_class = None
            tokenizer_class_tuple = TOKENIZER_MAPPING_NAMES.get(tokenizer_type, None)

            if tokenizer_class_tuple is None:
                raise ValueError(
                    f"Passed `tokenizer_type` {tokenizer_type} does not exist. `tokenizer_type` should be one of "
                    f"{', '.join(c for c in TOKENIZER_MAPPING_NAMES.keys())}."
                )

            tokenizer_class_name, tokenizer_fast_class_name = tokenizer_class_tuple

            if use_fast and tokenizer_fast_class_name is not None:
                tokenizer_class = tokenizer_class_from_name(tokenizer_fast_class_name)

            if tokenizer_class is None:
                tokenizer_class = tokenizer_class_from_name(tokenizer_class_name)

            if tokenizer_class is None:
                raise ValueError(f"Tokenizer class {tokenizer_class_name} is not currently imported.")

            return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

        # Next, let's try to use the tokenizer_config file to get the tokenizer class.
        tokenizer_config = get_tokenizer_config(pretrained_model_name_or_path, **kwargs)
        config_tokenizer_class = tokenizer_config.get("tokenizer_class")
        tokenizer_auto_map = None
        if "auto_map" in tokenizer_config:
            if isinstance(tokenizer_config["auto_map"], (tuple, list)):
                # Legacy format for dynamic tokenizers
                tokenizer_auto_map = tokenizer_config["auto_map"]
            else:
                tokenizer_auto_map = tokenizer_config["auto_map"].get("AutoTokenizer", None)

        # If that did not work, let's try to use the config.
        if config_tokenizer_class is None:
            if not isinstance(config, PretrainedConfig):
                config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
                )
            config_tokenizer_class = config.tokenizer_class
            if hasattr(config, "auto_map") and "AutoTokenizer" in config.auto_map:
                tokenizer_auto_map = config.auto_map["AutoTokenizer"]

        # If we have the tokenizer class from the tokenizer config or the model config we're good!
        if config_tokenizer_class is not None:
            tokenizer_class = None
            if tokenizer_auto_map is not None:
                if not trust_remote_code:
                    raise ValueError(
                        f"Loading {pretrained_model_name_or_path} requires you to execute the tokenizer file in that"
                        " repo on your local machine. Make sure you have read the code there to avoid malicious use,"
                        " then set the option `trust_remote_code=True` to remove this error."
                    )
                if kwargs.get("revision", None) is None:
                    logger.warning(
                        "Explicitly passing a `revision` is encouraged when loading a model with custom code to ensure"
                        " no malicious code has been contributed in a newer revision."
                    )

                if use_fast and tokenizer_auto_map[1] is not None:
                    class_ref = tokenizer_auto_map[1]
                else:
                    class_ref = tokenizer_auto_map[0]

                module_file, class_name = class_ref.split(".")
                tokenizer_class = get_class_from_dynamic_module(
                    pretrained_model_name_or_path, module_file + ".py", class_name, **kwargs
                )

            elif use_fast and not config_tokenizer_class.endswith("Fast"):
                tokenizer_class_candidate = f"{config_tokenizer_class}Fast"
                tokenizer_class = tokenizer_class_from_name(tokenizer_class_candidate)
            if tokenizer_class is None:
                tokenizer_class_candidate = config_tokenizer_class
                tokenizer_class = tokenizer_class_from_name(tokenizer_class_candidate)

            if tokenizer_class is None:
                raise ValueError(
                    f"Tokenizer class {tokenizer_class_candidate} does not exist or is not currently imported."
                )
            return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

        # Otherwise we have to be creative.
        # if model is an encoder decoder, the encoder tokenizer class is used by default
        if isinstance(config, EncoderDecoderConfig):
            if type(config.decoder) is not type(config.encoder):  # noqa: E721
                logger.warning(
                    f"The encoder model config class: {config.encoder.__class__} is different from the decoder model "
                    f"config class: {config.decoder.__class__}. It is not recommended to use the "
                    "`AutoTokenizer.from_pretrained()` method in this case. Please use the encoder and decoder "
                    "specific tokenizer classes."
                )
            config = config.encoder

        model_type = config_class_to_model_type(type(config).__name__)
        if model_type is not None:
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

    def register(config_class, slow_tokenizer_class=None, fast_tokenizer_class=None):
        """
        Register a new tokenizer in this mapping.


        Args:
            config_class ([`PretrainedConfig`]):
                The configuration corresponding to the model to register.
            slow_tokenizer_class ([`PretrainedTokenizer`], *optional*):
                The slow tokenizer to register.
            slow_tokenizer_class ([`PretrainedTokenizerFast`], *optional*):
                The fast tokenizer to register.
        """
        if slow_tokenizer_class is None and fast_tokenizer_class is None:
            raise ValueError("You need to pass either a `slow_tokenizer_class` or a `fast_tokenizer_class")
        if slow_tokenizer_class is not None and issubclass(slow_tokenizer_class, PreTrainedTokenizerFast):
            raise ValueError("You passed a fast tokenizer in the `slow_tokenizer_class`.")
        if fast_tokenizer_class is not None and issubclass(fast_tokenizer_class, PreTrainedTokenizer):
            raise ValueError("You passed a slow tokenizer in the `fast_tokenizer_class`.")

        if (
            slow_tokenizer_class is not None
            and fast_tokenizer_class is not None
            and issubclass(fast_tokenizer_class, PreTrainedTokenizerFast)
            and fast_tokenizer_class.slow_tokenizer_class != slow_tokenizer_class
        ):
            raise ValueError(
                "The fast tokenizer class you are passing has a `slow_tokenizer_class` attribute that is not "
                "consistent with the slow tokenizer class you passed (fast tokenizer has "
                f"{fast_tokenizer_class.slow_tokenizer_class} and you passed {slow_tokenizer_class}. Fix one of those "
                "so they match!"
            )

        # Avoid resetting a set slow/fast tokenizer if we are passing just the other ones.
        if config_class in TOKENIZER_MAPPING._extra_content:
            existing_slow, existing_fast = TOKENIZER_MAPPING[config_class]
            if slow_tokenizer_class is None:
                slow_tokenizer_class = existing_slow
            if fast_tokenizer_class is None:
                fast_tokenizer_class = existing_fast

        TOKENIZER_MAPPING.register(config_class, (slow_tokenizer_class, fast_tokenizer_class))
