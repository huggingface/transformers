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
"""Auto Tokenizer class."""

import importlib
import json
import os
from collections import OrderedDict
from typing import Any

from transformers.utils.import_utils import is_mistral_common_available

from ...configuration_utils import PreTrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ...modeling_gguf_pytorch_utils import load_gguf_checkpoint
from ...tokenization_utils_base import TOKENIZER_CONFIG_FILE
from ...utils import (
    extract_commit_hash,
    is_g2p_en_available,
    is_sentencepiece_available,
    is_tokenizers_available,
    logging,
)
from ...utils.hub import cached_file
from ..encoder_decoder import EncoderDecoderConfig
from .auto_factory import _LazyAutoMapping
from .configuration_auto import (
    CONFIG_MAPPING_NAMES,
    AutoConfig,
    config_class_to_model_type,
    model_type_to_module_name,
    replace_list_option_in_docstrings,
)


if is_tokenizers_available():
    from ...tokenization_utils_tokenizers import TokenizersBackend
else:
    TokenizersBackend = None

if is_sentencepiece_available():
    from ...tokenization_utils_sentencepiece import SentencePieceBackend
else:
    SentencePieceBackend = None

logger = logging.get_logger(__name__)

# V5: Simplified mapping - single tokenizer class per model type (always prefer tokenizers-based)
REGISTERED_TOKENIZER_CLASSES: dict[str, type[Any]] = {}
REGISTERED_FAST_ALIASES: dict[str, type[Any]] = {}

TOKENIZER_MAPPING_NAMES = OrderedDict[str, str | None](
    [
        ("aimv2", "CLIPTokenizer" if is_tokenizers_available() else None),
        ("albert", "AlbertTokenizer" if is_tokenizers_available() else None),
        ("align", "BertTokenizer" if is_tokenizers_available() else None),
        ("audioflamingo3", "Qwen2Tokenizer" if is_tokenizers_available() else None),
        ("aya_vision", "CohereTokenizer" if is_tokenizers_available() else None),
        ("bark", "BertTokenizer" if is_tokenizers_available() else None),
        ("bart", "RobertaTokenizer" if is_tokenizers_available() else None),
        ("barthez", "BarthezTokenizer" if is_tokenizers_available() else None),
        ("bartpho", "BartphoTokenizer"),
        ("bert", "BertTokenizer" if is_tokenizers_available() else None),
        ("bert-generation", "BertGenerationTokenizer" if is_sentencepiece_available() else None),
        ("bert-japanese", "BertJapaneseTokenizer"),
        ("bertweet", "BertweetTokenizer"),
        ("big_bird", "BigBirdTokenizer" if is_tokenizers_available() else None),
        ("bigbird_pegasus", "PegasusTokenizer" if is_tokenizers_available() else None),
        ("biogpt", "BioGptTokenizer"),
        ("blenderbot", "BlenderbotTokenizer" if is_tokenizers_available() else None),
        ("blenderbot-small", "BlenderbotSmallTokenizer"),
        ("blip", "BertTokenizer" if is_tokenizers_available() else None),
        ("blip-2", "GPT2Tokenizer" if is_tokenizers_available() else None),
        ("bridgetower", "RobertaTokenizer"),
        ("bros", "BertTokenizer" if is_tokenizers_available() else None),
        ("byt5", "ByT5Tokenizer"),
        ("camembert", "CamembertTokenizer" if is_tokenizers_available() else None),
        ("canine", "CanineTokenizer"),
        ("chinese_clip", "BertTokenizer" if is_tokenizers_available() else None),
        ("clap", "RobertaTokenizer"),
        ("clip", "CLIPTokenizer" if is_tokenizers_available() else None),
        ("clipseg", "CLIPTokenizer" if is_tokenizers_available() else None),
        ("clvp", "ClvpTokenizer"),
        ("code_llama", "CodeLlamaTokenizer" if is_tokenizers_available() else None),
        ("codegen", "GPT2Tokenizer" if is_tokenizers_available() else None),
        ("cohere", "CohereTokenizer" if is_tokenizers_available() else None),
        ("cohere2", "CohereTokenizer" if is_tokenizers_available() else None),
        ("colqwen2", "Qwen2Tokenizer" if is_tokenizers_available() else None),
        ("convbert", "BertTokenizer" if is_tokenizers_available() else None),
        ("cpm", "CpmTokenizer" if is_tokenizers_available() else None),
        ("cpmant", "CpmAntTokenizer"),
        ("ctrl", "CTRLTokenizer"),
        ("data2vec-audio", "Wav2Vec2CTCTokenizer"),
        ("data2vec-text", "RobertaTokenizer"),
        ("dbrx", "GPT2Tokenizer" if is_tokenizers_available() else None),
        ("deberta", "DebertaTokenizer" if is_tokenizers_available() else None),
        ("deberta-v2", "DebertaV2Tokenizer" if is_tokenizers_available() else None),
        ("dia", "DiaTokenizer"),
        ("distilbert", "BertTokenizer" if is_tokenizers_available() else None),
        ("dpr", "DPRQuestionEncoderTokenizer" if is_tokenizers_available() else None),
        ("electra", "BertTokenizer" if is_tokenizers_available() else None),
        ("emu3", "GPT2Tokenizer" if is_tokenizers_available() else None),
        ("ernie", "BertTokenizer" if is_tokenizers_available() else None),
        ("esm", "EsmTokenizer"),
        ("falcon_mamba", "GPTNeoXTokenizer" if is_tokenizers_available() else None),
        ("fastspeech2_conformer", "FastSpeech2ConformerTokenizer" if is_g2p_en_available() else None),
        ("flaubert", "FlaubertTokenizer"),
        ("flava", "BertTokenizer" if is_tokenizers_available() else None),
        ("flex_olmo", "GPT2Tokenizer" if is_tokenizers_available() else None),
        ("florence2", "BartTokenizer" if is_tokenizers_available() else None),
        ("fnet", "FNetTokenizer" if is_tokenizers_available() else None),
        ("fsmt", "FSMTTokenizer"),
        ("funnel", "FunnelTokenizer" if is_tokenizers_available() else None),
        ("gemma", "GemmaTokenizer" if is_tokenizers_available() else None),
        ("gemma2", "GemmaTokenizer" if is_tokenizers_available() else None),
        ("gemma3", "GemmaTokenizer" if is_tokenizers_available() else None),
        ("gemma3_text", "GemmaTokenizer" if is_tokenizers_available() else None),
        ("gemma3n", "GemmaTokenizer" if is_tokenizers_available() else None),
        ("gemma3n_text", "GemmaTokenizer" if is_tokenizers_available() else None),
        ("git", "BertTokenizer" if is_tokenizers_available() else None),
        ("glm", "TokenizersBackend" if is_tokenizers_available() else None),
        ("glm4", "TokenizersBackend" if is_tokenizers_available() else None),
        ("glm4_moe", "TokenizersBackend" if is_tokenizers_available() else None),
        ("glm4_moe_lite", "TokenizersBackend" if is_tokenizers_available() else None),
        ("glm4v", "TokenizersBackend" if is_tokenizers_available() else None),
        ("glm4v_moe", "TokenizersBackend" if is_tokenizers_available() else None),
        ("glm_image", "TokenizersBackend" if is_tokenizers_available() else None),
        ("glmasr", "TokenizersBackend" if is_tokenizers_available() else None),
        ("got_ocr2", "TokenizersBackend" if is_tokenizers_available() else None),
        ("gpt-sw3", "GPTSw3Tokenizer" if is_sentencepiece_available() else None),
        ("gpt2", "GPT2Tokenizer" if is_tokenizers_available() else None),
        ("gpt_bigcode", "GPT2Tokenizer" if is_tokenizers_available() else None),
        ("gpt_neo", "GPT2Tokenizer" if is_tokenizers_available() else None),
        ("gpt_neox", "GPTNeoXTokenizer" if is_tokenizers_available() else None),
        ("gpt_neox_japanese", "GPTNeoXJapaneseTokenizer"),
        ("gptj", "GPT2Tokenizer" if is_tokenizers_available() else None),
        ("granite", "GPT2Tokenizer"),
        ("granitemoe", "GPT2Tokenizer"),
        ("granitemoehybrid", "GPT2Tokenizer"),
        ("granitemoeshared", "GPT2Tokenizer"),
        ("grounding-dino", "BertTokenizer" if is_tokenizers_available() else None),
        ("groupvit", "CLIPTokenizer" if is_tokenizers_available() else None),
        ("herbert", "HerbertTokenizer" if is_tokenizers_available() else None),
        ("hubert", "Wav2Vec2CTCTokenizer"),
        ("ibert", "RobertaTokenizer"),
        ("idefics", "LlamaTokenizer" if is_tokenizers_available() else None),
        ("idefics2", "LlamaTokenizer" if is_tokenizers_available() else None),
        ("instructblip", "GPT2Tokenizer" if is_tokenizers_available() else None),
        ("instructblipvideo", "GPT2Tokenizer" if is_tokenizers_available() else None),
        ("internvl", "Qwen2Tokenizer" if is_tokenizers_available() else None),
        ("jais2", "GPT2Tokenizer" if is_tokenizers_available() else None),
        ("kosmos-2", "XLMRobertaTokenizer" if is_tokenizers_available() else None),
        ("lasr_ctc", "ParakeetTokenizer" if is_tokenizers_available() else None),
        ("lasr_encoder", "ParakeetTokenizer" if is_tokenizers_available() else None),
        ("layoutlm", "BertTokenizer" if is_tokenizers_available() else None),
        ("layoutlmv2", "LayoutLMv2Tokenizer" if is_tokenizers_available() else None),
        ("layoutlmv3", "LayoutLMv3Tokenizer" if is_tokenizers_available() else None),
        ("layoutxlm", "LayoutXLMTokenizer" if is_tokenizers_available() else None),
        ("led", "LEDTokenizer" if is_tokenizers_available() else None),
        ("lighton_ocr", "Qwen2TokenizerFast" if is_tokenizers_available() else None),
        ("lilt", "RobertaTokenizer" if is_tokenizers_available() else None),
        ("longformer", "RobertaTokenizer" if is_tokenizers_available() else None),
        ("longt5", "T5Tokenizer" if is_tokenizers_available() else None),
        ("luke", "LukeTokenizer"),
        ("lxmert", "LxmertTokenizer" if is_tokenizers_available() else None),
        ("m2m_100", "M2M100Tokenizer" if is_sentencepiece_available() else None),
        ("mamba", "GPTNeoXTokenizer" if is_tokenizers_available() else None),
        ("mamba2", "GPTNeoXTokenizer" if is_tokenizers_available() else None),
        ("marian", "MarianTokenizer" if is_sentencepiece_available() else None),
        ("markuplm", "MarkupLMTokenizer" if is_tokenizers_available() else None),
        ("mbart", "MBartTokenizer" if is_tokenizers_available() else None),
        ("mbart50", "MBart50Tokenizer" if is_tokenizers_available() else None),
        ("mega", "RobertaTokenizer"),
        ("megatron-bert", "BertTokenizer" if is_tokenizers_available() else None),
        ("metaclip_2", "XLMRobertaTokenizer" if is_tokenizers_available() else None),
        ("mgp-str", "MgpstrTokenizer"),
        (
            "ministral3",
            "MistralCommonBackend"
            if is_mistral_common_available()
            else ("TokenizersBackend" if is_tokenizers_available() else None),
        ),
        (
            "mistral",
            "MistralCommonBackend"
            if is_mistral_common_available()
            else ("TokenizersBackend" if is_tokenizers_available() else None),
        ),
        (
            "mistral3",
            "MistralCommonBackend"
            if is_mistral_common_available()
            else ("TokenizersBackend" if is_tokenizers_available() else None),
        ),
        (
            "mixtral",
            "MistralCommonBackend"
            if is_mistral_common_available()
            else ("TokenizersBackend" if is_tokenizers_available() else None),
        ),
        ("mluke", "MLukeTokenizer" if is_sentencepiece_available() else None),
        ("mm-grounding-dino", "BertTokenizer" if is_tokenizers_available() else None),
        ("mobilebert", "MobileBertTokenizer" if is_tokenizers_available() else None),
        ("mpnet", "MPNetTokenizer" if is_tokenizers_available() else None),
        ("mpt", "GPTNeoXTokenizer" if is_tokenizers_available() else None),
        ("mra", "RobertaTokenizer"),
        ("mt5", "T5Tokenizer" if is_tokenizers_available() else None),
        ("musicgen", "T5Tokenizer" if is_tokenizers_available() else None),
        ("musicgen_melody", "T5Tokenizer" if is_tokenizers_available() else None),
        ("mvp", "MvpTokenizer" if is_tokenizers_available() else None),
        ("myt5", "MyT5Tokenizer"),
        ("nezha", "BertTokenizer" if is_tokenizers_available() else None),
        ("nllb", "NllbTokenizer" if is_tokenizers_available() else None),
        ("nllb-moe", "NllbTokenizer" if is_tokenizers_available() else None),
        ("nougat", "NougatTokenizer" if is_tokenizers_available() else None),
        ("nystromformer", "AlbertTokenizer" if is_tokenizers_available() else None),
        ("olmo", "GPTNeoXTokenizer" if is_tokenizers_available() else None),
        ("olmo2", "GPTNeoXTokenizer" if is_tokenizers_available() else None),
        ("olmo3", "GPT2Tokenizer" if is_tokenizers_available() else None),
        ("olmoe", "GPTNeoXTokenizer" if is_tokenizers_available() else None),
        ("omdet-turbo", "CLIPTokenizer" if is_tokenizers_available() else None),
        ("oneformer", "CLIPTokenizer" if is_tokenizers_available() else None),
        ("openai-gpt", "OpenAIGPTTokenizer" if is_tokenizers_available() else None),
        ("opt", "GPT2Tokenizer" if is_tokenizers_available() else None),
        ("ovis2", "Qwen2Tokenizer" if is_tokenizers_available() else None),
        ("owlv2", "CLIPTokenizer" if is_tokenizers_available() else None),
        ("owlvit", "CLIPTokenizer" if is_tokenizers_available() else None),
        ("pegasus", "PegasusTokenizer" if is_tokenizers_available() else None),
        ("pegasus_x", "PegasusTokenizer" if is_tokenizers_available() else None),
        ("perceiver", "PerceiverTokenizer"),
        ("phi", "GPT2Tokenizer" if is_tokenizers_available() else None),
        ("phobert", "PhobertTokenizer"),
        ("pix2struct", "T5Tokenizer" if is_tokenizers_available() else None),
        (
            "pixtral",
            "MistralCommonBackend"
            if is_mistral_common_available()
            else ("TokenizersBackend" if is_tokenizers_available() else None),
        ),
        ("plbart", "PLBartTokenizer" if is_tokenizers_available() else None),
        ("prophetnet", "ProphetNetTokenizer"),
        ("qdqbert", "BertTokenizer" if is_tokenizers_available() else None),
        ("qwen2", "Qwen2Tokenizer" if is_tokenizers_available() else None),
        ("qwen2_5_omni", "Qwen2Tokenizer" if is_tokenizers_available() else None),
        ("qwen2_5_vl", "Qwen2Tokenizer" if is_tokenizers_available() else None),
        ("qwen2_audio", "Qwen2Tokenizer" if is_tokenizers_available() else None),
        ("qwen2_moe", "Qwen2Tokenizer" if is_tokenizers_available() else None),
        ("qwen2_vl", "Qwen2Tokenizer" if is_tokenizers_available() else None),
        ("qwen3", "Qwen2Tokenizer" if is_tokenizers_available() else None),
        ("qwen3_5", "Qwen3_5Tokenizer" if is_tokenizers_available() else None),
        ("qwen3_5_moe", "Qwen3_5Tokenizer" if is_tokenizers_available() else None),
        ("qwen3_moe", "Qwen2Tokenizer" if is_tokenizers_available() else None),
        ("qwen3_next", "Qwen2Tokenizer" if is_tokenizers_available() else None),
        ("qwen3_omni_moe", "Qwen2Tokenizer" if is_tokenizers_available() else None),
        ("qwen3_vl", "Qwen2Tokenizer" if is_tokenizers_available() else None),
        ("qwen3_vl_moe", "Qwen2Tokenizer" if is_tokenizers_available() else None),
        ("rag", "RagTokenizer"),
        ("realm", "BertTokenizer" if is_tokenizers_available() else None),
        ("recurrent_gemma", "GemmaTokenizer" if is_tokenizers_available() else None),
        ("reformer", "ReformerTokenizer" if is_tokenizers_available() else None),
        ("rembert", "RemBertTokenizer" if is_tokenizers_available() else None),
        ("retribert", "BertTokenizer" if is_tokenizers_available() else None),
        ("roberta", "RobertaTokenizer"),
        ("roberta-prelayernorm", "RobertaTokenizer"),
        ("roc_bert", "RoCBertTokenizer"),
        ("roformer", "RoFormerTokenizer" if is_tokenizers_available() else None),
        ("rwkv", "GPTNeoXTokenizer" if is_tokenizers_available() else None),
        ("sam3", "CLIPTokenizer" if is_tokenizers_available() else None),
        ("sam3_video", "CLIPTokenizer" if is_tokenizers_available() else None),
        ("seamless_m4t", "SeamlessM4TTokenizer" if is_tokenizers_available() else None),
        ("seamless_m4t_v2", "SeamlessM4TTokenizer" if is_tokenizers_available() else None),
        ("shieldgemma2", "GemmaTokenizer" if is_tokenizers_available() else None),
        ("siglip", "SiglipTokenizer" if is_sentencepiece_available() else None),
        ("siglip2", "Siglip2Tokenizer" if is_tokenizers_available() else None),
        ("speech_to_text", "Speech2TextTokenizer" if is_sentencepiece_available() else None),
        ("speecht5", "SpeechT5Tokenizer" if is_sentencepiece_available() else None),
        ("splinter", "SplinterTokenizer"),
        ("squeezebert", "BertTokenizer" if is_tokenizers_available() else None),
        ("stablelm", "GPTNeoXTokenizer" if is_tokenizers_available() else None),
        ("starcoder2", "GPT2Tokenizer" if is_tokenizers_available() else None),
        ("switch_transformers", "T5Tokenizer" if is_tokenizers_available() else None),
        ("t5", "T5Tokenizer" if is_tokenizers_available() else None),
        ("t5gemma", "GemmaTokenizer" if is_tokenizers_available() else None),
        ("tapas", "TapasTokenizer"),
        ("trocr", "XLMRobertaTokenizer" if is_tokenizers_available() else None),
        ("tvp", "BertTokenizer" if is_tokenizers_available() else None),
        ("udop", "UdopTokenizer" if is_tokenizers_available() else None),
        ("umt5", "T5Tokenizer" if is_tokenizers_available() else None),
        ("unispeech", "Wav2Vec2CTCTokenizer"),
        ("unispeech-sat", "Wav2Vec2CTCTokenizer"),
        ("vilt", "BertTokenizer" if is_tokenizers_available() else None),
        ("visual_bert", "BertTokenizer" if is_tokenizers_available() else None),
        ("vits", "VitsTokenizer"),
        (
            "voxtral",
            "MistralCommonBackend"
            if is_mistral_common_available()
            else ("TokenizersBackend" if is_tokenizers_available() else None),
        ),
        (
            "voxtral_realtime",
            "MistralCommonBackend"
            if is_mistral_common_available()
            else ("TokenizersBackend" if is_tokenizers_available() else None),
        ),
        ("wav2vec2", "Wav2Vec2CTCTokenizer"),
        ("wav2vec2-bert", "Wav2Vec2CTCTokenizer"),
        ("wav2vec2-conformer", "Wav2Vec2CTCTokenizer"),
        ("wav2vec2_phoneme", "Wav2Vec2PhonemeCTCTokenizer"),
        ("whisper", "WhisperTokenizer" if is_tokenizers_available() else None),
        ("xclip", "CLIPTokenizer" if is_tokenizers_available() else None),
        ("xglm", "XGLMTokenizer" if is_tokenizers_available() else None),
        ("xlm", "XLMTokenizer"),
        ("xlm-roberta", "XLMRobertaTokenizer" if is_tokenizers_available() else None),
        ("xlm-roberta-xl", "XLMRobertaTokenizer" if is_tokenizers_available() else None),
        ("xlnet", "XLNetTokenizer" if is_tokenizers_available() else None),
        ("xlstm", "GPTNeoXTokenizer" if is_tokenizers_available() else None),
        ("xmod", "XLMRobertaTokenizer" if is_tokenizers_available() else None),
        ("yoso", "AlbertTokenizer" if is_tokenizers_available() else None),
    ]
)

TOKENIZER_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TOKENIZER_MAPPING_NAMES)

CONFIG_TO_TYPE = {v: k for k, v in CONFIG_MAPPING_NAMES.items()}


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    with open(vocab_file, "r", encoding="utf-8") as reader:
        return json.load(reader)


def load_merges(merges_file):
    """Loads a merges file into a list."""
    merges = []
    with open(merges_file, "r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if line and not line.startswith("#"):
                merges.append(tuple(line.split()))
    return merges


def tokenizer_class_from_name(class_name: str) -> type[Any] | None:
    # Bloom tokenizer classes were removed but should map to the fast backend for BC
    if class_name in {"BloomTokenizer", "BloomTokenizerFast"}:
        return TokenizersBackend

    if class_name in REGISTERED_FAST_ALIASES:
        return REGISTERED_FAST_ALIASES[class_name]

    if class_name in REGISTERED_TOKENIZER_CLASSES:
        return REGISTERED_TOKENIZER_CLASSES[class_name]

    if class_name == "TokenizersBackend":
        return TokenizersBackend

    # V5: TOKENIZER_MAPPING_NAMES now maps to single strings, not tuples
    for module_name, tokenizer_class in TOKENIZER_MAPPING_NAMES.items():
        if tokenizer_class == class_name:
            module_name = model_type_to_module_name(module_name)
            if (
                module_name in ["mistral", "mistral3", "mixtral", "ministral", "ministral3", "pixtral", "voxtral"]
                and class_name == "MistralCommonBackend"
            ):
                module = importlib.import_module(".tokenization_mistral_common", "transformers")
            else:
                module = importlib.import_module(f".{module_name}", "transformers.models")
            try:
                return getattr(module, class_name)
            except AttributeError:
                continue

    for tokenizer in TOKENIZER_MAPPING._extra_content.values():
        if getattr(tokenizer, "__name__", None) == class_name:
            return tokenizer

    # We did not find the class, but maybe it's because a dep is missing. In that case, the class will be in the main
    # We did not find the class, but maybe it's because a dep is missing. In that case, the class will be in the main
    # init and we return the proper dummy to get an appropriate error message.
    main_module = importlib.import_module("transformers")
    if hasattr(main_module, class_name):
        return getattr(main_module, class_name)

    return None


def get_tokenizer_config(
    pretrained_model_name_or_path: str | os.PathLike[str],
    cache_dir: str | os.PathLike[str] | None = None,
    force_download: bool = False,
    proxies: dict[str, str] | None = None,
    token: bool | str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
    subfolder: str = "",
    **kwargs,
) -> dict[str, Any]:
    """
    Loads the tokenizer configuration from a pretrained model tokenizer configuration.

    Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
              huggingface.co.
            - a path to a *directory* containing a configuration file saved using the
              [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.

        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        proxies (`dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `hf auth login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.
        subfolder (`str`, *optional*, defaults to `""`):
            In case the tokenizer config is located inside a subfolder of the model repo on huggingface.co, you can
            specify the folder name here.

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `dict`: The configuration of the tokenizer.

    Examples:

    ```python
    # Download configuration from huggingface.co and cache.
    tokenizer_config = get_tokenizer_config("google-bert/bert-base-uncased")
    # This model does not have a tokenizer config so the result will be an empty dict.
    tokenizer_config = get_tokenizer_config("FacebookAI/xlm-roberta-base")

    # Save a pretrained tokenizer locally and you can reload its config
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
    tokenizer.save_pretrained("tokenizer-test")
    tokenizer_config = get_tokenizer_config("tokenizer-test")
    ```"""
    commit_hash = kwargs.get("_commit_hash")
    resolved_config_file = cached_file(
        pretrained_model_name_or_path,
        TOKENIZER_CONFIG_FILE,
        cache_dir=cache_dir,
        force_download=force_download,
        proxies=proxies,
        token=token,
        revision=revision,
        local_files_only=local_files_only,
        subfolder=subfolder,
        _raise_exceptions_for_gated_repo=False,
        _raise_exceptions_for_missing_entries=False,
        _raise_exceptions_for_connection_errors=False,
        _commit_hash=commit_hash,
    )
    if resolved_config_file is None:
        logger.info("Could not locate the tokenizer configuration file, will try to use the model config instead.")
        return {}
    commit_hash = extract_commit_hash(resolved_config_file, commit_hash)

    with open(resolved_config_file, encoding="utf-8") as reader:
        result = json.load(reader)
    result["_commit_hash"] = commit_hash
    return result


class AutoTokenizer:
    r"""
    This is a generic tokenizer class that will be instantiated as one of the tokenizer classes of the library when
    created with the [`AutoTokenizer.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        raise OSError(
            "AutoTokenizer is designed to be instantiated "
            "using the `AutoTokenizer.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    @replace_list_option_in_docstrings(TOKENIZER_MAPPING_NAMES)
    def from_pretrained(
        cls, pretrained_model_name_or_path, *inputs, **kwargs
    ) -> TokenizersBackend | SentencePieceBackend:
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
                    - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                      using the [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.
                    - A path or url to a single saved vocabulary file if and only if the tokenizer only requires a
                      single vocabulary file (like Bert or XLNet), e.g.: `./my_model_directory/vocab.txt`. (Not
                      applicable to all derived classes)
            inputs (additional positional arguments, *optional*):
                Will be passed along to the Tokenizer `__init__()` method.
            config ([`PreTrainedConfig`], *optional*)
                The configuration object used to determine the tokenizer class to instantiate.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download the model weights and configuration files and override the
                cached versions if they exist.
            proxies (`dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            subfolder (`str`, *optional*):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co (e.g. for
                facebook/rag-token-base), specify it here.
            tokenizer_type (`str`, *optional*):
                Tokenizer type to be loaded.
            backend (`str`, *optional*, defaults to `"tokenizers"`):
                Backend to use for tokenization. Valid options are:
                - `"tokenizers"`: Use the HuggingFace tokenizers library backend (default)
                - `"sentencepiece"`: Use the SentencePiece backend
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
        >>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

        >>> # Download vocabulary from huggingface.co (user-uploaded) and cache.
        >>> tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")

        >>> # If vocabulary files are in a directory (e.g. tokenizer was saved using *save_pretrained('./test/saved_model/')*)
        >>> # tokenizer = AutoTokenizer.from_pretrained("./test/bert_saved_model/")

        >>> # Download vocabulary from huggingface.co and define model-specific arguments
        >>> tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base", add_prefix_space=True)

        >>> # Explicitly use the tokenizers backend
        >>> tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer", backend="tokenizers")

        >>> # Explicitly use the sentencepiece backend
        >>> tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer", backend="sentencepiece")
        ```"""
        config = kwargs.pop("config", None)
        kwargs["_from_auto"] = True

        # V5: Always use fast tokenizers, ignore use_fast parameter
        _ = kwargs.pop("use_fast", None)
        tokenizer_type = kwargs.pop("tokenizer_type", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        gguf_file = kwargs.get("gguf_file")

        # First, let's see whether the tokenizer_type is passed so that we can leverage it
        if tokenizer_type is not None:
            tokenizer_class_name = TOKENIZER_MAPPING_NAMES.get(tokenizer_type, None)

            if tokenizer_class_name is None:
                raise ValueError(
                    f"Passed `tokenizer_type` {tokenizer_type} does not exist. `tokenizer_type` should be one of "
                    f"{', '.join(c for c in TOKENIZER_MAPPING_NAMES)}."
                )

            tokenizer_class = tokenizer_class_from_name(tokenizer_class_name)

            if tokenizer_class is None:
                raise ValueError(f"Tokenizer class {tokenizer_class_name} is not currently imported.")

            return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

        if gguf_file:
            gguf_path = cached_file(pretrained_model_name_or_path, gguf_file, **kwargs)
            config_dict = load_gguf_checkpoint(gguf_path, return_tensors=False)["config"]
            config = AutoConfig.for_model(**config_dict)
        elif config is None:
            try:
                config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
                )
            except Exception:
                config = PreTrainedConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        config_model_type = config.model_type

        # Next, let's try to use the tokenizer_config file to get the tokenizer class.
        tokenizer_config = get_tokenizer_config(pretrained_model_name_or_path, **kwargs)
        tokenizer_config_class = tokenizer_config.get("tokenizer_class", None)

        # Check for auto_map early to handle dynamic tokenizers properly
        tokenizer_auto_map = None
        if "auto_map" in tokenizer_config:
            if isinstance(tokenizer_config["auto_map"], (tuple, list)):
                # Legacy format for dynamic tokenizers
                tokenizer_auto_map = tokenizer_config["auto_map"]
            else:
                tokenizer_auto_map = tokenizer_config["auto_map"].get("AutoTokenizer", None)

        # if there is a config, we can check that the tokenizer class != than model class and can thus assume we need to use TokenizersBackend
        # Skip this early exit if auto_map is present (custom tokenizer with trust_remote_code)
        if (
            tokenizer_auto_map is None
            and tokenizer_config_class is not None
            and config_model_type is not None
            and config_model_type != ""
            and TOKENIZER_MAPPING_NAMES.get(config_model_type) is not None
            and TOKENIZER_MAPPING_NAMES.get(config_model_type).replace("Fast", "")
            != tokenizer_config_class.replace("Fast", "")
        ):
            # new model, but we ignore it unless the model type is the same
            if TokenizersBackend is not None:
                try:
                    return TokenizersBackend.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
                except Exception as e:
                    logger.debug(f"Failed to use TokenizersBackend: {e}")

            return tokenizer_class_from_name(tokenizer_config_class).from_pretrained(
                pretrained_model_name_or_path, *inputs, **kwargs
            )

        if "_commit_hash" in tokenizer_config:
            kwargs["_commit_hash"] = tokenizer_config["_commit_hash"]

        if tokenizer_config_class:
            tokenizer_config_class = tokenizer_config_class.replace("Fast", "")

        has_remote_code = tokenizer_auto_map is not None
        has_local_code = type(config) in TOKENIZER_MAPPING or (
            tokenizer_config_class is not None
            and (
                tokenizer_class_from_name(tokenizer_config_class) is not None
                or tokenizer_class_from_name(tokenizer_config_class + "Fast") is not None
            )
        )
        if has_remote_code:
            # V5: Always prefer fast tokenizer (index 1), fallback to slow (index 0)
            if tokenizer_auto_map[1] is not None:
                class_ref = tokenizer_auto_map[1]
            else:
                class_ref = tokenizer_auto_map[0]
            if "--" in class_ref:
                upstream_repo = class_ref.split("--")[0]
            else:
                upstream_repo = None
            trust_remote_code = resolve_trust_remote_code(
                trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code, upstream_repo
            )

        if has_remote_code and trust_remote_code:
            tokenizer_class = get_class_from_dynamic_module(class_ref, pretrained_model_name_or_path, **kwargs)
            _ = kwargs.pop("code_revision", None)
            tokenizer_class.register_for_auto_class()
            return tokenizer_class.from_pretrained(
                pretrained_model_name_or_path, *inputs, trust_remote_code=trust_remote_code, **kwargs
            )
        elif tokenizer_config_class is not None:
            tokenizer_class_candidate = tokenizer_config_class
            tokenizer_class = tokenizer_class_from_name(tokenizer_class_candidate)
            if tokenizer_class is None and not tokenizer_class_candidate.endswith("Fast"):
                tokenizer_class = tokenizer_class_from_name(tokenizer_class_candidate + "Fast")
            if tokenizer_class is not None and tokenizer_class.__name__ == "PythonBackend":
                tokenizer_class = TokenizersBackend
            # Fallback to TokenizersBackend if the class wasn't found
            if tokenizer_class is None:
                tokenizer_class = TokenizersBackend

            return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
        elif getattr(config, "tokenizer_class", None):
            _class = config.tokenizer_class
            if "PreTrainedTokenizerFast" not in _class:
                _class = _class.replace("Fast", "")
            tokenizer_class = tokenizer_class_from_name(_class)
            return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

        # Otherwise we have to be creative.
        # if model is an encoder decoder, the encoder tokenizer class is used by default
        if isinstance(config, EncoderDecoderConfig):
            if type(config.decoder) is not type(config.encoder):
                logger.warning(
                    f"The encoder model config class: {config.encoder.__class__} is different from the decoder model "
                    f"config class: {config.decoder.__class__}. It is not recommended to use the "
                    "`AutoTokenizer.from_pretrained()` method in this case. Please use the encoder and decoder "
                    "specific tokenizer classes."
                )
            config = config.encoder

        model_type = config_class_to_model_type(type(config).__name__) or getattr(config, "model_type", None)
        if model_type is not None:
            tokenizer_class = TOKENIZER_MAPPING.get(type(config), TokenizersBackend)
            if tokenizer_class is not None:
                return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

        # Fallback: try tokenizer_class from tokenizer_config.json
        tokenizer_config_class = tokenizer_config.get("tokenizer_class", None)
        if tokenizer_config_class is not None:
            if tokenizer_config_class != "TokenizersBackend" and "Fast" in tokenizer_config_class:
                tokenizer_config_class = tokenizer_config_class[:-4]
            tokenizer_class = tokenizer_class_from_name(tokenizer_config_class)
            if tokenizer_class is None and not tokenizer_config_class.endswith("Fast"):
                tokenizer_class = tokenizer_class_from_name(tokenizer_config_class + "Fast")
            if tokenizer_class is not None and tokenizer_class.__name__ == "PythonBackend":
                tokenizer_class = TokenizersBackend
            if tokenizer_class is None:
                tokenizer_class = TokenizersBackend
            return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

        raise ValueError(
            f"Unrecognized configuration class {config.__class__} to build an AutoTokenizer.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in TOKENIZER_MAPPING)}."
        )

    @staticmethod
    def register(
        config_class, tokenizer_class=None, slow_tokenizer_class=None, fast_tokenizer_class=None, exist_ok=False
    ):
        """
        Register a new tokenizer in this mapping.

        Args:
            config_class ([`PreTrainedConfig`]):
                The configuration corresponding to the model to register.
            tokenizer_class: The tokenizer class to register (V5 - preferred parameter).
            slow_tokenizer_class: (Deprecated) The slow tokenizer to register.
            fast_tokenizer_class: (Deprecated) The fast tokenizer to register.
        """
        if tokenizer_class is None:
            # Legacy: prefer fast over slow
            if fast_tokenizer_class is not None:
                tokenizer_class = fast_tokenizer_class
            elif slow_tokenizer_class is not None:
                tokenizer_class = slow_tokenizer_class
            else:
                raise ValueError("You need to pass a `tokenizer_class`")

        for candidate in (slow_tokenizer_class, fast_tokenizer_class, tokenizer_class):
            if candidate is not None:
                REGISTERED_TOKENIZER_CLASSES[candidate.__name__] = candidate

        if slow_tokenizer_class is not None and fast_tokenizer_class is not None:
            REGISTERED_FAST_ALIASES[slow_tokenizer_class.__name__] = fast_tokenizer_class

        TOKENIZER_MAPPING.register(config_class, tokenizer_class, exist_ok=exist_ok)


__all__ = ["TOKENIZER_MAPPING", "AutoTokenizer"]
