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
"""Auto Tokenizer class."""

import importlib
import inspect
import json
import os
import warnings
from collections import OrderedDict
from typing import Any, Optional, Union

import httpx
from huggingface_hub import list_repo_tree
from huggingface_hub.utils import (
    GatedRepoError,
    HfHubHTTPError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from transformers.utils.import_utils import is_mistral_common_available

from ...configuration_utils import PretrainedConfig
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
from ...utils.hub import cached_file, has_file
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
    from ...tokenization_tokenizers import TokenizersBackend
else:
    TokenizersBackend = None

if is_sentencepiece_available():
    from ...tokenization_sentencepiece import SentencePieceBackend
else:
    SentencePieceBackend = None

logger = logging.get_logger(__name__)

# V5: Simplified mapping - single tokenizer class per model type (always prefer tokenizers-based)
TOKENIZER_MAPPING_NAMES = OrderedDict[str, Optional[str]](
    [
        ("aimv2", "CLIPTokenizerFast" if is_tokenizers_available() else None),
        ("albert", "AlbertTokenizer" if is_tokenizers_available() else None),
        ("align", "BertTokenizer" if is_tokenizers_available() else None),
        ("arcee", "LlamaTokenizerFast" if is_tokenizers_available() else None),
        ("aria", "LlamaTokenizerFast" if is_tokenizers_available() else None),
        ("aya_vision", "CohereTokenizer" if is_tokenizers_available() else None),
        ("bark", "BertTokenizer" if is_tokenizers_available() else None),
        ("bart", "BartTokenizerFast"),
        ("barthez", "BarthezTokenizer" if is_tokenizers_available() else None),
        ("bartpho", "BartphoTokenizer"),
        ("bert", "BertTokenizer" if is_tokenizers_available() else None),
        ("bert-generation", "BertGenerationTokenizer" if is_sentencepiece_available() else None),
        ("bert-japanese", "BertJapaneseTokenizer"),
        ("bertweet", "BertweetTokenizer"),
        ("big_bird", "BigBirdTokenizer" if is_tokenizers_available() else None),
        ("bigbird_pegasus", "PegasusTokenizer" if is_tokenizers_available() else None),
        ("biogpt", "BioGptTokenizer"),
        ("bitnet", "PreTrainedTokenizerFast" if is_tokenizers_available() else None),
        ("blenderbot", "BlenderbotTokenizer" if is_tokenizers_available() else None),
        ("blenderbot-small", "BlenderbotSmallTokenizer"),
        ("blip", "BertTokenizer" if is_tokenizers_available() else None),
        ("blip-2", "GPT2Tokenizer" if is_tokenizers_available() else None),
        ("bloom", "TokenizersBackend" if is_tokenizers_available() else None),
        ("blt", "PreTrainedTokenizerFast" if is_tokenizers_available() else None),
        ("bridgetower", "RobertaTokenizer"),
        ("bros", "BertTokenizer" if is_tokenizers_available() else None),
        ("byt5", "ByT5Tokenizer"),
        ("camembert", "CamembertTokenizer" if is_tokenizers_available() else None),
        ("canine", "CanineTokenizer"),
        ("chameleon", "LlamaTokenizerFast" if is_tokenizers_available() else None),
        ("chinese_clip", "BertTokenizer" if is_tokenizers_available() else None),
        ("clap", "RobertaTokenizer"),
        ("clip", "CLIPTokenizer" if is_tokenizers_available() else None),
        ("clipseg", "CLIPTokenizer" if is_tokenizers_available() else None),
        ("clvp", "ClvpTokenizer"),
        ("code_llama", "CodeLlamaTokenizer" if is_tokenizers_available() else None),
        ("codegen", "CodeGenTokenizer" if is_tokenizers_available() else None),
        ("cohere", "CohereTokenizer" if is_tokenizers_available() else None),
        ("cohere2", "CohereTokenizer" if is_tokenizers_available() else None),
        ("colpali", "LlamaTokenizerFast" if is_tokenizers_available() else None),
        ("colqwen2", "Qwen2TokenizerFast" if is_tokenizers_available() else None),
        ("convbert", "ConvBertTokenizer" if is_tokenizers_available() else None),
        ("cpm", "CpmTokenizer" if is_tokenizers_available() else None),
        ("cpmant", "CpmAntTokenizer"),
        ("csm", "PreTrainedTokenizerFast" if is_tokenizers_available() else None),
        ("ctrl", "CTRLTokenizer"),
        ("data2vec-audio", "Wav2Vec2CTCTokenizer"),
        ("data2vec-text", "RobertaTokenizer"),
        ("dbrx", "GPT2Tokenizer" if is_tokenizers_available() else None),
        ("deberta", "DebertaTokenizer" if is_tokenizers_available() else None),
        ("deberta-v2", "DebertaV2Tokenizer" if is_tokenizers_available() else None),
        ("deepseek_v2", "LlamaTokenizerFast" if is_tokenizers_available() else None),
        ("deepseek_v3", "LlamaTokenizerFast" if is_tokenizers_available() else None),
        ("deepseek_vl", "LlamaTokenizerFast" if is_tokenizers_available() else None),
        ("deepseek_vl_hybrid", "LlamaTokenizerFast" if is_tokenizers_available() else None),
        ("dia", "DiaTokenizer"),
        ("diffllama", "LlamaTokenizerFast" if is_tokenizers_available() else None),
        ("distilbert", "DistilBertTokenizer" if is_tokenizers_available() else None),
        ("dpr", "DPRQuestionEncoderTokenizerFast" if is_tokenizers_available() else None),
        ("electra", "BertTokenizer" if is_tokenizers_available() else None),
        ("emu3", "GPT2Tokenizer" if is_tokenizers_available() else None),
        ("ernie", "BertTokenizer" if is_tokenizers_available() else None),
        ("ernie4_5", "LlamaTokenizerFast" if is_tokenizers_available() else None),
        ("ernie4_5_moe", "LlamaTokenizerFast" if is_tokenizers_available() else None),
        ("ernie_m", "ErnieMTokenizer" if is_sentencepiece_available() else None),
        ("esm", "EsmTokenizer"),
        ("exaone4", "GPT2Tokenizer" if is_tokenizers_available() else None),
        ("falcon", "PreTrainedTokenizerFast" if is_tokenizers_available() else None),
        ("falcon_mamba", "GPTNeoXTokenizerFast" if is_tokenizers_available() else None),
        ("fastspeech2_conformer", "FastSpeech2ConformerTokenizer" if is_g2p_en_available() else None),
        ("flaubert", "FlaubertTokenizer"),
        ("flex_olmo", "GPT2Tokenizer" if is_tokenizers_available() else None),
        ("fnet", "FNetTokenizerFast" if is_tokenizers_available() else None),
        ("fsmt", "FSMTTokenizer"),
        ("funnel", "FunnelTokenizer" if is_tokenizers_available() else None),
        ("gemma", "GemmaTokenizerFast" if is_tokenizers_available() else None),
        ("gemma2", "GemmaTokenizerFast" if is_tokenizers_available() else None),
        ("gemma3", "GemmaTokenizerFast" if is_tokenizers_available() else None),
        ("gemma3_text", "GemmaTokenizerFast" if is_tokenizers_available() else None),
        ("gemma3n", "GemmaTokenizerFast" if is_tokenizers_available() else None),
        ("gemma3n_text", "GemmaTokenizerFast" if is_tokenizers_available() else None),
        ("git", "BertTokenizer" if is_tokenizers_available() else None),
        ("glm", "PreTrainedTokenizerFast" if is_tokenizers_available() else None),
        ("glm4", "PreTrainedTokenizerFast" if is_tokenizers_available() else None),
        ("glm4_moe", "PreTrainedTokenizerFast" if is_tokenizers_available() else None),
        ("glm4v", "PreTrainedTokenizerFast" if is_tokenizers_available() else None),
        ("glm4v_moe", "PreTrainedTokenizerFast" if is_tokenizers_available() else None),
        ("gpt-sw3", "GPTSw3Tokenizer" if is_sentencepiece_available() else None),
        ("gpt2", "GPT2Tokenizer" if is_tokenizers_available() else None),
        ("gpt_bigcode", "GPT2Tokenizer" if is_tokenizers_available() else None),
        ("gpt_neo", "GPT2Tokenizer" if is_tokenizers_available() else None),
        ("gpt_neox", "GPTNeoXTokenizer" if is_tokenizers_available() else None),
        ("gpt_neox_japanese", "GPTNeoXJapaneseTokenizer"),
        ("gpt_oss", "PreTrainedTokenizerFast" if is_tokenizers_available() else None),
        ("gptj", "GPT2Tokenizer" if is_tokenizers_available() else None),
        ("gptsan-japanese", "GPTSanJapaneseTokenizer"),
        ("granite", "GPT2Tokenizer"),
        ("granitemoe", "GPT2Tokenizer"),
        ("granitemoehybrid", "GPT2Tokenizer"),
        ("granitemoeshared", "GPT2Tokenizer"),
        ("grounding-dino", "BertTokenizer" if is_tokenizers_available() else None),
        ("groupvit", "CLIPTokenizerFast" if is_tokenizers_available() else None),
        ("helium", "PreTrainedTokenizerFast" if is_tokenizers_available() else None),
        ("herbert", "HerbertTokenizer" if is_tokenizers_available() else None),
        ("hubert", "Wav2Vec2CTCTokenizer"),
        ("ibert", "RobertaTokenizer"),
        ("idefics", "LlamaTokenizerFast" if is_tokenizers_available() else None),
        ("idefics2", "LlamaTokenizerFast" if is_tokenizers_available() else None),
        ("idefics3", "LlamaTokenizerFast" if is_tokenizers_available() else None),
        ("instructblip", "GPT2Tokenizer" if is_tokenizers_available() else None),
        ("instructblipvideo", "GPT2Tokenizer" if is_tokenizers_available() else None),
        ("internvl", "Qwen2TokenizerFast" if is_tokenizers_available() else None),
        ("jamba", "LlamaTokenizerFast" if is_tokenizers_available() else None),
        ("janus", "LlamaTokenizerFast" if is_tokenizers_available() else None),
        ("jetmoe", "LlamaTokenizerFast" if is_tokenizers_available() else None),
        ("jukebox", "JukeboxTokenizer"),
        ("kosmos-2", "XLMRobertaTokenizerFast" if is_tokenizers_available() else None),
        ("kosmos-2.5", "PreTrainedTokenizerFast" if is_tokenizers_available() else None),
        ("layoutlm", "BertTokenizer" if is_tokenizers_available() else None),
        ("layoutlmv2", "LayoutLMv2TokenizerFast" if is_tokenizers_available() else None),
        ("layoutlmv3", "LayoutLMv3TokenizerFast" if is_tokenizers_available() else None),
        ("layoutxlm", "LayoutXLMTokenizerFast" if is_tokenizers_available() else None),
        ("led", "LEDTokenizerFast" if is_tokenizers_available() else None),
        ("lilt", "LayoutLMv3TokenizerFast" if is_tokenizers_available() else None),
        ("llama", "LlamaTokenizer" if is_tokenizers_available() else None),
        ("llama4", "LlamaTokenizerFast" if is_tokenizers_available() else None),
        ("llama4_text", "LlamaTokenizerFast" if is_tokenizers_available() else None),
        ("llava", "LlamaTokenizerFast" if is_tokenizers_available() else None),
        ("llava_next", "LlamaTokenizerFast" if is_tokenizers_available() else None),
        ("llava_next_video", "LlamaTokenizerFast" if is_tokenizers_available() else None),
        ("llava_onevision", "LlamaTokenizerFast" if is_tokenizers_available() else None),
        ("longformer", "LongformerTokenizerFast" if is_tokenizers_available() else None),
        ("longt5", "T5Tokenizer" if is_tokenizers_available() else None),
        ("luke", "LukeTokenizer"),
        ("lxmert", "BertTokenizer" if is_tokenizers_available() else None),
        ("m2m_100", "M2M100Tokenizer" if is_sentencepiece_available() else None),
        ("mamba", "GPTNeoXTokenizerFast" if is_tokenizers_available() else None),
        ("mamba2", "GPTNeoXTokenizerFast" if is_tokenizers_available() else None),
        ("marian", "MarianTokenizer" if is_sentencepiece_available() else None),
        ("mbart", "MBartTokenizer" if is_tokenizers_available() else None),
        ("mbart50", "MBart50TokenizerFast" if is_tokenizers_available() else None),
        ("mega", "RobertaTokenizer"),
        ("megatron-bert", "BertTokenizer" if is_tokenizers_available() else None),
        ("metaclip_2", "XLMRobertaTokenizerFast" if is_tokenizers_available() else None),
        ("mgp-str", "MgpstrTokenizer"),
        ("minimax", "GPT2Tokenizer" if is_tokenizers_available() else None),
        (
            "mistral",
            "MistralCommonTokenizer"
            if is_mistral_common_available()
            else ("LlamaTokenizerFast" if is_tokenizers_available() else None),
        ),
        (
            "mixtral",
            "MistralCommonTokenizer"
            if is_mistral_common_available()
            else ("LlamaTokenizerFast" if is_tokenizers_available() else None),
        ),
        ("mllama", "LlamaTokenizerFast" if is_tokenizers_available() else None),
        ("mluke", "MLukeTokenizer" if is_sentencepiece_available() else None),
        ("mm-grounding-dino", "BertTokenizer" if is_tokenizers_available() else None),
        ("mobilebert", "BertTokenizer" if is_tokenizers_available() else None),
        ("modernbert", "PreTrainedTokenizerFast" if is_tokenizers_available() else None),
        ("moonshine", "PreTrainedTokenizerFast" if is_tokenizers_available() else None),
        ("moshi", "PreTrainedTokenizerFast" if is_tokenizers_available() else None),
        ("mpnet", "MPNetTokenizer" if is_tokenizers_available() else None),
        ("mpt", "GPTNeoXTokenizerFast" if is_tokenizers_available() else None),
        ("mra", "RobertaTokenizer"),
        ("mt5", "MT5TokenizerFast" if is_tokenizers_available() else None),
        ("musicgen", "T5Tokenizer" if is_tokenizers_available() else None),
        ("musicgen_melody", "T5Tokenizer" if is_tokenizers_available() else None),
        ("mvp", "MvpTokenizerFast" if is_tokenizers_available() else None),
        ("myt5", "MyT5Tokenizer"),
        ("nemotron", "PreTrainedTokenizerFast" if is_tokenizers_available() else None),
        ("nezha", "BertTokenizer" if is_tokenizers_available() else None),
        ("nllb", "NllbTokenizerFast" if is_tokenizers_available() else None),
        ("nllb-moe", "NllbTokenizerFast" if is_tokenizers_available() else None),
        ("nougat", "NougatTokenizer" if is_tokenizers_available() else None),
        ("nystromformer", "AlbertTokenizerFast" if is_tokenizers_available() else None),
        ("olmo", "GPTNeoXTokenizerFast" if is_tokenizers_available() else None),
        ("olmo2", "GPTNeoXTokenizerFast" if is_tokenizers_available() else None),
        ("olmo3", "GPT2Tokenizer" if is_tokenizers_available() else None),
        ("olmoe", "GPTNeoXTokenizerFast" if is_tokenizers_available() else None),
        ("omdet-turbo", "CLIPTokenizerFast" if is_tokenizers_available() else None),
        ("oneformer", "CLIPTokenizerFast" if is_tokenizers_available() else None),
        ("openai-gpt", "OpenAIGPTTokenizer" if is_tokenizers_available() else None),
        ("opt", "GPT2Tokenizer" if is_tokenizers_available() else None),
        ("owlv2", "CLIPTokenizerFast" if is_tokenizers_available() else None),
        ("owlvit", "CLIPTokenizerFast" if is_tokenizers_available() else None),
        ("paligemma", "LlamaTokenizerFast" if is_tokenizers_available() else None),
        ("pegasus", "PegasusTokenizer" if is_tokenizers_available() else None),
        ("pegasus_x", "PegasusTokenizer" if is_tokenizers_available() else None),
        ("perceiver", "PerceiverTokenizer"),
        ("persimmon", "LlamaTokenizerFast" if is_tokenizers_available() else None),
        ("phi", "CodeGenTokenizer" if is_tokenizers_available() else None),
        ("phi3", "LlamaTokenizerFast" if is_tokenizers_available() else None),
        ("phimoe", "LlamaTokenizerFast" if is_tokenizers_available() else None),
        ("phobert", "PhobertTokenizer"),
        ("pix2struct", "T5Tokenizer" if is_tokenizers_available() else None),
        (
            "pixtral",
            "MistralCommonTokenizer"
            if is_mistral_common_available()
            else ("PreTrainedTokenizerFast" if is_tokenizers_available() else None),
        ),
        ("plbart", "PLBartTokenizer" if is_sentencepiece_available() else None),
        ("prophetnet", "ProphetNetTokenizer"),
        ("qdqbert", "BertTokenizer" if is_tokenizers_available() else None),
        ("qwen2", "Qwen2TokenizerFast" if is_tokenizers_available() else None),
        ("qwen2_5_omni", "Qwen2TokenizerFast" if is_tokenizers_available() else None),
        ("qwen2_5_vl", "Qwen2TokenizerFast" if is_tokenizers_available() else None),
        ("qwen2_audio", "Qwen2TokenizerFast" if is_tokenizers_available() else None),
        ("qwen2_moe", "Qwen2TokenizerFast" if is_tokenizers_available() else None),
        ("qwen2_vl", "Qwen2TokenizerFast" if is_tokenizers_available() else None),
        ("qwen3", "Qwen2TokenizerFast" if is_tokenizers_available() else None),
        ("qwen3_moe", "Qwen2TokenizerFast" if is_tokenizers_available() else None),
        ("qwen3_next", "Qwen2TokenizerFast" if is_tokenizers_available() else None),
        ("qwen3_omni_moe", "Qwen2TokenizerFast" if is_tokenizers_available() else None),
        ("qwen3_vl", "Qwen2TokenizerFast" if is_tokenizers_available() else None),
        ("qwen3_vl_moe", "Qwen2TokenizerFast" if is_tokenizers_available() else None),
        ("rag", "RagTokenizer"),
        ("realm", "BertTokenizer" if is_tokenizers_available() else None),
        ("recurrent_gemma", "GemmaTokenizerFast" if is_tokenizers_available() else None),
        ("reformer", "ReformerTokenizer" if is_tokenizers_available() else None),
        ("rembert", "RemBertTokenizer" if is_tokenizers_available() else None),
        ("retribert", "BertTokenizer" if is_tokenizers_available() else None),
        ("roberta", "RobertaTokenizer"),
        ("roberta-prelayernorm", "RobertaTokenizer"),
        ("roc_bert", "RoCBertTokenizer"),
        ("roformer", "RoFormerTokenizerFast" if is_tokenizers_available() else None),
        ("rwkv", "GPTNeoXTokenizerFast" if is_tokenizers_available() else None),
        ("seamless_m4t", "SeamlessM4TTokenizerFast" if is_tokenizers_available() else None),
        ("seamless_m4t_v2", "SeamlessM4TTokenizerFast" if is_tokenizers_available() else None),
        ("shieldgemma2", "GemmaTokenizerFast" if is_tokenizers_available() else None),
        ("siglip", "SiglipTokenizer" if is_sentencepiece_available() else None),
        ("siglip2", "GemmaTokenizerFast" if is_tokenizers_available() else None),
        ("smollm3", "PreTrainedTokenizerFast" if is_tokenizers_available() else None),
        ("speech_to_text", "Speech2TextTokenizer" if is_sentencepiece_available() else None),
        ("speech_to_text_2", "Speech2Text2Tokenizer"),
        ("speecht5", "SpeechT5Tokenizer" if is_sentencepiece_available() else None),
        ("splinter", "SplinterTokenizer"),
        ("squeezebert", "BertTokenizer" if is_tokenizers_available() else None),
        ("stablelm", "GPTNeoXTokenizerFast" if is_tokenizers_available() else None),
        ("starcoder2", "GPT2Tokenizer" if is_tokenizers_available() else None),
        ("switch_transformers", "T5Tokenizer" if is_tokenizers_available() else None),
        ("t5", "T5Tokenizer" if is_tokenizers_available() else None),
        ("t5gemma", "GemmaTokenizerFast" if is_tokenizers_available() else None),
        ("tapas", "TapasTokenizer"),
        ("tapex", "TapexTokenizer"),
        ("transfo-xl", "TransfoXLTokenizer"),
        ("tvp", "BertTokenizer" if is_tokenizers_available() else None),
        ("udop", "UdopTokenizerFast" if is_tokenizers_available() else None),
        ("umt5", "T5Tokenizer" if is_tokenizers_available() else None),
        ("video_llava", "LlamaTokenizerFast" if is_tokenizers_available() else None),
        ("vilt", "BertTokenizer" if is_tokenizers_available() else None),
        ("vipllava", "LlamaTokenizerFast" if is_tokenizers_available() else None),
        ("visual_bert", "BertTokenizer" if is_tokenizers_available() else None),
        ("vits", "VitsTokenizer"),
        (
            "voxtral",
            "MistralCommonTokenizer"
            if is_mistral_common_available()
            else ("LlamaTokenizerFast" if is_tokenizers_available() else None),
        ),
        ("wav2vec2", "Wav2Vec2CTCTokenizer"),
        ("wav2vec2-bert", "Wav2Vec2CTCTokenizer"),
        ("wav2vec2-conformer", "Wav2Vec2CTCTokenizer"),
        ("wav2vec2_phoneme", "Wav2Vec2PhonemeCTCTokenizer"),
        ("whisper", "WhisperTokenizerFast" if is_tokenizers_available() else None),
        ("xclip", "CLIPTokenizerFast" if is_tokenizers_available() else None),
        ("xglm", "XGLMTokenizer" if is_tokenizers_available() else None),
        ("xlm", "XLMTokenizer"),
        ("xlm-prophetnet", "XLMProphetNetTokenizer" if is_sentencepiece_available() else None),
        ("xlm-roberta", "XLMRobertaTokenizer" if is_tokenizers_available() else None),
        ("xlm-roberta-xl", "XLMRobertaTokenizer" if is_tokenizers_available() else None),
        ("xlnet", "XLNetTokenizerFast" if is_tokenizers_available() else None),
        ("xlstm", "GPTNeoXTokenizerFast" if is_tokenizers_available() else None),
        ("xmod", "XLMRobertaTokenizerFast" if is_tokenizers_available() else None),
        ("yoso", "AlbertTokenizer" if is_tokenizers_available() else None),
        ("zamba", "LlamaTokenizerFast" if is_tokenizers_available() else None),
        ("zamba2", "LlamaTokenizerFast" if is_tokenizers_available() else None),
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


def tokenizer_class_from_name(class_name: str) -> Union[type[Any], None]:
    if class_name == "PreTrainedTokenizerFast":
        return TokenizersBackend

    # V5: TOKENIZER_MAPPING_NAMES now maps to single strings, not tuples
    for module_name, tokenizer_class in TOKENIZER_MAPPING_NAMES.items():
        if tokenizer_class == class_name:
            module_name = model_type_to_module_name(module_name)
            if module_name in ["mistral", "mixtral", "ministral"] and class_name == "MistralCommonTokenizer":
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
    # init and we return the proper dummy to get an appropriate error message.
    main_module = importlib.import_module("transformers")
    if hasattr(main_module, class_name):
        return getattr(main_module, class_name)

    return None


def _find_sentencepiece_model_file(pretrained_model_name_or_path, **kwargs):
    """
    Find any .model file (SentencePiece model) in the model directory or Hub repo.

    Args:
        pretrained_model_name_or_path: Path to local directory or model id on the Hub
        **kwargs: Additional arguments like revision, token, cache_dir, local_files_only, subfolder

    Returns:
        The filename of the .model file if found, None otherwise
    """
    # First try tokenizer.model (most common)
    try:
        if has_file(
            pretrained_model_name_or_path,
            "tokenizer.model",
            revision=kwargs.get("revision"),
            token=kwargs.get("token"),
            cache_dir=kwargs.get("cache_dir"),
            local_files_only=kwargs.get("local_files_only", False),
        ):
            return "tokenizer.model"
    except Exception:
        pass

    subfolder = kwargs.get("subfolder", "")
    local_files_only = kwargs.get("local_files_only", False)

    # If it's a local directory, list all files and find any .model file
    if os.path.isdir(pretrained_model_name_or_path):
        dir_path = (
            os.path.join(pretrained_model_name_or_path, subfolder) if subfolder else pretrained_model_name_or_path
        )
        if os.path.isdir(dir_path):
            for filename in os.listdir(dir_path):
                if filename.endswith(".model"):
                    return filename
    # Otherwise, try to list files from the Hub
    elif not local_files_only:
        try:
            model_files = [
                entry.path if not subfolder else entry.path.removeprefix(f"{subfolder}/")
                for entry in list_repo_tree(
                    repo_id=pretrained_model_name_or_path,
                    revision=kwargs.get("revision"),
                    path_in_repo=subfolder if subfolder else None,
                    recursive=False,
                    token=kwargs.get("token"),
                )
                if entry.path.endswith(".model")
            ]
            if model_files:
                # Return the first .model file found
                return model_files[0]
        except (GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError):
            # These are valid errors that should be raised
            raise
        except (HfHubHTTPError, OfflineModeIsEnabled, httpx.NetworkError):
            # Connection errors - fall through to return None
            pass

    return None


def _load_tokenizers_backend(tokenizer_class, pretrained_model_name_or_path, inputs, kwargs):
    """
    Load a tokenizer using only the tokenizers backend (no SentencePiece fallback).

    This function attempts to load with the following priority:
    1. If tokenizer.json exists, load directly
    2. If any .model file (SPM) exists, try extracting vocab and merges
    3. If vocab.json and merges.txt exist, load with those

    Args:
        tokenizer_class: The tokenizer class to instantiate
        pretrained_model_name_or_path: Path or model id
        inputs: Additional positional arguments for tokenizer init
        kwargs: Additional keyword arguments

    Returns:
        An instantiated tokenizer object

    Raises:
        ValueError: If tokenizer could not be loaded with tokenizers backend
    """
    # Try tokenizer.json first
    try:
        tokenizer_json_exists = has_file(
            pretrained_model_name_or_path,
            "tokenizer.json",
            revision=kwargs.get("revision"),
            token=kwargs.get("token"),
            cache_dir=kwargs.get("cache_dir"),
            local_files_only=kwargs.get("local_files_only", False),
        )
    except Exception:
        tokenizer_json_exists = False

    if tokenizer_json_exists:
        return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

    # Try extracting from SentencePiece model
    spm_file = _find_sentencepiece_model_file(pretrained_model_name_or_path, **kwargs)
    if spm_file is not None:
        try:
            resolved_spm = cached_file(
                pretrained_model_name_or_path,
                spm_file,
                cache_dir=kwargs.get("cache_dir"),
                force_download=kwargs.get("force_download", False),
                proxies=kwargs.get("proxies"),
                token=kwargs.get("token"),
                revision=kwargs.get("revision"),
                local_files_only=kwargs.get("local_files_only", False),
                subfolder=kwargs.get("subfolder", ""),
            )
        except Exception:
            resolved_spm = None

        if resolved_spm is not None:
            try:
                from ...tokenization_sentencepiece import SentencePieceExtractor

                fast_sig = inspect.signature(getattr(tokenizer_class, "__init__", tokenizer_class))
                if "vocab" in fast_sig.parameters and "merges" in fast_sig.parameters:
                    try:
                        vocab, merges = SentencePieceExtractor(resolved_spm).extract()
                        return tokenizer_class.from_pretrained(
                            pretrained_model_name_or_path, *inputs, vocab=vocab, merges=merges, **kwargs
                        )
                    except Exception:
                        pass
            except Exception:
                pass

    # Try vocab.json + merges.txt
    try:
        vocab_file = cached_file(
            pretrained_model_name_or_path,
            "vocab.json",
            cache_dir=kwargs.get("cache_dir"),
            force_download=kwargs.get("force_download", False),
            proxies=kwargs.get("proxies"),
            token=kwargs.get("token"),
            revision=kwargs.get("revision"),
            local_files_only=kwargs.get("local_files_only", False),
            subfolder=kwargs.get("subfolder", ""),
        )
        vocab_json_exists = True
    except Exception:
        vocab_json_exists = False

    try:
        merges_file = cached_file(
            pretrained_model_name_or_path,
            "merges.txt",
            cache_dir=kwargs.get("cache_dir"),
            force_download=kwargs.get("force_download", False),
            proxies=kwargs.get("proxies"),
            token=kwargs.get("token"),
            revision=kwargs.get("revision"),
            local_files_only=kwargs.get("local_files_only", False),
        )
        merges_txt_exists = True
    except Exception:
        merges_txt_exists = False

    if vocab_json_exists and merges_txt_exists:
        vocab = load_vocab(vocab_file)
        merges = load_merges(merges_file)
        return tokenizer_class.from_pretrained(
            pretrained_model_name_or_path, *inputs, vocab=vocab, merges=merges, **kwargs
        )

    # If all methods failed, raise an error
    raise ValueError(
        f"Could not load tokenizer from {pretrained_model_name_or_path} using tokenizers backend. "
        "No tokenizer.json, vocab.json+merges.txt, or compatible SentencePiece model found."
    )


def _try_load_tokenizer_with_fallbacks(tokenizer_class, pretrained_model_name_or_path, inputs, kwargs):
    """
    Try to load a tokenizer with backend selection.

    This function routes to the appropriate backend based on the 'backend' parameter:
    - "tokenizers" (default): Uses HuggingFace tokenizers library backend
    - "sentencepiece": Uses SentencePiece backend

    For the tokenizers backend, attempts to load with the following priority:
    1. If tokenizer.json exists, load directly
    2. If any .model file (SPM) exists, try extracting vocab and merges
    3. If vocab.json and merges.txt exist, load with those
    4. Fallback to SentencePieceBackend if available

    Args:
        tokenizer_class: The tokenizer class to instantiate (can be None)
        pretrained_model_name_or_path: Path or model id
        inputs: Additional positional arguments for tokenizer init
        kwargs: Additional keyword arguments (may include 'backend' parameter, defaults to "tokenizers")

    Returns:
        An instantiated tokenizer object

    Raises:
        ValueError: If no tokenizer could be loaded
    """
    # Extract the backend parameter - default to "tokenizers" to prioritize tokenizers backend
    backend = kwargs.pop("backend", "tokenizers")

    # Validate backend parameter
    if backend not in ["sentencepiece", "tokenizers"]:
        logger.warning(
            f"Invalid backend '{backend}' specified. Valid options are 'tokenizers' or 'sentencepiece'. "
            "Defaulting to 'tokenizers' backend."
        )
        backend = "tokenizers"

    # Route to SentencePiece backend if requested
    if backend == "sentencepiece":
        if SentencePieceBackend is None:
            raise ValueError(
                "SentencePiece backend was requested but sentencepiece is not installed. "
                "Please install it with: pip install sentencepiece"
            )
        logger.info("Loading tokenizer with SentencePiece backend")
        return SentencePieceBackend.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

    # Route to tokenizers backend (default)
    if backend == "tokenizers":
        if tokenizer_class is not None:
            # Check if tokenizer_class inherits from PreTrainedTokenizer (but not from TokenizersBackend/SentencePieceBackend)
            # These are edge cases with custom logic (e.g., BioGptTokenizer with Moses tokenization)
            from ...tokenization_utils import PreTrainedTokenizer

            # Build list of backend classes to check against
            backend_classes = [TokenizersBackend] if TokenizersBackend else []
            if SentencePieceBackend:
                backend_classes.append(SentencePieceBackend)

            if issubclass(tokenizer_class, PreTrainedTokenizer) and not any(
                issubclass(tokenizer_class, bc) for bc in backend_classes
            ):
                logger.info("Loading tokenizer with custom PreTrainedTokenizer backend (edge case)")
                return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

            if TokenizersBackend is None:
                raise ValueError(
                    "Tokenizers backend is the default but tokenizers library is not installed. "
                    "Please install it with: pip install tokenizers"
                )
            logger.info("Loading tokenizer with tokenizers backend")
            return _load_tokenizers_backend(tokenizer_class, pretrained_model_name_or_path, inputs, kwargs)

        # If no tokenizer class but tokenizers backend requested, fall back to SentencePiece if available
        spm_file = _find_sentencepiece_model_file(pretrained_model_name_or_path, **kwargs)
        if spm_file is not None and SentencePieceBackend is not None:
            logger.info(
                f"Tokenizers backend was requested but no tokenizer class found. "
                f"Falling back to SentencePieceBackend since {spm_file} file was found."
            )
            return SentencePieceBackend.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

        raise ValueError(
            f"Could not load tokenizer from {pretrained_model_name_or_path}. "
            "No tokenizer class could be determined and no SentencePiece model found."
        )


def get_tokenizer_config(
    pretrained_model_name_or_path: Union[str, os.PathLike[str]],
    cache_dir: Optional[Union[str, os.PathLike[str]]] = None,
    force_download: bool = False,
    resume_download: Optional[bool] = None,
    proxies: Optional[dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
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
        resume_download:
            Deprecated and ignored. All downloads are now resumed by default when possible.
            Will be removed in v5 of Transformers.
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
    use_auth_token = kwargs.pop("use_auth_token", None)
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        token = use_auth_token

    commit_hash = kwargs.get("_commit_hash")
    resolved_config_file = cached_file(
        pretrained_model_name_or_path,
        TOKENIZER_CONFIG_FILE,
        cache_dir=cache_dir,
        force_download=force_download,
        resume_download=resume_download,
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
                    - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                      using the [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.
                    - A path or url to a single saved vocabulary file if and only if the tokenizer only requires a
                      single vocabulary file (like Bert or XLNet), e.g.: `./my_model_directory/vocab.txt`. (Not
                      applicable to all derived classes)
            inputs (additional positional arguments, *optional*):
                Will be passed along to the Tokenizer `__init__()` method.
            config ([`PretrainedConfig`], *optional*)
                The configuration object used to determine the tokenizer class to instantiate.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download the model weights and configuration files and override the
                cached versions if they exist.
            resume_download:
                Deprecated and ignored. All downloads are now resumed by default when possible.
                Will be removed in v5 of Transformers.
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
        use_auth_token = kwargs.pop("use_auth_token", None)
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if kwargs.get("token") is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            kwargs["token"] = use_auth_token

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

        # Next, let's try to use the tokenizer_config file to get the tokenizer class.
        tokenizer_config = get_tokenizer_config(pretrained_model_name_or_path, **kwargs)
        if "_commit_hash" in tokenizer_config:
            kwargs["_commit_hash"] = tokenizer_config["_commit_hash"]
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
                if gguf_file:
                    gguf_path = cached_file(pretrained_model_name_or_path, gguf_file, **kwargs)
                    config_dict = load_gguf_checkpoint(gguf_path, return_tensors=False)["config"]
                    config = AutoConfig.for_model(**config_dict)
                else:
                    config = AutoConfig.from_pretrained(
                        pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
                    )
            config_tokenizer_class = config.tokenizer_class
            if hasattr(config, "auto_map") and "AutoTokenizer" in config.auto_map:
                tokenizer_auto_map = config.auto_map["AutoTokenizer"]

        if (
            config_tokenizer_class is not None
            and config_tokenizer_class != "PreTrainedTokenizerFast"
            and "Fast" in config_tokenizer_class
        ):
            config_tokenizer_class = config_tokenizer_class[:-4]

        has_remote_code = tokenizer_auto_map is not None
        has_local_code = type(config) in TOKENIZER_MAPPING or (
            config_tokenizer_class is not None
            and (
                tokenizer_class_from_name(config_tokenizer_class) is not None
                or tokenizer_class_from_name(config_tokenizer_class + "Fast") is not None
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
        elif config_tokenizer_class is not None:
            fast_tokenizer_class = None
            if fast_tokenizer_class is None:
                tokenizer_class_candidate = config_tokenizer_class
                tokenizer_class = tokenizer_class_from_name(tokenizer_class_candidate)
            else:
                tokenizer_class = fast_tokenizer_class

            return _try_load_tokenizer_with_fallbacks(tokenizer_class, pretrained_model_name_or_path, inputs, kwargs)

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
            tokenizer_class = TOKENIZER_MAPPING[type(config)]

            if tokenizer_class is not None:
                return _try_load_tokenizer_with_fallbacks(
                    tokenizer_class, pretrained_model_name_or_path, inputs, kwargs
                )
            else:
                raise ValueError(
                    "This tokenizer cannot be instantiated. Please make sure you have `sentencepiece` installed "
                    "in order to use this tokenizer."
                )

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
            config_class ([`PretrainedConfig`]):
                The configuration corresponding to the model to register.
            tokenizer_class: The tokenizer class to register (V5 - preferred parameter).
            slow_tokenizer_class: (Deprecated) The slow tokenizer to register.
            fast_tokenizer_class: (Deprecated) The fast tokenizer to register.
        """
        # V5: Support both new single-tokenizer API and legacy slow/fast API for BC for now
        if tokenizer_class is None:
            # Legacy: prefer fast over slow
            if fast_tokenizer_class is not None:
                tokenizer_class = fast_tokenizer_class
            elif slow_tokenizer_class is not None:
                tokenizer_class = slow_tokenizer_class
            else:
                raise ValueError("You need to pass a `tokenizer_class`")

        TOKENIZER_MAPPING.register(config_class, tokenizer_class, exist_ok=exist_ok)


__all__ = ["TOKENIZER_MAPPING", "AutoTokenizer"]
