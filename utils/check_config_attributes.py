# Copyright 2023 The HuggingFace Inc. team.
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

import inspect
import os
import re

from transformers.configuration_utils import PreTrainedConfig
from transformers.utils import direct_transformers_import


# All paths are set with the intent you should run this script from the root of the repo with the command
# python utils/check_config_docstrings.py
PATH_TO_TRANSFORMERS = "src/transformers"


# This is to make sure the transformers module imported is the one in the repo.
transformers = direct_transformers_import(PATH_TO_TRANSFORMERS)

CONFIG_MAPPING = transformers.models.auto.configuration_auto.CONFIG_MAPPING

# Usually of small list of allowed attrs, but can be True to allow all
SPECIAL_CASES_TO_ALLOW = {
    "ExaoneMoeConfig": ["first_k_dense_replace"],  # BC for other frameworks
    "AfmoeConfig": ["global_attn_every_n_layers", "rope_scaling"],
    "xLSTMConfig": ["add_out_norm", "chunkwise_kernel", "sequence_kernel", "step_kernel"],
    "Lfm2Config": ["full_attn_idxs"],
    "DiaConfig": ["delay_pattern"],
    "BambaConfig": ["attn_layer_indices"],
    "Dots1Config": ["max_window_layers"],
    "JambaConfig": ["attn_layer_offset", "attn_layer_period", "expert_layer_offset", "expert_layer_period"],
    "JetMoeConfig": ["output_router_logits"],
    "Phi3Config": ["embd_pdrop"],
    "EncodecConfig": ["overlap"],
    "XcodecConfig": ["sample_rate", "audio_channels"],
    "RecurrentGemmaConfig": ["block_types"],
    "MambaConfig": ["expand"],
    "FalconMambaConfig": ["expand"],
    "FSMTConfig": ["langs", "common_kwargs", "early_stopping", "length_penalty", "max_length", "num_beams"],
    "GPTNeoConfig": ["attention_types"],
    "BlenderbotConfig": ["encoder_no_repeat_ngram_size"],
    "EsmConfig": ["is_folding_model"],
    "Mask2FormerConfig": ["ignore_value"],
    "OneFormerConfig": ["ignore_value", "norm"],
    "T5Config": ["feed_forward_proj"],
    "MT5Config": ["feed_forward_proj", "tokenizer_class"],
    "UMT5Config": ["feed_forward_proj", "tokenizer_class"],
    "LongT5Config": ["feed_forward_proj"],
    "Pop2PianoConfig": ["feed_forward_proj"],
    "BioGptConfig": ["layer_norm_eps"],
    "GLPNConfig": ["layer_norm_eps"],
    "SegformerConfig": ["layer_norm_eps"],
    "CvtConfig": ["layer_norm_eps"],
    "PerceiverConfig": ["layer_norm_eps"],
    "InformerConfig": ["num_static_real_features", "num_time_features"],
    "TimeSeriesTransformerConfig": ["num_static_real_features", "num_time_features"],
    "AutoformerConfig": ["num_static_real_features", "num_time_features"],
    "SamVisionConfig": ["mlp_ratio"],
    "Sam3VisionConfig": ["backbone_feature_sizes"],
    "SamHQVisionConfig": ["mlp_ratio"],
    "ClapAudioConfig": ["num_classes"],
    "SpeechT5HifiGanConfig": ["sampling_rate"],
    "UdopConfig": ["feed_forward_proj"],
    "ZambaConfig": ["attn_layer_offset", "attn_layer_period"],
    "MllamaVisionConfig": ["supported_aspect_ratios"],
    "LEDConfig": ["classifier_dropout"],
    "GPTNeoXConfig": ["rotary_emb_base"],
    "ShieldGemma2Config": ["mm_tokens_per_image", "vision_config"],
    "Llama4VisionConfig": ["multi_modal_projector_bias", "norm_eps"],
    "ModernBertConfig": ["local_attention", "reference_compile"],
    "ModernBertDecoderConfig": ["global_attn_every_n_layers", "local_attention", "local_rope_theta"],
    "SmolLM3Config": ["no_rope_layer_interval"],
    "Gemma3nVisionConfig": ["architecture", "do_pooling", "model_args"],
    "HiggsAudioV2Config": ["audio_bos_token", "audio_stream_bos_id", "audio_stream_eos_id"],
    "HiggsAudioV2TokenizerConfig": ["downsample_factor"],
    "CsmConfig": ["tie_codebooks_embeddings"],
    "DeepseekV2Config": ["norm_topk_prob"],
    "SeamlessM4TConfig": True,
    "SeamlessM4Tv2Config": True,
    "ConditionalDetrConfig": True,
    "DabDetrConfig": True,
    "SwitchTransformersConfig": True,
    "DetrConfig": True,
    "DFineConfig": True,
    "GroundingDinoConfig": True,
    "MMGroundingDinoConfig": True,
    "RTDetrConfig": True,
    "RTDetrV2Config": True,
    "YolosConfig": True,
    "Llama4TextConfig": True,
    "DPRConfig": True,
    "FuyuConfig": True,
    "LayoutXLMConfig": True,
    "CLIPSegConfig": True,
    "DeformableDetrConfig": True,
    "DinatConfig": True,
    "DonutSwinConfig": True,
    "FastSpeech2ConformerConfig": True,
    "LayoutLMv2Config": True,
    "MaskFormerSwinConfig": True,
    "MptConfig": True,
    "MptAttentionConfig": True,
    "RagConfig": True,
    "SpeechT5Config": True,
    "SwinConfig": True,
    "Swin2SRConfig": True,
    "Swinv2Config": True,
    "TableTransformerConfig": True,
    "TapasConfig": True,
    "UniSpeechConfig": True,
    "UniSpeechSatConfig": True,
    "WavLMConfig": True,
    "WhisperConfig": True,
    "JukeboxPriorConfig": True,
    "Pix2StructTextConfig": True,
    "IdeficsConfig": True,
    "IdeficsVisionConfig": True,
    "IdeficsPerceiverConfig": True,
    "GptOssConfig": True,
    "LwDetrConfig": True,
}

# Common and important attributes, even if they do not always appear in the modeling files (can be a regex pattern)
ATTRIBUTES_TO_ALLOW = (
    # Inits related
    "initializer_range",
    "init_std",
    "initializer_factor",
    "tie_word_embeddings",
    # Special tokens
    "bos_index",
    "eos_index",
    "pad_index",
    "unk_index",
    "mask_index",
    r".+_token_id",
    r".+_token_index",
    # Processors
    "image_seq_length",
    "video_seq_length",
    "image_size",
    "text_config",  # may appear as `get_text_config()`
    "use_cache",
    "out_features",
    "out_indices",
    "sampling_rate",
    # backbone related arguments passed to load_backbone
    "use_pretrained_backbone",
    "backbone",
    "backbone_config",
    "use_timm_backbone",
    "backbone_kwargs",
    # rope attributes may not appear directly in the modeling but are used
    "rope_theta",
    "partial_rotary_factor",
    "max_position_embeddings",
    "pretraining_tp",
    "use_sliding_window",
    "max_window_layers",
    # vision attributes that may be used indirectly via merge_with_config_defaults
    "vision_feature_layer",
    "vision_feature_select_strategy",
    "vision_aspect_ratio",
)


def check_attribute_being_used(config_class, attributes, default_value, source_strings):
    """Check if any name in `attributes` is used in one of the strings in `source_strings`

    Args:
        config_class (`type`):
            The configuration class for which the arguments in its `__init__` will be checked.
        attributes (`List[str]`):
            The name of an argument (or attribute) and its variant names if any.
        default_value (`Any`):
            A default value for the attribute in `attributes` assigned in the `__init__` of `config_class`.
        source_strings (`List[str]`):
            The python source code strings in the same modeling directory where `config_class` is defined. The file
            containing the definition of `config_class` should be excluded.
    """
    # If we can find the attribute used, then it's all good
    for attribute in attributes:
        for modeling_source in source_strings:
            # check if we can find `config.xxx`, `getattr(config, "xxx", ...)` or `getattr(self.config, "xxx", ...)`
            if (
                f"config.{attribute}" in modeling_source
                or f'getattr(config, "{attribute}"' in modeling_source
                or f'getattr(self.config, "{attribute}"' in modeling_source
                or (
                    "TextConfig" in config_class.__name__
                    and f"config.get_text_config().{attribute}" in modeling_source
                )
            ):
                return True
            # Deal with multi-line cases
            elif (
                re.search(
                    rf'getattr[ \t\v\n\r\f]*\([ \t\v\n\r\f]*(self\.)?config,[ \t\v\n\r\f]*"{attribute}"',
                    modeling_source,
                )
                is not None
            ):
                return True

    # Special cases to be allowed even if not found as used
    for attribute in attributes:
        # Allow if the default value in the configuration class is different from the one in `PreTrainedConfig`
        if (attribute == "is_encoder_decoder" and default_value is True) or attribute == "tie_word_embeddings":
            return True
        # General exceptions for all models
        elif any(re.search(exception, attribute) for exception in ATTRIBUTES_TO_ALLOW):
            return True
        # Model-specific exceptions
        elif config_class.__name__ in SPECIAL_CASES_TO_ALLOW:
            model_exceptions = SPECIAL_CASES_TO_ALLOW[config_class.__name__]
            # Can be true to allow all attributes, or a list of specific allowed attributes
            if (isinstance(model_exceptions, bool) and model_exceptions) or attribute in model_exceptions:
                return True

    return False


def check_config_attributes_being_used(config_class):
    """Check the arguments in `__init__` of `config_class` are used in the modeling files in the same directory

    Args:
        config_class (`type`):
            The configuration class for which the arguments in its `__init__` will be checked.
    """
    # Get the parameters in `__init__` of the configuration class, and the default values if any
    signature = dict(inspect.signature(config_class.__init__).parameters)
    parameter_names = [x for x in list(signature.keys()) if x not in ["self", "kwargs"]]
    parameter_defaults = [signature[param].default for param in parameter_names]

    # If `attribute_map` exists, an attribute can have different names to be used in the modeling files, and as long
    # as one variant is used, the test should pass
    reversed_attribute_map = {}
    if len(config_class.attribute_map) > 0:
        reversed_attribute_map = {v: k for k, v in config_class.attribute_map.items()}

    # Get the path to modeling source files
    config_source_file = inspect.getsourcefile(config_class)
    model_dir = os.path.dirname(config_source_file)
    modeling_paths = [os.path.join(model_dir, fn) for fn in os.listdir(model_dir) if fn.startswith("modeling_")]

    # Get the source code strings
    modeling_sources = []
    for path in modeling_paths:
        if os.path.isfile(path):
            with open(path, encoding="utf8") as fp:
                modeling_sources.append(fp.read())

    unused_attributes = []
    for config_param, default_value in zip(parameter_names, parameter_defaults):
        # `attributes` here is all the variant names for `config_param`
        attributes = [config_param]
        # some configuration classes have non-empty `attribute_map`, and both names could be used in the
        # corresponding modeling files. As long as one of them appears, it is fine.
        if config_param in reversed_attribute_map:
            attributes.append(reversed_attribute_map[config_param])

        if not check_attribute_being_used(config_class, attributes, default_value, modeling_sources):
            unused_attributes.append(attributes[0])

    return sorted(unused_attributes)


def check_config_attributes():
    """Check the arguments in `__init__` of all configuration classes are used in python files"""
    configs_with_unused_attributes = {}
    for _config_class in list(CONFIG_MAPPING.values()):
        # Skip deprecated models
        if "models.deprecated" in _config_class.__module__:
            continue
        # Some config classes are not in `CONFIG_MAPPING` (e.g. `CLIPVisionConfig`, `Blip2VisionConfig`, etc.)
        config_classes_in_module = [
            cls
            for name, cls in inspect.getmembers(
                inspect.getmodule(_config_class),
                lambda x: inspect.isclass(x)
                and issubclass(x, PreTrainedConfig)
                and inspect.getmodule(x) == inspect.getmodule(_config_class),
            )
        ]
        for config_class in config_classes_in_module:
            unused_attributes = check_config_attributes_being_used(config_class)
            if len(unused_attributes) > 0:
                configs_with_unused_attributes[config_class.__name__] = unused_attributes

    if len(configs_with_unused_attributes) > 0:
        error = "The following configuration classes contain unused attributes in the corresponding modeling files:\n"
        for name, attributes in configs_with_unused_attributes.items():
            error += f"{name}: {attributes}\n"

        raise ValueError(error)


if __name__ == "__main__":
    check_config_attributes()
