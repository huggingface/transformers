# coding=utf-8
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

import importlib
import inspect
import os
import re


# All paths are set with the intent you should run this script from the root of the repo with the command
# python utils/check_config_docstrings.py
PATH_TO_TRANSFORMERS = "src/transformers"


# This is to make sure the transformers module imported is the one in the repo.
spec = importlib.util.spec_from_file_location(
    "transformers",
    os.path.join(PATH_TO_TRANSFORMERS, "__init__.py"),
    submodule_search_locations=[PATH_TO_TRANSFORMERS],
)
transformers = spec.loader.load_module()

CONFIG_MAPPING = transformers.models.auto.configuration_auto.CONFIG_MAPPING

SPECIAL_CASES_TO_ALLOW = {
    # used as `self.bert_model = BertModel(config, ...)`
    "DPRConfig": True,
    # not used in modeling files, but it's an important information
    "FSMTConfig": ["langs"],
    # used internally in the configuration class file
    "GPTNeoConfig": ["attention_types"],
    # used internally in the configuration class file
    "EsmConfig": ["is_folding_model"],
    # used during training (despite we don't have training script for these models yet)
    "Mask2FormerConfig": ["ignore_value"],
    # `ignore_value` used during training (despite we don't have training script for these models yet)
    # `norm` used in conversion script (despite not using in the modeling file)
    "OneFormerConfig": ["ignore_value", "norm"],
    # used during preprocessing and collation, see `collating_graphormer.py`
    "GraphormerConfig": ["spatial_pos_max"],
    # used internally in the configuration class file
    "T5Config": ["feed_forward_proj"],
    # used internally in the configuration class file
    # `tokenizer_class` get default value `T5Tokenizer` intentionally
    "MT5Config": ["feed_forward_proj", "tokenizer_class"],
    # used internally in the configuration class file
    "LongT5Config": ["feed_forward_proj"],
    # used internally in the configuration class file
    "SwitchTransformersConfig": ["feed_forward_proj"],
    # having default values other than `1e-5` - we can't fix them without breaking
    "BioGptConfig": ["layer_norm_eps"],
    # having default values other than `1e-5` - we can't fix them without breaking
    "GLPNConfig": ["layer_norm_eps"],
    # having default values other than `1e-5` - we can't fix them without breaking
    "SegformerConfig": ["layer_norm_eps"],
    # having default values other than `1e-5` - we can't fix them without breaking
    "CvtConfig": ["layer_norm_eps"],
    # having default values other than `1e-5` - we can't fix them without breaking
    "PerceiverConfig": ["layer_norm_eps"],
    # having default values other than `1e-5` - we can't fix them without breaking
    "RetriBertConfig": ["layer_norm_eps"],
    # having default values other than `1e-5` - we can't fix them without breaking
    "TrajectoryTransformerConfig": ["layer_norm_eps"],
}

# TODO (ydshieh): Check the failing cases, try to fix them or move some cases to the above block once we are sure
SPECIAL_CASES_TO_ALLOW.update(
    {
        "CLIPSegConfig": True,
        "DeformableDetrConfig": True,
        "DetaConfig": True,
        "DinatConfig": True,
        "DonutSwinConfig": True,
        "EfficientFormerConfig": True,
        "FSMTConfig": True,
        "JukeboxConfig": True,
        "LayoutLMv2Config": True,
        "MaskFormerSwinConfig": True,
        "MT5Config": True,
        "NatConfig": True,
        "OneFormerConfig": True,
        "PerceiverConfig": True,
        "RagConfig": True,
        "RetriBertConfig": True,
        "SpeechT5Config": True,
        "SwinConfig": True,
        "Swin2SRConfig": True,
        "Swinv2Config": True,
        "SwitchTransformersConfig": True,
        "TableTransformerConfig": True,
        "TapasConfig": True,
        "TimeSeriesTransformerConfig": True,
        "TrajectoryTransformerConfig": True,
        "TransfoXLConfig": True,
        "UniSpeechConfig": True,
        "UniSpeechSatConfig": True,
        "VanConfig": True,
        "WavLMConfig": True,
        "WhisperConfig": True,
    }
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
    attribute_used = False
    for attribute in attributes:
        for modeling_source in source_strings:
            # check if we can find `config.xxx`, `getattr(config, "xxx", ...)` or `getattr(self.config, "xxx", ...)`
            if (
                f"config.{attribute}" in modeling_source
                or f'getattr(config, "{attribute}"' in modeling_source
                or f'getattr(self.config, "{attribute}"' in modeling_source
            ):
                attribute_used = True
            # Deal with multi-line cases
            elif (
                re.search(
                    rf'getattr[ \t\v\n\r\f]*\([ \t\v\n\r\f]*(self\.)?config,[ \t\v\n\r\f]*"{attribute}"',
                    modeling_source,
                )
                is not None
            ):
                attribute_used = True
            # `SequenceSummary` is called with `SequenceSummary(config)`
            elif attribute in [
                "summary_type",
                "summary_use_proj",
                "summary_activation",
                "summary_last_dropout",
                "summary_proj_to_labels",
                "summary_first_dropout",
            ]:
                if "SequenceSummary" in modeling_source:
                    attribute_used = True
            if attribute_used:
                break
        if attribute_used:
            break

    # common and important attributes, even if they do not always appear in the modeling files
    attributes_to_allow = [
        "bos_index",
        "eos_index",
        "pad_index",
        "unk_index",
        "mask_index",
        "image_size",
        "use_cache",
    ]
    attributes_used_in_generation = ["encoder_no_repeat_ngram_size"]

    # Special cases to be allowed
    case_allowed = True
    if not attribute_used:
        case_allowed = False
        for attribute in attributes:
            # Allow if the default value in the configuration class is different from the one in `PretrainedConfig`
            if attribute in ["is_encoder_decoder"] and default_value is True:
                case_allowed = True
            elif attribute in ["tie_word_embeddings"] and default_value is False:
                case_allowed = True

            # Allow cases without checking the default value in the configuration class
            elif attribute in attributes_to_allow + attributes_used_in_generation:
                case_allowed = True
            elif attribute.endswith("_token_id"):
                case_allowed = True

            # configuration class specific cases
            if not case_allowed:
                allowed_cases = SPECIAL_CASES_TO_ALLOW.get(config_class.__name__, [])
                case_allowed = allowed_cases is True or attribute in allowed_cases

    return attribute_used or case_allowed


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
    # Let's check against all frameworks: as long as one framework uses an attribute, we are good.
    modeling_paths = [os.path.join(model_dir, fn) for fn in os.listdir(model_dir) if fn.startswith("modeling_")]

    # Get the source code strings
    modeling_sources = []
    for path in modeling_paths:
        if os.path.isfile(path):
            with open(path) as fp:
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
    """Check the arguments in `__init__` of all configuration classes are used in  python files"""
    configs_with_unused_attributes = {}
    for config_class in list(CONFIG_MAPPING.values()):
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
