# Copyright 2026 Mistral AI and The HuggingFace Inc. team. All rights reserved.
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

import copy
import dataclasses
import json
import unittest
from enum import Enum
from typing import Any

from huggingface_hub import hf_hub_download
from parameterized import parameterized

from transformers import Ministral3Config, Mistral3Config, Mistral4Config, MistralConfig
from transformers.integrations.mistral.native_config import mistral_native_config_from_params
from transformers.integrations.mistral.params_conversion import (
    _MISTRAL_EXTRAS_DOC,
    mistral_native_config_from_hf_config,
    mistral_native_config_to_hf_config,
)
from transformers.testing_utils import slow


# Real published repos, paired with the config.json section their params.json must reproduce.
# A section means the `vision_encoder` block is dropped and only the text config is compared.
_CONFIG_PARITY_CASES = [
    ("mistral", "mistralai/Mistral-7B-Instruct-v0.3", None),
    ("ministral3", "mistralai/Ministral-3-3B-Instruct-2512", "text_config"),
    ("mistral4", "mistralai/Mistral-Small-4-119B-2603", "text_config"),
    ("mistral3_vlm", "mistralai/Ministral-3-3B-Instruct-2512", None),
    ("mistral3_vlm_moe_text", "mistralai/Mistral-Small-4-119B-2603", None),
    ("devstral2", "mistralai/Devstral-2-123B-Instruct-2512", None),
    ("mistral_medium_3_5", "mistralai/Mistral-Medium-3.5-128B", None),
    # Mistral3Config wrapping a plain MistralConfig text config (no yarn), unlike every other VLM
    # case above (all yarn, so Ministral3Config or Mistral4Config text configs).
    ("mistral3_vlm_plain_text", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", None),
    ("mistral3_vlm_plain_text", "mistralai/Magistral-Small-2509", None),
]

# HF config classes for the full (non-sectioned) parity cases above, used by
# `TestReverseFromPublishedConfig` to load each published `config.json` as the right type.
_CONFIG_CLASSES: dict[str, type[MistralConfig | Ministral3Config | Mistral4Config | Mistral3Config]] = {
    "mistral": MistralConfig,
    "mistral3_vlm": Mistral3Config,
    "mistral3_vlm_moe_text": Mistral3Config,
    "devstral2": Ministral3Config,
    "mistral_medium_3_5": Mistral3Config,
    "mistral3_vlm_plain_text": Mistral3Config,
}

# Per-case keys to exclude from `_collect_mismatches`, keyed by the case name in
# `_CONFIG_PARITY_CASES`. Reserved for known, explained deviations, not a general escape hatch.
_KNOWN_DEVIATIONS: dict[str, frozenset[str]] = {
    # Devstral-2-123B-Instruct-2512 is text-only, yet publishes `model.vision_tower` and
    # `model.multi_modal_projector` in `quantization_config.modules_to_not_convert`: an artifact of
    # `convert_ministral3_weights_to_hf.py` hard-coding those regardless of `is_vision`.
    "devstral2": frozenset({"modules_to_not_convert"}),
}

# Converter-computed values the published config omits entirely, keyed by case name. Unlike
# `_KNOWN_DEVIATIONS` (which excludes a published value known to be wrong), these assert what the
# converter must produce for a key `_collect_mismatches` never reaches, because the published
# config carries no value at all to compare against (either because a superset key lives outside
# the compared section, or the schema has no place for it).
_ADDITIONAL_EXPECTATIONS: dict[str, dict[str, Any]] = {
    # No `head_dim` and no `rope_parameters.rope_type` in this legacy (pre-rope_parameters)
    # published config; both are still converter-computed and must be right.
    "mistral": {"head_dim": 128, "rope_parameters": {"rope_type": "default"}},
    # `quantization_config` lives outside `text_config` in the published (VLM) document, so the
    # text-only conversion this case exercises is never checked against it at all.
    "ministral3": {
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "static",
            "modules_to_not_convert": ["lm_head"],
            "weight_block_size": None,
        }
    },
    # Neither `rope_parameters.partial_rotary_factor` (an MLA-only field) nor `mistral_extras`
    # (which no published config carries) appear in the published document at all.
    "mistral4": {
        "rope_parameters": {"partial_rotary_factor": 0.5},
        "mistral_extras": {
            "moe": {"expert_parallel": 1, "expert_model_parallel": 1, "route_every_n": 1},
            "_comment": _MISTRAL_EXTRAS_DOC,
        },
    },
    # The published document omits the outer `tie_word_embeddings` key entirely (only
    # `text_config.tie_word_embeddings` is present); `Mistral3Config`'s own default (`True`)
    # happens to match, which is exactly what makes an injected wrong value unfalsifiable.
    "mistral3_vlm": {"tie_word_embeddings": True},
}

# Not derivable from params.json. Published configs also carry HF defaults that params.json never
# feeds (`initializer_range`, `pretraining_tp`, ...); if a transformers release changes one of
# those, extend this set rather than changing the converter.
_METADATA_KEYS = frozenset({"transformers_version", "architectures", "dtype", "torch_dtype", "_name_or_path"})

# Mistral-Small-4-119B-2603 publishes `mlp_bias` in its text config, but Mistral4Config has no such
# attribute. Excluded by name rather than by a generic unknown-key rule, which would mask real gaps.
_STALE_PUBLISHED_KEYS = frozenset({"mlp_bias"})


def _fold_legacy_rope_theta(section: dict[str, Any]) -> dict[str, Any]:
    """Move a pre-`rope_parameters` top-level `rope_theta` into `rope_parameters`."""
    if "rope_theta" not in section:
        return section
    section = dict(section)
    rope_theta = section.pop("rope_theta")
    section["rope_parameters"] = {**section.get("rope_parameters", {}), "rope_theta": rope_theta}
    return section


def _fold_legacy_rope_type(rope_parameters: dict[str, Any]) -> dict[str, Any]:
    """Rewrite the deprecated `type` spelling of `rope_type`, which current configs no longer emit.

    See the back-compat handling in `modeling_rope_utils.py`. Published configs carry `type` either
    alone or alongside an identical `rope_type`; a disagreeing pair is left alone so it still fails.
    """
    if "type" not in rope_parameters:
        return rope_parameters
    rope_type = rope_parameters.get("rope_type", rope_parameters["type"])
    if rope_type != rope_parameters["type"]:
        return rope_parameters
    return {k: v for k, v in rope_parameters.items() if k != "type"} | {"rope_type": rope_type}


def _normalize_published_section(section: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a published config section into the schema current transformers emits."""
    section = _fold_legacy_rope_theta(section)
    normalized = {}
    for key, value in section.items():
        if key == "rope_parameters" and isinstance(value, dict):
            value = _fold_legacy_rope_type(value)
        elif isinstance(value, dict):
            value = _normalize_published_section(value)
        normalized[key] = value
    return normalized


def _collect_mismatches(
    expected: dict[str, Any],
    actual: dict[str, Any],
    path: str = "",
    excluded_keys: frozenset[str] = frozenset(),
) -> list[str]:
    """Report every published key that is missing from or differs in `actual`.

    Extra keys in `actual` are ignored: current transformers emits defaults that predate the
    published configs, and those are not conversion errors. `excluded_keys` adds per-case
    exclusions on top of the module-wide `_METADATA_KEYS` and `_STALE_PUBLISHED_KEYS`.
    """
    mismatches = []
    for key, expected_value in expected.items():
        if key in _METADATA_KEYS or key in _STALE_PUBLISHED_KEYS or key in excluded_keys:
            continue
        where = f"{path}.{key}" if path else key
        if key not in actual:
            mismatches.append(f"{where}: missing, expected {expected_value!r}")
            continue
        actual_value = actual[key]
        if key == "modules_to_not_convert" and isinstance(expected_value, list) and isinstance(actual_value, list):
            # Only membership matters: `Quantizer.get_modules_to_not_convert` deduplicates this list
            # before use. Ministral-3-3B-Instruct-2512 ships it concatenated with itself.
            if set(expected_value) != set(actual_value):
                mismatches.append(
                    f"{where}: expected {sorted(set(expected_value))!r}, got {sorted(set(actual_value))!r}"
                )
        elif isinstance(expected_value, dict) and isinstance(actual_value, dict):
            mismatches += _collect_mismatches(expected_value, actual_value, where, excluded_keys=excluded_keys)
        elif expected_value != actual_value:
            mismatches.append(f"{where}: expected {expected_value!r}, got {actual_value!r}")
    return mismatches


class TestNativeToHFPublishedConfigs(unittest.TestCase):
    """Published params.json files must reproduce the config.json published alongside them."""

    _documents: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}

    @classmethod
    def _get_documents(cls, repo: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Download and cache (params.json, config.json) for a repo."""
        if repo not in cls._documents:
            with open(hf_hub_download(repo, "params.json")) as f:
                params = json.load(f)
            with open(hf_hub_download(repo, "config.json")) as f:
                config = json.load(f)
            cls._documents[repo] = (params, config)
        params, config = cls._documents[repo]
        return copy.deepcopy(params), copy.deepcopy(config)

    @parameterized.expand(_CONFIG_PARITY_CASES)
    @slow
    def test_reproduces_published_config(self, name: str, repo: str, section: str | None) -> None:
        params, published = self._get_documents(repo)
        if section is not None:
            params.pop("vision_encoder", None)

        actual = json.loads(
            mistral_native_config_to_hf_config(mistral_native_config_from_params(params)).to_json_string()
        )
        expected = _normalize_published_section(published[section] if section else published)

        mismatches = _collect_mismatches(expected, actual, excluded_keys=_KNOWN_DEVIATIONS.get(name, frozenset()))
        # Keys the published document omits entirely, so the pass above never reaches them, but
        # that the converter still computes and must get right (see `_ADDITIONAL_EXPECTATIONS`).
        mismatches += _collect_mismatches(_ADDITIONAL_EXPECTATIONS.get(name, {}), actual)
        self.assertEqual(mismatches, [], "\n".join([f"{repo} does not round-trip:", *mismatches]))

    def test_mlp_bias_is_still_absent(self) -> None:
        """`mlp_bias` may only stay in `_STALE_PUBLISHED_KEYS` while Mistral4Config lacks it."""
        self.assertFalse(hasattr(Mistral4Config(), "mlp_bias"))


def _native_config_to_comparable_dict(value: Any) -> Any:
    """Render a `MistralNativeConfig` (or one of its section dataclasses) as a `params.json`-
    shaped structure.

    Native section dataclasses (`YarnArgs`, `Llama4Scaling`, `QuantizationArgs`, `MOEModelArgs`,
    `VisionEncoderArgs`) use the same field names as their `params.json` counterparts one-for-one,
    so this is a direct render rather than a schema translation.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _native_config_to_comparable_dict(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    return value


# Native fields no published `config.json` can recover: a published config never carries
# `mistral_extras` (it is a transformers-internal attribute, never published), so the reverse
# converter falls back to these fields' native dataclass defaults regardless of the repo's real
# native value. Mistral-Medium-3.5-128B already demonstrates the resulting silent mismatch for the
# image token ids: its own `params.json` sets both to -1, but with no `mistral_extras` to recover
# that from, reversing its published `config.json` yields the default 12/13 instead.
_REVERSE_UNRECOVERABLE_FIELDS = frozenset(
    {
        "expert_parallel",
        "expert_model_parallel",
        "route_every_n",
        "max_image_size",
        "image_break_token_id",
        "image_end_token_id",
    }
)


class TestReverseFromPublishedConfig(unittest.TestCase):
    """A published `config.json`, converted back to native, must reproduce that repo's own
    `params.json` on the subset of fields a published config can possibly recover.

    Not full equality: no published config carries `mistral_extras`, so every field in
    `_REVERSE_UNRECOVERABLE_FIELDS` is unrecoverable by construction (see the comment there).
    Restricted to the `_CONFIG_PARITY_CASES` entries with `section=None`: the sectioned
    (text-config-only) cases have no published `config.json` of their own to load as a
    standalone HF config.
    """

    @parameterized.expand([(name, repo) for name, repo, section in _CONFIG_PARITY_CASES if section is None])
    @slow
    def test_reproduces_published_params(self, name: str, repo: str) -> None:
        params, _ = TestNativeToHFPublishedConfigs._get_documents(repo)
        hf_config = _CONFIG_CLASSES[name].from_pretrained(repo)

        native = mistral_native_config_from_hf_config(hf_config)
        actual = _native_config_to_comparable_dict(native)

        mismatches = _collect_mismatches(params, actual, excluded_keys=_REVERSE_UNRECOVERABLE_FIELDS)
        self.assertEqual(mismatches, [], "\n".join([f"{repo} does not reverse-convert:", *mismatches]))


if __name__ == "__main__":
    unittest.main()
