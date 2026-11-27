# Copyright 2022 The HuggingFace Inc. team.
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

"""
Checks that all model configs have defaults that match the checkpoint declared in their docstring.

For every model_type in CONFIG_MAPPING:
  1. find the declared checkpoint unless it it skipped via `CONFIG_CLASSES_TO_IGNORE_FOR_DOCSTRING_CHECKPOINT_CHECK`)
  2. build the default `ModelConfigClass()` with no kwargs
  3. download the `config.json` of the declared checkpoint
  4. compare every key present in JSON against the default config dict, recursing into
    `sub_configs` (text_config, vision_config, ...) using the *sub-config's own* defaults
"""

import argparse
import datetime
import inspect
import json
import math
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from enum import Enum

from check_config_docstrings import CONFIG_CLASSES_TO_IGNORE_FOR_DOCSTRING_CHECKPOINT_CHECK, _re_checkpoint
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, GatedRepoError, RepositoryNotFoundError

from transformers.models.auto.configuration_auto import CONFIG_MAPPING, CONFIG_MAPPING_NAMES


CHECKER_CONFIG = {
    "name": "config_defaults",
    "label": "Check config defaults",
    "cache_globs": [],
    "check_args": [],
    "fix_args": None,
}

IGNORE_KEYS = {
    "model_type",
    "architectures",
    "transformers_version",
    "_name_or_path",
    "_commit_hash",
    "auto_map",
    "torch_dtype",
    "dtype",
    "id2label",
    "label2id",
    "problem_type",
}

ROOT = os.getcwd().split("utils")[0]
DOCS_PATH = os.path.join(ROOT, "docs/source/en/model_doc")
CUTOFF = datetime.date(2026, 11, 15)

MODEL_NAME_TO_DOC_MAPPING = {
    "dinov3_convnext": "dinov3",
    "dinov3_vit": "dinov3",
    "kosmos2": "kosmos-2",
    "openai": "openai-gpt",
    "vitpose_backbone": "vitpose",
    "x_clip": "xclip",
    "timm_backbone": "timm_wrapper",
    # underscore -> hyphen in some models
    "audio_spectrogram_transformer": "audio-spectrogram-transformer",
    "bert_generation": "bert-generation",
    "blenderbot_small": "blenderbot-small",
    "blip_2": "blip-2",
    "dab_detr": "dab-detr",
    "deberta_v2": "deberta-v2",
    "encoder_decoder": "encoder-decoder",
    "grounding_dino": "grounding-dino",
    "megatron_bert": "megatron-bert",
    "mgp_str": "mgp-str",
    "mm_grounding_dino": "mm-grounding-dino",
    "modernbert_decoder": "modernbert-decoder",
    "nllb_moe": "nllb-moe",
    "omdet_turbo": "omdet-turbo",
    "roberta_prelayernorm": "roberta-prelayernorm",
    "sew_d": "sew-d",
    "speech_encoder_decoder": "speech-encoder-decoder",
    "table_transformer": "table-transformer",
    "unispeech_sat": "unispeech-sat",
    "vision_encoder_decoder": "vision-encoder-decoder",
    "vision_text_dual_encoder": "vision-text-dual-encoder",
    "wav2vec2_bert": "wav2vec2-bert",
    "wav2vec2_conformer": "wav2vec2-conformer",
    "xlm_roberta": "xlm-roberta",
    "xlm_roberta_xl": "xlm-roberta-xl",
}

_re_contribution_date = "contributed to Hugging Face Transformers on "
_re_date = re.compile(re.escape(_re_contribution_date) + r"(\d{4}-\d{2}-\d{2})")


class CheckStatus(str, Enum):
    NO_CHECKPOINT = "NO_CHECKPOINT"
    SKIPPED_LEGACY_MODEL = "SKIPPED_LEGACY_MODEL"
    DEFAULT_INIT_ERROR = "DEFAULT_INIT_ERROR"
    GATED_REPO = "GATED_REPO"
    CONFIG_NOT_FOUND = "CONFIG_NOT_FOUND"
    FETCH_ERROR = "FETCH_ERROR"
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"


@dataclass
class ConfigCheckResult:
    model_type: str
    checkpoint: str | None = None
    status: CheckStatus = CheckStatus.NO_CHECKPOINT
    diffs: list = field(default_factory=list)
    detail: str = ""


FAILING_STATUSES = (CheckStatus.MISMATCH, CheckStatus.NO_CHECKPOINT, CheckStatus.DEFAULT_INIT_ERROR)


def normalize(value):
    "Normalize a value by converting all tuples into list and making sure dict keys are all `str`"
    if isinstance(value, (list, tuple)):
        return [normalize(x) for x in value]
    if isinstance(value, dict):
        return {str(k): normalize(x) for k, x in value.items()}
    return value


def is_field_equal(a, b):
    "Check whether two values are equal for given config field"
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(is_field_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(is_field_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return a == b


def find_config_diffs(default_cls, loaded_json, path=""):
    """List of (key_path, default_value, hub_value) for keys on the hub that differ from the default."""
    diffs = []
    default_dict = default_cls.to_dict()
    sub_configs = getattr(type(default_cls), "sub_configs")

    for key, hub_value in loaded_json.items():
        if key in IGNORE_KEYS or key.startswith("_"):
            continue

        # If subconfig, recurse over its own `init_kwargs` from SubConfigSpec
        if key in sub_configs:
            sub_configs_default = getattr(default_cls, key, None)
            if sub_configs_default is None:
                if hub_value is not None:
                    diffs.append(
                        {
                            "field_name": path + key,
                            "hub_value": hub_value,
                            "code_value": "<sub_configs_default> is `None`",
                        }
                    )
                continue

            if isinstance(hub_value, dict):
                hub_mt = hub_value.get("model_type")
                if hub_mt is not None and hub_mt != sub_configs_default.model_type:
                    diffs.append(
                        {
                            "field_name": f"{path}{key}.model_type",
                            "hub_value": hub_mt,
                            "code_value": sub_configs_default.model_type,
                        }
                    )
                diffs += find_config_diffs(sub_configs_default, hub_value, f"{path}{key}.")
            continue

        if key not in default_dict:
            # extra key on the hub, not a default-drift problem
            # common-case: legacy keys that get converted in post-init, can't check it
            continue

        if not is_field_equal(normalize(default_dict[key]), normalize(hub_value)):
            diffs.append(
                {
                    "field_name": path + key,
                    "hub_value": hub_value,
                    "code_value": default_dict[key],
                }
            )
    return diffs


def is_model_old(config_cls):
    model_dir = os.path.dirname(inspect.getsourcefile(config_cls))
    model_name = os.path.basename(model_dir)  # e.g. "bert/llama"
    model_name = MODEL_NAME_TO_DOC_MAPPING.get(model_name, model_name)

    # timm is the only exception with no addition date as it is not a real model
    if model_name == "timm_wrapper":
        return True

    model_card = os.path.join(DOCS_PATH, f"{model_name}.md")
    if not os.path.isfile(model_card):
        raise ValueError(f"No model card found for {model_card}")

    with open(model_card, encoding="utf-8") as f:
        match = _re_date.search(f.read())

    if match is None:
        raise ValueError(f"No contribution date found in {model_name}")

    date = datetime.date.fromisoformat(match.group(1))
    return date < CUTOFF


def prepare_config_check(model_type: str) -> tuple[ConfigCheckResult, object | None]:
    """Find the checkpoint declared in the config docstring and instantiate the default config.

    Returns the result record plus the default config instance (None if the model is
    skipped or the default config could not be built).
    """
    result = ConfigCheckResult(model_type=model_type)
    config_class = CONFIG_MAPPING[model_type]

    checkpoint_match = _re_checkpoint.search(inspect.getsource(config_class))
    if checkpoint_match:
        result.checkpoint = checkpoint_match.group(1)

    # Legacy configs can't be changed because of backward compatibility, so skip them.
    is_legacy = is_model_old(config_class)
    is_ignored = config_class.__name__ in CONFIG_CLASSES_TO_IGNORE_FOR_DOCSTRING_CHECKPOINT_CHECK
    if is_legacy or is_ignored:
        result.status = CheckStatus.SKIPPED_LEGACY_MODEL
        return result, None

    try:
        default_config = config_class()
    except Exception as error:
        result.status = CheckStatus.DEFAULT_INIT_ERROR
        result.detail = repr(error)[:200]
        return result, None

    return result, default_config


def compare_defaults(result: ConfigCheckResult, default_config) -> ConfigCheckResult:
    """Compare the defaults of the config class in transformers with the config.json on the Hub."""
    if result.checkpoint is None:
        result.status = CheckStatus.NO_CHECKPOINT
        return result

    try:
        # Download the raw file instead of using `from_pretrained`, because `from_pretrained`
        # can post-process the downloaded json and hide differences.
        config_path = hf_hub_download(result.checkpoint, "config.json")
        with open(config_path, encoding="utf8") as f:
            hub_config = json.load(f)
    except GatedRepoError:  # must come before RepositoryNotFoundError, it's a subclass
        result.status = CheckStatus.GATED_REPO
        return result
    except (RepositoryNotFoundError, EntryNotFoundError):
        result.status = CheckStatus.CONFIG_NOT_FOUND
        return result
    except Exception as error:
        result.status = CheckStatus.FETCH_ERROR
        result.detail = repr(error)[:200]
        return result

    result.diffs = find_config_diffs(default_config, hub_config)
    result.status = CheckStatus.MISMATCH if result.diffs else CheckStatus.MATCH
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-to-check", nargs="*", help="Only check these model types.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--save-json", default="output.json", help="Write full report here.")
    parser.add_argument("--verbose", action="store_true", help="Print every diff for each mismatching model.")
    return parser.parse_args()


def run_checks(model_types, num_workers):
    """Prepare every model, then compare the ones that have a default config against the Hub."""
    skipped_results = []  # models that never reach the comparison
    pending_results = []
    pending_default_configs = []

    for model_type in model_types:
        result, default_config = prepare_config_check(model_type)
        if default_config is None:
            skipped_results.append(result)
        else:
            pending_results.append(result)
            pending_default_configs.append(default_config)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        compared_results = list(executor.map(compare_defaults, pending_results, pending_default_configs))

    return sorted(skipped_results + compared_results, key=lambda r: r.model_type)


def print_summary(all_results):
    status_counts = Counter(result.status for result in all_results)
    num_match = status_counts[CheckStatus.MATCH]
    num_comparable = num_match + status_counts[CheckStatus.MISMATCH]

    print("\n================ SUMMARY ================")
    if num_comparable:
        print(f"\nmatch rate (of {num_comparable} comparable): {num_match / num_comparable:.1%}")
    else:
        print("\nno comparable configs")
    for status, count in sorted(status_counts.items()):
        print(f"{status.value:25s} {count}")


def save_report(all_results, path):
    with open(path, "w", encoding="utf8") as f:
        json.dump([asdict(result) for result in all_results], f, indent=2, default=str)
    print(f"report written to {path}")


def print_verbose_diffs(mismatched_results):
    for result in mismatched_results:
        print(f"\n{result.model_type} ({result.checkpoint})")
        for diff in result.diffs:
            print(f"    {diff}")
    if mismatched_results:
        raise Exception(
            f"Fix these {len(mismatched_results)} config(s) with defaults that differ from their Hub checkpoint."
        )


def build_failure_error(failed_results):
    sections = []

    mismatched = [r for r in failed_results if r.status == CheckStatus.MISMATCH]
    if mismatched:
        lines = [f"{len(mismatched)} config(s) have defaults that differ from their Hub checkpoint:"]
        for result in mismatched:
            lines.append(f"\n- {result.model_type} (checkpoint: {result.checkpoint})")
        lines.append(
            "\nHow to fix: make the defaults in the config class `__init__` match the `config.json` of the "
            "checkpoint linked in the class docstring, or link a checkpoint that matches the defaults. "
            "If changing a default would break backward compatibility, update the docstring link instead."
        )
        sections.append("\n".join(lines))

    no_checkpoint = [r for r in failed_results if r.status == CheckStatus.NO_CHECKPOINT]
    if no_checkpoint:
        names = "\n".join(f"- {r.model_type}" for r in no_checkpoint)
        sections.append(
            f"{len(no_checkpoint)} config(s) have no valid checkpoint link in their docstring:\n{names}\n\n"
            "How to fix: add a link like [myorg/mymodel](https://huggingface.co/myorg/mymodel) to the config "
            "class docstring. The link text must match the URL. See `utils/check_config_docstrings.py`."
        )

    init_errors = [r for r in failed_results if r.status == CheckStatus.DEFAULT_INIT_ERROR]
    if init_errors:
        lines = [f"{len(init_errors)} config(s) can't be instantiated with default arguments:"]
        for result in init_errors:
            lines.append(f"\n- {result.model_type}: {result.detail}")
        lines.append("\nHow to fix: every argument of the config `__init__` needs a default value.")
        sections.append("\n".join(lines))

    model_list = " ".join(r.model_type for r in failed_results)
    sections.append(
        "For full details on the failing models (including values), run:\n"
        f"    python utils/check_config_defaults.py --models-to-check {model_list} --verbose"
    )
    return ValueError("\n\n".join(sections))


def check_config_defaults_match_checkpoint():
    args = parse_args()

    model_types = sorted(CONFIG_MAPPING_NAMES)
    if args.models_to_check:
        requested = set(args.models_to_check)
        model_types = [m for m in model_types if m in requested]

    all_results = run_checks(model_types, args.workers)

    print_summary(all_results)
    if args.save_json:
        save_report(all_results, args.save_json)

    failed_results = [r for r in all_results if r.status in FAILING_STATUSES]
    if args.verbose:
        print_verbose_diffs([r for r in failed_results if r.status == CheckStatus.MISMATCH])

    elif failed_results:
        raise build_failure_error(failed_results)


if __name__ == "__main__":
    check_config_defaults_match_checkpoint()
