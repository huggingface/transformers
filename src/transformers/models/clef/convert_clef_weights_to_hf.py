# Copyright 2026 The HuggingFace Team. All rights reserved.
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
Converts a folder that ships a backbone with a decision head in its own format to a native Clef checkpoint (`config.json`
of `model_type: clef`, its weights with the head in float32, its tokenizer, `decision.json` and its template):

- a clef checkpoint or a folder trained from one: the weights of the backbone, `joint_head.safetensors` and
  `joint_head_config.json` (the joint head, or the cosine head when its `head_type` is `cosine_span_v1`), with an optional
  `unsloth_decision_config.json` whose temperatures (noul, choice, score) become those of the decision config;
- a remote-code package around a native model, such as a reranker, whose weights are those of that model under a single
  prefix next to a task head: the prefix is stripped, the task head dropped, and a new head of `head_type` initialized.

The backbone is the bare model of the `model_type` of the `config.json` of the folder, loaded strictly. No code of the
folder is ever imported.

Example:
    python src/transformers/models/clef/convert_clef_weights_to_hf.py --source path/to/checkpoint --output path/to/clef
    python src/transformers/models/clef/convert_clef_weights_to_hf.py --source ericlewis/ultrabert-150m-reranker \
        --output path/to/clef --head_type cosine
"""

import argparse
import json
import tempfile
from contextlib import ExitStack
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

from transformers import AutoConfig, AutoModel, AutoTokenizer, ClefConfig, ClefForDecision
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_FOR_MASKED_LM_MAPPING_NAMES
from transformers.utils import logging


logger = logging.get_logger(__name__)

HEAD_WEIGHTS, HEAD_CONFIG, CALIBRATION = (
    "joint_head.safetensors",
    "joint_head_config.json",
    "unsloth_decision_config.json",
)
COSINE_HEAD_TYPE = "cosine_span_v1"
QUESTION_TYPES = ("noul", "choice", "score")  # the order of the temperatures of the calibration file
# Settings of the config of the folder that describe its remote or task classes, not the backbone
TASK_CONFIG_KEYS = ("auto_map", "architectures", "id2label", "label2id", "num_labels", "problem_type")
SAFETENSORS_DTYPES = {"F64": torch.float64, "F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}
JOINT_SIZES = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_evidence_layers", "num_attention_heads")
COSINE_SIZES = ("cosine_width", "cosine_initial_scale")


def source_folder(source: str | Path) -> Path:
    """A local folder, or the snapshot of a repo of the Hub without its code nor its pickled files."""
    if Path(source).is_dir():
        return Path(source)
    return Path(snapshot_download(str(source), ignore_patterns=["*.py", "*.bin", "*.pt", "*.pth", "*.pkl", "*.ckpt"]))


def read_head_config(folder: Path) -> dict | None:
    """The config of the decision head of the folder, `None` without a head; raises when one of its files is missing."""
    present = [(folder / name).is_file() for name in (HEAD_WEIGHTS, HEAD_CONFIG)]
    if any(present) and not all(present):
        found, missing = (HEAD_WEIGHTS, HEAD_CONFIG) if present[0] else (HEAD_CONFIG, HEAD_WEIGHTS)
        raise ValueError(f"{folder} has {found} but not {missing}: its decision head is incomplete.")
    return json.loads((folder / HEAD_CONFIG).read_text(encoding="utf-8")) if all(present) else None


def weight_files(folder: Path) -> list[Path]:
    """The safetensors files of the weights of the backbone of the folder."""
    index = folder / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        return [folder / name for name in sorted(set(weight_map.values()))]
    if (folder / "model.safetensors").is_file():
        return [folder / "model.safetensors"]
    raise ValueError(f"{folder} has no `model.safetensors` (nor index): only safetensors weights are converted.")


def open_weights(files: list[Path], stack: ExitStack) -> dict:
    """The open safetensors file of each weight of `files`, by name: the weights are read when used."""
    weights = {}
    for file in files:
        handle = stack.enter_context(safe_open(file, framework="pt"))
        weights.update(dict.fromkeys(handle.keys(), handle))
    return weights


def backbone_prefix(keys: set[str], expected: set[str], model_name: str) -> str:
    """
    The prefix of the weights of the backbone among `keys`: none, the one of its class with a head (`model.`, `bert.`),
    or the one of a remote wrapper (`backbone.`). Exactly one must hold every weight of the backbone (`expected`), and
    a prefix must hold nothing else.
    """

    def under(prefix):
        return {key[len(prefix) :] for key in keys if key.startswith(prefix)}

    candidates = [""] + sorted({key.split(".", 1)[0] + "." for key in keys if "." in key})
    matches = [prefix for prefix in candidates if expected <= under(prefix)]
    if len(matches) > 1:
        raise ValueError(
            f"The weights of {model_name} are ambiguous: they are complete under each of the prefixes {matches}."
        )
    if not matches:
        closest = max(candidates, key=lambda prefix: len(expected & under(prefix)))
        missing = sorted(expected - under(closest))
        raise ValueError(
            f"The weights of {model_name} are incomplete: under the prefix {closest!r}, {len(missing)} of its weights "
            f"are missing, such as {missing[:5]}."
        )
    prefix = matches[0]
    unexpected = sorted(under(prefix) - expected) if prefix else []
    if unexpected:
        raise ValueError(
            f"The weights under the prefix {prefix!r} are not those of {model_name}: it has no {unexpected[:5]}."
        )
    return prefix


def calibrated_decision_config(folder: Path, prompt_style: str):
    """The decision config of Clef in `prompt_style`, with the temperatures and `max_len` of the calibration file."""
    decision_config, max_len = ClefConfig.prompt_decision_config(prompt_style), None
    if (folder / CALIBRATION).is_file():
        calibration = json.loads((folder / CALIBRATION).read_text(encoding="utf-8"))
        if calibration.get("layout", "clef") != "clef":
            raise ValueError(f"{folder / CALIBRATION} has the layout {calibration['layout']!r}, not clef's.")
        if (temperature := calibration.get("temperature")) is not None:
            if len(temperature) != len(QUESTION_TYPES):
                raise ValueError(f"The temperatures of {folder / CALIBRATION} are not those of {QUESTION_TYPES}.")
            scale = float(calibration.get("head_temperature", 1.0))
            decision_config.temperature = {
                kind: float(value) * scale for kind, value in zip(QUESTION_TYPES, temperature)
            }
        max_len = calibration.get("max_len")
    return decision_config, max_len


def clef_config(
    folder: Path,
    text_config: dict,
    head_config: dict | None,
    head_type: str | None,
    global_pooling: str | None,
    prompt_style: str | None,
    sizes: dict,
) -> ClefConfig:
    """The config of the converted model: the head of the folder, or a new one of `head_type` and `sizes`."""
    bidirectional = text_config["model_type"] in MODEL_FOR_MASKED_LM_MAPPING_NAMES
    kwargs = {
        "global_pooling": global_pooling or ("first" if bidirectional else "last"),
        "prompt_style": prompt_style or ("plain" if bidirectional else "chatml"),
    }
    if head_config is None:
        kwargs.update(head_type=head_type or "joint", backbone_hidden_size=text_config["hidden_size"], **sizes)
    else:
        if sizes:
            raise ValueError(f"The sizes of the head come from {folder / HEAD_CONFIG}, got {sorted(sizes)}.")
        declared = head_config.get("head_type")  # none for the joint head of clef
        if declared not in (None, COSINE_HEAD_TYPE):
            raise ValueError(f"{folder / HEAD_CONFIG} has an unknown head type {declared!r}.")
        source_type = "cosine" if declared == COSINE_HEAD_TYPE else "joint"
        if head_type not in (None, source_type):
            raise ValueError(f"{folder} ships a {source_type} head, not a {head_type} one.")
        if head_config["hidden_size"] != text_config["hidden_size"]:
            raise ValueError(
                f"The head of {folder} reads states of size {head_config['hidden_size']}, its backbone gives "
                f"{text_config['hidden_size']}."
            )
        if source_type == "cosine":
            kwargs.update(
                head_type="cosine",
                backbone_hidden_size=head_config["hidden_size"],
                cosine_width=head_config["width"],
                cosine_initial_scale=head_config["initial_scale"],
            )
        else:
            kwargs.update(head_type="joint", joint_head_config=head_config)
    return ClefConfig(text_config=text_config, **kwargs)


def convert_clef_checkpoint(
    source: str | Path,
    output: str | Path,
    *,
    safe_serialization: bool = True,
    head_type: str | None = None,
    global_pooling: str | None = None,
    prompt_style: str | None = None,
    seed: int = 0,
    **head_sizes,
) -> None:
    """
    Converts the folder (or repo of the Hub) `source` to a native Clef checkpoint in `output`.

    Args:
        source (`str` or `Path`): the folder of the checkpoint, see the documentation of this module.
        output (`str` or `Path`): the folder of the converted checkpoint.
        safe_serialization (`bool`, defaults to `True`): Transformers saves safetensors only, `False` is refused.
        head_type (`str`, *optional*): `"joint"` (the default) or `"cosine"`, the head of a folder without one. A folder
            with a head keeps it, and refuses another type.
        global_pooling, prompt_style (`str`, *optional*): see [`ClefConfig`]. By default `"first"` and `"plain"` for a
            bidirectional backbone (one of a masked language model), `"last"` and `"chatml"` otherwise.
        seed (`int`, defaults to 0): the seed of the initialization of a new head.
        head_sizes: the sizes of a new head (`hidden_size`, `intermediate_size`, `num_hidden_layers`,
            `num_evidence_layers` and `num_attention_heads` of a joint head, `cosine_width` and `cosine_initial_scale`
            of a cosine head). A folder with a head takes its sizes from its `joint_head_config.json`.
    """
    if not safe_serialization:
        raise ValueError("Transformers saves safetensors weights only: `safe_serialization=False` is not supported.")
    if head_type not in (None, "joint", "cosine"):
        raise ValueError(f"`head_type` must be 'joint' or 'cosine', got {head_type!r}.")
    unknown = set(head_sizes) - set(JOINT_SIZES) - set(COSINE_SIZES)
    if unknown:
        raise TypeError(f"Unknown sizes of the head: {sorted(unknown)}.")
    sizes = {key: value for key, value in head_sizes.items() if value is not None}
    folder, output = source_folder(source), Path(output)
    head_config = read_head_config(folder)

    text_config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    text_config = {key: value for key, value in text_config.items() if key not in TASK_CONFIG_KEYS}
    config = clef_config(folder, text_config, head_config, head_type, global_pooling, prompt_style, sizes)
    decision_config, max_len = calibrated_decision_config(folder, config.prompt_style)

    with ExitStack() as stack:
        weights = open_weights(weight_files(folder), stack)
        # The weights the bare model of the backbone expects, without allocating them
        with torch.device("meta"):
            backbone = AutoModel.from_config(AutoConfig.for_model(**text_config))
        expected = set(backbone.state_dict())
        prefix = backbone_prefix(set(weights), expected, type(backbone).__name__)
        backbone_keys = [key for key in weights if key.startswith(prefix) and key[len(prefix) :] in expected]
        backbone_dtype = SAFETENSORS_DTYPES.get(
            weights[backbone_keys[0]].get_slice(backbone_keys[0]).get_dtype(), torch.float32
        )
        state = {f"model.{key[len(prefix) :]}": (weights[key], key) for key in backbone_keys}
        others = sorted(set(weights) - set(backbone_keys))

        # The joint head reads the language modeling head of a causal backbone, unless it is tied to its embeddings
        reads_lm_head = config.head_type == "joint" and text_config["model_type"] in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        if reads_lm_head and not config.tie_word_embeddings:
            if "lm_head.weight" not in weights:
                raise ValueError(f"The joint head reads the language modeling head, which {folder} does not have.")
            state["lm_head.weight"] = (weights["lm_head.weight"], "lm_head.weight")
            others.remove("lm_head.weight")
        if others:
            logger.warning(f"Dropped the weights of {folder} that the decision model does not read: {others}.")

        if head_config is not None:
            head = open_weights([folder / HEAD_WEIGHTS], stack)
            # The cosine head keeps its names; the joint head is renamed by the conversion mapping of clef
            state.update(
                {(f"cosine_head.{k}" if config.head_type == "cosine" else k): (v, k) for k, v in head.items()}
            )

        # Staged as a checkpoint of Clef, which `from_pretrained` loads (renaming the joint head) and checks
        staging = stack.enter_context(tempfile.TemporaryDirectory())
        config.save_pretrained(staging)
        save_file(
            {key: handle.get_tensor(name) for key, (handle, name) in state.items()},
            f"{staging}/model.safetensors",
            {"format": "pt"},
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model, info = ClefForDecision.from_pretrained(staging, dtype=torch.float32, output_loading_info=True)
    new_head = {key for key in model.state_dict() if not key.startswith(("model.", "lm_head."))}
    allowed_missing = new_head if head_config is None else set()
    problems = {
        "missing": set(info["missing_keys"]) - allowed_missing,
        "unexpected": set(info["unexpected_keys"]),
        "mismatched": set(info["mismatched_keys"]),
        "errors": list(info["error_msgs"]),
    }
    if any(problems.values()):
        raise ValueError(f"The weights of {folder} do not load into {type(model).__name__}: {problems}.")

    # The backbone in the precision of the source, the head in float32
    model.model.to(backbone_dtype)
    model.config.text_config.dtype = backbone_dtype
    if getattr(model, "lm_head", None) is not None:
        model.lm_head.to(backbone_dtype)
    model.decision_config = decision_config
    tokenizer = AutoTokenizer.from_pretrained(folder, trust_remote_code=False)
    if max_len is not None:
        tokenizer.model_max_length = int(max_len)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    logger.info(f"Converted {source} to a Clef checkpoint with a {config.head_type} head in {output}.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="The folder (or repo of the Hub) of the checkpoint.")
    parser.add_argument("--output", required=True, help="The folder of the converted checkpoint.")
    parser.add_argument("--head_type", choices=["joint", "cosine"], help="The head of a folder without one.")
    parser.add_argument("--global_pooling", choices=["last", "first"])
    parser.add_argument("--prompt_style", choices=["chatml", "plain"])
    parser.add_argument("--seed", type=int, default=0, help="The seed of the initialization of a new head.")
    for size in JOINT_SIZES + COSINE_SIZES:
        parser.add_argument(f"--{size}", type=float if size == "cosine_initial_scale" else int)
    args = vars(parser.parse_args(argv))
    convert_clef_checkpoint(args.pop("source"), args.pop("output"), **args)


if __name__ == "__main__":
    main()
