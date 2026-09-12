# Copyright 2026 The HuggingFace Team. All rights reserved.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

"""Convert an original Chronos-2 checkpoint to the native Transformers configuration.

The released ``amazon/chronos-2`` checkpoint already uses the tensor names expected by
the native implementation. Conversion is therefore deliberately an identity operation
for tensors: this script migrates the legacy ``model_type='t5'`` configuration to
``Chronos2Config`` and proves that every tensor key, shape, dtype, and value is preserved.

Example:

```bash
python src/transformers/models/chronos2/convert_chronos2_original_to_hf.py \
    --checkpoint amazon/chronos-2 \
    --revision 29ec3766d36d6f73f0696f85560a422f50e8498c \
    --output_path ./chronos-2-transformers
```

Pass ``--verify-original`` to additionally compare a deterministic grouped forecast
against the implementation from the optional ``chronos-forecasting`` package. This
script has no upload option and refuses to overwrite an existing output path.
"""

import argparse
import json
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from transformers import AutoConfig, AutoModel, AutoModelForTimeSeriesPrediction, Chronos2Config, Chronos2Model
from transformers.modeling_utils import load_state_dict
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME


DEFAULT_CHECKPOINT = "amazon/chronos-2"
DEFAULT_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
REQUIRED_CHRONOS_CONFIG_FIELDS = {
    "context_length",
    "input_patch_size",
    "input_patch_stride",
    "output_patch_size",
    "quantiles",
}


def _resolve_checkpoint(
    checkpoint: str,
    revision: str | None,
    cache_dir: str | None,
    token: str | None,
) -> Path:
    """Resolve a local directory or download the required files from the Hub."""
    local_path = Path(checkpoint).expanduser()
    if local_path.exists():
        if not local_path.is_dir():
            raise ValueError(f"A local checkpoint must be a directory, but received: {local_path}")
        return local_path.resolve()

    resolved_revision = revision or (DEFAULT_REVISION if checkpoint == DEFAULT_CHECKPOINT else None)
    snapshot_path = snapshot_download(
        repo_id=checkpoint,
        revision=resolved_revision,
        cache_dir=cache_dir,
        token=token,
        allow_patterns=[
            "config.json",
            "*.safetensors",
            "*.safetensors.index.json",
            "pytorch_model*.bin",
            "pytorch_model.bin.index.json",
        ],
    )
    return Path(snapshot_path).resolve()


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _migrate_config(checkpoint_dir: Path) -> Chronos2Config:
    """Validate the legacy Chronos-2 metadata and construct a native config."""
    config_path = checkpoint_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Chronos-2 config not found: {config_path}")

    legacy_config = _read_json(config_path)
    legacy_model_type = legacy_config.get("model_type")
    if legacy_model_type not in {"t5", "chronos2"}:
        raise ValueError(
            "Expected a released Chronos-2 config with `model_type='t5'` or an already converted "
            f"config with `model_type='chronos2'`, but found {legacy_model_type!r}."
        )

    architectures = legacy_config.get("architectures")
    if architectures != ["Chronos2Model"]:
        raise ValueError(
            "The checkpoint does not identify the expected Chronos-2 architecture: "
            f"`architectures` is {architectures!r}."
        )

    chronos_config = legacy_config.get("chronos_config")
    if not isinstance(chronos_config, Mapping):
        raise ValueError("A Chronos-2 checkpoint must contain a mapping-valued `chronos_config`.")
    missing_fields = REQUIRED_CHRONOS_CONFIG_FIELDS.difference(chronos_config)
    if missing_fields:
        raise ValueError(f"`chronos_config` is missing required fields: {sorted(missing_fields)}")

    config_kwargs = dict(legacy_config)
    config_kwargs.pop("model_type", None)
    config_kwargs.pop("transformers_version", None)
    config_kwargs.pop("_name_or_path", None)
    config_kwargs.pop("dense_act_fn", None)
    config_kwargs.pop("is_gated_act", None)
    # This points to the external chronos-forecasting pipeline and is not part of
    # the prepared-tensor native Transformers API.
    config_kwargs.pop("chronos_pipeline_class", None)
    config_kwargs["architectures"] = ["Chronos2Model"]

    config = Chronos2Config(**config_kwargs)
    if config.model_type != "chronos2":
        raise AssertionError(f"Config migration produced unexpected model type {config.model_type!r}.")
    return config


def _resolve_weight_files(checkpoint_dir: Path) -> list[Path]:
    """Return all checkpoint shards, preferring safetensors when both formats exist."""
    for index_name in (SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_INDEX_NAME):
        index_path = checkpoint_dir / index_name
        if not index_path.is_file():
            continue

        index = _read_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise ValueError(f"Invalid or empty `weight_map` in {index_path}.")

        checkpoint_root = checkpoint_dir.resolve()
        shard_paths = []
        for shard_name in sorted(set(weight_map.values())):
            shard_path = (checkpoint_dir / shard_name).resolve()
            if checkpoint_root != shard_path.parent and checkpoint_root not in shard_path.parents:
                raise ValueError(f"Checkpoint index points outside its directory: {shard_name}")
            if not shard_path.is_file():
                raise FileNotFoundError(f"Checkpoint shard listed in {index_path} was not found: {shard_path}")
            shard_paths.append(shard_path)
        return shard_paths

    for weights_name in (SAFE_WEIGHTS_NAME, WEIGHTS_NAME):
        weights_path = checkpoint_dir / weights_name
        if weights_path.is_file():
            return [weights_path.resolve()]

    raise FileNotFoundError(
        f"No {SAFE_WEIGHTS_NAME}, {SAFE_WEIGHTS_INDEX_NAME}, {WEIGHTS_NAME}, or {WEIGHTS_INDEX_NAME} "
        f"was found in {checkpoint_dir}."
    )


def _iter_checkpoint_tensors(weight_files: list[Path]) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield checkpoint tensors one shard at a time to keep peak memory bounded."""
    for weight_file in weight_files:
        if weight_file.suffix == ".safetensors":
            with safe_open(weight_file, framework="pt", device="cpu") as checkpoint:
                for key in checkpoint.keys():
                    yield key, checkpoint.get_tensor(key)
        else:
            state_dict = load_state_dict(weight_file, map_location="cpu")
            for key, tensor in state_dict.items():
                yield key, tensor
            del state_dict


def _assert_loading_info(loading_info: dict, source: str) -> None:
    failures = {
        field: loading_info.get(field, [])
        for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if loading_info.get(field)
    }
    if failures:
        raise RuntimeError(f"Non-strict load from {source}: {failures}")


def _assert_identity_checkpoint(model: Chronos2Model, weight_files: list[Path], source: str) -> int:
    """Assert exact tensor key, shape, dtype, and value identity against ``model``."""
    model_state = model.state_dict()
    seen_keys: set[str] = set()
    unexpected_keys: list[str] = []
    shape_mismatches: list[str] = []
    dtype_mismatches: list[str] = []
    value_mismatches: list[str] = []

    for key, source_tensor in _iter_checkpoint_tensors(weight_files):
        if key in seen_keys:
            raise ValueError(f"Duplicate tensor key across checkpoint shards: {key}")
        seen_keys.add(key)

        if key not in model_state:
            unexpected_keys.append(key)
            continue

        model_tensor = model_state[key].detach().cpu()
        if source_tensor.shape != model_tensor.shape:
            shape_mismatches.append(f"{key}: {tuple(source_tensor.shape)} != {tuple(model_tensor.shape)}")
            continue
        if source_tensor.dtype != model_tensor.dtype:
            dtype_mismatches.append(f"{key}: {source_tensor.dtype} != {model_tensor.dtype}")
            continue
        if not torch.equal(source_tensor, model_tensor):
            value_mismatches.append(key)

    missing_keys = sorted(set(model_state).difference(seen_keys))
    failures = {
        "missing_keys": missing_keys,
        "unexpected_keys": sorted(unexpected_keys),
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches": dtype_mismatches,
        "value_mismatches": value_mismatches,
    }
    failures = {name: values for name, values in failures.items() if values}
    if failures:
        raise RuntimeError(f"Identity audit failed for {source}: {failures}")

    return len(seen_keys)


def _load_strict(checkpoint_dir: Path, config: Chronos2Config) -> Chronos2Model:
    model, loading_info = Chronos2Model.from_pretrained(
        checkpoint_dir,
        config=config,
        dtype="auto",
        output_loading_info=True,
    )
    _assert_loading_info(loading_info, str(checkpoint_dir))
    model.eval()
    return model


def _verify_saved_config(output_dir: Path) -> None:
    saved_config = _read_json(output_dir / "config.json")
    if saved_config.get("model_type") != "chronos2":
        raise AssertionError("The converted config was not saved with `model_type='chronos2'`.")
    if saved_config.get("architectures") != ["Chronos2Model"]:
        raise AssertionError("The converted config did not preserve `architectures=['Chronos2Model']`.")

    auto_config = AutoConfig.from_pretrained(output_dir)
    if not isinstance(auto_config, Chronos2Config):
        raise AssertionError(
            f"AutoConfig resolved the converted checkpoint to {type(auto_config).__name__}, not Chronos2Config."
        )

    for auto_class in (AutoModel, AutoModelForTimeSeriesPrediction):
        auto_model, loading_info = auto_class.from_pretrained(
            output_dir,
            dtype="auto",
            trust_remote_code=False,
            output_loading_info=True,
        )
        _assert_loading_info(loading_info, f"{output_dir} through {auto_class.__name__}")
        if not isinstance(auto_model, Chronos2Model):
            raise AssertionError(
                f"{auto_class.__name__} resolved the converted checkpoint to {type(auto_model).__name__}, "
                "not Chronos2Model."
            )
        del auto_model


def _verify_original_parity(
    source_dir: Path,
    converted_dir: Path,
    config: Chronos2Config,
    atol: float,
    rtol: float,
) -> tuple[float, float]:
    """Compare a deterministic grouped forecast with the optional Amazon package."""
    try:
        from chronos import Chronos2Model as OriginalChronos2Model
    except ImportError as error:
        raise ImportError(
            "`--verify-original` requires the optional Amazon implementation. Install it with "
            "`pip install chronos-forecasting` and retry."
        ) from error

    original_model = OriginalChronos2Model.from_pretrained(source_dir, attn_implementation="eager").eval()
    native_model = Chronos2Model.from_pretrained(converted_dir, attn_implementation="eager").eval()

    forecast_config = config.chronos_config
    if not isinstance(forecast_config, Mapping):
        forecast_config = vars(forecast_config)
    input_patch_size = int(forecast_config["input_patch_size"])
    output_patch_size = int(forecast_config["output_patch_size"])
    num_output_patches = min(2, int(forecast_config.get("max_output_patches", 2)))
    context_length = 2 * input_patch_size
    future_length = num_output_patches * output_patch_size

    context_steps = torch.arange(context_length, dtype=torch.float32)
    context = torch.stack(
        [
            (context_steps - context_steps.mean()) / max(context_length, 1),
            torch.sin(context_steps / 3.0),
            torch.cos(context_steps / 5.0),
        ]
    )
    context[0, 1] = torch.nan
    group_ids = torch.tensor([0, 0, 1], dtype=torch.long)

    future_steps = torch.arange(future_length, dtype=torch.float32)
    future_covariates = torch.full((3, future_length), torch.nan, dtype=torch.float32)
    future_covariates[1] = torch.cos(future_steps / 4.0)
    future_target = torch.stack(
        [
            torch.sin(future_steps / 2.0),
            torch.cos(future_steps / 4.0),
            (future_steps + 1.0) / max(future_length, 1),
        ]
    )
    future_target[2, -1] = torch.nan

    forward_kwargs = {
        "context": context,
        "group_ids": group_ids,
        "future_covariates": future_covariates,
        "num_output_patches": num_output_patches,
        "future_target": future_target,
    }
    with torch.no_grad():
        original_output = original_model(**forward_kwargs)
        native_output = native_model(**forward_kwargs)

    torch.testing.assert_close(
        native_output.quantile_preds,
        original_output.quantile_preds,
        atol=atol,
        rtol=rtol,
        equal_nan=True,
    )
    torch.testing.assert_close(native_output.loss, original_output.loss, atol=atol, rtol=rtol, equal_nan=True)

    prediction_max_diff = (
        (native_output.quantile_preds.float() - original_output.quantile_preds.float()).abs().max().item()
    )
    loss_diff = (native_output.loss.float() - original_output.loss.float()).abs().item()
    return prediction_max_diff, loss_diff


def convert_checkpoint(
    checkpoint: str,
    output_path: str,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | None = None,
    verify_original: bool = False,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> None:
    source_dir = _resolve_checkpoint(checkpoint, revision=revision, cache_dir=cache_dir, token=token)
    output_dir = Path(output_path).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output path {output_dir}. Choose a new path or remove it explicitly."
        )
    if output_dir == source_dir:
        raise ValueError("The conversion output path must differ from the source checkpoint path.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    config = _migrate_config(source_dir)
    source_weight_files = _resolve_weight_files(source_dir)
    model = _load_strict(source_dir, config=config)
    source_key_count = _assert_identity_checkpoint(model, source_weight_files, source=str(source_dir))
    print(f"Strict source load and identity audit passed for {source_key_count} tensors.")

    with tempfile.TemporaryDirectory(prefix="chronos2-conversion-", dir=output_dir.parent) as temporary_dir:
        staging_dir = Path(temporary_dir)
        model.save_pretrained(staging_dir, safe_serialization=True)
        del model
        _verify_saved_config(staging_dir)

        converted_config = Chronos2Config.from_pretrained(staging_dir)
        reloaded_model = _load_strict(staging_dir, config=converted_config)
        converted_weight_files = _resolve_weight_files(staging_dir)
        converted_key_count = _assert_identity_checkpoint(
            reloaded_model,
            converted_weight_files,
            source=str(staging_dir),
        )
        if converted_key_count != source_key_count:
            raise AssertionError(
                f"Tensor count changed during conversion: {source_key_count} -> {converted_key_count}."
            )
        # This second source audit proves that save/reload retained the original
        # values, not merely a self-consistent converted checkpoint.
        _assert_identity_checkpoint(reloaded_model, source_weight_files, source=str(source_dir))
        del reloaded_model

        if verify_original:
            prediction_diff, loss_diff = _verify_original_parity(
                source_dir,
                staging_dir,
                config=config,
                atol=atol,
                rtol=rtol,
            )
            print(
                "Original/native parity passed: "
                f"prediction max abs diff={prediction_diff:.8g}, loss abs diff={loss_diff:.8g}."
            )

        staging_dir.rename(output_dir)

    print(f"Converted Chronos-2 checkpoint written to {output_dir}.")
    print("No files were uploaded.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Local checkpoint directory or Hub model ID (default: amazon/chronos-2).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help=(
            "Hub revision. The pinned official revision is used when the default checkpoint is selected; "
            "this argument is ignored for a local directory."
        ),
    )
    parser.add_argument(
        "--output_path", required=True, help="New directory in which to save the converted checkpoint."
    )
    parser.add_argument("--cache_dir", default=None, help="Optional Hugging Face Hub cache directory.")
    parser.add_argument("--token", default=None, help="Optional Hugging Face Hub read token.")
    parser.add_argument(
        "--verify-original",
        action="store_true",
        help="Compare deterministic outputs with the optional chronos-forecasting implementation.",
    )
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance for optional output parity.")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance for optional output parity.")
    args = parser.parse_args()

    convert_checkpoint(
        checkpoint=args.checkpoint,
        output_path=args.output_path,
        revision=args.revision,
        cache_dir=args.cache_dir,
        token=args.token,
        verify_original=args.verify_original,
        atol=args.atol,
        rtol=args.rtol,
    )


if __name__ == "__main__":
    main()
