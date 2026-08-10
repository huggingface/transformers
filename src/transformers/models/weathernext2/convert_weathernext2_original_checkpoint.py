# Copyright 2026 Google DeepMind and The HuggingFace Inc. team. All rights reserved.
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
"""Converts an original WeatherNext 2 checkpoint to the Transformers format.

The upstream checkpoints are Haiku parameter trees saved as `.npz`, published alongside a Fiddle
JSON config that carries both the architecture hyper-parameters and the normalization statistics.
Both live in a public bucket:

```bash
BASE=https://storage.googleapis.com/dm_graphcast/weathernext2
curl -o params.npz  "$BASE/params/WeatherNextCyclones_Mini_%3C2024.npz"
curl -o config.json "$BASE/configs/WeatherNextCyclones_Mini.json"   # or take it from the repo

python src/transformers/models/weathernext2/convert_weathernext2_original_checkpoint.py \
    --checkpoint_path params.npz \
    --fiddle_config_path config.json \
    --output_dir weathernext2-mini
```

Two structural differences are worth knowing about:

* Haiku stores linear weights as `[in, out]`; `nn.Linear` wants `[out, in]`, so every weight is
  transposed.
* The encoders and decoder keep one weight array *per input variable*, so a variable can be added or
  removed at fine-tuning time without disturbing the rest. Mathematically this is a single matmul
  over the concatenated inputs, which is what we store: the converter stacks the per-variable arrays
  in the canonical channel order defined by [`WeatherNext2Config.input_channel_layout`].
"""

import argparse
import json
import re
from typing import Any

import numpy as np
import torch

from transformers.models.weathernext2.configuration_weathernext2 import WeatherNext2Config
from transformers.models.weathernext2.modeling_weathernext2 import WeatherNext2ForWeatherForecasting
from transformers.models.weathernext2.processing_weathernext2 import WeatherNext2Processor


PARAM_PREFIX = "params:multimodality_forward/"
SPATIAL_NODE_FEATURES = "sin_lat,sin_lon,cos_lon"
SPATIAL_EDGE_FEATURES = "distance,rel_x,rel_y,rel_z"


# -------------------------------------------------------------------------------------------------
# Fiddle config
# -------------------------------------------------------------------------------------------------


def load_fiddle_config(path: str) -> dict[str, Any]:
    """Materializes a Fiddle JSON graph into plain Python containers."""
    document = json.load(open(path))
    objects = document["objects"]

    def field_name(key: str) -> str:
        match = re.search(r"name='([^']*)'", key) or re.search(r"key='([^']*)'", key) or re.search(r"index=(\d+)", key)
        return match.group(1) if match else key

    def resolve(node):
        kind = node.get("type")
        if kind == "leaf":
            return node.get("value")
        if kind == "pyref":
            return f"{node['module']}.{node['name']}"
        if kind != "ref":
            return None
        obj = objects[node["key"]]
        items = obj.get("items")
        if items is None:
            return None
        obj_type = obj.get("type")
        if isinstance(obj_type, dict) and obj_type.get("name") in ("tuple", "list"):
            return [resolve(value) for _, value in items]
        return {field_name(key): resolve(value) for key, value in items}

    return resolve(document["root"])


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def config_from_fiddle(fiddle: dict[str, Any], grid_latitudes: int, grid_longitudes: int) -> WeatherNext2Config:
    task = fiddle["task"]
    architecture = fiddle["predictor_kwargs"]["noisy_function_kwargs"]
    transformer = architecture["mesh_model_ctor"]["transformer_kwargs"]
    latent = architecture["latent_dense_kwargs"]

    shifted = {}
    for variable, (function, kwargs) in (architecture.get("per_var_activation_fns") or {}).items():
        if not function.endswith("shifted_activation") or not kwargs["activation_fn"].endswith("sigmoid"):
            raise ValueError(f"Unsupported output activation for {variable!r}: {function}")
        shifted[variable] = -float(kwargs["input_offset"])

    input_duration_hours = int(re.fullmatch(r"(\d+)h", task["input_duration"]).group(1))
    time_step_hours = 6

    return WeatherNext2Config(
        hidden_size=transformer["d_model"],
        intermediate_size=transformer["ffw_hidden"],
        num_hidden_layers=transformer["num_layers"],
        num_attention_heads=transformer["num_heads"],
        edge_hidden_size=architecture["points_to_mesh_model_ctor"]["edge_encoder_dense_kwargs"]["output_size"],
        noise_channels=architecture["norm_conditioning_latent_dense_kwargs"]["output_size"],
        mesh_splits=architecture["mesh_num_splits"],
        attention_k_hop=transformer["attention_k_hop"],
        ball_query_radius_fraction=architecture["points_to_mesh_model_ctor"]["ball_query_radius_fraction"],
        aggregate_normalization=_optional_float(
            architecture["points_to_mesh_model_ctor"]["deep_gnn_kwargs"].get("aggregate_normalization")
        ),
        grid_latitudes=grid_latitudes,
        grid_longitudes=grid_longitudes,
        input_variables=tuple(task["input_variables"]),
        target_variables=tuple(task["target_variables"]),
        forcing_variables=tuple(task["forcing_variables"]),
        pressure_levels=tuple(task["pressure_levels"]),
        num_input_timesteps=input_duration_hours // time_step_hours,
        time_step_hours=time_step_hours,
        sigmoid_shifted_outputs=shifted,
        hidden_act="gelu_pytorch_tanh",
        mlp_act="silu" if latent["activation"] == "swish" else latent["activation"],
    )


def statistics_from_fiddle(fiddle: dict[str, Any], name: str) -> dict[str, Any]:
    data_vars = fiddle["predictor_wrappers"][0]["kwargs"][name]["data"]["data_vars"]
    return {variable: entry["data"] for variable, entry in data_vars.items()}


def nan_filled_variables_from_fiddle(fiddle: dict[str, Any]) -> list[str]:
    """Variables the original `NaNCleaner` wrapper fills before the network sees them."""
    variables = []
    for wrapper in fiddle["predictor_wrappers"]:
        if wrapper["constructor"].endswith("NaNCleaner"):
            variables.append(wrapper["kwargs"]["var_to_clean"])
    return variables


# -------------------------------------------------------------------------------------------------
# Parameter names
# -------------------------------------------------------------------------------------------------


def split_weight_name(config: WeatherNext2Config, variable: str, time_offset: int | None, prefix: str) -> str:
    """Rebuilds the per-variable weight name used by the original `xarray_dense` encoders.

    The name records the coordinates the array covers, e.g.
    `w_input_temperature_level=50,...,1000_time=-21600`: pressure levels first (they stay inside one
    array), then the time slice (which gets its own array).
    """
    name = f"w_{prefix}{variable}"
    if variable in config.atmospheric_variables:
        name += "_level=" + ",".join(str(level) for level in config.pressure_levels)
    if time_offset is not None:
        name += f"_time={time_offset * 3600}"
    return name


def stacked_input_weight(
    params: dict[str, np.ndarray],
    config: WeatherNext2Config,
    module: str,
    layout: list[tuple[str, int | None, int]],
    spatial_features: str,
) -> np.ndarray:
    """Concatenates the split first matmul into one `[out, in]` weight."""
    parts = [params[f"{module}/split_input_matmul:w_spatial_feature={spatial_features}"]]
    for variable, time_offset, _ in layout:
        prefix = "forcing_" if time_offset is not None and time_offset > 0 else "input_"
        parts.append(params[f"{module}/split_input_matmul:{split_weight_name(config, variable, time_offset, prefix)}"])
    return np.concatenate(parts, axis=0).T


def stacked_output_weight(
    params: dict[str, np.ndarray], config: WeatherNext2Config, module: str
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenates the split output linear into one `[out, in]` weight and `[out]` bias."""
    weights, biases = [], []
    for variable, time_offset, _ in config.target_channel_layout:
        name = split_weight_name(config, variable, time_offset, prefix="")
        weights.append(params[f"{module}/split_output_linear:{name}"])
        biases.append(params[f"{module}/split_output_linear:{name.replace('w_', 'b_', 1)}"])
    return np.concatenate(weights, axis=1).T, np.concatenate(biases, axis=0)


# -------------------------------------------------------------------------------------------------
# Conversion
# -------------------------------------------------------------------------------------------------


def convert_state_dict(params: dict[str, np.ndarray], config: WeatherNext2Config) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()

    def take(name: str) -> np.ndarray:
        consumed.add(name)
        return params[name]

    def put(target: str, value: np.ndarray) -> None:
        state_dict[target] = torch.from_numpy(np.ascontiguousarray(value))

    def convert_conditioned_mlp(target: str, module: str) -> None:
        """The `Linear -> act -> Linear -> LayerNorm -> FiLM` block, minus its first weight."""
        put(f"{target}.in_proj.bias", take(f"{module}/shared_dense/mlp/linear_0:b"))
        put(f"{target}.out_proj.weight", take(f"{module}/shared_dense/mlp/linear_1:w").T)
        put(f"{target}.out_proj.bias", take(f"{module}/shared_dense/mlp/linear_1:b"))
        film = f"{module}/shared_dense/normalization/linear_norm_conditioning/linear"
        put(f"{target}.norm.film.linear.weight", take(f"{film}:w").T)
        put(f"{target}.norm.film.linear.bias", take(f"{film}:b"))

    # --- noise, encoders -------------------------------------------------------------------
    put(
        "model.noise_encoder.weight",
        take(
            "global_norm_conditioning_encoder/split_input_matmul:"
            f"w_input_noise_noise_channels=range({config.noise_channels})"
        ).T,
    )

    for target, module, layout in (
        ("model.grid_encoder", "grid_encoder", config.input_channel_layout),
        ("model.mesh_encoder", "mesh_encoder", config.mesh_channel_layout),
    ):
        put(f"{target}.in_proj.weight", stacked_input_weight(params, config, module, layout, SPATIAL_NODE_FEATURES))
        for variable, time_offset, _ in layout:
            prefix = "forcing_" if time_offset is not None and time_offset > 0 else "input_"
            consumed.add(f"{module}/split_input_matmul:{split_weight_name(config, variable, time_offset, prefix)}")
        consumed.add(f"{module}/split_input_matmul:w_spatial_feature={SPATIAL_NODE_FEATURES}")
        convert_conditioned_mlp(target, module)

    # --- graph networks --------------------------------------------------------------------
    for target, module, edge_set, receiver in (
        ("model.grid_to_mesh", "grid_to_mesh_gnn", "points_to_mesh_nodes", "mesh"),
        ("model.mesh_to_grid", "mesh_to_grid_gnn", "mesh_to_points_nodes", "point"),
    ):
        edge_module = f"{module}/edge_encoder"
        put(
            f"{target}.edge_encoder.in_proj.weight",
            take(f"{edge_module}/split_input_matmul:w_spatial_feature={SPATIAL_EDGE_FEATURES}").T,
        )
        convert_conditioned_mlp(f"{target}.edge_encoder", edge_module)

        gnn = f"{module}/deep_gnn"
        put(f"{target}.edge_update.edge_proj.weight", take(f"{gnn}/processor_edges_0_edge_{edge_set}:w").T)
        put(f"{target}.edge_update.sender_proj.weight", take(f"{gnn}/processor_edges_0_sender_{edge_set}:w").T)
        if receiver == "point":
            # Only the mesh-to-grid direction folds the receiver's own features into the message.
            put(
                f"{target}.edge_update.receiver_proj.weight",
                take(f"{gnn}/processor_edges_0_receiver_{edge_set}:w").T,
            )
        edge_update = f"{gnn}/processor_edges_0_{edge_set}"
        put(f"{target}.edge_update.bias", take(f"{edge_update}/mlp/linear_0:b"))
        put(f"{target}.edge_update.out_proj.weight", take(f"{edge_update}/mlp/linear_1:w").T)
        put(f"{target}.edge_update.out_proj.bias", take(f"{edge_update}/mlp/linear_1:b"))
        film = f"{edge_update}/normalization/linear_norm_conditioning/linear"
        put(f"{target}.edge_update.norm.film.linear.weight", take(f"{film}:w").T)
        put(f"{target}.edge_update.norm.film.linear.bias", take(f"{film}:b"))

        for node_target, node_set in (("mesh_node_update", "mesh_nodes"), ("grid_node_update", "point_nodes")):
            node_module = f"{gnn}/processor_nodes_0_{node_set}"
            put(f"{target}.{node_target}.in_proj.weight", take(f"{node_module}/mlp/linear_0:w").T)
            put(f"{target}.{node_target}.in_proj.bias", take(f"{node_module}/mlp/linear_0:b"))
            put(f"{target}.{node_target}.out_proj.weight", take(f"{node_module}/mlp/linear_1:w").T)
            put(f"{target}.{node_target}.out_proj.bias", take(f"{node_module}/mlp/linear_1:b"))
            film = f"{node_module}/normalization/linear_norm_conditioning/linear"
            put(f"{target}.{node_target}.norm.film.linear.weight", take(f"{film}:w").T)
            put(f"{target}.{node_target}.norm.film.linear.bias", take(f"{film}:b"))

    # --- mesh transformer ------------------------------------------------------------------
    for layer_idx in range(config.num_hidden_layers):
        block = f"mesh_transformer/transformer/block_{layer_idx:02d}"
        target = f"model.mesh_transformer.layers.{layer_idx}"
        for projection in ("q", "k", "v"):
            put(f"{target}.self_attn.{projection}_proj.weight", take(f"{block}/mha_proj_{projection}:w").T)
        put(f"{target}.self_attn.o_proj.weight", take(f"{block}/mha_final:w").T)
        put(f"{target}.self_attn.o_proj.bias", take(f"{block}/mha_final:b"))
        put(f"{target}.mlp.fc1.weight", take(f"{block}/ffw_up:w").T)
        put(f"{target}.mlp.fc1.bias", take(f"{block}/ffw_up:b"))
        put(f"{target}.mlp.fc2.weight", take(f"{block}/ffw_down:w").T)
        put(f"{target}.mlp.fc2.bias", take(f"{block}/ffw_down:b"))
        # Haiku names the two FiLM layers of a block by call order, so the second gets a `_1` suffix.
        for norm, suffix in (("input_layernorm", ""), ("post_attention_layernorm", "_1")):
            film = f"{block}/block_{layer_idx:02d}_norm_conditioning{suffix}/linear"
            put(f"{target}.{norm}.film.linear.weight", take(f"{film}:w").T)
            put(f"{target}.{norm}.film.linear.bias", take(f"{film}:b"))

    film = "mesh_transformer/transformer/transformer_final_norm_conditioning/linear"
    put("model.mesh_transformer.norm.film.linear.weight", take(f"{film}:w").T)
    put("model.mesh_transformer.norm.film.linear.bias", take(f"{film}:b"))

    # --- decoder ---------------------------------------------------------------------------
    put("decoder_proj.weight", take("grid_decoder/shared_dense/mlp/linear_0:w").T)
    put("decoder_proj.bias", take("grid_decoder/shared_dense/mlp/linear_0:b"))
    output_weight, output_bias = stacked_output_weight(params, config, "grid_decoder")
    put("output_proj.weight", output_weight)
    put("output_proj.bias", output_bias)
    for variable, time_offset, _ in config.target_channel_layout:
        name = split_weight_name(config, variable, time_offset, prefix="")
        consumed.add(f"grid_decoder/split_output_linear:{name}")
        consumed.add(f"grid_decoder/split_output_linear:{name.replace('w_', 'b_', 1)}")

    unconsumed = sorted(set(params) - consumed)
    if unconsumed:
        raise ValueError(f"{len(unconsumed)} checkpoint parameters were not converted, e.g. {unconsumed[:5]}")
    return state_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", required=True, help="Original `.npz` parameter file.")
    parser.add_argument("--fiddle_config_path", required=True, help="Matching Fiddle JSON config.")
    parser.add_argument("--output_dir", required=True, help="Where to save the converted model.")
    parser.add_argument(
        "--grid_latitudes", type=int, default=None, help="Defaults to 181 for the 1 degree mini model."
    )
    parser.add_argument("--grid_longitudes", type=int, default=None)
    parser.add_argument("--push_to_hub", default=None, help="Optional Hub repository id.")
    args = parser.parse_args()

    archive = np.load(args.checkpoint_path, allow_pickle=True)
    params = {key[len(PARAM_PREFIX) :]: archive[key] for key in archive.files if key.startswith(PARAM_PREFIX)}
    print(f"Loaded {len(params)} parameter arrays ({sum(v.size for v in params.values()) / 1e6:.1f}M values).")

    fiddle = load_fiddle_config(args.fiddle_config_path)
    # The mini model is trained at 1 degree, the rest at 0.25.
    is_mini = fiddle["predictor_kwargs"]["noisy_function_kwargs"]["mesh_num_splits"] < 6
    grid_latitudes = args.grid_latitudes or (181 if is_mini else 721)
    grid_longitudes = args.grid_longitudes or (360 if is_mini else 1440)

    config = config_from_fiddle(fiddle, grid_latitudes, grid_longitudes)
    state_dict = convert_state_dict(params, config)

    model = WeatherNext2ForWeatherForecasting(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected keys: {unexpected}")
    if missing:
        raise ValueError(f"Missing keys: {missing}")
    print(f"Converted {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters.")

    processor = WeatherNext2Processor(
        input_variables=config.input_variables,
        target_variables=config.target_variables,
        forcing_variables=config.forcing_variables,
        atmospheric_variables=config.atmospheric_variables,
        static_variables=config.static_variables,
        global_variables=config.global_variables,
        pressure_levels=config.pressure_levels,
        mean_by_level=statistics_from_fiddle(fiddle, "mean_by_level"),
        stddev_by_level=statistics_from_fiddle(fiddle, "stddev_by_level"),
        diffs_stddev_by_level=statistics_from_fiddle(fiddle, "diffs_stddev_by_level"),
        nan_filled_variables=nan_filled_variables_from_fiddle(fiddle),
        num_input_timesteps=config.num_input_timesteps,
        time_step_hours=config.time_step_hours,
        grid_latitudes=grid_latitudes,
        grid_longitudes=grid_longitudes,
    )

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}.")
    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub)
        processor.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()
