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
"""WeatherNext 2 model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


ATMOSPHERIC_VARIABLES = (
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "specific_humidity",
)
STATIC_VARIABLES = ("geopotential_at_surface", "land_sea_mask")
GLOBAL_VARIABLES = ("year_progress_sin", "year_progress_cos")

CYCLONE_VARIABLES = (
    "cyclone_exists_gaussian_unit_mode",
    "cyclone_all_wind_disc",
    "cyclone_usa_wind_disc",
    "cyclone_usa_r34_ne_radius_disc",
    "cyclone_usa_r34_se_radius_disc",
    "cyclone_usa_r34_sw_radius_disc",
    "cyclone_usa_r34_nw_radius_disc",
    "cyclone_usa_r50_ne_radius_disc",
    "cyclone_usa_r50_se_radius_disc",
    "cyclone_usa_r50_sw_radius_disc",
    "cyclone_usa_r50_nw_radius_disc",
    "cyclone_usa_r64_ne_radius_disc",
    "cyclone_usa_r64_se_radius_disc",
    "cyclone_usa_r64_sw_radius_disc",
    "cyclone_usa_r64_nw_radius_disc",
    "cyclone_usa_rmw_disc",
    "cyclone_usa_pres_disc",
)

DEFAULT_INPUT_VARIABLES = (
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "specific_humidity",
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_v_component_of_wind",
    "10m_u_component_of_wind",
    "sea_surface_temperature",
    "geopotential_at_surface",
    "land_sea_mask",
    "year_progress_sin",
    "year_progress_cos",
    "day_progress_sin",
    "day_progress_cos",
)
DEFAULT_TARGET_VARIABLES = DEFAULT_INPUT_VARIABLES[:11] + ("total_precipitation_6hr",) + CYCLONE_VARIABLES
DEFAULT_FORCING_VARIABLES = ("year_progress_sin", "year_progress_cos", "day_progress_sin", "day_progress_cos")
DEFAULT_PRESSURE_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)


@auto_docstring(checkpoint="kashif/weathernext2")
@strict
class WeatherNext2Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`WeatherNext2Model`]. It is used to instantiate a
    WeatherNext 2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a configuration similar to the 0.25 degree
    [kashif/weathernext2](https://huggingface.co/kashif/weathernext2) architecture.

    WeatherNext 2 is a Functional Generative Network (FGN, https://huggingface.co/papers/2506.10772): an encode-process-decode graph network over an icosahedral
    mesh, made probabilistic by a single global noise vector that modulates every normalization layer. One forward pass
    advances the global atmospheric state by `time_step`; ensembles are produced by drawing several noise vectors.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Latent width of the grid points, the mesh nodes and the transformer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Width of the feed-forward layer inside each mesh transformer block.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of mesh transformer blocks.
        num_attention_heads (`int`, *optional*, defaults to 6):
            Number of attention heads in the mesh transformer.
        edge_hidden_size (`int`, *optional*, defaults to 32):
            Latent width of the grid/mesh graph edges.
        noise_channels (`int`, *optional*, defaults to 32):
            Dimension of the global noise vector, and of the conditioning vector it is projected to.
        hidden_act (`str`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            Activation of the mesh transformer feed-forward layer. The original implementation calls `jax.nn.gelu`,
            which defaults to the tanh approximation.
        mlp_act (`str`, *optional*, defaults to `"silu"`):
            Activation of every other multi-layer perceptron (encoders, graph network, decoder).
        mesh_splits (`int`, *optional*, defaults to 6):
            Number of times the base icosahedron is subdivided. `n` splits give `10 * 4**n + 2` mesh nodes, so 5 gives
            10242 nodes and 6 gives 40962.
        attention_k_hop (`int`, *optional*, defaults to 32):
            Radius, in mesh edges, of the local attention neighbourhood. Each mesh node attends to every node reachable
            within this many hops.
        ball_query_radius_fraction (`float`, *optional*, defaults to 0.6):
            Radius used to connect grid points to mesh nodes, as a fraction of the longest mesh edge.
        aggregate_normalization (`float`, *optional*):
            Constant the summed grid-to-mesh messages are divided by. Used by the 0.25 degree checkpoints to keep
            activations stable across grid resolutions; `None` disables it.
        grid_latitudes (`int`, *optional*, defaults to 721):
            Number of latitudes, from -90 to 90 inclusive.
        grid_longitudes (`int`, *optional*, defaults to 1440):
            Number of longitudes, from 0 eastwards.
        input_variables (`tuple(str)`, *optional*):
            Variables fed to the model, in the channel order the model expects.
        target_variables (`tuple(str)`, *optional*):
            Variables the model predicts, in the channel order it produces.
        forcing_variables (`tuple(str)`, *optional*):
            Variables known ahead of time and supplied for the *predicted* time step.
        atmospheric_variables (`tuple(str)`, *optional*):
            Subset of the variables that has a pressure-level dimension.
        static_variables (`tuple(str)`, *optional*):
            Subset of the input variables that is constant in time.
        global_variables (`tuple(str)`, *optional*):
            Subset of the variables that has neither a latitude nor a longitude dimension. These are broadcast over the
            grid and additionally fed to the mesh encoder.
        pressure_levels (`tuple(int)`, *optional*):
            Pressure levels in hPa, ascending.
        num_input_timesteps (`int`, *optional*, defaults to 2):
            Number of past states the model conditions on.
        time_step_hours (`int`, *optional*, defaults to 6):
            Hours advanced by one forward pass.
        sigmoid_shifted_outputs (`dict[str, float]`, *optional*):
            Target variables that get `sigmoid(x - shift)` applied after decoding, mapped to their shift.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation of the truncated normal initializer for all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            Epsilon of every layer normalization.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout ratio of the attention probabilities.

    ```python
    >>> from transformers import WeatherNext2Config, WeatherNext2Model

    >>> configuration = WeatherNext2Config()
    >>> model = WeatherNext2Model(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "weathernext2"
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 24
    num_attention_heads: int = 6
    edge_hidden_size: int = 32
    noise_channels: int = 32
    hidden_act: str = "gelu_pytorch_tanh"
    mlp_act: str = "silu"

    mesh_splits: int = 6
    attention_k_hop: int = 32
    ball_query_radius_fraction: float = 0.6
    aggregate_normalization: float | int | None = 4.0

    grid_latitudes: int = 721
    grid_longitudes: int = 1440

    input_variables: list[str] | tuple[str, ...] = DEFAULT_INPUT_VARIABLES
    target_variables: list[str] | tuple[str, ...] = DEFAULT_TARGET_VARIABLES
    forcing_variables: list[str] | tuple[str, ...] = DEFAULT_FORCING_VARIABLES
    atmospheric_variables: list[str] | tuple[str, ...] = ATMOSPHERIC_VARIABLES
    static_variables: list[str] | tuple[str, ...] = STATIC_VARIABLES
    global_variables: list[str] | tuple[str, ...] = GLOBAL_VARIABLES
    pressure_levels: list[int] | tuple[int, ...] = DEFAULT_PRESSURE_LEVELS

    num_input_timesteps: int = 2
    time_step_hours: int = 6
    sigmoid_shifted_outputs: dict[str, float] | None = None

    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    attention_dropout: float = 0.0

    def __post_init__(self, **kwargs):
        if self.sigmoid_shifted_outputs is None:
            self.sigmoid_shifted_outputs = (
                {"cyclone_exists_gaussian_unit_mode": 2.0}
                if "cyclone_exists_gaussian_unit_mode" in self.target_variables
                else {}
            )
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})."
            )
        unknown_static = set(self.static_variables) - set(self.input_variables)
        if unknown_static:
            raise ValueError(f"`static_variables` {sorted(unknown_static)} are not in `input_variables`.")
        unknown_shifted = set(self.sigmoid_shifted_outputs or {}) - set(self.target_variables)
        if unknown_shifted:
            raise ValueError(f"`sigmoid_shifted_outputs` {sorted(unknown_shifted)} are not in `target_variables`.")

    # ---------------------------------------------------------------------------------------
    # Channel layout. The encoders and the decoder are plain linear layers over a flat channel
    # axis, so both the model and the processor need one shared description of what that axis
    # contains. `*_channel_layout` is that description: an ordered list of
    # `(variable, time_offset_hours or None, num_levels)`.
    # ---------------------------------------------------------------------------------------

    def num_levels(self, variable: str) -> int:
        return len(self.pressure_levels) if variable in self.atmospheric_variables else 1

    @property
    def input_channel_layout(self) -> list[tuple[str, int | None, int]]:
        """Channels the grid encoder consumes, after the three spatial features.

        Dynamic input variables appear once per past time step, static variables once with no time,
        and forcings once at the predicted time step.
        """
        layout: list[tuple[str, int | None, int]] = []
        past_offsets = [
            -(self.num_input_timesteps - 1 - i) * self.time_step_hours for i in range(self.num_input_timesteps)
        ]
        for variable in self.input_variables:
            if variable in self.static_variables:
                layout.append((variable, None, self.num_levels(variable)))
            else:
                layout.extend((variable, offset, self.num_levels(variable)) for offset in past_offsets)
        layout.extend(
            (variable, self.time_step_hours, self.num_levels(variable)) for variable in self.forcing_variables
        )
        return layout

    @property
    def mesh_channel_layout(self) -> list[tuple[str, int | None, int]]:
        """Channels the mesh encoder consumes, after the three spatial features.

        Only the variables with no spatial extent reach the mesh encoder directly.
        """
        return [entry for entry in self.input_channel_layout if entry[0] in self.global_variables]

    @property
    def target_channel_layout(self) -> list[tuple[str, int | None, int]]:
        return [(variable, self.time_step_hours, self.num_levels(variable)) for variable in self.target_variables]

    @property
    def num_grid_input_channels(self) -> int:
        return 3 + sum(levels for _, _, levels in self.input_channel_layout)

    @property
    def num_mesh_input_channels(self) -> int:
        return 3 + sum(levels for _, _, levels in self.mesh_channel_layout)

    @property
    def num_output_channels(self) -> int:
        return sum(levels for _, _, levels in self.target_channel_layout)

    @property
    def num_grid_points(self) -> int:
        return self.grid_latitudes * self.grid_longitudes

    @property
    def num_mesh_nodes(self) -> int:
        return 10 * 4**self.mesh_splits + 2


__all__ = ["WeatherNext2Config"]
