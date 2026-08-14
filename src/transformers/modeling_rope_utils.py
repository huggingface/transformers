# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, TypedDict

from .utils import is_torch_available, logging


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch

if TYPE_CHECKING:
    from .configuration_utils import PreTrainedConfig


def dynamic_rope_update(rope_forward):
    """
    Decorator function to update the RoPE parameters in the forward pass, if the model is using a dynamic RoPE
    (i.e. a RoPE implementation that may recompute its frequencies in the forward pass).

    Args:
        rope_forward (Callable):
            The forward pass of the RoPE implementation.

    Returns:
        The decorated forward pass.
    """

    def longrope_frequency_update(self, position_ids, device, layer_type=None):
        """Longrope uses long factor if sequence is larger than original pretraining length, short otherwise."""
        seq_len = torch.max(position_ids) + 1

        if layer_type is None:
            rope_type = self.rope_type
            original_inv_freq = self.original_inv_freq
            prefix = ""
            original_max_position_embeddings = self.config.rope_parameters["original_max_position_embeddings"]
        else:
            rope_type = self.rope_type[layer_type]
            original_inv_freq = getattr(self, f"{layer_type}_original_inv_freq")
            prefix = f"{layer_type}_"
            original_max_position_embeddings = self.config.rope_parameters[layer_type][
                "original_max_position_embeddings"
            ]

        if seq_len > original_max_position_embeddings:
            if not hasattr(self, f"{layer_type}_long_inv_freq"):
                rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
                long_inv_freq, _ = rope_init_fn(
                    self.config,
                    device,
                    seq_len=original_max_position_embeddings + 1,
                    layer_type=layer_type,
                )
            self.register_buffer(f"{prefix}inv_freq", long_inv_freq, persistent=False)
            setattr(self, f"{prefix}long_inv_freq", long_inv_freq)
        else:
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            original_inv_freq = original_inv_freq.to(device)
            self.register_buffer(f"{prefix}inv_freq", original_inv_freq, persistent=False)
            setattr(self, f"{prefix}original_inv_freq", original_inv_freq)

    def dynamic_frequency_update(self, position_ids, device, layer_type=None):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if layer_type is None:
            rope_type = self.rope_type
            max_seq_len_cached = self.max_seq_len_cached
            original_inv_freq = self.original_inv_freq
            prefix = ""
        else:
            rope_type = self.rope_type[layer_type]
            max_seq_len_cached = getattr(self, f"{layer_type}_max_seq_len_cached", self.max_seq_len_cached)
            original_inv_freq = getattr(self, f"{layer_type}_original_inv_freq")
            prefix = f"{layer_type}_"

        if seq_len > max_seq_len_cached:  # growth
            rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
            inv_freq, self.attention_scaling = rope_init_fn(
                self.config,
                device,
                seq_len=seq_len,
                layer_type=layer_type,
            )
            # TODO joao: may break with compilation
            self.register_buffer(f"{prefix}inv_freq", inv_freq, persistent=False)
            setattr(self, f"{prefix}max_seq_len_cached", seq_len)

        if seq_len < self.original_max_seq_len and max_seq_len_cached > self.original_max_seq_len:  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            original_inv_freq = original_inv_freq.to(device)
            self.register_buffer(f"{prefix}inv_freq", original_inv_freq, persistent=False)
            setattr(self, f"{prefix}original_inv_freq", original_inv_freq)
            setattr(self, f"{prefix}max_seq_len_cached", self.original_max_seq_len)

    @wraps(rope_forward)
    def wrapper(self, x, position_ids, layer_type=None):
        rope_type = self.rope_type if layer_type is None else self.rope_type[layer_type]
        kwargs = {"layer_type": layer_type} if layer_type is not None else {}
        if "dynamic" in rope_type:
            dynamic_frequency_update(self, position_ids, device=x.device, **kwargs)
        elif rope_type == "longrope":
            longrope_frequency_update(self, position_ids, device=x.device, **kwargs)
        return rope_forward(self, x, position_ids, **kwargs)

    return wrapper


def _compute_linear_scaling_rope_parameters(
    config: PreTrainedConfig | None = None,
    device: torch.device | None = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
) -> tuple[torch.Tensor, float]:
    """
    Computes the inverse frequencies with linear scaling. Credits to the Reddit user /u/kaiokendev
    Args:
        config ([`~transformers."PreTrainedConfig"`]):
            The model configuration. This function assumes that the config will provide at least the following
            properties:

            *   rope_theta (`float`, *optional*): The base wavelength from which the inverse frequencies will be derived. Defaults to `config.default_theta` if omitted.
            *   hidden_size (`int`): The numerator when deriving a head_dim, if not provided directly.
            *   num_attention_heads (`int`): The denominator when deriving a head_dim, if not provided directly.

            Additionally, this function will make use of the following properties if they are found in the config:

            *   head_dim (`int`, *optional*): The size of the key-value heads in the model. If None, this value will be
                derived as hidden_size // num_attention_heads.
            *   partial_rotary_factor (`float`, *optional*): If less than 1.0, inverse frequencies will be returned for
                the first fraction of the head_dim. Defaults to 1.0.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.

    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    # Sloppy workaround for per-layer-config in gemma4 family which raises error for other models (DS4)
    try:
        config = config.per_layer_config[layer_type] if layer_type is not None else config
    except ValueError:
        pass

    # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
    config.standardize_rope_params()
    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters
    factor = rope_parameters_dict["factor"]

    # Gets the default RoPE parameters
    base = rope_parameters_dict["rope_theta"]
    partial_rotary_factor = rope_parameters_dict.get("partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))

    # Then applies linear scaling to the frequencies.
    # NOTE: originally, scaling was applied to the position_ids. However, we get `embs = inv_freq @ position_ids`, so
    # applying scaling to the inverse frequencies is equivalent.
    inv_freq /= factor
    return inv_freq, attention_factor


def _compute_proportional_rope_parameters(
    config: PreTrainedConfig | None = None,
    device: torch.device | None = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
    head_dim_key: str = "head_dim",
) -> tuple[torch.Tensor, float]:
    """
    Computes the inverse frequencies with proportional RoPE.

    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration. This function assumes that the config will provide at least the following
            properties:

            *   rope_theta (`float`, *optional*): The base wavelength from which the inverse frequencies will be derived. Defaults to `config.default_theta` if omitted.
            *   hidden_size (`int`): The numerator when deriving a head_dim, if not provided directly.
            *   num_attention_heads (`int`): The denominator when deriving a head_dim, if not provided directly.

            Additionally, this function will make use of the following properties if they are found in the config:

            *   head_dim (`int`, *optional*): The size of the key-value heads in the model. If None, this value will be
                derived as hidden_size // num_attention_heads.
            *   partial_rotary_factor (`float`, *optional*, defaults to 1.0): The proportion of the embedding dimension
                to apply rotary positional encoding, e.g., [0.0, 0.25, 0.5, 0.75, 1.0]. Unlike other RoPE functions
                that use this parameter, proportional RoPE will always return an encoding that is the size of
                `head_dim`.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.

    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    # Sloppy workaround for per-layer-config in gemma4 family which raises error for other models (DS4)
    try:
        config = config.per_layer_config[layer_type] if layer_type is not None else config
    except ValueError:
        pass

    # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
    config.standardize_rope_params()
    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters

    head_dim = getattr(config, head_dim_key, None) or config.hidden_size // config.num_attention_heads
    base = rope_parameters_dict["rope_theta"]
    factor = rope_parameters_dict.get("factor", 1.0)
    rope_proportion = rope_parameters_dict.get("partial_rotary_factor", 1.0)

    attention_factor = 1.0  # Unused in this type of RoPE

    rope_angles = int(rope_proportion * head_dim // 2)

    inv_freq_rotated = 1.0 / (
        base
        ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / head_dim)
    )

    nope_angles = head_dim // 2 - rope_angles
    if nope_angles > 0:
        inv_freq = torch.cat(
            (
                inv_freq_rotated,
                torch.zeros(nope_angles, dtype=torch.float32, device=device),
            ),
            dim=0,
        )
    else:
        inv_freq = inv_freq_rotated

    inv_freq /= factor
    return inv_freq, attention_factor


def _compute_dynamic_ntk_parameters(
    config: PreTrainedConfig | None = None,
    device: torch.device | None = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
) -> tuple[torch.Tensor, float]:
    """
    Computes the inverse frequencies with NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla

    Args:
        config ([`~transformers."PreTrainedConfig"`]):
            The model configuration. This function assumes that the config will provide at least the following
            properties:

            *   rope_theta (`float`, *optional*): The base wavelength from which the inverse frequencies will be derived. Defaults to `config.default_theta` if omitted.
            *   hidden_size (`int`): The numerator when deriving a head_dim, if not provided directly.
            *   num_attention_heads (`int`): The denominator when deriving a head_dim, if not provided directly.
            *   max_position_embeddings (`int`): The default sequence length used to update the dynamic RoPE at
                inference time
            *   rope_parameters (`dict[str, float]`): The standard RoPE scaling parameters, from which `factor`
                will be accessed. The value of `factor` is used to determine the new base frequency, along with the
                current sequence length (seq_len), the maximum positional embeddings (max_position_embeddings), and the
                computed dimensionality (dim) of the rotary embeddings. If seq_len <= max_position_embeddings, this
                factor has no effect. If seq_len <= max_position_embeddings, this factor effectively stretches the
                context window using an exponent derived from `dim`.

            Additionally, this function will make use of the following properties if they are found in the config:

            *   head_dim (`int`, *optional*): The size of the key-value heads in the model. If None, this value will be
                derived as hidden_size // num_attention_heads.
            *   partial_rotary_factor (`float`, *optional*): If less than 1.0, inverse frequencies will be returned for
                the first fraction of the head_dim. Defaults to 1.0.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length, used to update the dynamic RoPE at inference time. If `None` or shorter than
            max_position_embeddings, this value will be overridden by max_position_embeddings.

    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    # Sloppy workaround for per-layer-config in gemma4 family which raises error for other models (DS4)
    try:
        config = config.per_layer_config[layer_type] if layer_type is not None else config
    except ValueError:
        pass

    # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
    config.standardize_rope_params()
    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters

    base = rope_parameters_dict["rope_theta"]
    partial_rotary_factor = rope_parameters_dict.get("partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    factor = rope_parameters_dict["factor"]
    attention_factor = 1.0  # Unused in this type of RoPE

    # seq_len: default to max_position_embeddings, e.g. at init time
    if seq_len is None:
        seq_len = config.max_position_embeddings
    elif isinstance(seq_len, torch.Tensor):
        seq_len = torch.maximum(
            seq_len,
            torch.tensor(config.max_position_embeddings, dtype=seq_len.dtype, device=seq_len.device),
        )
    else:
        seq_len = max(seq_len, config.max_position_embeddings)

    # Compute the inverse frequencies
    base = base * ((factor * seq_len / config.max_position_embeddings) - (factor - 1)) ** (dim / (dim - 2))
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, attention_factor


def _compute_yarn_parameters(
    config: PreTrainedConfig,
    device: torch.device | None = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
) -> tuple[torch.Tensor, float]:
    """
    Computes the inverse frequencies with NTK scaling. Please refer to the
    [original paper](https://huggingface.co/papers/2309.00071)

    Args:
        config ([`~transformers."PreTrainedConfig"`]):
            The model configuration. This function assumes that the config will provide at least the following
            properties:

            *   rope_theta (`float`, *optional*): The base wavelength from which the inverse frequencies will be derived. Defaults to `config.default_theta` if omitted.
            *   hidden_size (`int`): The numerator when deriving a head_dim, if not provided directly.
            *   num_attention_heads (`int`): The denominator when deriving a head_dim, if not provided directly.
            *   max_position_embeddings (`int`): The maximum length of the positional embeddings.
            *   rope_parameters (`dict[str, float | int]`): The standard RoPE scaling parameters, from which the following
                keys will be accessed:
                *   `attention_factor` (`float`, *optional*): The scaling factor to be applied to the computed cos/sin.
                    If None, the value is inferred from `factor`, `mscale`, and `mscale_all_dim` as available.
                *   `beta_fast` (`float`, *optional*, defaults to 32): Parameter to set the boundary for extrapolation
                    (only) in the linear ramp function.
                *   `beta_slow` (`float`, *optional*, defaults to 1): Parameter to set the boundary for interpolation
                    (only) in the linear ramp function.
                *   `factor` (`float`, *optional*): The scaling factor applied when interpolating the position IDs to
                    extend the possible context length. Additionally, if `attention_factor` is None, the log of this
                    value is used to compute a value for `attention_factor`, possibly in conjunction with `mscale` and
                    `mscale_all_dim`, if provided.
                *   `mscale` (`float`, *optional*): If `attention_factor` is None and both `mscale` and
                    `mscale_all_dim` are provided, `mscale` acts scalar augmenting `log(factor)` when computing the
                    numerator for the inferred value of `attention_factor`. If not provided, `attention_factor` will be
                    calculated based on `factor` only.
                *   `mscale_all_dim` (`float`, *optional*): If `attention_factor` is None and both `mscale` and
                    `mscale_all_dim` are provided, `mscale_all_dim` acts scalar augmenting `log(factor)` when computing
                    the denominator for the inferred value of `attention_factor`. If not provided, `attention_factor`
                    will be calculated based on `factor` only.
                *   `original_max_position_embeddings` (`int`): The original max position embeddings used during pretraining.
                *   `truncate` (`bool`, *optional*): Whether to truncate the correction range.

            Additionally, this function will make use of the following properties if they are found in the config:

            *   head_dim (`int`, *optional*): The size of the key-value heads in the model. If None, this value will be
                derived as hidden_size // num_attention_heads.
            *   partial_rotary_factor (`float`, *optional*, defaults to 1.0): If less than 1.0, inverse frequencies
                will be returned for the first fraction of the head_dim.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.

    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin.
    """
    # Sloppy workaround for per-layer-config in gemma4 family which raises error for other models (DS4)
    try:
        config = config.per_layer_config[layer_type] if layer_type is not None else config
    except ValueError:
        pass

    # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
    config.standardize_rope_params()
    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters

    base = rope_parameters_dict["rope_theta"]
    partial_rotary_factor = rope_parameters_dict.get("partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)

    factor = rope_parameters_dict["factor"]
    attention_factor = rope_parameters_dict.get("attention_factor")
    mscale = rope_parameters_dict.get("mscale")
    mscale_all_dim = rope_parameters_dict.get("mscale_all_dim")
    original_max_position_embeddings = rope_parameters_dict["original_max_position_embeddings"]

    # NOTE: DeepSeek-V3 (and potentially other models) have `original_max_position_embeddings` field
    # containing the pretrained value. They use the ratio between `max_position_embeddings` and this value
    # to compute the default attention scaling factor, instead of using `factor`.
    if factor is None:
        factor = config.max_position_embeddings / original_max_position_embeddings

    def get_mscale(scale, mscale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    # Sets the attention factor as suggested in the paper
    if attention_factor is None:
        if mscale and mscale_all_dim:
            attention_factor = float(get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim))
        else:
            attention_factor = get_mscale(factor)

    # Optional config options
    # beta_fast/beta_slow: as suggested in the paper, default to 32/1 (correspondingly)
    beta_fast = rope_parameters_dict.get("beta_fast") or 32
    beta_slow = rope_parameters_dict.get("beta_slow") or 1

    # Compute the inverse frequencies
    def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
        """Inverse dimension formula to find the dimension based on the number of rotations"""
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings, truncate):
        """Find dimension range bounds based on rotations"""
        low = find_correction_dim(low_rot, dim, base, max_position_embeddings)
        high = find_correction_dim(high_rot, dim, base, max_position_embeddings)
        if truncate:
            low = math.floor(low)
            high = math.ceil(high)
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001  # Prevent singularity

        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # Note on variable naming: "interpolation" comes from the original technique, where we interpolate the position IDs
    # to expand the possible context length. In other words, interpolation = apply scaling factor.
    pos_freqs = base ** (torch.arange(0, dim, 2).to(device=device, dtype=torch.float) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    truncate = config.rope_parameters.get("truncate", True)
    low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_max_position_embeddings, truncate)

    # Get n-dimensional rotational scaling corrected for extrapolation
    inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2).to(device=device, dtype=torch.float)
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
        + inv_freq_extrapolation * inv_freq_extrapolation_factor
    )
    return inv_freq, attention_factor


def _compute_longrope_parameters(
    config: PreTrainedConfig,
    device: torch.device | None = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
) -> tuple[torch.Tensor, float]:
    """
    Computes the inverse frequencies with LongRoPE scaling. Please refer to the
    [original implementation](https://github.com/microsoft/LongRoPE)

    Args:
        config ([`~transformers."PreTrainedConfig"`]):
            The model configuration. This function assumes that the config will provide at least the following
            properties:

            *   rope_theta (`float`, *optional*): The base wavelength from which the inverse frequencies will be derived. Defaults to `config.default_theta` if omitted.
            *   hidden_size (`int`): The numerator when deriving a head_dim, if not provided directly.
            *   num_attention_heads (`int`): The denominator when deriving a head_dim, if not provided directly.
            *   max_position_embeddings (`int`): The maximum length of the positional embeddings.
            *   original_max_position_embeddings (`int`, *optional*): The original max position embeddings used during
                pretraining. If not provided, defaults to `max_position_embeddings`.
            *   rope_parameters (`dict[str, float]`): The standard RoPE scaling parameters, from which the following keys
                will be accessed:
                *   `attention_factor` (`float`, *optional*): The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, inferred from
                    the value of `factor`.
                *   `factor` (`float`, *optional*): The scaling factor to apply to the RoPE embeddings. If both
                    `max_position_embeddings` and `original_max_position_embeddings` are provided, this value will be
                    overridden s the ratio between those values.
                *   `long_factor` (`float`, *optional*): The scale factor applied when computing the inverse
                    frequencies if `seq_len` is provided and greater than `original_max_position_embeddings`.
                *   `short_factor` (`float`, *optional*): The scale factor applied when computing the inverse
                    frequencies if `seq_len` is None or less-than-or-equal-to `original_max_position_embeddings`.

            Additionally, this function will make use of the following properties if they are found in the config:

            *   head_dim (`int`, *optional*): The size of the key-value heads in the model. If None, this value will be
                derived as hidden_size // num_attention_heads.
            *   partial_rotary_factor (`float`, *optional*, defaults to 1.0): If less than 1.0, inverse frequencies
                will be returned for the first fraction of the head_dim.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length.

    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin.
    """
    # Sloppy workaround for per-layer-config in gemma4 family which raises error for other models (DS4)
    try:
        config = config.per_layer_config[layer_type] if layer_type is not None else config
    except ValueError:
        pass

    # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
    config.standardize_rope_params()
    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters

    base = rope_parameters_dict["rope_theta"]
    partial_rotary_factor = rope_parameters_dict.get("partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)

    long_factor = rope_parameters_dict["long_factor"]
    short_factor = rope_parameters_dict["short_factor"]
    factor = rope_parameters_dict.get("factor")
    attention_factor = rope_parameters_dict.get("attention_factor")
    original_max_position_embeddings = rope_parameters_dict["original_max_position_embeddings"]

    # NOTE: Phi3 (and potentially other models) modify `max_position_embeddings` and have a
    # `original_max_position_embeddings` field containing the pretrained value. They use the ratio between these two
    # values to compute the default attention scaling factor, instead of using `factor`.
    if factor is None:
        factor = config.max_position_embeddings / original_max_position_embeddings

    # Sets the attention factor as suggested in the paper
    if attention_factor is None:
        if factor <= 1.0:
            attention_factor = 1.0
        else:
            attention_factor = math.sqrt(1 + math.log(factor) / math.log(original_max_position_embeddings))

    # Compute the inverse frequencies -- scaled based on the target sequence length
    if seq_len and seq_len > original_max_position_embeddings:
        ext_factors = torch.tensor(long_factor, dtype=torch.float32, device=device)
    else:
        ext_factors = torch.tensor(short_factor, dtype=torch.float32, device=device)
    inv_freq_shape = torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim
    inv_freq = 1.0 / (ext_factors * base**inv_freq_shape)

    return inv_freq, attention_factor


def _compute_llama3_parameters(
    config: PreTrainedConfig,
    device: torch.device | None = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
) -> tuple[torch.Tensor, float]:
    """
    Computes the inverse frequencies for llama 3.1.

    Args:
        config ([`~transformers."PreTrainedConfig"`]):
            The model configuration. This function assumes that the config will provide at least the following
            properties:

            *   rope_theta (`float`, *optional*): The base wavelength from which the inverse frequencies will be derived. Defaults to `config.default_theta` if omitted.
            *   hidden_size (`int`): The numerator when deriving a head_dim, if not provided directly.
            *   num_attention_heads (`int`): The denominator when deriving a head_dim, if not provided directly.
            *   rope_parameters (`dict[str, float | int]`): The standard RoPE scaling parameters, from which the following
                keys will be accessed:
                *   `factor` (`float`, *optional*): The scaling factor applied to the inverse frequencies when 1) the
                    wavelength is greater than `low_freq_wavelen` prior to smoothing, and 2) to all inverse frequencies
                    during smoothing.
                *   `high_freq_factor` (`float`): The scale factor used to compute `high_freq_wavelen` and
                    the value for the denominator of the smoothing factor prior to the `low_freq_factor` shift.
                *   `low_freq_factor` (`float`): The scale factor used to compute `low_freq_wavelen` and
                    the shift applied to the numerator and denominator of the smoothing factor.
                    frequencies if `seq_len` is None or less-than-or-equal-to `original_max_position_embeddings`.
                *   `original_max_position_embeddings` (`int`): The original max position embeddings used
                    during pretraining. If not provided, the function falls back to `max_position_embeddings`.

            Additionally, this function will make use of the following properties if they are found in the config:

            *   head_dim (`int`, *optional*): The size of the key-value heads in the model. If None, this value will be
                derived as hidden_size // num_attention_heads.
            *   partial_rotary_factor (`float`, *optional*): If less than 1.0, inverse frequencies will be returned for
                the first fraction of the head_dim. Defaults to 1.0.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin.
    """
    # Sloppy workaround for per-layer-config in gemma4 family which raises error for other models (DS4)
    try:
        config = config.per_layer_config[layer_type] if layer_type is not None else config
    except ValueError:
        pass

    # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
    config.standardize_rope_params()
    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters

    # Gets the default RoPE parameters
    base = rope_parameters_dict["rope_theta"]
    partial_rotary_factor = rope_parameters_dict.get("partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))

    factor = rope_parameters_dict["factor"]  # `8` in the original implementation
    low_freq_factor = rope_parameters_dict["low_freq_factor"]  # `1` in the original implementation
    high_freq_factor = rope_parameters_dict["high_freq_factor"]  # `4` in the original implementation
    old_context_len = rope_parameters_dict["original_max_position_embeddings"]  # `8192` in the original implementation

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq_llama, attention_factor


# This maps the "rope_type" string field in rope config to the corresponding function to compute the RoPE parameters
# from the model config. You can append new {'rope_type': callable} pairs to this rope_parameters to enable custom RoPE
# parameterizations, as long as the callable has the same signature.
ROPE_INIT_FUNCTIONS: dict[str, Callable[..., tuple[torch.Tensor, float]]] = {
    "linear": _compute_linear_scaling_rope_parameters,
    "dynamic": _compute_dynamic_ntk_parameters,
    "yarn": _compute_yarn_parameters,
    "longrope": _compute_longrope_parameters,
    "llama3": _compute_llama3_parameters,
    "proportional": _compute_proportional_rope_parameters,
}


class RopeParameters(TypedDict):
    """
    Args:
        rope_theta (`float`, *optional*, defaults to `RotaryEmbeddingConfigMixin.default_theta`):
            The base period of the RoPE embeddings. Optional in serialized configs — if omitted,
            the model's `default_theta` (typically 10000.0) is used.
        rope_type (`str`, *optional*, defaults to "default"):
            The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
            'llama3'], with 'default' being the original RoPE implementation.
        partial_rotary_factor (`float`, *optional*):
            The percentage of the query and key head embedding on which RoPE will be applied.
        factor (`float`, *optional*):
            Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
            most scaling types, a `factor` of x will enable the model to handle sequences of length x *
            original maximum pre-trained length.
        original_max_position_embeddings (`int`, *optional*):
            Used with 'yarn', 'longrope' and 'llama3'. The original max position embeddings used during
            pretraining.
        attention_factor (`float`, *optional*):
            Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
            computation. If unspecified, it defaults to value recommended by the implementation, using the
            `factor` field to infer the suggested value.
        beta_fast (`float`, *optional*):
            Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
            ramp function. If unspecified, it defaults to 32.
        beta_slow (`float`, *optional*):
            Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
            ramp function. If unspecified, it defaults to 1.
        short_factor (`list[float]`, *optional*):
            Only used with 'longrope'. The scaling factor to be applied to short contexts (<
            `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
            size divided by the number of attention heads divided by 2
        long_factor (`list[float]`, *optional*):
            Only used with 'longrope'. The scaling factor to be applied to long contexts (<
            `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
            size divided by the number of attention heads divided by 2
        low_freq_factor (`float`, *optional*):
            Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
        high_freq_factor (`float`, *optional*):
            Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
    """

    rope_theta: float | None
    rope_type: str | None
    partial_rotary_factor: float | None
    factor: float | None
    original_max_position_embeddings: int | None
    attention_factor: float | None
    beta_fast: float | None
    beta_slow: float | None
    short_factor: list[float] | None
    long_factor: list[float] | None
    low_freq_factor: float | None
    high_freq_factor: float | None


class RotaryEmbeddingConfigMixin:
    """
    A Mixin containing the functionality to standardize and validate RoPE parameters.
    """

    default_theta = 10_000.0
    ignore_keys_at_rope_validation = set()

    def convert_rope_params_to_dict(self, **kwargs):
        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or self.rope_parameters
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else {}

        # Standardize and validate the correctness of rotary position embeddings parameters. Priority for these parameters is:
        # 1. Values in `rope_parameters` dict (where they should be after standardization)
        # 2. Values in `kwargs` (i.e. it's in config.json but not MyConfig.__init__'s args)
        # 3. Values in the config's attributes (i.e. it's in MyConfig.__init__'s args)
        # 4. Default values (i.e. not present at all but other RoPE parameters are present)
        rope_theta = kwargs.pop("rope_theta", getattr(self, "rope_theta", self.default_theta))
        self.rope_parameters.setdefault("rope_theta", rope_theta)

        partial_rotary_factor = kwargs.get("partial_rotary_factor", getattr(self, "partial_rotary_factor", None))
        if partial_rotary_factor is not None:
            self.rope_parameters.setdefault("partial_rotary_factor", partial_rotary_factor)
            self.ignore_keys_at_rope_validation = set(self.ignore_keys_at_rope_validation or []) | {
                "partial_rotary_factor"
            }

        self.standardize_rope_params()
        return kwargs

    def standardize_rope_params(self):
        """
        Helper to standardize the config's rope params field by ensuring the params are defined for each
        later type. For old model the fn will duplicate a single rope param in each layer type (backward compatibility)
        """
        # Move `rope_theta` and `partial_rotary_factor` to the `rope_parameters`, if not there yet
        rope_theta = getattr(self, "rope_theta", None)
        partial_rotary_factor = getattr(self, "partial_rotary_factor", None)
        rope_parameters = getattr(self, "rope_parameters", None) or {}

        # Deepseekv4 has `layer_types` which are different from `_rope_type_labels`
        layer_types = getattr(self, "_rope_type_labels", getattr(self, "layer_types", None))

        # Case 0: no RoPE params defined
        if not (rope_parameters or rope_theta):
            # partial_rotary_factor without rope_theta is invalid, so we don't check for it here
            logger.warning("`standardize_rope_params` was called but no RoPE parameters were found.")
            return
        # Case 1: RoPE param keys do not intersect with possible `layer_types` -> one global dict
        elif layer_types is None or rope_parameters == {} or not set(rope_parameters.keys()).issubset(layer_types):
            rope_parameters.setdefault("rope_type", rope_parameters.get("type", "default"))
            rope_parameters.setdefault("rope_theta", rope_theta)
            if partial_rotary_factor is not None:
                rope_parameters["partial_rotary_factor"] = partial_rotary_factor

            # Move pretraining-time maximum length to rope parameter dict for RoPE types with scaling
            if rope_parameters["rope_type"] in ["llama3", "yarn", "longrope"]:
                if hasattr(self, "original_max_position_embeddings"):
                    # NOTE: Phi3 (and potentially other models) save `original_max_position_embeddings` field
                    # containing the pretrained value outside rope parameters. This is an exception case where we
                    # give priority to `self.original_max_position_embeddings
                    self.rope_parameters["original_max_position_embeddings"] = self.original_max_position_embeddings
                else:
                    self.rope_parameters.setdefault("original_max_position_embeddings", self.max_position_embeddings)

        # Case 2: different RoPE for each layer -> several params as nested dict
        else:
            for layer_type in set(layer_types):
                rope_parameters[layer_type].setdefault("rope_type", rope_parameters[layer_type].get("type", "default"))
                rope_parameters[layer_type].setdefault("rope_theta", rope_theta)
                if partial_rotary_factor is not None:
                    rope_parameters[layer_type]["partial_rotary_factor"] = partial_rotary_factor

                if rope_parameters[layer_type]["rope_type"] in ["llama3", "yarn", "longrope"]:
                    self.rope_parameters[layer_type].setdefault(
                        "original_max_position_embeddings", self.max_position_embeddings
                    )

        self.rope_parameters = rope_parameters

    def validate_rope(self: PreTrainedConfig):
        """
        Validate the RoPE config arguments, given a `"PreTrainedConfig"` object
        """
        # Don't validate if no rope_parameters found (`None`) or if it's an empty dict
        # Note that validation runs every time a new config is created, even if config is non-RoPE
        rope_parameters_dict = getattr(self, "rope_parameters", None)
        if not rope_parameters_dict:
            return

        if getattr(self, "layer_types", None) is not None and set(rope_parameters_dict.keys()).issubset(
            self.layer_types
        ):
            pass
        else:
            rope_parameters_dict = {"full_attention": rope_parameters_dict}

        for rope_parameters in rope_parameters_dict.values():
            rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
            validation_fn = getattr(self, f"_validate_{rope_type}_rope_parameters", None)
            rope_parameters["rope_type"] = rope_type

            if validation_fn is not None:
                validation_fn(rope_parameters, ignore_keys=self.ignore_keys_at_rope_validation)
            else:
                logger.warning(
                    f"Missing validation function in 'RotaryEmbeddingConfigMixin' for 'rope_type'='{rope_type}'"
                )

    def _validate_default_rope_parameters(self, rope_parameters: dict, ignore_keys: set | None = None):
        required_keys = {"rope_type"}
        optional_keys = {"rope_theta"}
        received_keys = set(rope_parameters.keys())
        rope_type = rope_parameters["rope_type"]
        self._check_received_keys(
            rope_type, received_keys, required_keys, optional_keys=optional_keys, ignore_keys=ignore_keys
        )

    def _validate_linear_rope_parameters(self, rope_parameters: dict, ignore_keys: set | None = None):
        required_keys = {"rope_type", "factor"}
        optional_keys = {"rope_theta"}
        received_keys = set(rope_parameters.keys())
        rope_type = rope_parameters["rope_type"]
        self._check_received_keys(
            rope_type, received_keys, required_keys, optional_keys=optional_keys, ignore_keys=ignore_keys
        )

        factor = rope_parameters["factor"]
        if factor is None or not isinstance(factor, (float, int)) or factor < 1.0:
            logger.warning(f"`rope_parameters`'s factor field must be a float or int >= 1, got {factor}")

    def _validate_dynamic_rope_parameters(self, rope_parameters: dict, ignore_keys: set | None = None):
        required_keys = {"rope_type", "factor"}
        optional_keys = {"rope_theta"}
        received_keys = set(rope_parameters.keys())
        rope_type = rope_parameters["rope_type"]
        self._check_received_keys(
            rope_type, received_keys, required_keys, optional_keys=optional_keys, ignore_keys=ignore_keys
        )

        factor = rope_parameters["factor"]
        if factor is None or not isinstance(factor, (float, int)) or factor < 1.0:
            logger.warning(f"`rope_parameters`'s factor field must be a float or int >= 1, got {factor}")

    def _validate_yarn_rope_parameters(self, rope_parameters: dict, ignore_keys: set | None = None):
        required_keys = {"rope_type", "factor", "original_max_position_embeddings"}
        optional_keys = {
            "rope_theta",
            "attention_factor",
            "beta_fast",
            "beta_slow",
            "mscale",
            "mscale_all_dim",
            "truncate",
        }
        received_keys = set(rope_parameters.keys())
        rope_type = rope_parameters["rope_type"]
        self._check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

        factor = rope_parameters["factor"]
        if factor is None or not isinstance(factor, (float, int)) or factor < 1.0:
            logger.warning(f"`rope_parameters`'s factor field must be a float or int >= 1, got {factor}")

        attention_factor = rope_parameters.get("attention_factor")
        if attention_factor is not None and (not isinstance(attention_factor, float) or attention_factor < 0):
            logger.warning(
                f"`rope_parameters`'s attention_factor field must be a float greater than 0, got {attention_factor}"
            )
        beta_fast = rope_parameters.get("beta_fast")
        if beta_fast is not None and not isinstance(beta_fast, (float, int)):
            logger.warning(f"`rope_parameters`'s beta_fast field must be a float or int, got {beta_fast}")
        beta_slow = rope_parameters.get("beta_slow")
        if beta_slow is not None and not isinstance(beta_slow, (float, int)):
            logger.warning(f"`rope_parameters`'s beta_slow field must be a float or int, got {beta_slow}")

        if (beta_fast or 32) < (beta_slow or 1):
            logger.warning(
                f"`rope_parameters`'s beta_fast field must be greater than beta_slow, got beta_fast={beta_fast} "
                f"(defaults to 32 if None) and beta_slow={beta_slow} (defaults to 1 if None)"
            )

        # Double-check: `factor` should be the ratio between the pre-yarn and post-yarn context lengths.
        # NOTE: we might get `implicit_factor == 1` if config's `original_max_position_embeddings` was
        # inferred from `max_position_embeddings` during standardization
        original_max_position_embeddings = rope_parameters["original_max_position_embeddings"]
        implicit_factor = self.max_position_embeddings / original_max_position_embeddings
        if implicit_factor != factor and implicit_factor != 1:
            logger.warning_once(
                f"The explicitly set RoPE scaling factor (config.rope_parameters['factor'] = {factor}) does not match "
                "the ratio implicitly set by other parameters (implicit factor = "
                "post-yarn context length / pre-yarn context length = "
                "config.max_position_embeddings / config.rope_parameters['original_max_position_embeddings'] = "
                f"{implicit_factor}). Using the explicit factor ({factor}) in YaRN. This may cause unexpected "
                "behaviour in model usage, please correct the 'original_max_position_embeddings' fields in the model config."
            )

    def _validate_longrope_rope_parameters(self, rope_parameters: dict, ignore_keys: set | None = None):
        required_keys = {"rope_type", "short_factor", "long_factor", "original_max_position_embeddings"}
        optional_keys = {"rope_theta", "attention_factor", "factor"}
        received_keys = set(rope_parameters.keys())
        rope_type = rope_parameters["rope_type"]
        self._check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

        partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(self, "head_dim", self.hidden_size // self.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)

        short_factor = rope_parameters.get("short_factor")
        if not (isinstance(short_factor, list) and all(isinstance(x, (int, float)) for x in short_factor)):
            logger.warning(f"`rope_parameters`'s short_factor field must be a list of numbers, got {short_factor}")
        if len(short_factor) != dim // 2:
            logger.warning(
                f"`rope_parameters`'s short_factor field must have length {dim // 2}, got {len(short_factor)}"
            )

        long_factor = rope_parameters.get("long_factor")
        if not (isinstance(long_factor, list) and all(isinstance(x, (int, float)) for x in long_factor)):
            logger.warning(f"`rope_parameters`'s long_factor field must be a list of numbers, got {long_factor}")
        if len(long_factor) != dim // 2:
            logger.warning(
                f"`rope_parameters`'s long_factor field must have length {dim // 2}, got {len(long_factor)}"
            )

        factor = rope_parameters.get("factor")
        original_max_position_embeddings = rope_parameters["original_max_position_embeddings"]

        # Handle Phi3 divergence: we prefer the use of `attention_factor` and/or `factor` over
        # `original_max_position_embeddings` to compute internal variables. The latter is undesirable
        if factor is None and original_max_position_embeddings is not None:
            logger.warning_once(
                "This model config has set a `rope_parameters['original_max_position_embeddings']` field, to be used together with "
                "`max_position_embeddings` to determine a scaling factor. Please set the `factor` field of `rope_parameters`"
                "with this ratio instead -- we recommend the use of this field over `original_max_position_embeddings`, "
                "as it is compatible with most model architectures."
            )
        elif factor is None and original_max_position_embeddings is None:
            logger.warning("Missing required keys in `rope_parameters`: 'factor'")
        elif not isinstance(factor, (float, int)) or factor < 1.0:
            logger.warning(f"`rope_parameters`'s factor field must be a float or int >= 1, got {factor}")

        attention_factor = rope_parameters.get("attention_factor")
        if attention_factor is not None and (not isinstance(attention_factor, (float, int)) or attention_factor < 0.0):
            logger.warning(
                f"`rope_parameters`'s attention_factor field must be a float or int greater than 0, got {attention_factor}"
            )

    def _validate_llama3_rope_parameters(self, rope_parameters: dict, ignore_keys: set | None = None):
        required_keys = {
            "rope_type",
            "factor",
            "original_max_position_embeddings",
            "low_freq_factor",
            "high_freq_factor",
            "rope_theta",
        }
        rope_type = rope_parameters["rope_type"]
        received_keys = set(rope_parameters.keys())
        self._check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)

        factor = rope_parameters["factor"]
        if factor is None or not isinstance(factor, (float, int)) or factor < 1.0:
            logger.warning(f"`rope_parameters`'s factor field must be a float or int >= 1, got {factor}")

        low_freq_factor = rope_parameters["low_freq_factor"]
        high_freq_factor = rope_parameters["high_freq_factor"]
        if low_freq_factor is None or not isinstance(low_freq_factor, (float, int)):
            logger.warning(f"`rope_parameters`'s low_freq_factor field must be a float, or int got {low_freq_factor}")
        if high_freq_factor is None or not isinstance(high_freq_factor, (float, int)):
            logger.warning(
                f"`rope_parameters`'s high_freq_factor field must be a float or int, got {high_freq_factor}"
            )
        if high_freq_factor <= low_freq_factor:
            logger.warning(
                "`rope_parameters`'s high_freq_factor field must be greater than low_freq_factor, got high_freq_factor="
                f"{high_freq_factor} and low_freq_factor={low_freq_factor}"
            )

        original_max_position_embeddings = rope_parameters["original_max_position_embeddings"]
        if original_max_position_embeddings is None or not isinstance(original_max_position_embeddings, int):
            logger.warning(
                "`rope_parameters`'s original_max_position_embeddings field must be an integer, got "
                f"{original_max_position_embeddings}"
            )
        if original_max_position_embeddings >= self.max_position_embeddings:
            logger.warning(
                "`rope_parameters`'s original_max_position_embeddings field must be less than max_position_embeddings, got "
                f"{original_max_position_embeddings} and max_position_embeddings={self.max_position_embeddings}"
            )

    def _validate_proportional_rope_parameters(self, rope_parameters: dict, ignore_keys: set | None = None):
        required_keys = {"rope_type", "rope_theta"}
        rope_type = rope_parameters["rope_type"]
        received_keys = set(rope_parameters.keys())
        self._check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)

        partial_rotary_factor = rope_parameters.get("partial_rotary_factor")
        if partial_rotary_factor is None:
            logger.warning(
                "`rope_parameters`'s partial_rotary_factor is None. This will default to 1.0 in the computation, "
                "making this equivalent to the linear_scaling RoPE type. Provide a value in the range [0.0, 1.0) to "
                "make use of the proportional RoPE functionality."
            )

    @staticmethod
    def _check_received_keys(
        rope_type: str,
        received_keys: set,
        required_keys: set,
        optional_keys: set | None = None,
        ignore_keys: set | None = None,
    ):
        """Compare the received keys in `config.rope_parameters` against the expected and optional keys"""
        # BC: "rope_type" was originally "type" -- let's check for "rope_type" when "type" is present
        if "type" in received_keys:
            received_keys -= {"type"}
            required_keys.add("rope_type")

        optional_keys = optional_keys or set()
        if "partial_rotary_factor" not in optional_keys:
            optional_keys.add("partial_rotary_factor")

        # Some models need to store model-specific keys, and we don't want to throw warning at them
        if ignore_keys is not None:
            received_keys -= set(ignore_keys)

        missing_keys = required_keys - received_keys
        if missing_keys:
            raise KeyError(f"Missing required keys in `rope_parameters` for 'rope_type'='{rope_type}': {missing_keys}")

        unused_keys = received_keys - required_keys - optional_keys
        if unused_keys:
            logger.warning(f"Unrecognized keys in `rope_parameters` for 'rope_type'='{rope_type}': {unused_keys}")


def rope_config_validation(config: RotaryEmbeddingConfigMixin, ignore_keys: set | None = None):
    """
    This is a deprecated function.
    It has been kept for backward compatibility with custom code models.
    """
    warnings.warn(
        "`rope_config_validation` is deprecated and has been removed. "
        "Its functionality has been moved to RotaryEmbeddingConfigMixin.validate_rope method. "
        "PreTrainedConfig inherits this class, so please call self.validate_rope() instead. "
        "Also, make sure to use the new rope_parameters syntax. "
        "You can call self.standardize_rope_params() in the meantime.",
        FutureWarning,
    )
    config.standardize_rope_params()
    config.validate_rope()


# ── Multimodal (M-RoPE) decoder positions ────────────────────────────────────
# Where a multimodal model's decoder position ids come from: text tokens keep 1D positions while each
# image/video span gets its own axes. `get_mrope_index` is the entry point, `_MROPE_LAYOUT_FUNCTIONS` the
# layouts; a model declares which one it uses (and the knobs it needs) on its config.


def get_mrope_vision_positions(
    start_position: int,
    grid_thw: list[int] | torch.Tensor,
    temporal_merge_size: int = 1,
    spatial_merge_size: int = 1,
    time_interval: float = 1,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """3D M-RoPE positions (temporal, height, width) for one image/video grid, in the *decoder* sequence.

    This is the decoder-side counterpart to [`get_vision_position_ids`] (which produces the vision
    *encoder*'s rotary positions): it lays out a single grid's `(T, H, W)` patches as three position axes,
    each offset by `start_position` (the running position in the token sequence). The temporal axis is
    additionally scaled by `time_interval` (video temporal spacing). Merge sizes downscale the grid the way
    the vision backbone does. Pure function of its arguments — the model passes its config-derived spec.

    Args:
        start_position: running position offset of this grid in the decoder sequence.
        grid_thw: `(3,)` `(T, H, W)` grid of the image/video after patch embedding.
        temporal_merge_size: temporal backbone merge factor (`T // temporal_merge_size`).
        spatial_merge_size: spatial backbone merge factor (`H, W // spatial_merge_size`).
        time_interval: spacing between consecutive temporal position ids. May be fractional, in which case
            the scaled temporal ids are truncated to `dtype`.
        dtype: dtype of the returned positions.
        device: device for the returned tensor.

    Returns:
        `(3, T'*H'*W')` — temporal/height/width position ids, offset by `start_position`.
    """
    dtype = dtype if dtype is not None else torch.long
    llm_grid_t = grid_thw[0].item() // temporal_merge_size
    llm_grid_h = grid_thw[1].item() // spatial_merge_size
    llm_grid_w = grid_thw[2].item() // spatial_merge_size
    position_temporal = (torch.arange(llm_grid_t, device=device) * time_interval).to(dtype)
    position_height = torch.arange(llm_grid_h, dtype=dtype, device=device) + start_position
    position_width = torch.arange(llm_grid_w, dtype=dtype, device=device) + start_position
    t_grid, h_grid, w_grid = torch.meshgrid(position_temporal, position_height, position_width, indexing="ij")
    vision_position_ids = torch.stack([t_grid, h_grid, w_grid], dim=0).reshape(3, -1)
    vision_position_ids[0] += start_position  # temporal offset, after the time_interval scaling
    return vision_position_ids


def uses_mrope(config) -> bool:
    """Whether `config` declares M-RoPE at all — its text rope parameters carry an `mrope_section` (the
    per-axis head split only multi-axis models have). `False` for plain decoders and for VLMs that keep 1D
    text positions (Llava & co.), which is the signal to leave `position_ids` alone."""
    text_config = config.get_text_config()
    rope_parameters = getattr(text_config, "rope_parameters", None)
    if isinstance(rope_parameters, dict) and "mrope_section" in rope_parameters:
        return True
    return "mrope_section" in (getattr(text_config, "ignore_keys_at_rope_validation", None) or ())


def get_mrope_text_positions(
    attention_mask: torch.Tensor,
    num_axes: int = 3,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain 1D positions broadcast over `num_axes` M-RoPE axes: `(position_ids, rope_deltas)`.

    What a multi-axis model falls back to when a sequence has no vision span to lay out — every token
    (including audio ones, which carry no spatial axes of their own) just counts up, padded slots keeping
    position 1.
    """
    dtype = dtype if dtype is not None else torch.long
    position_ids = attention_mask.to(dtype).cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    position_ids = position_ids.unsqueeze(0).expand(num_axes, -1, -1)
    max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
    return position_ids, max_position_ids + 1 - attention_mask.sum(dim=-1, keepdim=True)


def _mrope_place_positions(
    input_ids: torch.LongTensor,
    mm_token_type_ids: torch.IntTensor | None,
    num_axes: int,
    attention_mask: torch.Tensor | None,
    positions_for_sequence,
    dtype: torch.dtype | None = None,
    padded_position: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared frame every M-RoPE layout sits in: `(position_ids, rope_deltas)`.

    Per batch row it drops the padded tokens, asks `positions_for_sequence(token_ids, token_types)` for that
    sequence's `(num_axes, seq)` positions — which is where the layouts differ — scatters them back onto the
    unpadded slots, and records the delta `generate` advances decode positions by. `dtype` defaults to
    `input_ids`' (a layout that lays out fractional temporal positions asks for a float one) and
    `padded_position` is what the masked-out slots keep.
    """
    position_ids = torch.full(
        (num_axes, input_ids.shape[0], input_ids.shape[1]),
        padded_position,
        dtype=dtype or input_ids.dtype,
        device=input_ids.device,
    )
    rope_deltas = []
    for batch_idx, token_ids in enumerate(input_ids):
        token_types = mm_token_type_ids[batch_idx] if mm_token_type_ids is not None else None
        valid_tokens = None
        if attention_mask is not None:
            valid_tokens = attention_mask[batch_idx].bool()
            token_ids = token_ids[valid_tokens]
            token_types = token_types[valid_tokens] if token_types is not None else None

        positions = positions_for_sequence(token_ids, token_types)
        positions = positions.to(device=position_ids.device, dtype=position_ids.dtype)

        if valid_tokens is not None:
            position_ids[:, batch_idx, valid_tokens] = positions
        else:
            position_ids[:, batch_idx] = positions
        rope_deltas.append(positions.max() + 1 - len(token_ids))
    return position_ids, torch.tensor(rope_deltas, device=input_ids.device).unsqueeze(1)


def _mrope_index_interleaved_runs(
    input_ids: torch.LongTensor,
    mm_token_type_ids: torch.IntTensor,
    *,
    spatial_merge_size: int,
    attention_mask: torch.Tensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    second_per_grid_ts: torch.Tensor | None = None,
    tokens_per_second: float | None = None,
    split_video_frames: bool = False,
    video_temporal_merge_size: int = 1,
    **unused,
) -> tuple[torch.Tensor, torch.Tensor]:
    """3-axis M-RoPE laid out run by run: `(position_ids, mrope_deltas)`.

    Text runs use standard 1D positions (broadcast across the 3 axes); image/video runs use 3D
    (temporal, height, width) positions from [`get_mrope_vision_positions`]. Runs are delimited by
    `mm_token_type_ids` (0=text, 1=image, 2=video). A pure function driven entirely by the caller's spec —
    each VLM's `get_rope_index` becomes a thin wrapper passing its config-derived values, so the layout
    lives here instead of being copied per model.

    Args:
        input_ids: `(batch, seq)` token ids.
        mm_token_type_ids: `(batch, seq)` per-token modality marker (0=text, 1=image, 2=video).
        spatial_merge_size: spatial backbone merge factor.
        attention_mask: `(batch, seq)` padding mask; positions are placed on the unpadded tokens.
        image_grid_thw: `(num_images, 3)` grids, consumed in order for image runs.
        video_grid_thw: `(num_videos, 3)` grids, consumed in order for video runs.
        second_per_grid_ts: `(num_videos,)` per-video seconds-per-temporal-grid.
        tokens_per_second: when set, video temporal spacing scales by `tokens_per_second *
            second_per_grid_ts`; otherwise consecutive temporal ids are 1 apart.
        split_video_frames: expand each video grid to one `T=1` row per frame before placement, for
            processors that separate frames with timestamps.
        video_temporal_merge_size: temporal backbone merge factor, applied to video grids only.

    Returns:
        position_ids: `(3, batch, seq)` long.
        mrope_position_deltas: `(batch, 1)` — `max_position + 1 - unpadded_len` per batch row.
    """
    if split_video_frames and video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    grid_iters = {
        1: iter(image_grid_thw) if image_grid_thw is not None else None,
        2: iter(video_grid_thw) if video_grid_thw is not None else None,
    }
    second_per_grid_iter = (
        iter(second_per_grid_ts) if second_per_grid_ts is not None else iter([1] * input_ids.shape[1])
    )

    def positions_for_sequence(token_ids, token_types):
        input_type_group = []
        for key, group in itertools.groupby(enumerate(token_types.tolist()), lambda x: x[1]):
            group = list(group)
            start_index = group[0][0]
            end_index = group[-1][0] + 1
            input_type_group.append((key, start_index, end_index))

        current_pos = 0
        llm_pos_ids_list = []
        for modality_type, start_idx, end_idx in input_type_group:
            if modality_type == 0:
                text_len = end_idx - start_idx
                llm_pos_ids_list.append(_mrope_positions_block(text_len, current_pos, device=token_ids.device))
                current_pos += text_len
            else:
                grid_thw = next(grid_iters[modality_type])
                if modality_type == 2 and tokens_per_second is not None:
                    time_interval = tokens_per_second * int(next(second_per_grid_iter))
                else:
                    time_interval = 1
                vision_position_ids = get_mrope_vision_positions(
                    current_pos,
                    grid_thw,
                    temporal_merge_size=video_temporal_merge_size if modality_type == 2 else 1,
                    spatial_merge_size=spatial_merge_size,
                    time_interval=time_interval,
                    device=token_ids.device,
                )
                llm_pos_ids_list.append(vision_position_ids)
                current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size
        return torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)

    return _mrope_place_positions(input_ids, mm_token_type_ids, 3, attention_mask, positions_for_sequence)


def _mrope_index_indexed_images(
    input_ids: torch.LongTensor,
    mm_token_type_ids: torch.IntTensor,
    *,
    num_axes: int,
    spatial_merge_size: int,
    attention_mask: torch.Tensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    **unused,
) -> tuple[torch.Tensor, torch.Tensor]:
    """M-RoPE over a per-image ordinal instead of a temporal axis: `(position_ids, rope_deltas)`.

    HunYuanVL's layout. Every token starts from plain 1D positions (a text baseline that is *overwritten*,
    not accumulated) and each image span replaces its slice with `(width, height, image_index)` on the last
    three of `mrope_section`'s axes — earlier axes keep the 1D positions. The pooled image grid carries one
    newline-style token per row, so the width channel spans `w + 1` (see `get_mrope_image_positions`), and a
    span may include the two image-boundary tokens.

    Returns:
        position_ids: `(num_axes, batch, seq)` long — unlike the other layouts this is not fixed at 3 axes,
            since `mrope_section` decides how many there are and only the last 3 carry the image.
        rope_deltas: `(batch, 1)` — `max_position + 1 - unpadded_len` per batch row.
    """
    if num_axes < 3:
        raise ValueError(f"An indexed-image M-RoPE layout needs at least 3 axes, got {num_axes}.")

    grid_iter = iter(image_grid_thw) if image_grid_thw is not None else None
    image_index = 0

    def positions_for_sequence(token_ids, token_types):
        nonlocal image_index
        current_position_ids = torch.arange(token_ids.shape[-1], dtype=input_ids.dtype, device=token_ids.device)
        current_position_ids = current_position_ids.view(1, -1).expand(num_axes, -1).clone()
        if grid_iter is None:
            return current_position_ids

        for modality_type, group in itertools.groupby(enumerate(token_types.tolist()), lambda x: x[1]):
            if modality_type != 1:  # image == 1; this layout has no video/audio modality
                continue
            group = list(group)
            span_start, span_end = group[0][0], group[-1][0] + 1
            try:
                grid_thw = next(grid_iter)
            except StopIteration as error:
                raise ValueError("Found more image placeholder spans than entries in `image_grid_thw`.") from error

            vision_position_ids = get_mrope_image_positions(
                grid_thw[1:], spatial_merge_size=spatial_merge_size, device=token_ids.device
            )
            grid_tokens = vision_position_ids.shape[1]
            span_length = span_end - span_start
            if span_length == grid_tokens + 2:  # span includes the image-boundary tokens
                grid_start = span_start + 1
            elif span_length == grid_tokens:
                grid_start = span_start
            else:
                raise ValueError(
                    "Image placeholder span length does not match `image_grid_thw`: "
                    f"span_length={span_length}, expected {grid_tokens} or {grid_tokens + 2}."
                )

            grid_end = grid_start + grid_tokens
            offset = num_axes - 3
            current_position_ids[offset : offset + 2, grid_start:grid_end] = vision_position_ids.to(
                dtype=input_ids.dtype
            )
            current_position_ids[offset + 2, grid_start:grid_end] = image_index
            image_index += 1
        return current_position_ids

    position_ids, rope_deltas = _mrope_place_positions(
        input_ids, mm_token_type_ids, num_axes, attention_mask, positions_for_sequence
    )
    if image_grid_thw is not None and image_index != len(image_grid_thw):
        raise ValueError(
            "Number of image placeholder spans does not match `image_grid_thw`: "
            f"spans={image_index}, images={len(image_grid_thw)}."
        )
    return position_ids, rope_deltas


def get_mrope_image_positions(
    grid_hw: list[int] | torch.Tensor,
    spatial_merge_size: int = 1,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """`(2, tokens)` `(width, height)` indices for one pooled image grid, for the indexed-image layout.

    The vision merger appends one newline-style token per image row, so the width channel spans `w + 1`
    positions while the height channel repeats each row id over that extra column.
    """
    grid_h, grid_w = (int(value) for value in grid_hw)
    llm_grid_h = grid_h // spatial_merge_size
    llm_grid_w = grid_w // spatial_merge_size
    position_height, position_width = torch.meshgrid(
        torch.arange(llm_grid_h, dtype=torch.long, device=device),
        torch.arange(llm_grid_w + 1, dtype=torch.long, device=device),
        indexing="ij",
    )
    return torch.stack([position_width.flatten(), position_height.flatten()], dim=0)


def _mrope_positions_block(
    length: int | torch.Tensor,
    start_position: int | torch.Tensor,
    num_axes: int = 3,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """`(num_axes, length)` block of consecutive positions from `start_position`, the same on every axis.

    What a text run — or an audio span, which is 1D in time — contributes to a multi-axis layout.
    """
    dtype = dtype if dtype is not None else torch.long
    return torch.arange(int(length), dtype=dtype, device=device).view(1, -1).expand(num_axes, -1) + start_position


def _mrope_temporal_chunks(
    temporal_positions: torch.Tensor, positions_per_chunk: int, start_position: int | torch.Tensor
) -> list[tuple[int, int]]:
    """`(start, end)` slices cutting a monotonic temporal axis into successive `positions_per_chunk` ranges.

    Used to interleave a video and its own audio track in time: both are cut at the same chunk boundaries
    (`position_id_per_seconds * seconds_per_chunk` positions ≈ one chunk of wall-clock time), then emitted
    chunk by chunk. `start_position` is the span's own origin, subtracted before chunking.
    """
    chunks = []
    chunk_start, chunk_index = 0, 1
    for index in range(len(temporal_positions)):
        if temporal_positions[index] - start_position >= chunk_index * positions_per_chunk:
            chunks.append((chunk_start, index))
            chunk_start = index
            chunk_index += 1
    chunks.append((chunk_start, len(temporal_positions)))
    return chunks


def _mrope_index_audio_chunked(
    input_ids: torch.LongTensor,
    mm_token_type_ids: torch.IntTensor | None = None,
    *,
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    audio_token_id: int,
    vision_start_token_id: int,
    audio_start_token_id: int,
    position_id_per_seconds: float,
    seconds_per_chunk: float,
    attention_mask: torch.Tensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    second_per_grid_ts: torch.Tensor | None = None,
    audio_seqlens: torch.LongTensor | None = None,
    use_audio_in_video: bool = False,
    **unused,
) -> tuple[torch.Tensor, torch.Tensor]:
    """3-axis M-RoPE over image, video **and audio** spans, audio-in-video interleaved in time chunks.

    Qwen2.5-Omni's layout. Spans are located by scanning `input_ids` for placeholder tokens rather than by
    grouping `mm_token_type_ids` (which this family does not produce): at each step the *nearest* upcoming
    image/video/audio placeholder wins, and its span contributes a one-position `bos` block, the modality's
    own positions, and a one-position `eos` block, each starting one past the previous block's maximum.
    Audio is 1D in time, images and videos are 3D (see [`get_mrope_vision_positions`]), and the temporal axis
    counts `position_id_per_seconds` positions per second of media. With `use_audio_in_video`, a video and
    its soundtrack share one span: both are cut into `seconds_per_chunk` chunks by
    [`_mrope_temporal_chunks`] and emitted video-chunk-then-audio-chunk, wrapped in doubled bos/eos blocks.

    Args:
        input_ids: `(batch, seq)` token ids.
        mm_token_type_ids: unused — spans come from the placeholder token ids.
        spatial_merge_size: spatial backbone merge factor.
        image_token_id, video_token_id, audio_token_id: placeholder token ids marking each modality's span.
        vision_start_token_id, audio_start_token_id: the tokens opening a vision/audio span, used to count
            the spans in the sequence.
        position_id_per_seconds: temporal positions per second of media.
        seconds_per_chunk: audio-in-video interleaving granularity, in seconds.
        attention_mask: `(batch, seq)` padding mask; positions are placed on the unpadded tokens.
        image_grid_thw: `(num_images, 3)` grids, consumed in order for image spans.
        video_grid_thw: `(num_videos, 3)` grids, consumed in order for video spans.
        second_per_grid_ts: `(num_videos,)` per-video seconds-per-temporal-grid.
        audio_seqlens: `(num_audios,)` mel-frame lengths, consumed in order for audio spans.
        use_audio_in_video: lay a video out interleaved with its own audio track.

    Returns:
        position_ids: `(3, batch, seq)` long, masked-out slots left at 1.
        mrope_position_deltas: `(batch, 1)` — `max_position + 1 - unpadded_len` per batch row.
    """
    image_idx, video_idx, audio_idx = 0, 0, 0

    def positions_for_sequence(token_ids, _token_types):
        nonlocal image_idx, video_idx, audio_idx
        device = token_ids.device
        input_tokens = token_ids.tolist()
        blocks = []

        def next_position():
            return blocks[-1].max() + 1 if blocks else 0

        def block(length, start_position=None):
            start_position = next_position() if start_position is None else start_position
            return _mrope_positions_block(length, start_position, device=device)

        vision_start_indices = torch.argwhere(token_ids == vision_start_token_id).squeeze(1)
        vision_tokens = token_ids[vision_start_indices + 1]
        num_audios = (token_ids == audio_start_token_id).sum()
        num_images = (vision_tokens == image_token_id).sum()
        num_videos = (
            (vision_tokens == audio_start_token_id).sum()
            if use_audio_in_video
            else (vision_tokens == video_token_id).sum()
        )
        remain_images, remain_videos, remain_audios = num_images, num_videos, num_audios
        num_spans = num_images + num_audios if use_audio_in_video else num_images + num_videos + num_audios

        st = 0
        for _ in range(num_spans):
            unreachable = len(input_tokens) + 1
            ed_image = (
                input_tokens.index(image_token_id, st)
                if image_token_id in input_tokens and remain_images > 0
                else unreachable
            )
            ed_video = (
                input_tokens.index(video_token_id, st)
                if video_token_id in input_tokens and remain_videos > 0
                else unreachable
            )
            ed_audio = (
                input_tokens.index(audio_token_id, st)
                if audio_token_id in input_tokens and remain_audios > 0
                else unreachable
            )
            min_ed = min(ed_image, ed_video, ed_audio)
            # the span's own opening token sits between the text run and the placeholders (two of them for
            # audio-in-video, which opens with both a vision and an audio start token)
            text_len = min_ed - st - (2 if min_ed == ed_video and use_audio_in_video else 1)
            if text_len != 0:
                blocks.append(block(text_len))
            bos_len = eos_len = 1

            if min_ed == ed_audio:
                blocks.append(block(bos_len))
                audio_len = _mrope_audio_length_pooled(audio_seqlens[audio_idx])
                blocks.append(block(audio_len))
                blocks.append(block(eos_len))
                st += int(text_len + bos_len + audio_len + eos_len)
                audio_idx += 1
                remain_audios -= 1

            elif min_ed == ed_image:
                blocks.append(block(bos_len))
                grid_thw = image_grid_thw[image_idx]
                blocks.append(
                    get_mrope_vision_positions(
                        next_position(),
                        grid_thw,
                        spatial_merge_size=spatial_merge_size,
                        time_interval=position_id_per_seconds,
                        device=device,
                    )
                )
                image_len = grid_thw.prod() // (spatial_merge_size**2)
                blocks.append(block(eos_len))
                st += int(text_len + bos_len + image_len + eos_len)
                image_idx += 1
                remain_images -= 1

            elif min_ed == ed_video and not use_audio_in_video:
                blocks.append(block(bos_len))
                grid_thw = video_grid_thw[video_idx]
                blocks.append(
                    get_mrope_vision_positions(
                        next_position(),
                        grid_thw,
                        spatial_merge_size=spatial_merge_size,
                        time_interval=float(second_per_grid_ts[video_idx]) * position_id_per_seconds,
                        device=device,
                    )
                )
                video_len = grid_thw.prod() // (spatial_merge_size**2)
                blocks.append(block(eos_len))
                st += int(text_len + bos_len + video_len + eos_len)
                video_idx += 1
                remain_videos -= 1

            elif min_ed == ed_video and use_audio_in_video:
                bos_position = next_position()
                blocks.append(block(bos_len, bos_position))
                blocks.append(block(bos_len, bos_position))

                span_start = next_position()
                grid_thw = video_grid_thw[video_idx]
                audio_len = _mrope_audio_length_pooled(audio_seqlens[audio_idx])
                audio_positions = block(audio_len, span_start)
                video_positions = get_mrope_vision_positions(
                    span_start,
                    grid_thw,
                    spatial_merge_size=spatial_merge_size,
                    time_interval=float(second_per_grid_ts[video_idx]) * position_id_per_seconds,
                    device=device,
                )
                positions_per_chunk = int(position_id_per_seconds * seconds_per_chunk)
                video_chunks = _mrope_temporal_chunks(video_positions[0], positions_per_chunk, span_start)
                audio_chunks = _mrope_temporal_chunks(audio_positions[0], positions_per_chunk, span_start)
                for video_chunk, audio_chunk in itertools.zip_longest(video_chunks, audio_chunks):
                    if video_chunk is not None:
                        blocks.append(video_positions[:, video_chunk[0] : video_chunk[1]])
                    if audio_chunk is not None:
                        blocks.append(audio_positions[:, audio_chunk[0] : audio_chunk[1]])
                video_len = grid_thw.prod() // (spatial_merge_size**2)

                eos_position = next_position()
                blocks.append(block(eos_len, eos_position))
                blocks.append(block(eos_len, eos_position))
                st += int(text_len + 2 * bos_len + audio_len + video_len + 2 * eos_len)
                audio_idx += 1
                video_idx += 1
                remain_videos -= 1
                remain_audios -= 1

        if st < len(input_tokens):
            blocks.append(block(len(input_tokens) - st))
        return torch.cat(blocks, dim=1).reshape(3, -1)

    return _mrope_place_positions(
        input_ids, mm_token_type_ids, 3, attention_mask, positions_for_sequence, padded_position=1
    )


def _mrope_index_audio_merged(
    input_ids: torch.LongTensor,
    mm_token_type_ids: torch.IntTensor | None = None,
    *,
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    audio_token_id: int,
    vision_start_token_id: int,
    audio_start_token_id: int,
    position_id_per_seconds: float,
    audio_window_size: int,
    attention_mask: torch.Tensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    second_per_grid_ts: torch.Tensor | None = None,
    audio_seqlens: torch.LongTensor | None = None,
    use_audio_in_video: bool = False,
    **unused,
) -> tuple[torch.Tensor, torch.Tensor]:
    """3-axis M-RoPE over image, video and audio spans, audio-in-video merged position by position.

    Qwen3-Omni's layout. Like [`_mrope_index_audio_chunked`] it locates spans by scanning `input_ids`, but
    it scans for the *opening* tokens (`vision_start_token_id` / `audio_start_token_id`) — so the text run
    ends where the opening token starts, and positions advance by counting emitted tokens instead of
    re-reading the previous block's maximum. Audio-in-video is merged position by position (whichever of the
    two streams is earlier in time goes next) rather than in chunks, and its span is wrapped in two-position
    bos/eos blocks. Positions are float, since a fractional `second_per_grid_ts` leaves the temporal axis
    fractional here.

    Args:
        input_ids: `(batch, seq)` token ids.
        mm_token_type_ids: unused — spans come from the opening token ids.
        spatial_merge_size: spatial backbone merge factor.
        image_token_id, video_token_id, audio_token_id: placeholder token ids marking each modality's span.
        vision_start_token_id, audio_start_token_id: the tokens opening a vision/audio span.
        position_id_per_seconds: temporal positions per second of media.
        audio_window_size: audio encoder window (`audio_config.n_window`), which sets how many decoder
            tokens an audio of a given mel length takes.
        attention_mask: `(batch, seq)` padding mask; positions are placed on the unpadded tokens.
        image_grid_thw: `(num_images, 3)` grids, consumed in order for image spans.
        video_grid_thw: `(num_videos, 3)` grids, consumed in order for video spans.
        second_per_grid_ts: `(num_videos,)` per-video seconds-per-temporal-grid.
        audio_seqlens: `(num_audios,)` mel-frame lengths, consumed in order for audio spans.
        use_audio_in_video: lay a video out interleaved with its own audio track.

    Returns:
        position_ids: `(3, batch, seq)` float.
        mrope_position_deltas: `(batch, 1)` — `max_position + 1 - unpadded_len` per batch row.
    """
    image_idx, video_idx, audio_idx = 0, 0, 0

    def positions_for_sequence(token_ids, _token_types):
        nonlocal image_idx, video_idx, audio_idx
        device = token_ids.device
        input_tokens = token_ids.tolist()
        blocks = []

        def block(length, start_position):
            return _mrope_positions_block(length, start_position, dtype=torch.float, device=device)

        vision_start_indices = torch.argwhere(token_ids == vision_start_token_id).squeeze(1)
        vision_tokens = token_ids[vision_start_indices + 1]
        num_audios = (token_ids == audio_start_token_id).sum()
        num_images = (vision_tokens == image_token_id).sum()
        num_videos = (
            (vision_tokens == audio_start_token_id).sum()
            if use_audio_in_video
            else (vision_tokens == video_token_id).sum()
        )
        remain_images, remain_videos, remain_audios = num_images, num_videos, num_audios
        num_spans = num_images + num_audios if use_audio_in_video else num_images + num_videos + num_audios

        st = 0
        for _ in range(num_spans):
            start_position = blocks[-1].max() + 1 if blocks else 0
            unreachable = len(input_tokens) + 1
            ed_vision_start = (
                input_tokens.index(vision_start_token_id, st)
                if (image_token_id in input_tokens or video_token_id in input_tokens)
                and (remain_videos > 0 or remain_images > 0)
                else unreachable
            )
            ed_audio_start = (
                input_tokens.index(audio_start_token_id, st)
                if audio_token_id in input_tokens and remain_audios > 0
                else unreachable
            )
            min_ed = min(ed_vision_start, ed_audio_start)

            text_len = min_ed - st
            if text_len != 0:
                blocks.append(block(text_len, start_position))
                start_position += text_len

            audio_in_video = min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start
            bos_len = eos_len = 2 if audio_in_video else 1
            blocks.append(block(bos_len, start_position))
            start_position += bos_len

            if min_ed == ed_audio_start:
                audio_len = _mrope_audio_length_windowed(audio_seqlens[audio_idx], audio_window_size)
                blocks.append(block(audio_len, start_position))
                st += int(text_len + bos_len + audio_len + eos_len)
                audio_idx += 1
                remain_audios -= 1

            elif min_ed == ed_vision_start and token_ids[ed_vision_start + 1] == image_token_id:
                grid_thw = image_grid_thw[image_idx]
                blocks.append(
                    get_mrope_vision_positions(
                        start_position,
                        grid_thw,
                        spatial_merge_size=spatial_merge_size,
                        time_interval=position_id_per_seconds,
                        dtype=torch.float,
                        device=device,
                    )
                )
                st += int(text_len + bos_len + grid_thw.prod() // (spatial_merge_size**2) + eos_len)
                image_idx += 1
                remain_images -= 1

            elif min_ed == ed_vision_start and token_ids[ed_vision_start + 1] == video_token_id:
                grid_thw = video_grid_thw[video_idx]
                blocks.append(
                    get_mrope_vision_positions(
                        start_position,
                        grid_thw,
                        spatial_merge_size=spatial_merge_size,
                        time_interval=float(second_per_grid_ts[video_idx]) * position_id_per_seconds,
                        dtype=torch.float,
                        device=device,
                    )
                )
                st += int(text_len + bos_len + grid_thw.prod() // (spatial_merge_size**2) + eos_len)
                video_idx += 1
                remain_videos -= 1

            elif audio_in_video:
                grid_thw = video_grid_thw[video_idx]
                audio_len = _mrope_audio_length_windowed(audio_seqlens[audio_idx], audio_window_size)
                audio_positions = block(audio_len, start_position)
                video_positions = get_mrope_vision_positions(
                    start_position,
                    grid_thw,
                    spatial_merge_size=spatial_merge_size,
                    time_interval=float(second_per_grid_ts[video_idx]) * position_id_per_seconds,
                    dtype=torch.float,
                    device=device,
                )
                # merge the two streams by temporal position, one token at a time, video first on a tie
                video_pos, audio_pos = 0, 0
                while video_pos < video_positions.shape[-1] and audio_pos < audio_positions.shape[-1]:
                    if video_positions[0][video_pos] <= audio_positions[0][audio_pos]:
                        blocks.append(video_positions[:, video_pos : video_pos + 1])
                        video_pos += 1
                    else:
                        blocks.append(audio_positions[:, audio_pos : audio_pos + 1])
                        audio_pos += 1
                if video_pos < video_positions.shape[-1]:
                    blocks.append(video_positions[:, video_pos:])
                if audio_pos < audio_positions.shape[-1]:
                    blocks.append(audio_positions[:, audio_pos:])
                video_len = grid_thw.prod() // (spatial_merge_size**2)
                st += int(text_len + bos_len + audio_len + video_len + eos_len)
                audio_idx += 1
                video_idx += 1
                remain_videos -= 1
                remain_audios -= 1

            blocks.append(block(eos_len, blocks[-1].max() + 1))

        if st < len(input_tokens):
            blocks.append(block(len(input_tokens) - st, blocks[-1].max() + 1 if blocks else 0))
        return torch.cat(blocks, dim=1).reshape(3, -1)

    return _mrope_place_positions(
        input_ids, mm_token_type_ids, 3, attention_mask, positions_for_sequence, dtype=torch.float
    )


def _mrope_audio_length_pooled(audio_seqlen: torch.Tensor | int) -> torch.Tensor | int:
    """Decoder tokens one audio takes in the chunked layout: two stride-2 convs, then a stride-2 pooler."""
    return ((audio_seqlen - 1) // 2 + 1 - 2) // 2 + 1


def _mrope_audio_length_windowed(audio_seqlen: torch.Tensor | int, window_size: int) -> torch.Tensor | int:
    """Decoder tokens one audio takes in the merged layout: three stride-2 convs over fixed-size windows,
    each full `2 * window_size`-frame window collapsing to 13 tokens."""
    chunk_len = window_size * 2
    feat_len = (audio_seqlen % chunk_len - 1) // 2 + 1
    return ((feat_len - 1) // 2 + 1 - 1) // 2 + 1 + (audio_seqlen // chunk_len) * 13


# Layout name -> implementation, the way `ACT2FN` maps an activation name. A model declares which one it
# uses with `config.mrope_layout`; a family whose positions are laid out differently adds its function here
# (and names it in its config) rather than growing an existing layout. Every layout is a pure function of
# `(config, inputs)`, which is what lets a model's `get_rope_index` be a one-line wrapper AND lets the
# exporters compute positions from a saved config with no model instance.
_MROPE_LAYOUT_FUNCTIONS: dict[str, callable] = {
    "interleaved_runs": _mrope_index_interleaved_runs,
    "indexed_images": _mrope_index_indexed_images,
    "audio_chunked": _mrope_index_audio_chunked,
    "audio_merged": _mrope_index_audio_merged,
}


def get_mrope_index(
    config,
    input_ids: torch.LongTensor,
    mm_token_type_ids: torch.IntTensor | None = None,
    *,
    attention_mask: torch.Tensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    second_per_grid_ts: torch.Tensor | None = None,
    audio_seqlens: torch.LongTensor | None = None,
    use_audio_in_video: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """M-RoPE decoder positions for a `vision + text` sequence: `(position_ids, rope_deltas)`.

    Runs the layout `config.mrope_layout` declares, with that layout's knobs read off the config here — the
    one place config is turned into M-RoPE parameters, so it works from a saved config with no model
    instance (a VLM's `get_rope_index` is a one-line wrapper over this, and the exporters drive exported
    VLMs through the very same call). The layouts themselves (`_MROPE_LAYOUT_FUNCTIONS`) take plain values,
    so they stay usable and testable on their own.

    Args:
        config: the model config, declaring `mrope_layout` and its layout's knobs.
        input_ids: `(batch, seq)` token ids.
        mm_token_type_ids: `(batch, seq)` per-token modality marker (0=text, 1=image, 2=video); layouts that
            locate their spans by placeholder token id instead do not need it.
        image_grid_thw: `(num_images, 3)` grids, consumed in order for image spans.
        video_grid_thw: `(num_videos, 3)` grids, consumed in order for video spans.
        attention_mask: `(batch, seq)` padding mask; positions are placed on the unpadded tokens.
        second_per_grid_ts: `(num_videos,)` per-video seconds-per-temporal-grid.
        audio_seqlens: `(num_audios,)` mel-frame lengths, consumed in order for audio spans.
        use_audio_in_video: lay each video out interleaved with its own audio track.

    A layout ignores the tensors it has no use for (an image-only layout takes no video grids).

    Returns:
        position_ids: `(num_axes, batch, seq)` — 3 axes for every layout except `indexed_images`, which
            takes its axis count from `mrope_section`; long, except layouts whose temporal axis is
            fractional (`audio_merged`), which return float.
        rope_deltas: `(batch, 1)` — `max_position + 1 - unpadded_len` per batch row, in every layout.
    """
    layout_name = getattr(config, "mrope_layout", None)
    layout = _MROPE_LAYOUT_FUNCTIONS.get(layout_name)
    if layout is None:
        raise NotImplementedError(
            f"`{getattr(config, 'model_type', type(config).__name__)}` declares "
            f"`mrope_layout={layout_name!r}`, which is not implemented here (known layouts: "
            f"{sorted(_MROPE_LAYOUT_FUNCTIONS)}). Its own `get_rope_index` is the reference: port it to a "
            "`_MROPE_LAYOUT_FUNCTIONS` entry (a pure function of its inputs) and name it in the config."
        )
    vision_config = getattr(config, "vision_config", config)
    audio_config = getattr(config, "audio_config", config)
    rope_parameters = getattr(config.get_text_config(), "rope_parameters", None) or {}
    return layout(
        input_ids,
        mm_token_type_ids,
        spatial_merge_size=vision_config.spatial_merge_size,
        tokens_per_second=getattr(vision_config, "tokens_per_second", None),
        video_temporal_merge_size=getattr(vision_config, "temporal_merge_size", None) or 1,
        split_video_frames=getattr(vision_config, "timestamped_video_frames", False),
        num_axes=len(rope_parameters.get("mrope_section", [])),
        image_token_id=getattr(config, "image_token_id", None),
        video_token_id=getattr(config, "video_token_id", None),
        audio_token_id=getattr(config, "audio_token_id", None),
        vision_start_token_id=getattr(config, "vision_start_token_id", None),
        audio_start_token_id=getattr(config, "audio_start_token_id", None),
        position_id_per_seconds=getattr(config, "position_id_per_seconds", None),
        seconds_per_chunk=getattr(config, "seconds_per_chunk", None),
        audio_window_size=getattr(audio_config, "n_window", None),
        attention_mask=attention_mask,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        audio_seqlens=audio_seqlens,
        use_audio_in_video=use_audio_in_video,
    )
