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

import math
from functools import wraps
from typing import Optional, TypedDict

from .configuration_utils import PreTrainedConfig
from .utils import is_torch_available, logging


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch


def standardize_rope_params(config, rope_theta: float | dict[str, float] | None = None):
    """
    Helper to standardize the config's rope params field by ensuring the params are defined for each
    later type. For old model the fn will duplicate a single rope param in each layer type (backward compatibility)
    """
    rope_parameters = getattr(config, "rope_parameters", None)
    layer_types = getattr(config, "layer_types", None)
    if rope_theta is None:
        rope_theta = getattr(config, "rope_theta", None)

    # Case 1: one RoPE theat = one RoPE param per model without nesting
    if not isinstance(rope_theta, dict):
        if rope_parameters is None:
            rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}
        else:
            # BC: if there is a 'type' field, copy it it to 'rope_type'.
            rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
            rope_theta = rope_parameters.get("rope_theta") or rope_theta
            rope_parameters.update({"rope_theta": rope_theta, "rope_type": rope_type})
        config.rope_parameters = rope_parameters

    # Case 2: different RoPE for each layer as nested dict
    else:
        rope_parameters_per_layer_type = {}
        for layer_type in layer_types:
            if rope_parameters is None:
                rope_parameters_per_layer_type[layer_type] = {
                    "rope_type": "default",
                    "rope_theta": rope_theta[layer_type],
                }
            else:
                is_field_in_new_format = any(layer_type in rope_parameters for layer_type in layer_types)
                if not is_field_in_new_format:
                    curr_rope_type = rope_parameters.get("rope_type", rope_parameters.get("type"))
                    rope_parameters_per_layer_type[layer_type] = {
                        **rope_parameters,
                        "rope_type": curr_rope_type,
                        "rope_theta": rope_theta[layer_type],
                    }
                else:
                    curr_rope_type = rope_parameters[layer_type].get(
                        "rope_type", rope_parameters[layer_type].get("type")
                    )
                    rope_parameters_per_layer_type[layer_type] = {
                        **rope_parameters[layer_type],
                        "rope_type": curr_rope_type,
                        "rope_theta": rope_theta[layer_type],
                    }
            config.rope_parameters = rope_parameters_per_layer_type


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
        original_max_position_embeddings = getattr(
            self.config, "original_max_position_embeddings", self.config.max_position_embeddings
        )
        if layer_type is None:
            rope_type = self.rope_type
            original_inv_freq = self.original_inv_freq
            prefix = ""
        else:
            rope_type = self.rope_type[layer_type]
            original_inv_freq = getattr(self, f"{layer_type}_original_inv_freq")
            prefix = f"{layer_type}_"

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
            setattr(self, f"{layer_type}_max_seq_len_cached", seq_len)

        if seq_len < self.original_max_seq_len and max_seq_len_cached > self.original_max_seq_len:  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            original_inv_freq = original_inv_freq.to(device)
            self.register_buffer(f"{prefix}inv_freq", original_inv_freq, persistent=False)
            setattr(self, f"{prefix}original_inv_freq", original_inv_freq)
            setattr(self, f"{layer_type}_max_seq_len_cached", self.original_max_seq_len)

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
    config: Optional[PreTrainedConfig] = None,
    device: Optional["torch.device"] = None,
    seq_len: Optional[int] = None,
    layer_type: Optional[str] = None,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies with linear scaling. Credits to the Reddit user /u/kaiokendev
    Args:
        config ([`~transformers.PreTrainedConfig`]):
            The model configuration. This function assumes that the config will provide at least the following
            properties:

            *   rope_theta (`float`): The base wavelength from which the inverse frequencies will be derived.
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
    # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
    standardize_rope_params(config)
    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters
    factor = rope_parameters_dict["factor"]

    # Gets the default RoPE parameters
    base = rope_parameters_dict["rope_theta"]
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
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


def _compute_dynamic_ntk_parameters(
    config: Optional[PreTrainedConfig] = None,
    device: Optional["torch.device"] = None,
    seq_len: Optional[int] = None,
    layer_type: Optional[str] = None,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies with NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla

    Args:
        config ([`~transformers.PreTrainedConfig`]):
            The model configuration. This function assumes that the config will provide at least the following
            properties:

            *   rope_theta (`float`): The base wavelength from which the inverse frequencies will be derived.
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
    # TODO (joao): use the new `original_max_position_embeddings` from rope_parameters
    # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
    standardize_rope_params(config)
    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters

    base = rope_parameters_dict["rope_theta"]
    partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    max_position_embeddings = config.max_position_embeddings
    factor = rope_parameters_dict["factor"]
    attention_factor = 1.0  # Unused in this type of RoPE

    # seq_len: default to max_position_embeddings, e.g. at init time
    if seq_len is None:
        seq_len = max_position_embeddings
    elif isinstance(seq_len, torch.Tensor):
        seq_len = torch.maximum(
            seq_len,
            torch.tensor(max_position_embeddings, dtype=seq_len.dtype, device=seq_len.device),
        )
    else:
        seq_len = max(seq_len, max_position_embeddings)

    # Compute the inverse frequencies
    base = base * ((factor * seq_len / max_position_embeddings) - (factor - 1)) ** (dim / (dim - 2))
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, attention_factor


def _compute_yarn_parameters(
    config: PreTrainedConfig,
    device: "torch.device",
    seq_len: Optional[int] = None,
    layer_type: Optional[str] = None,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies with NTK scaling. Please refer to the
    [original paper](https://huggingface.co/papers/2309.00071)

    Args:
        config ([`~transformers.PreTrainedConfig`]):
            The model configuration. This function assumes that the config will provide at least the following
            properties:

            *   rope_theta (`float`): The base wavelength from which the inverse frequencies will be derived.
            *   hidden_size (`int`): The numerator when deriving a head_dim, if not provided directly.
            *   num_attention_heads (`int`): The denominator when deriving a head_dim, if not provided directly.
            *   max_position_embeddings (`int`): The maximum length of the positional embeddings.
            *   rope_parameters (`dict[str, float | int]`): The standard RoPE scaling parameters, from which the following
                keys will be accessed:
                *   `attention_factor` (`float`, *optional*): The scaling factor to be applied to the computed cos/sin.
                    If None, the value is inferred from `factor`, `mscale`, and `mscale_all_dim` as avaialble.
                *   `beta_fast` (`float`, *optional*, defaults to 32): Parameter to set the boundary for extrapolation
                    (only) in the linear ramp function.
                *   `beta_slow` (`float`, *optional*, defaults to 1): Parameter to set the boundary for interpolation
                    (only) in the linear ramp function.
                *   `factor` (`float`, *optional*): The scaling factor applied when interpolating the position IDs to
                    extend the possible context length. Additionally, if `attention_factor` is None, the log of this
                    value is used to compute a value for `attention_factor`, possibly in conjunciton with `mscale` and
                    `mscale_all_dim`, if provided.
                *   `mscale` (`float`, *optional*): If `attention_factor` is None and both `mscale` and
                    `mscale_all_dim` are provided, `mscale` acts scalar augmenting `log(factor)` when computing the
                    numerator for the inferred value of `attention_factor`. If not provided, `attention_factor` will be
                    calculated based on `factor` only.
                *   `mscale_all_dim` (`float`, *optional*): If `attention_factor` is None and both `mscale` and
                    `mscale_all_dim` are provided, `mscale_all_dim` acts scalar augmenting `log(factor)` when computing
                    the denominator for the inferred value of `attention_factor`. If not provided, `attention_factor`
                    will be calculated based on `factor` only.
                *   `original_max_position_embeddings` (`int`, *optional*): The original max position embeddings used
                    during pretraining. If not provided, the function falls back to `max_position_embeddings`.
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
    # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
    standardize_rope_params(config)
    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters

    base = rope_parameters_dict["rope_theta"]
    partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)

    factor = rope_parameters_dict["factor"]
    attention_factor = rope_parameters_dict.get("attention_factor")
    mscale = rope_parameters_dict.get("mscale")
    mscale_all_dim = rope_parameters_dict.get("mscale_all_dim")

    # NOTE: DeekSeek-V3 (and potentially other models) modify `max_position_embeddings` and have a
    # `original_max_position_embeddings` field containing the pretrained value. They use the ratio between these two
    # values to compute the default attention scaling factor, instead of using `factor`.
    if "original_max_position_embeddings" in rope_parameters_dict:
        original_max_position_embeddings = rope_parameters_dict["original_max_position_embeddings"]
        factor = config.max_position_embeddings / original_max_position_embeddings
    else:
        original_max_position_embeddings = config.max_position_embeddings

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
    device: "torch.device",
    seq_len: Optional[int] = None,
    layer_type: Optional[str] = None,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies with LongRoPE scaling. Please refer to the
    [original implementation](https://github.com/microsoft/LongRoPE)

    Args:
        config ([`~transformers.PreTrainedConfig`]):
            The model configuration. This function assumes that the config will provide at least the following
            properties:

            *   rope_theta (`float`): The base wavelength from which the inverse frequencies will be derived.
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
    # TODO (joao): use the new `original_max_position_embeddings` from rope_parameters
    # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
    standardize_rope_params(config)
    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters

    base = rope_parameters_dict["rope_theta"]
    partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)

    long_factor = rope_parameters_dict["long_factor"]
    short_factor = rope_parameters_dict["short_factor"]
    factor = rope_parameters_dict.get("factor")
    attention_factor = rope_parameters_dict.get("attention_factor")

    # NOTE: Phi3 (and potentially other models) modify `max_position_embeddings` and have a
    # `original_max_position_embeddings` field containing the pretrained value. They use the ratio between these two
    # values to compute the default attention scaling factor, instead of using `factor`.
    if original_max_position_embeddings := getattr(config, "original_max_position_embeddings", None):
        factor = config.max_position_embeddings / original_max_position_embeddings
    else:
        original_max_position_embeddings = config.max_position_embeddings

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
    device: "torch.device",
    seq_len: Optional[int] = None,
    layer_type: Optional[str] = None,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies for llama 3.1.

    Args:
        config ([`~transformers.PreTrainedConfig`]):
            The model configuration. This function assumes that the config will provide at least the following
            properties:

            *   rope_theta (`float`): The base wavelength from which the inverse frequencies will be derived.
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
    # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
    standardize_rope_params(config)
    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters

    # Gets the default RoPE parameters
    base = rope_parameters_dict["rope_theta"]
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
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
ROPE_INIT_FUNCTIONS = {
    "linear": _compute_linear_scaling_rope_parameters,
    "dynamic": _compute_dynamic_ntk_parameters,
    "yarn": _compute_yarn_parameters,
    "longrope": _compute_longrope_parameters,
    "llama3": _compute_llama3_parameters,
}


def _check_received_keys(
    rope_type: str,
    received_keys: set,
    required_keys: set,
    optional_keys: Optional[set] = None,
    ignore_keys: Optional[set] = None,
):
    """Compare the received keys in `config.rope_parameters` against the expected and optional keys"""
    # BC: "rope_type" was originally "type" -- let's check for "rope_type" when "type" is present
    if "type" in received_keys:
        received_keys -= {"type"}
        required_keys.add("rope_type")

    # Some models need to store model-specific keys, and we don't want to throw warning at them
    if ignore_keys is not None:
        received_keys -= ignore_keys

    missing_keys = required_keys - received_keys
    if missing_keys:
        raise KeyError(f"Missing required keys in `rope_parameters` for 'rope_type'='{rope_type}': {missing_keys}")

    if optional_keys is not None:
        unused_keys = received_keys - required_keys - optional_keys
    else:
        unused_keys = received_keys - required_keys
    if unused_keys:
        logger.warning(f"Unrecognized keys in `rope_parameters` for 'rope_type'='{rope_type}': {unused_keys}")


def _validate_default_rope_parameters(
    rope_parameters: dict, config: Optional[PreTrainedConfig] = None, ignore_keys: Optional[set] = None
):
    required_keys = {"rope_type", "rope_theta"}
    received_keys = set(rope_parameters.keys())
    rope_type = rope_parameters["rope_type"]
    _check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)


def _validate_linear_scaling_rope_parameters(
    rope_parameters: dict, config: Optional[PreTrainedConfig] = None, ignore_keys: Optional[set] = None
):
    required_keys = {"rope_type", "factor", "rope_theta"}
    received_keys = set(rope_parameters.keys())
    rope_type = rope_parameters["rope_type"]
    _check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)

    factor = rope_parameters["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        logger.warning(f"`rope_parameters`'s factor field must be a float >= 1, got {factor}")


def _validate_dynamic_scaling_rope_parameters(
    rope_parameters: dict, config: Optional[PreTrainedConfig] = None, ignore_keys: Optional[set] = None
):
    # TODO (joao): update logic for the inclusion of `original_max_position_embeddings`
    optional_keys = {"original_max_position_embeddings"}
    required_keys = {"rope_type", "factor"}
    received_keys = set(rope_parameters.keys())
    rope_type = rope_parameters["rope_type"]
    _check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

    factor = rope_parameters["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        logger.warning(f"`rope_parameters`'s factor field must be a float >= 1, got {factor}")


def _validate_yarn_parameters(
    rope_parameters: dict, config: Optional[PreTrainedConfig] = None, ignore_keys: Optional[set] = None
):
    required_keys = {"rope_type", "factor", "rope_theta"}
    optional_keys = {
        "attention_factor",
        "beta_fast",
        "beta_slow",
        "original_max_position_embeddings",
        "mscale",
        "mscale_all_dim",
    }
    received_keys = set(rope_parameters.keys())
    rope_type = rope_parameters["rope_type"]
    _check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

    factor = rope_parameters["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        logger.warning(f"`rope_parameters`'s factor field must be a float >= 1, got {factor}")

    attention_factor = rope_parameters.get("attention_factor")
    if attention_factor is not None and (not isinstance(attention_factor, float) or attention_factor < 0):
        logger.warning(
            f"`rope_parameters`'s attention_factor field must be a float greater than 0, got {attention_factor}"
        )
    beta_fast = rope_parameters.get("beta_fast")
    if beta_fast is not None and not isinstance(beta_fast, float):
        logger.warning(f"`rope_parameters`'s beta_fast field must be a float, got {beta_fast}")
    beta_slow = rope_parameters.get("beta_slow")
    if beta_slow is not None and not isinstance(beta_slow, float):
        logger.warning(f"`rope_parameters`'s beta_slow field must be a float, got {beta_slow}")

    if (beta_fast or 32) < (beta_slow or 1):
        logger.warning(
            f"`rope_parameters`'s beta_fast field must be greater than beta_slow, got beta_fast={beta_fast} "
            f"(defaults to 32 if None) and beta_slow={beta_slow} (defaults to 1 if None)"
        )

    # Models should set `config.rope_parameters["original_max_position_embeddings"]` to their original (pre-yarn) context
    # length, with `config.max_position_embeddings` corresponding to their post-yarn context length.
    # However, for BC purposes, we allow the former to be unset.
    original_max_position_embeddings = config.rope_parameters.get("original_max_position_embeddings")
    if original_max_position_embeddings is not None:
        # Double-check: `factor` should be the ratio between the pre-yarn and post-yarn context lengths.
        implicit_factor = config.max_position_embeddings / original_max_position_embeddings
        if implicit_factor != factor:
            logger.warning_once(
                f"The explicitly set RoPE scaling factor (config.rope_parameters['factor'] = {factor}) does not match "
                "the ratio implicitly set by other parameters (implicit factor = "
                "post-yarn context length / pre-yarn context length = "
                "config.max_position_embeddings / config.rope_parameters['original_max_position_embeddings'] = "
                f"{implicit_factor}). Using the explicit factor ({factor}) in YaRN. This may cause unexpected "
                "behaviour in model usage, please correct the 'max_position_embeddings' fields in the model config."
            )
    # No `config.rope_parameters["original_max_position_embeddings"]`. Is `config.max_position_embeddings` the
    # pre-yarn or the post-yarn context length?
    # BC: we assume it is the pre-yarn context length.
    else:
        logger.warning_once(
            "config.rope_parameters['original_max_position_embeddings'], the pre-yarn context length, is unset. We will "
            "**assume** config.max_position_embeddings holds the pre-yarn context length. Some use cases may expect "
            "config.max_position_embeddings to hold the post-yarn context length (pre-yarn context length * "
            "factor) -- we recommend updating both fields for optimal downstream model usage."
        )


def _validate_longrope_parameters(rope_parameters: dict, config: PreTrainedConfig, ignore_keys: Optional[set] = None):
    required_keys = {"rope_type", "short_factor", "long_factor", "rope_theta"}
    # TODO (joao): update logic for the inclusion of `original_max_position_embeddings`
    optional_keys = {"attention_factor", "factor", "original_max_position_embeddings"}
    received_keys = set(rope_parameters.keys())
    rope_type = rope_parameters["rope_type"]
    _check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)

    short_factor = rope_parameters.get("short_factor")
    if not isinstance(short_factor, list) and all(isinstance(x, (int, float)) for x in short_factor):
        logger.warning(f"`rope_parameters`'s short_factor field must be a list of numbers, got {short_factor}")
    if len(short_factor) != dim // 2:
        logger.warning(f"`rope_parameters`'s short_factor field must have length {dim // 2}, got {len(short_factor)}")

    long_factor = rope_parameters.get("long_factor")
    if not isinstance(long_factor, list) and all(isinstance(x, (int, float)) for x in long_factor):
        logger.warning(f"`rope_parameters`'s long_factor field must be a list of numbers, got {long_factor}")
    if len(long_factor) != dim // 2:
        logger.warning(f"`rope_parameters`'s long_factor field must have length {dim // 2}, got {len(long_factor)}")

    # Handle Phi3 divergence: prefer the use of `attention_factor` and/or `factor` over
    # `original_max_position_embeddings` to compute internal variables. The latter lives outside `rope_parameters` and is
    # unique to longrope (= undesirable)
    if hasattr(config, "original_max_position_embeddings"):
        logger.warning_once(
            "This model has set a `original_max_position_embeddings` field, to be used together with "
            "`max_position_embeddings` to determine a scaling factor. Please set the `factor` field of `rope_parameters`"
            "with this ratio instead -- we recommend the use of this field over `original_max_position_embeddings`, "
            "as it is compatible with most model architectures."
        )
    else:
        factor = rope_parameters.get("factor")
        if factor is None:
            logger.warning("Missing required keys in `rope_parameters`: 'factor'")
        elif not isinstance(factor, float) or factor < 1.0:
            logger.warning(f"`rope_parameters`'s factor field must be a float >= 1, got {factor}")

        attention_factor = rope_parameters.get("attention_factor")
        if attention_factor is not None:
            if not isinstance(attention_factor, float) or attention_factor < 0.0:
                logger.warning(
                    f"`rope_parameters`'s attention_factor field must be a float greater than 0, got {attention_factor}"
                )


def _validate_llama3_parameters(rope_parameters: dict, config: PreTrainedConfig, ignore_keys: Optional[set] = None):
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
    _check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)

    factor = rope_parameters["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        logger.warning(f"`rope_parameters`'s factor field must be a float >= 1, got {factor}")

    low_freq_factor = rope_parameters["low_freq_factor"]
    high_freq_factor = rope_parameters["high_freq_factor"]
    if low_freq_factor is None or not isinstance(low_freq_factor, float):
        logger.warning(f"`rope_parameters`'s low_freq_factor field must be a float, got {low_freq_factor}")
    if high_freq_factor is None or not isinstance(high_freq_factor, float):
        logger.warning(f"`rope_parameters`'s high_freq_factor field must be a float, got {high_freq_factor}")
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
    if original_max_position_embeddings >= config.max_position_embeddings:
        logger.warning(
            "`rope_parameters`'s original_max_position_embeddings field must be less than max_position_embeddings, got "
            f"{original_max_position_embeddings} and max_position_embeddings={config.max_position_embeddings}"
        )


# Like `ROPE_INIT_FUNCTIONS`, this validation function mapping can be dynamically updated for custom RoPE types.
ROPE_VALIDATION_FUNCTIONS = {
    "default": _validate_default_rope_parameters,
    "linear": _validate_linear_scaling_rope_parameters,
    "dynamic": _validate_dynamic_scaling_rope_parameters,
    "yarn": _validate_yarn_parameters,
    "longrope": _validate_longrope_parameters,
    "llama3": _validate_llama3_parameters,
}


def rope_config_validation(config: PreTrainedConfig, ignore_keys: Optional[set] = None):
    """
    Validate the RoPE config arguments, given a `PreTrainedConfig` object
    """
    rope_parameters_dict = getattr(config, "rope_parameters", None)  # not a default parameter in `PreTrainedConfig`
    if rope_parameters_dict is None:
        return

    if getattr(config, "layer_types", None) is not None and all(
        key in config.layer_types for key in rope_parameters_dict.keys()
    ):
        pass
    else:
        rope_parameters_dict = {"full_attention": rope_parameters_dict}

    for rope_parameters in rope_parameters_dict.values():
        rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
        validation_fn = ROPE_VALIDATION_FUNCTIONS.get(rope_type)

        rope_parameters["rope_type"] = rope_type
        # BC: "rope_theta" was originally saved in config
        rope_parameters["rope_theta"] = rope_parameters.get("rope_theta", getattr(config, "rope_theta", None))

        if validation_fn is not None:
            validation_fn(rope_parameters, config=config, ignore_keys=ignore_keys)
        else:
            logger.warning(
                f"Missing validation function mapping in `ROPE_VALIDATION_FUNCTIONS` for 'rope_type'='{rope_type}'"
            )


class RopeParameters(TypedDict):
    """
    Args:
        rope_theta (`float`):
            The base period of the RoPE embeddings.
        rope_type (`str`, *optional*, defaults to "default"):
            The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
            'llama3'], with 'default' being the original RoPE implementation.
        factor (`float`, *optional*):
            Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
            most scaling types, a `factor` of x will enable the model to handle sequences of length x *
            original maximum pre-trained length.
        original_max_position_embeddings (`int`, *optional*):
            Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
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

    rope_theta: float
    rope_type: Optional[str]
    factor: Optional[float]
    original_max_position_embeddings: Optional[int]
    attention_factor: Optional[float]
    beta_fast: Optional[float]
    beta_slow: Optional[float]
    short_factor: Optional[list[float]]
    long_factor: Optional[list[float]]
    low_freq_factor: Optional[float]
    high_freq_factor: Optional[float]
