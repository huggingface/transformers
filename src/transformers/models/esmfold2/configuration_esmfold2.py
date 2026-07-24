# Copyright 2026 BioHub and The HuggingFace Inc. team. All rights reserved.
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
"""ESMFold2 model configuration."""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..esmc.configuration_esmc import EsmcConfig


logger = logging.get_logger(__name__)


@strict
class EsmFold2AtomEncoderConfig(PreTrainedConfig):
    """Configuration for the SWA atom encoder used by the inputs embedder (3D-RoPE atom transformer)."""

    hidden_size: int | None = 128
    token_hidden_size: int | None = 768
    num_hidden_layers: int | None = 3
    num_attention_heads: int | None = 4
    expansion_ratio: int | None = 2
    ffn_intermediate_size: int | None = None  # derived from expansion_ratio/hidden_size in the parent
    spatial_rope_base_frequency: float | None = 20.0
    n_spatial_rope_pairs_per_axis: int | None = 2
    n_uid_rope_pairs: int | None = 10
    uid_rope_base_frequency: float | None = 10000.0


@strict
class EsmFold2DiffusionModuleConfig(PreTrainedConfig):
    """Configuration for the diffusion denoiser (atom encoder/decoder + token transformer with pair bias)."""

    sigma_data: float | None = 16.0
    atom_hidden_size: int | None = 128
    token_hidden_size: int | None = 768
    fourier_dim: int | None = 256
    atom_num_blocks: int | None = 3
    atom_num_heads: int | None = 4
    token_num_blocks: int | None = 12
    token_num_heads: int | None = 16
    transition_multiplier: int | None = 2
    atom_expansion_ratio: int | None = 2
    atom_ffn_intermediate_size: int | None = None  # derived in the parent
    token_transition_intermediate_size: int | None = None  # derived in the parent


@strict
class EsmFold2StructureHeadConfig(PreTrainedConfig):
    """Configuration for the diffusion structure-prediction head (distogram + the diffusion sampler)."""

    sub_configs = {"diffusion_module": EsmFold2DiffusionModuleConfig}

    diffusion_module: dict | EsmFold2DiffusionModuleConfig | None = None
    distogram_bins: int | None = 128
    gamma_0: float | None = 0.605
    gamma_min: float | None = 1.107
    noise_scale: float | None = 0.0
    step_scale: float | None = 1.0
    inference_s_max: float | None = 160.0
    inference_s_min: float | None = 4e-4
    inference_p: float | None = 8.0
    inference_num_steps: int | None = 68
    max_inference_sigma: float | None = 256.0

    def __post_init__(self, **kwargs):
        if self.diffusion_module is None:
            self.diffusion_module = EsmFold2DiffusionModuleConfig()
        elif isinstance(self.diffusion_module, dict):
            self.diffusion_module = EsmFold2DiffusionModuleConfig(**self.diffusion_module)
        super().__post_init__(**kwargs)


@strict
class EsmFold2ConfidenceHeadConfig(PreTrainedConfig):
    """Configuration for the confidence head (pLDDT / PAE / PDE / pTM)."""

    num_hidden_layers: int | None = 4
    num_plddt_bins: int | None = 50
    num_pde_bins: int | None = 64
    num_pae_bins: int | None = 64
    min_dist: float | None = 2.0
    max_dist: float | None = 52.0
    distogram_bins: int | None = 128
    eps: float | None = 1e-6  # additive guard for masked-mean denominators (empty chains / all-padding rows)


@strict
class EsmFold2MsaEncoderConfig(PreTrainedConfig):
    """Configuration for the MSA encoder (outer-product-mean + pair-weighted averaging + transition)."""

    overwrite: bool | None = True
    divide_outer_before_proj: bool | None = False
    hidden_size: int | None = 128
    outer_hidden_size: int | None = 32
    num_hidden_layers: int | None = 4
    num_attention_heads: int | None = 8
    head_width: int | None = 32
    transition_intermediate_size: int | None = None  # derived in the parent


@strict
class EsmFold2LmEncoderConfig(PreTrainedConfig):
    """Configuration for the language-model (ESMC) hidden-state encoder folded into the trunk."""

    num_hidden_layers: int | None = 4
    lm_dropout: float | None = 0.25
    per_loop_lm_dropout: bool | None = True


@auto_docstring(checkpoint="biohub/ESMFold2")
@strict
class EsmFold2Config(PreTrainedConfig):
    r"""
    hidden_size (`int`, *optional*, defaults to 384):
        Single-representation width.
    pairwise_hidden_size (`int`, *optional*, defaults to 256):
        Pair-representation width.
    single_inputs_size (`int`, *optional*, defaults to 451):
        Width of the concatenated single-input features fed to the trunk and diffusion conditioning.
    transition_expansion_ratio (`int`, *optional*, defaults to 4):
        Expansion ratio for the pair / MSA transition FFNs (used to derive their intermediate sizes).
    pair_transition_intermediate_size (`int`, *optional*):
        Pair-transition FFN width; defaults to `transition_expansion_ratio * pairwise_hidden_size`.
    sliding_window (`int`, *optional*, defaults to 128):
        Sliding-window size (token-index distance) for the atom-stack attention.
    chunk_size (`int`, *optional*, defaults to 64):
        Chunk size for the memory-heavy pair-/MSA-stream ops. `None` disables chunking.
    n_relative_residx_bins (`int`, *optional*, defaults to 32):
        Number of relative residue-index bins in the relative-position encoding.
    n_relative_chain_bins (`int`, *optional*, defaults to 2):
        Number of relative chain-index bins in the relative-position encoding.
    num_loops (`int`, *optional*, defaults to 10):
        Number of trunk refinement loops.
    num_diffusion_samples (`int`, *optional*, defaults to 8):
        Number of parallel structure samples drawn by the diffusion sampler.
    num_res_types (`int`, *optional*, defaults to 33):
        Number of residue types.
    max_atomic_number (`int`, *optional*, defaults to 128):
        Size of the element one-hot in the atom features.
    char_vocab_size (`int`, *optional*, defaults to 64):
        Character-vocabulary size for the encoded atom names.
    max_chars (`int`, *optional*, defaults to 4):
        Number of characters per encoded atom name.
    max_atoms_per_token (`int`, *optional*, defaults to 23):
        Maximum number of atoms per token.
    atom_feature_dim (`int`, *optional*):
        Atom feature width; derived from `max_atomic_number`, `char_vocab_size` and `max_chars` when unset.
    folding_trunk_num_hidden_layers (`int`, *optional*, defaults to 24):
        Number of pair-update blocks in the folding trunk.
    parcae_num_coda_layers (`int`, *optional*, defaults to 2):
        Number of pair-update blocks in the parcae coda.
    atom_encoder ([`EsmFold2AtomEncoderConfig`], *optional*):
        Configuration for the inputs-embedder SWA atom encoder.
    structure_head ([`EsmFold2StructureHeadConfig`], *optional*):
        Configuration for the diffusion structure-prediction head (holds the `diffusion_module` sub-config).
    confidence_head ([`EsmFold2ConfidenceHeadConfig`], *optional*):
        Configuration for the confidence head.
    msa_encoder ([`EsmFold2MsaEncoderConfig`], *optional*):
        Configuration for the MSA encoder.
    lm_encoder ([`EsmFold2LmEncoderConfig`], *optional*):
        Configuration for the language-model hidden-state encoder.
    esmc_config ([`EsmcConfig`], *optional*):
        Configuration for the bundled ESMC backbone.
    """

    model_type = "esmfold2"
    sub_configs = {
        "esmc_config": EsmcConfig,
        "atom_encoder": EsmFold2AtomEncoderConfig,
        "structure_head": EsmFold2StructureHeadConfig,
        "confidence_head": EsmFold2ConfidenceHeadConfig,
        "msa_encoder": EsmFold2MsaEncoderConfig,
        "lm_encoder": EsmFold2LmEncoderConfig,
    }

    hidden_size: int | None = 384
    pairwise_hidden_size: int | None = 256
    single_inputs_size: int | None = 451
    transition_expansion_ratio: int | None = 4
    pair_transition_intermediate_size: int | None = None
    sliding_window: int | None = 128
    chunk_size: int | None = 64
    n_relative_residx_bins: int | None = 32
    n_relative_chain_bins: int | None = 2
    num_loops: int | None = 10
    num_diffusion_samples: int | None = 8
    num_res_types: int | None = 33
    max_atomic_number: int | None = 128
    char_vocab_size: int | None = 64
    max_chars: int | None = 4
    max_atoms_per_token: int | None = 23
    atom_feature_dim: int | None = None
    folding_trunk_num_hidden_layers: int | None = 24
    parcae_num_coda_layers: int | None = 2

    atom_encoder: dict | EsmFold2AtomEncoderConfig | None = None
    structure_head: dict | EsmFold2StructureHeadConfig | None = None
    confidence_head: dict | EsmFold2ConfidenceHeadConfig | None = None
    msa_encoder: dict | EsmFold2MsaEncoderConfig | None = None
    lm_encoder: dict | EsmFold2LmEncoderConfig | None = None
    esmc_config: dict | EsmcConfig | None = None

    def __post_init__(self, **kwargs):
        def _init_nested(cls, val):
            if val is None:
                return cls()
            if isinstance(val, dict):
                return cls(**val)
            return val

        self.esmc_config = _init_nested(EsmcConfig, self.esmc_config)
        self.atom_encoder = _init_nested(EsmFold2AtomEncoderConfig, self.atom_encoder)
        self.structure_head = _init_nested(EsmFold2StructureHeadConfig, self.structure_head)
        self.confidence_head = _init_nested(EsmFold2ConfidenceHeadConfig, self.confidence_head)
        self.msa_encoder = _init_nested(EsmFold2MsaEncoderConfig, self.msa_encoder)
        self.lm_encoder = _init_nested(EsmFold2LmEncoderConfig, self.lm_encoder)

        # Atom featurization width: 3 (xyz) + 1 (charge) + 1 (mask) + element one-hot + atom-name-char one-hots.
        if self.atom_feature_dim is None:
            self.atom_feature_dim = 3 + 1 + 1 + self.max_atomic_number + self.char_vocab_size * self.max_chars

        # SwiGLU FFN widths derived from the stream widths when not set explicitly (matches the reference
        # feed-forward blocks). Atom-stack FFNs are rounded up to a multiple of 256 (hardware-aligned width).
        if self.pair_transition_intermediate_size is None:
            self.pair_transition_intermediate_size = self.transition_expansion_ratio * self.pairwise_hidden_size
        atom = self.atom_encoder
        if atom.ffn_intermediate_size is None:
            atom.ffn_intermediate_size = (atom.expansion_ratio * (atom.hidden_size // 3) * 2 + 255) // 256 * 256
        diff = self.structure_head.diffusion_module
        if diff.atom_ffn_intermediate_size is None:
            diff.atom_ffn_intermediate_size = (
                (diff.atom_expansion_ratio * (diff.atom_hidden_size // 3) * 2 + 255) // 256 * 256
            )
        if diff.token_transition_intermediate_size is None:
            diff.token_transition_intermediate_size = diff.transition_multiplier * diff.token_hidden_size
        if self.msa_encoder.transition_intermediate_size is None:
            self.msa_encoder.transition_intermediate_size = (
                self.transition_expansion_ratio * self.msa_encoder.hidden_size
            )

        super().__post_init__(**kwargs)


# Only the parent config is public; the sub-configs are internal (mirrors the pre-flattening layout).
__all__ = ["EsmFold2Config"]
