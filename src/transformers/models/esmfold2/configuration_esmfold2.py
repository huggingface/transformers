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


@auto_docstring(
    custom_intro="""
    Configuration for a sliding-window atom transformer with 3D RoPE.

    The same architecture is instantiated twice, so this config is used twice: once for the inputs
    embedder (`EsmFold2Config.atom_encoder`) and once for the diffusion denoiser
    (`EsmFold2Config.structure_head.diffusion_module.atom_encoder`). The field names are identical at
    both sites, so every atom-stack module simply picks the sub-config for its call site.
    """
)
@strict
class EsmFold2AtomEncoderConfig(PreTrainedConfig):
    r"""
    hidden_size (`int`, *optional*, defaults to 128):
        Per-atom hidden width of the atom transformer.
    output_dim (`int`, *optional*):
        Width this stack aggregates to when scattering atoms back into tokens. Derived by the parent
        config: the diffusion token width for the denoiser, and whatever makes the single-inputs
        feature concat total `single_inputs_size` for the inputs embedder.
    num_hidden_layers (`int`, *optional*, defaults to 3):
        Number of sliding-window atom-transformer blocks.
    num_attention_heads (`int`, *optional*, defaults to 4):
        Number of attention heads in each atom-transformer block.
    head_dim (`int`, *optional*):
        Per-head width. Derived as `hidden_size // num_attention_heads` if unset.
    expansion_ratio (`int`, *optional*, defaults to 2):
        Expansion ratio used to derive `ffn_intermediate_size`.
    ffn_intermediate_size (`int`, *optional*):
        SwiGLU feed-forward width. Derived from `expansion_ratio` and `hidden_size` if unset, rounded
        up to a multiple of 256.
    spatial_rope_base_frequency (`float`, *optional*, defaults to 20.0):
        Base frequency for the spatial (x/y/z) half of the 3D rotary embedding.
    n_spatial_rope_pairs_per_axis (`int`, *optional*, defaults to 2):
        Number of rotary frequency pairs allocated to each spatial axis.
    n_uid_rope_pairs (`int`, *optional*, defaults to 10):
        Number of rotary frequency pairs allocated to the per-atom space UID.
    uid_rope_base_frequency (`float`, *optional*, defaults to 10000.0):
        Base frequency for the space-UID half of the 3D rotary embedding.
    """

    hidden_size: int | None = 128
    output_dim: int | None = None  # atom->token aggregation width; derived in the parent
    num_hidden_layers: int | None = 3
    num_attention_heads: int | None = 4
    head_dim: int | None = None  # derived from hidden_size/num_attention_heads in the parent
    expansion_ratio: int | None = 2
    ffn_intermediate_size: int | None = None  # derived from expansion_ratio/hidden_size in the parent
    spatial_rope_base_frequency: float | None = 20.0
    n_spatial_rope_pairs_per_axis: int | None = 2
    n_uid_rope_pairs: int | None = 10
    uid_rope_base_frequency: float | None = 10000.0


@auto_docstring(
    custom_intro="Configuration for the diffusion denoiser: an atom encoder/decoder around a token "
    "transformer with pair bias."
)
@strict
class EsmFold2DiffusionModuleConfig(PreTrainedConfig):
    r"""
    sigma_data (`float`, *optional*, defaults to 16.0):
        Data noise scale of the EDM preconditioning; sets the scale at which coordinates are normalized.
    token_hidden_size (`int`, *optional*, defaults to 768):
        Token-stream width of the diffusion transformer. Also the width the atom stack projects to and
        from, so `atom_encoder.output_dim` mirrors it.
    fourier_dim (`int`, *optional*, defaults to 256):
        Width of the Fourier noise-level embedding.
    token_num_blocks (`int`, *optional*, defaults to 12):
        Number of token-transformer blocks (one attention + one transition each).
    token_num_heads (`int`, *optional*, defaults to 16):
        Number of attention heads in the token transformer; also the width of each block's pair bias.
    transition_multiplier (`int`, *optional*, defaults to 2):
        Multiplier used to derive `token_transition_intermediate_size`.
    token_transition_intermediate_size (`int`, *optional*):
        SwiGLU width of the token transitions. Derived as
        `transition_multiplier * token_hidden_size` if unset.
    atom_encoder ([`EsmFold2AtomEncoderConfig`], *optional*):
        Configuration for the denoiser's atom encoder/decoder stack.
    """

    sub_configs = {"atom_encoder": EsmFold2AtomEncoderConfig}

    sigma_data: float | None = 16.0
    token_hidden_size: int | None = 768
    fourier_dim: int | None = 256
    token_num_blocks: int | None = 12
    token_num_heads: int | None = 16
    transition_multiplier: int | None = 2
    token_transition_intermediate_size: int | None = None  # derived in the parent
    atom_encoder: dict | EsmFold2AtomEncoderConfig | None = None

    def __post_init__(self, **kwargs):
        if self.atom_encoder is None:
            self.atom_encoder = EsmFold2AtomEncoderConfig()
        elif isinstance(self.atom_encoder, dict):
            self.atom_encoder = EsmFold2AtomEncoderConfig(**self.atom_encoder)
        super().__post_init__(**kwargs)


@auto_docstring(
    custom_intro="Configuration for the structure-prediction head: the distogram output plus the "
    "diffusion sampler's noise schedule."
)
@strict
class EsmFold2StructureHeadConfig(PreTrainedConfig):
    r"""
    diffusion_module ([`EsmFold2DiffusionModuleConfig`], *optional*):
        Configuration for the denoiser this head samples from.
    distogram_bins (`int`, *optional*, defaults to 128):
        Number of distance bins predicted by the distogram head.
    gamma_0 (`float`, *optional*, defaults to 0.605):
        Churn factor applied at noise levels above `gamma_min` (extra noise re-injected before a step).
    gamma_min (`float`, *optional*, defaults to 1.107):
        Noise level below which no churn is applied.
    noise_scale (`float`, *optional*, defaults to 0.0):
        Scale of the noise added by the churn step. `0.0` makes sampling a deterministic ODE.
    step_scale (`float`, *optional*, defaults to 1.0):
        Scale applied to each denoising update.
    inference_s_max (`float`, *optional*, defaults to 160.0):
        Highest sigma of the Karras noise schedule, in units of `sigma_data`.
    inference_s_min (`float`, *optional*, defaults to 4e-4):
        Lowest sigma of the Karras noise schedule, in units of `sigma_data`.
    inference_p (`float`, *optional*, defaults to 8.0):
        Exponent shaping the Karras schedule between `inference_s_min` and `inference_s_max`.
    inference_num_steps (`int`, *optional*, defaults to 68):
        Default number of denoising steps.
    max_inference_sigma (`float`, *optional*, defaults to 256.0):
        Cap on the schedule: the high-sigma tail above this is truncated and the cap re-prepended, so
        sampling still starts from it.
    """

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


@auto_docstring(
    custom_intro="Configuration for the confidence head, which predicts pLDDT, PAE, PDE, resolved-atom "
    "probability and the pTM/ipTM summaries."
)
@strict
class EsmFold2ConfidenceHeadConfig(PreTrainedConfig):
    r"""
    num_hidden_layers (`int`, *optional*, defaults to 4):
        Number of pair-update blocks in the head's own folding trunk.
    num_plddt_bins (`int`, *optional*, defaults to 50):
        Number of bins in the per-atom pLDDT distribution.
    num_pde_bins (`int`, *optional*, defaults to 64):
        Number of bins in the predicted-distance-error distribution.
    num_pae_bins (`int`, *optional*, defaults to 64):
        Number of bins in the predicted-aligned-error distribution.
    min_dist (`float`, *optional*, defaults to 2.0):
        Lower edge (Å) of the head's distance binning.
    max_dist (`float`, *optional*, defaults to 52.0):
        Upper edge (Å) of the head's distance binning.
    distogram_bins (`int`, *optional*, defaults to 128):
        Number of distance bins used to embed predicted inter-atom distances.
    eps (`float`, *optional*, defaults to 1e-6):
        Additive guard for masked-mean denominators (empty chains / all-padding rows). Kept as an
        explicit `+ eps` rather than clamping the denominator, to reproduce the reference numerics.
    """

    num_hidden_layers: int | None = 4
    num_plddt_bins: int | None = 50
    num_pde_bins: int | None = 64
    num_pae_bins: int | None = 64
    min_dist: float | None = 2.0
    max_dist: float | None = 52.0
    distogram_bins: int | None = 128
    eps: float | None = 1e-6  # additive guard for masked-mean denominators (empty chains / all-padding rows)


@auto_docstring(
    custom_intro="Configuration for the MSA encoder: outer-product-mean into the pair stream, "
    "pair-weighted averaging back into the MSA stream, and the triangle updates."
)
@strict
class EsmFold2MsaEncoderConfig(PreTrainedConfig):
    r"""
    overwrite (`bool`, *optional*, defaults to `True`):
        Whether the MSA-conditioned pair representation replaces the injected pair each trunk loop
        rather than being added to it.
    divide_outer_before_proj (`bool`, *optional*, defaults to `False`):
        Order of the outer-product-mean normalization: `False` computes `Wout(outer) / n_valid` (the
        projection bias is scaled too), `True` computes `Wout(outer / n_valid)`. Different released
        checkpoints were trained with different orderings.
    hidden_size (`int`, *optional*, defaults to 128):
        Width of the MSA stream.
    outer_hidden_size (`int`, *optional*, defaults to 32):
        Per-side width of the outer-product-mean projection; the outer product is this value squared.
    num_hidden_layers (`int`, *optional*, defaults to 4):
        Number of MSA encoder blocks. The last one updates only the pair stream, since its MSA output
        would be discarded.
    num_attention_heads (`int`, *optional*, defaults to 8):
        Number of heads in the pair-weighted averaging.
    head_width (`float`, *optional*, defaults to 32):
        Per-head width of the pair-weighted averaging.
    transition_intermediate_size (`int`, *optional*):
        SwiGLU width of the MSA-stream transition. Derived from the parent's
        `transition_expansion_ratio` and `hidden_size` if unset.
    outer_product_chunk_size (`int`, *optional*):
        Chunk size for the outer-product-mean einsum, off by default. Unlike the other chunked ops,
        chunking this one is not always bit-exact in bf16, so enable it only to trade exactness for
        peak memory on long sequences (the intermediate drops from `[B, L, L, c*d]` to
        `[B, chunk, L, c*d]`).
    """

    overwrite: bool | None = True
    divide_outer_before_proj: bool | None = False
    hidden_size: int | None = 128
    outer_hidden_size: int | None = 32
    num_hidden_layers: int | None = 4
    num_attention_heads: int | None = 8
    head_width: int | None = 32
    transition_intermediate_size: int | None = None
    outer_product_chunk_size: int | None = None


@auto_docstring(custom_intro="Configuration for the language-model (ESMC) hidden-state encoder folded into the trunk.")
@strict
class EsmFold2LmEncoderConfig(PreTrainedConfig):
    r"""
    num_hidden_layers (`int`, *optional*, defaults to 4):
        Number of pair-update blocks refining the projected ESMC hidden states.
    lm_dropout (`float`, *optional*, defaults to 0.25):
        Dropout applied to the projected language-model pair representation.
    per_loop_lm_dropout (`bool`, *optional*, defaults to `True`):
        Whether to resample that dropout on every trunk loop rather than once per fold.
    """

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
        diff = self.structure_head.diffusion_module

        # The diffusion atom stack reuses the inputs embedder's 3D-RoPE settings, as the reference
        # does, so mirror them rather than letting a checkpoint carry inconsistent copies.
        for rope_field in (
            "spatial_rope_base_frequency",
            "n_spatial_rope_pairs_per_axis",
            "n_uid_rope_pairs",
            "uid_rope_base_frequency",
        ):
            setattr(diff.atom_encoder, rope_field, getattr(self.atom_encoder, rope_field))

        # Same derivations at both atom-stack call sites, now that the sub-configs share field names.
        for atom in (self.atom_encoder, diff.atom_encoder):
            if atom.head_dim is None:
                atom.head_dim = atom.hidden_size // atom.num_attention_heads
            if atom.ffn_intermediate_size is None:
                atom.ffn_intermediate_size = (atom.expansion_ratio * (atom.hidden_size // 3) * 2 + 255) // 256 * 256

        # Atom->token aggregation widths. The diffusion atom stack feeds the token transformer, so it
        # aggregates to the token-stream width. The inputs embedder's encoding is instead concatenated
        # with the residue-type/profile/deletion features into ``single_inputs``, so its width is
        # whatever makes that concat total ``single_inputs_size`` -- deriving it here keeps the two
        # consistent by construction (they used to be independent fields that merely agreed).
        if diff.atom_encoder.output_dim is None:
            diff.atom_encoder.output_dim = diff.token_hidden_size
        if self.atom_encoder.output_dim is None:
            self.atom_encoder.output_dim = self.single_inputs_size - (2 * self.num_res_types + 1)

        if diff.token_transition_intermediate_size is None:
            diff.token_transition_intermediate_size = diff.transition_multiplier * diff.token_hidden_size
        if self.msa_encoder.transition_intermediate_size is None:
            self.msa_encoder.transition_intermediate_size = (
                self.transition_expansion_ratio * self.msa_encoder.hidden_size
            )

        super().__post_init__(**kwargs)


# Only the parent config is public; the sub-configs are internal (mirrors the pre-flattening layout).
__all__ = ["EsmFold2Config"]
