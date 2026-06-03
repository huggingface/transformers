# Copyright 2026 Biohub. All rights reserved.
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

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ...configuration_utils import PretrainedConfig  # type: ignore[import]

# ---------------------------------------------------------------------------
# Nested dataclass configs
# ---------------------------------------------------------------------------

_DEFAULT_ESMC_HF_REPO = "biohub/ESMC-6B"


@dataclass
class MSAEncoderConfig:
    """Config for the optional MSA encoder module (Large MSA models only)."""

    enabled: bool = False
    d_msa: int = 128
    d_hidden: int = 32
    n_layers: int = 4
    n_heads_msa: int = 8
    msa_head_width: int = 32


@dataclass
class ParcaeConfig:
    """Release-only config for the parcae diffusion-loop scheduler."""

    enabled: bool = True
    poisson_mean: float = 3.0
    min_steps: int = 1
    max_steps: int | None = 6
    coda_n_layers: int = 2


@dataclass
class LMEncoderConfig:
    """Release-only config for the LM-side pair encoder."""

    enabled: bool = True
    n_layers: int = 4
    lm_dropout: float = 0.25
    per_loop_lm_dropout: bool = True


@dataclass
class AtomAttentionConfig:
    """Config for SWA atom encoder/decoder with 3D RoPE."""

    d_atom: int = 128
    d_token: int = 768
    n_blocks: int = 3
    n_heads: int = 4
    swa_window_size: int = 128
    expansion_ratio: int = 2
    # 3D RoPE config
    spatial_rope_base_frequency: float = 20.0
    n_spatial_rope_pairs_per_axis: int = 2
    n_uid_rope_pairs: int = 10
    uid_rope_base_frequency: float = 10000.0


@dataclass
class FoldingTrunkConfig:
    n_layers: int = 24
    n_heads: int = 8
    dropout: float = 0.0


@dataclass
class InputsEmbedderConfig:
    d_inputs: int = 451
    atom_encoder: AtomAttentionConfig = field(default_factory=AtomAttentionConfig)

    def __post_init__(self):
        if isinstance(self.atom_encoder, dict):
            self.atom_encoder = AtomAttentionConfig(**self.atom_encoder)


@dataclass
class DiffusionModuleConfig:
    """Config for the DiffusionModule."""

    sigma_data: float = 16.0
    c_atom: int = 128
    c_token: int = 768
    c_z: int = 256
    c_s_inputs: int = 451
    fourier_dim: int = 256
    relpos_r_max: int = 32
    relpos_s_max: int = 2
    atom_num_blocks: int = 3
    atom_num_heads: int = 4
    token_num_blocks: int = 12
    token_num_heads: int = 16
    transition_multiplier: int = 2


@dataclass
class DiffusionStructureHeadConfig:
    """Config for the diffusion-based structure prediction head."""

    diffusion_module: DiffusionModuleConfig = field(
        default_factory=DiffusionModuleConfig
    )
    distogram_bins: int = 128

    # Training noise: sigma ~ sigma_data * exp(mu + sigma * N(0,1))
    train_noise_log_mean: float = -1.2
    train_noise_log_std: float = 1.5

    # Sampling defaults (ODE)
    gamma_0: float = 0.605
    gamma_min: float = 1.107
    noise_scale: float = 0.0
    step_scale: float = 1.0

    # Inference schedule defaults
    inference_s_max: float = 160.0
    inference_s_min: float = 4e-4
    inference_p: float = 8.0
    inference_num_steps: int = 68

    def __post_init__(self):
        if isinstance(self.diffusion_module, dict):
            self.diffusion_module = DiffusionModuleConfig(**self.diffusion_module)


@dataclass
class ConfidenceHeadConfig:
    enabled: bool = True
    num_plddt_bins: int = 50
    num_pde_bins: int = 64
    num_pae_bins: int = 64
    min_dist: float = 2.0
    max_dist: float = 52.0
    distogram_bins: int = 128
    folding_trunk: FoldingTrunkConfig = field(
        default_factory=lambda: FoldingTrunkConfig(n_layers=4)
    )

    def __post_init__(self):
        if isinstance(self.folding_trunk, dict):
            self.folding_trunk = FoldingTrunkConfig(**self.folding_trunk)


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class ESMFold2Config(PretrainedConfig):
    """
    Configuration for the ESMFold2 structure prediction model.

    Uses SWA atom encoders with 3D RoPE, a diffusion transformer,
    a folding trunk, and an ESMC 6B PLM backbone.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control
    the model outputs. Read the documentation from [`PretrainedConfig`] for more
    information.

    Args:
        d_single (`int`, defaults to 384):
            Dimensionality of single (per-residue) representations.
        d_pair (`int`, defaults to 256):
            Dimensionality of pair (residue-residue) representations.
        n_relative_residx_bins (`int`, defaults to 32):
            Number of bins for relative residue index encoding.
        n_relative_chain_bins (`int`, defaults to 2):
            Number of bins for relative chain encoding.
        num_loops (`int`, defaults to 10):
            Number of trunk loops for iterative refinement.
        num_diffusion_samples (`int`, defaults to 8):
            Number of parallel structure predictions to generate.
        lm_dropout (`float`, defaults to 0.0):
            Dropout probability on LM pair embeddings. When > 0, dropout is
            applied with ``training=True`` (including at inference) to match
            the experimental training recipe used by binder design.
        force_lm_dropout_during_inference (`bool`, defaults to False):
            When True, apply ``lm_dropout`` even when ``model.eval()`` and
            ``lm_dropout`` > 0. Binder-design loads set this to True.
        disable_msa_features (`bool`, defaults to False):
            When True, zero out MSA-derived ``profile`` and ``deletion_mean``
            before the inputs embedder (experimental medium/large checkpoints).
        inputs (`InputsEmbedderConfig`):
            Configuration for the inputs embedder module.
        folding_trunk (`FoldingTrunkConfig`):
            Configuration for the folding trunk.
        structure_head (`DiffusionStructureHeadConfig`):
            Configuration for the diffusion-based structure prediction head.
        confidence_head (`ConfidenceHeadConfig`):
            Configuration for the confidence prediction head.

    Examples:

    ```python
    >>> from transformers import ESMFold2Config, ESMFold2Model

    >>> # Initializing an ESMFold2 configuration
    >>> configuration = ESMFold2Config(type="release")

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = ESMFold2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "esmfold2"
    has_no_defaults_at_init = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.type: str = kwargs.get("type", "release")
        if self.type != "release":
            raise ValueError(
                "ESMFold2Config.type must be 'release' (the 'experimental' variant "
                f"is not included in this release), got {self.type!r}"
            )

        # Top-level scalar fields
        self.d_single: int = kwargs.get("d_single", 384)
        self.d_pair: int = kwargs.get("d_pair", 256)
        self.n_relative_residx_bins: int = kwargs.get("n_relative_residx_bins", 32)
        self.n_relative_chain_bins: int = kwargs.get("n_relative_chain_bins", 2)
        self.num_loops: int = kwargs.get("num_loops", 10)
        self.num_diffusion_samples: int = kwargs.get("num_diffusion_samples", 8)
        # If True, ``profile`` / ``deletion_mean`` are zeroed before the inputs
        # embedder.
        self.disable_msa_features: bool = kwargs.get("disable_msa_features", False)
        self.lm_dropout: float = kwargs.get("lm_dropout", 0.0)
        self.force_lm_dropout_during_inference: bool = kwargs.get(
            "force_lm_dropout_during_inference", False
        )

        self.lm_d_model: int = kwargs.get("lm_d_model", 2560)
        self.lm_num_layers: int = kwargs.get("lm_num_layers", 80)
        # Required, no default — every shipped HF export must name its ESMC backbone.
        self.esmc_id: str = kwargs.get("esmc_id", _DEFAULT_ESMC_HF_REPO)

        def _init_nested(cls, val):
            if isinstance(val, cls):
                return val
            if isinstance(val, dict):
                return cls(**val)
            return cls()

        self.inputs = _init_nested(InputsEmbedderConfig, kwargs.get("inputs"))
        self.folding_trunk = _init_nested(
            FoldingTrunkConfig, kwargs.get("folding_trunk")
        )
        self.structure_head = _init_nested(
            DiffusionStructureHeadConfig, kwargs.get("structure_head")
        )
        self.confidence_head = _init_nested(
            ConfidenceHeadConfig, kwargs.get("confidence_head")
        )
        self.msa_encoder = _init_nested(MSAEncoderConfig, kwargs.get("msa_encoder"))
        # Release-only modules — ignored when ``type == "experimental"``.
        self.parcae = _init_nested(ParcaeConfig, kwargs.get("parcae"))
        self.lm_encoder = _init_nested(LMEncoderConfig, kwargs.get("lm_encoder"))
        # If True, MSA encoder output replaces the pair stream; if False, it is added.
        self.msa_encoder_overwrite: bool = bool(
            kwargs.get("msa_encoder_overwrite", True)
        )

    def to_dict(self):
        output = super().to_dict()
        output["inputs"] = asdict(self.inputs)
        output["folding_trunk"] = asdict(self.folding_trunk)
        output["structure_head"] = asdict(self.structure_head)
        output["confidence_head"] = asdict(self.confidence_head)
        output["msa_encoder"] = asdict(self.msa_encoder)
        output["parcae"] = asdict(self.parcae)
        output["lm_encoder"] = asdict(self.lm_encoder)
        return output


__all__ = ["ESMFold2Config", "MSAEncoderConfig", "ParcaeConfig", "LMEncoderConfig"]
