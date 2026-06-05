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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)

_DEFAULT_ESMC_HF_REPO = "biohub/ESMC-6B"


# ---------------------------------------------------------------------------
# Nested sub-configs (registered via the parent's ``sub_configs``)
# ---------------------------------------------------------------------------


@strict
class MSAEncoderConfig(PreTrainedConfig):
    """Config for the optional MSA encoder module (Large MSA models only)."""

    enabled: bool | None = False
    d_msa: int | None = 128
    d_hidden: int | None = 32
    n_layers: int | None = 4
    n_heads_msa: int | None = 8
    msa_head_width: int | None = 32


@strict
class ParcaeConfig(PreTrainedConfig):
    """Release-only config for the parcae diffusion-loop scheduler."""

    enabled: bool | None = True
    poisson_mean: float | None = 3.0
    min_steps: int | None = 1
    max_steps: int | None = 6
    coda_n_layers: int | None = 2


@strict
class LMEncoderConfig(PreTrainedConfig):
    """Release-only config for the LM-side pair encoder."""

    enabled: bool | None = True
    n_layers: int | None = 4
    lm_dropout: float | None = 0.25
    per_loop_lm_dropout: bool | None = True


@strict
class AtomAttentionConfig(PreTrainedConfig):
    """Config for the SWA atom encoder/decoder with 3D RoPE."""

    d_atom: int | None = 128
    d_token: int | None = 768
    n_blocks: int | None = 3
    n_heads: int | None = 4
    swa_window_size: int | None = 128
    expansion_ratio: int | None = 2
    spatial_rope_base_frequency: float | None = 20.0
    n_spatial_rope_pairs_per_axis: int | None = 2
    n_uid_rope_pairs: int | None = 10
    uid_rope_base_frequency: float | None = 10000.0


@strict
class FoldingTrunkConfig(PreTrainedConfig):
    """Config for a pairwise folding trunk stack."""

    n_layers: int | None = 24
    n_heads: int | None = 8
    dropout: float | None = 0.0


@strict
class InputsEmbedderConfig(PreTrainedConfig):
    """Config for the inputs embedder (wraps the atom encoder)."""

    sub_configs = {"atom_encoder": AtomAttentionConfig}

    d_inputs: int | None = 451
    atom_encoder: dict | AtomAttentionConfig | None = None

    def __post_init__(self, **kwargs):
        if self.atom_encoder is None:
            self.atom_encoder = AtomAttentionConfig()
        elif isinstance(self.atom_encoder, dict):
            self.atom_encoder = AtomAttentionConfig(**self.atom_encoder)
        super().__post_init__(**kwargs)


@strict
class DiffusionModuleConfig(PreTrainedConfig):
    """Config for the DiffusionModule."""

    sigma_data: float | None = 16.0
    c_atom: int | None = 128
    c_token: int | None = 768
    c_z: int | None = 256
    c_s_inputs: int | None = 451
    fourier_dim: int | None = 256
    relpos_r_max: int | None = 32
    relpos_s_max: int | None = 2
    atom_num_blocks: int | None = 3
    atom_num_heads: int | None = 4
    token_num_blocks: int | None = 12
    token_num_heads: int | None = 16
    transition_multiplier: int | None = 2


@strict
class DiffusionStructureHeadConfig(PreTrainedConfig):
    """Config for the diffusion-based structure prediction head."""

    sub_configs = {"diffusion_module": DiffusionModuleConfig}

    diffusion_module: dict | DiffusionModuleConfig | None = None
    distogram_bins: int | None = 128
    # Training noise: sigma ~ sigma_data * exp(mu + sigma * N(0,1))
    train_noise_log_mean: float | None = -1.2
    train_noise_log_std: float | None = 1.5
    # Sampling defaults (ODE)
    gamma_0: float | None = 0.605
    gamma_min: float | None = 1.107
    noise_scale: float | None = 0.0
    step_scale: float | None = 1.0
    # Inference schedule defaults
    inference_s_max: float | None = 160.0
    inference_s_min: float | None = 4e-4
    inference_p: float | None = 8.0
    inference_num_steps: int | None = 68

    def __post_init__(self, **kwargs):
        if self.diffusion_module is None:
            self.diffusion_module = DiffusionModuleConfig()
        elif isinstance(self.diffusion_module, dict):
            self.diffusion_module = DiffusionModuleConfig(**self.diffusion_module)
        super().__post_init__(**kwargs)


@strict
class ConfidenceHeadConfig(PreTrainedConfig):
    """Config for the confidence prediction head."""

    sub_configs = {"folding_trunk": FoldingTrunkConfig}

    enabled: bool | None = True
    num_plddt_bins: int | None = 50
    num_pde_bins: int | None = 64
    num_pae_bins: int | None = 64
    min_dist: float | None = 2.0
    max_dist: float | None = 52.0
    distogram_bins: int | None = 128
    folding_trunk: dict | FoldingTrunkConfig | None = None

    def __post_init__(self, **kwargs):
        if self.folding_trunk is None:
            self.folding_trunk = FoldingTrunkConfig(n_layers=4)
        elif isinstance(self.folding_trunk, dict):
            self.folding_trunk = FoldingTrunkConfig(**self.folding_trunk)
        super().__post_init__(**kwargs)


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@auto_docstring(checkpoint="biohub/ESMFold2")
@strict
class ESMFold2Config(PreTrainedConfig):
    r"""
    type (`str`, *optional*, defaults to `"release"`):
        Architecture variant. Only `"release"` is supported in this port (the
        `"experimental"` variant is deferred to a follow-up).
    d_single (`int`, *optional*, defaults to 384):
        Dimensionality of single (per-residue) representations.
    d_pair (`int`, *optional*, defaults to 256):
        Dimensionality of pair (residue-residue) representations.
    n_relative_residx_bins (`int`, *optional*, defaults to 32):
        Number of bins for relative residue index encoding.
    n_relative_chain_bins (`int`, *optional*, defaults to 2):
        Number of bins for relative chain encoding.
    num_loops (`int`, *optional*, defaults to 10):
        Number of trunk loops for iterative refinement.
    num_diffusion_samples (`int`, *optional*, defaults to 8):
        Number of parallel structure predictions to generate.
    lm_d_model (`int`, *optional*, defaults to 2560):
        Hidden size of the ESMC language-model backbone.
    lm_num_layers (`int`, *optional*, defaults to 80):
        Number of layers in the ESMC language-model backbone.
    esmc_id (`str`, *optional*, defaults to `"biohub/ESMC-6B"`):
        Hub id of the ESMC backbone, loaded separately (the ESMFold2 checkpoint
        does not bundle ESMC weights).
    msa_encoder_overwrite (`bool`, *optional*, defaults to `True`):
        If `True`, MSA encoder output replaces the pair stream; if `False`, it
        is added.
    inputs (`InputsEmbedderConfig`, *optional*):
        Configuration for the inputs embedder module.
    folding_trunk (`FoldingTrunkConfig`, *optional*):
        Configuration for the folding trunk.
    structure_head (`DiffusionStructureHeadConfig`, *optional*):
        Configuration for the diffusion-based structure prediction head.
    confidence_head (`ConfidenceHeadConfig`, *optional*):
        Configuration for the confidence prediction head.
    msa_encoder (`MSAEncoderConfig`, *optional*):
        Configuration for the optional MSA encoder.
    parcae (`ParcaeConfig`, *optional*):
        Configuration for the parcae diffusion-loop scheduler.
    lm_encoder (`LMEncoderConfig`, *optional*):
        Configuration for the LM-side pair encoder.

    Examples:

    ```python
    >>> from transformers import ESMFold2Config, ESMFold2Model

    >>> # Initializing an ESMFold2 configuration
    >>> configuration = ESMFold2Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = ESMFold2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "esmfold2"
    sub_configs = {
        "inputs": InputsEmbedderConfig,
        "folding_trunk": FoldingTrunkConfig,
        "structure_head": DiffusionStructureHeadConfig,
        "confidence_head": ConfidenceHeadConfig,
        "msa_encoder": MSAEncoderConfig,
        "parcae": ParcaeConfig,
        "lm_encoder": LMEncoderConfig,
    }

    type: str | None = "release"
    d_single: int | None = 384
    d_pair: int | None = 256
    n_relative_residx_bins: int | None = 32
    n_relative_chain_bins: int | None = 2
    num_loops: int | None = 10
    num_diffusion_samples: int | None = 8
    lm_d_model: int | None = 2560
    lm_num_layers: int | None = 80
    esmc_id: str | None = _DEFAULT_ESMC_HF_REPO
    msa_encoder_overwrite: bool | None = True
    inputs: dict | InputsEmbedderConfig | None = None
    folding_trunk: dict | FoldingTrunkConfig | None = None
    structure_head: dict | DiffusionStructureHeadConfig | None = None
    confidence_head: dict | ConfidenceHeadConfig | None = None
    msa_encoder: dict | MSAEncoderConfig | None = None
    parcae: dict | ParcaeConfig | None = None
    lm_encoder: dict | LMEncoderConfig | None = None

    def __post_init__(self, **kwargs):
        if self.type != "release":
            raise ValueError(
                "ESMFold2Config.type must be 'release' (the 'experimental' variant "
                f"is not included in this release), got {self.type!r}"
            )

        def _init_nested(cls, val):
            if val is None:
                return cls()
            if isinstance(val, dict):
                return cls(**val)
            return val

        self.inputs = _init_nested(InputsEmbedderConfig, self.inputs)
        self.folding_trunk = _init_nested(FoldingTrunkConfig, self.folding_trunk)
        self.structure_head = _init_nested(DiffusionStructureHeadConfig, self.structure_head)
        self.confidence_head = _init_nested(ConfidenceHeadConfig, self.confidence_head)
        self.msa_encoder = _init_nested(MSAEncoderConfig, self.msa_encoder)
        self.parcae = _init_nested(ParcaeConfig, self.parcae)
        self.lm_encoder = _init_nested(LMEncoderConfig, self.lm_encoder)

        super().__post_init__(**kwargs)


__all__ = ["ESMFold2Config"]
