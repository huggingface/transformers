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
from ..esmc.configuration_esmc import ESMCConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="biohub/ESMFold2")
@strict
class ESMFold2Config(PreTrainedConfig):
    r"""
    type (`str`, *optional*, defaults to `"release"`):
        Architecture variant. Only `"release"` is supported in this port (the
        `"experimental"` variant is deferred to a follow-up).
    hidden_size (`int`, *optional*, defaults to 384):
        Dimensionality of single (per-residue) representations.
    pairwise_hidden_size (`int`, *optional*, defaults to 256):
        Dimensionality of pair (residue-residue) representations.
    single_inputs_size (`int`, *optional*, defaults to 451):
        Width of the single-inputs tensor produced by the inputs embedder (consumed by the
        diffusion conditioning, confidence head and MSA encoder).
    transition_expansion_ratio (`int`, *optional*, defaults to 4):
        Expansion ratio for the pair- and MSA-stream SwiGLU transition FFNs (matches the
        reference ESMFold2 feed-forward blocks).
    pair_transition_intermediate_size (`int`, *optional*):
        Hidden size of the pair-stream SwiGLU transitions (folding trunk, LM encoder, parcae
        coda and MSA encoder). Derived as `transition_expansion_ratio * pairwise_hidden_size` if not set.
    sliding_window (`int`, *optional*, defaults to 128):
        Sliding-window size (in valid tokens) for the atom-encoder SWA attention.
    n_relative_residx_bins (`int`, *optional*, defaults to 32):
        Number of bins for relative residue index encoding.
    n_relative_chain_bins (`int`, *optional*, defaults to 2):
        Number of bins for relative chain encoding.
    num_loops (`int`, *optional*, defaults to 10):
        Number of trunk loops for iterative refinement.
    num_diffusion_samples (`int`, *optional*, defaults to 8):
        Number of parallel structure predictions to generate.
    num_res_types (`int`, *optional*, defaults to 33):
        Number of residue types for the one-hot residue/MSA features.
    max_atomic_number (`int`, *optional*, defaults to 128):
        Size of the element (atomic-number) one-hot in the atom featurization.
    char_vocab_size (`int`, *optional*, defaults to 64):
        Vocabulary size for the per-character atom-name encoding.
    max_chars (`int`, *optional*, defaults to 4):
        Maximum number of characters per atom name.
    msa_encoder_overwrite (`bool`, *optional*, defaults to `True`):
        If `True`, MSA encoder output replaces the pair stream; if `False`, it is added.
    folding_trunk_num_hidden_layers (`int`, *optional*, defaults to 24):
        Number of pairformer layers in the main folding trunk.
    atom_encoder_hidden_size (`int`, *optional*, defaults to 128):
        Atom-level width of the SWA atom encoder.
    atom_encoder_token_hidden_size (`int`, *optional*, defaults to 768):
        Token-level width of the SWA atom encoder.
    atom_encoder_num_hidden_layers (`int`, *optional*, defaults to 3):
        Number of blocks in the SWA atom encoder.
    atom_encoder_num_attention_heads (`int`, *optional*, defaults to 4):
        Number of attention heads in the SWA atom encoder.
    atom_encoder_expansion_ratio (`int`, *optional*, defaults to 2):
        Expansion ratio for the atom-encoder SwiGLU FFN.
    atom_encoder_ffn_intermediate_size (`int`, *optional*):
        SwiGLU FFN width for the atom encoder. Derived as
        `(expansion_ratio * (atom_encoder_hidden_size // 3) * 2)` rounded up to a multiple of 256 if not set.
    atom_encoder_spatial_rope_base_frequency (`float`, *optional*, defaults to 20.0):
        Base frequency of the 3D spatial RoPE in the atom encoder.
    atom_encoder_n_spatial_rope_pairs_per_axis (`int`, *optional*, defaults to 2):
        Number of spatial RoPE frequency pairs per spatial axis.
    atom_encoder_n_uid_rope_pairs (`int`, *optional*, defaults to 10):
        Number of RoPE frequency pairs used for the atom unique-id encoding.
    atom_encoder_uid_rope_base_frequency (`float`, *optional*, defaults to 10000.0):
        Base frequency of the atom unique-id RoPE.
    diffusion_sigma_data (`float`, *optional*, defaults to 16.0):
        EDM `sigma_data` for the diffusion module.
    diffusion_atom_hidden_size (`int`, *optional*, defaults to 128):
        Atom-level width of the diffusion module's atom encoder/decoder.
    diffusion_token_hidden_size (`int`, *optional*, defaults to 768):
        Token-level width of the diffusion module.
    diffusion_fourier_dim (`int`, *optional*, defaults to 256):
        Dimensionality of the Fourier noise embedding.
    diffusion_atom_num_blocks (`int`, *optional*, defaults to 3):
        Number of blocks in the diffusion atom encoder/decoder.
    diffusion_atom_num_heads (`int`, *optional*, defaults to 4):
        Number of attention heads in the diffusion atom encoder/decoder.
    diffusion_token_num_blocks (`int`, *optional*, defaults to 12):
        Number of blocks in the diffusion token transformer.
    diffusion_token_num_heads (`int`, *optional*, defaults to 16):
        Number of attention heads in the diffusion token transformer.
    diffusion_transition_multiplier (`int`, *optional*, defaults to 2):
        Multiplier for the diffusion transition FFN widths.
    diffusion_atom_expansion_ratio (`int`, *optional*, defaults to 2):
        Expansion ratio for the diffusion atom-encoder SwiGLU FFN.
    diffusion_atom_ffn_intermediate_size (`int`, *optional*):
        SwiGLU FFN width for the diffusion atom encoder. Derived as
        `(atom_expansion_ratio * (diffusion_atom_hidden_size // 3) * 2)` rounded up to a multiple of 256 if not set.
    diffusion_token_transition_intermediate_size (`int`, *optional*):
        Diffusion token transition FFN width. Derived as
        `diffusion_transition_multiplier * diffusion_token_hidden_size` if not set.
    structure_head_distogram_bins (`int`, *optional*, defaults to 128):
        Number of distogram bins predicted by the structure head.
    structure_head_gamma_0 (`float`, *optional*, defaults to 0.605):
        Sampling churn parameter `gamma_0`.
    structure_head_gamma_min (`float`, *optional*, defaults to 1.107):
        Minimum sigma below which no churn is applied.
    structure_head_noise_scale (`float`, *optional*, defaults to 0.0):
        Scale of the churn noise added during sampling.
    structure_head_step_scale (`float`, *optional*, defaults to 1.0):
        Scale applied to each denoising update step.
    structure_head_inference_s_max (`float`, *optional*, defaults to 160.0):
        Maximum sigma of the inference noise schedule.
    structure_head_inference_s_min (`float`, *optional*, defaults to 4e-4):
        Minimum sigma of the inference noise schedule.
    structure_head_inference_p (`float`, *optional*, defaults to 8.0):
        Power-law exponent of the inference noise schedule.
    structure_head_inference_num_steps (`int`, *optional*, defaults to 68):
        Default number of sampling steps.
    confidence_head_num_hidden_layers (`int`, *optional*, defaults to 4):
        Number of pairformer layers in the confidence head's folding trunk.
    confidence_head_num_plddt_bins (`int`, *optional*, defaults to 50):
        Number of pLDDT bins.
    confidence_head_num_pde_bins (`int`, *optional*, defaults to 64):
        Number of PDE bins.
    confidence_head_num_pae_bins (`int`, *optional*, defaults to 64):
        Number of PAE bins.
    confidence_head_min_dist (`float`, *optional*, defaults to 2.0):
        Minimum distance of the confidence-head distance binning.
    confidence_head_max_dist (`float`, *optional*, defaults to 52.0):
        Maximum distance of the confidence-head distance binning.
    confidence_head_distogram_bins (`int`, *optional*, defaults to 128):
        Number of distogram bins consumed by the confidence head.
    msa_encoder_enabled (`bool`, *optional*, defaults to `False`):
        Whether to build and run the optional MSA encoder (Large MSA models only).
    msa_encoder_hidden_size (`int`, *optional*, defaults to 128):
        MSA-stream width of the MSA encoder.
    msa_encoder_outer_hidden_size (`int`, *optional*, defaults to 32):
        Hidden size of the MSA encoder outer-product-mean projection.
    msa_encoder_num_hidden_layers (`int`, *optional*, defaults to 4):
        Number of MSA encoder blocks.
    msa_encoder_num_attention_heads (`int`, *optional*, defaults to 8):
        Number of MSA pair-weighted-averaging heads.
    msa_encoder_head_width (`int`, *optional*, defaults to 32):
        Per-head width of the MSA pair-weighted averaging.
    msa_encoder_transition_intermediate_size (`int`, *optional*):
        SwiGLU FFN width of the MSA-stream transition. Derived as
        `transition_expansion_ratio * msa_encoder_hidden_size` if not set.
    lm_encoder_enabled (`bool`, *optional*, defaults to `True`):
        Whether to build and run the LM-side pair encoder.
    lm_encoder_num_hidden_layers (`int`, *optional*, defaults to 4):
        Number of pairformer layers in the LM-side pair encoder.
    lm_encoder_lm_dropout (`float`, *optional*, defaults to 0.25):
        Dropout probability applied to the LM pair stream.
    lm_encoder_per_loop_lm_dropout (`bool`, *optional*, defaults to `True`):
        If `True`, LM dropout is resampled on every trunk loop (applied even at inference).
    parcae_num_coda_layers (`int`, *optional*, defaults to 2):
        Number of pairformer layers in the parcae coda.
    esmc_config (`ESMCConfig`, *optional*):
        Configuration of the bundled ESMC language-model backbone. Defaults to the
        ESMC-6B configuration. The backbone weights are part of the ESMFold2
        checkpoint (built with `AutoModel.from_config(esmc_config)`).

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
    sub_configs = {"esmc_config": ESMCConfig}

    type: str | None = "release"

    # Shared / top-level dims
    hidden_size: int | None = 384
    pairwise_hidden_size: int | None = 256
    single_inputs_size: int | None = 451
    transition_expansion_ratio: int | None = 4
    pair_transition_intermediate_size: int | None = None
    sliding_window: int | None = 128
    n_relative_residx_bins: int | None = 32
    n_relative_chain_bins: int | None = 2
    num_loops: int | None = 10
    num_diffusion_samples: int | None = 8
    num_res_types: int | None = 33
    max_atomic_number: int | None = 128
    char_vocab_size: int | None = 64
    max_chars: int | None = 4
    msa_encoder_overwrite: bool | None = True

    # Folding trunk
    folding_trunk_num_hidden_layers: int | None = 24

    # SWA atom encoder (inputs embedder)
    atom_encoder_hidden_size: int | None = 128
    atom_encoder_token_hidden_size: int | None = 768
    atom_encoder_num_hidden_layers: int | None = 3
    atom_encoder_num_attention_heads: int | None = 4
    atom_encoder_expansion_ratio: int | None = 2
    atom_encoder_ffn_intermediate_size: int | None = None
    atom_encoder_spatial_rope_base_frequency: float | None = 20.0
    atom_encoder_n_spatial_rope_pairs_per_axis: int | None = 2
    atom_encoder_n_uid_rope_pairs: int | None = 10
    atom_encoder_uid_rope_base_frequency: float | None = 10000.0

    # Diffusion module
    diffusion_sigma_data: float | None = 16.0
    diffusion_atom_hidden_size: int | None = 128
    diffusion_token_hidden_size: int | None = 768
    diffusion_fourier_dim: int | None = 256
    diffusion_atom_num_blocks: int | None = 3
    diffusion_atom_num_heads: int | None = 4
    diffusion_token_num_blocks: int | None = 12
    diffusion_token_num_heads: int | None = 16
    diffusion_transition_multiplier: int | None = 2
    diffusion_atom_expansion_ratio: int | None = 2
    diffusion_atom_ffn_intermediate_size: int | None = None
    diffusion_token_transition_intermediate_size: int | None = None

    # Diffusion structure head (sampling)
    structure_head_distogram_bins: int | None = 128
    structure_head_gamma_0: float | None = 0.605
    structure_head_gamma_min: float | None = 1.107
    structure_head_noise_scale: float | None = 0.0
    structure_head_step_scale: float | None = 1.0
    structure_head_inference_s_max: float | None = 160.0
    structure_head_inference_s_min: float | None = 4e-4
    structure_head_inference_p: float | None = 8.0
    structure_head_inference_num_steps: int | None = 68

    # Confidence head
    confidence_head_num_hidden_layers: int | None = 4
    confidence_head_num_plddt_bins: int | None = 50
    confidence_head_num_pde_bins: int | None = 64
    confidence_head_num_pae_bins: int | None = 64
    confidence_head_min_dist: float | None = 2.0
    confidence_head_max_dist: float | None = 52.0
    confidence_head_distogram_bins: int | None = 128

    # MSA encoder
    msa_encoder_enabled: bool | None = False
    msa_encoder_hidden_size: int | None = 128
    msa_encoder_outer_hidden_size: int | None = 32
    msa_encoder_num_hidden_layers: int | None = 4
    msa_encoder_num_attention_heads: int | None = 8
    msa_encoder_head_width: int | None = 32
    msa_encoder_transition_intermediate_size: int | None = None

    # LM-side pair encoder
    lm_encoder_enabled: bool | None = True
    lm_encoder_num_hidden_layers: int | None = 4
    lm_encoder_lm_dropout: float | None = 0.25
    lm_encoder_per_loop_lm_dropout: bool | None = True

    # Parcae diffusion-loop scheduler
    parcae_num_coda_layers: int | None = 2

    # Bundled ESMC language-model backbone
    esmc_config: dict | ESMCConfig | None = None

    def __post_init__(self, **kwargs):
        if self.type != "release":
            raise ValueError(
                "ESMFold2Config.type must be 'release' (the 'experimental' variant "
                f"is not included in this release), got {self.type!r}"
            )

        if self.esmc_config is None:
            self.esmc_config = ESMCConfig()
        elif isinstance(self.esmc_config, dict):
            self.esmc_config = ESMCConfig(**self.esmc_config)

        # SwiGLU FFN widths that are derived from the stream widths when not set explicitly
        # (matches the reference ESMFold2 feed-forward blocks). The atom-stack FFNs are rounded
        # up to a multiple of 256 (hardware-aligned width).
        if self.pair_transition_intermediate_size is None:
            self.pair_transition_intermediate_size = self.transition_expansion_ratio * self.pairwise_hidden_size
        if self.atom_encoder_ffn_intermediate_size is None:
            self.atom_encoder_ffn_intermediate_size = (
                (self.atom_encoder_expansion_ratio * (self.atom_encoder_hidden_size // 3) * 2 + 255) // 256 * 256
            )
        if self.diffusion_atom_ffn_intermediate_size is None:
            self.diffusion_atom_ffn_intermediate_size = (
                (self.diffusion_atom_expansion_ratio * (self.diffusion_atom_hidden_size // 3) * 2 + 255) // 256 * 256
            )
        if self.diffusion_token_transition_intermediate_size is None:
            self.diffusion_token_transition_intermediate_size = (
                self.diffusion_transition_multiplier * self.diffusion_token_hidden_size
            )
        if self.msa_encoder_transition_intermediate_size is None:
            self.msa_encoder_transition_intermediate_size = (
                self.transition_expansion_ratio * self.msa_encoder_hidden_size
            )

        super().__post_init__(**kwargs)


__all__ = ["ESMFold2Config"]
