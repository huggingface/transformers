# Copyright 2026 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
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
"""MossAudioTokenizer model configuration"""

from typing import Any

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="OpenMOSS-Team/MOSS-Audio-Tokenizer")
@strict
class MossAudioTokenizerConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MossAudioTokenizerModel`]. It is used to instantiate a
    MossAudioTokenizer model according to the specified arguments, defining the model architecture.

    Instantiating a configuration with the defaults will yield a similar configuration to that of the
    [VoiceAgentGroup/moss_audio_tokenizer](https://huggingface.co/VoiceAgentGroup/moss_audio_tokenizer) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    version (`str`, *optional*):
        Version string accepted for checkpoint compatibility.
    sampling_rate (`int`, *optional*, defaults to 24000):
        The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).
    downsample_rate (`int`, *optional*, defaults to 1920):
        Total downsampling rate from waveform to tokens.
    causal_transformer_context_duration (`float`, *optional*, defaults to 10.0):
        Context duration in seconds for causal transformer.
    encoder_kwargs (`list[dict]`, *optional*):
        List of encoder module configurations. Each dict specifies a module type and its parameters.
    decoder_kwargs (`list[dict]`, *optional*):
        List of decoder module configurations in execution order.
    quantizer_type (`str`, *optional*, defaults to `"rlfq"`):
        Quantizer type. Options include `"rvq"`, `"spec_rvq"`, `"rlfq"`, `"random_prefix_rlfq"`.
    quantizer_kwargs (`dict`, *optional*):
        Configuration for the quantizer including `input_dim`, `rvq_dim`, `output_dim`, `num_quantizers`,
        `codebook_size`, and `codebook_dim`.

    Example:

    ```python
    >>> from transformers import MossAudioTokenizerModel, MossAudioTokenizerConfig

    >>> # Initializing a MossAudioTokenizer style configuration
    >>> configuration = MossAudioTokenizerConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = MossAudioTokenizerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "moss-audio-tokenizer"

    # Backward-compatible alias used by some checkpoints.
    attribute_map = {"sample_rate": "sampling_rate"}

    sampling_rate: int
    downsample_rate: int
    causal_transformer_context_duration: int | float
    encoder_kwargs: list[dict[str, Any]]
    decoder_kwargs: list[dict[str, Any]]
    quantizer_type: str
    quantizer_kwargs: dict[str, Any]

    def __init__(
        self,
        version: str | None = None,
        sampling_rate: int = 24000,
        downsample_rate: int = 1920,
        causal_transformer_context_duration: int | float = 10.0,
        encoder_kwargs: list[dict[str, Any]] | None = None,
        decoder_kwargs: list[dict[str, Any]] | None = None,
        quantizer_type: str = "rlfq",
        quantizer_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        r"""
        version (`str`, *optional*):
            Version string accepted for checkpoint compatibility.
        sampling_rate (`int`, *optional*, defaults to 24000):
            The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).
        downsample_rate (`int`, *optional*, defaults to 1920):
            Total downsampling rate from waveform to tokens.
        causal_transformer_context_duration (`int` or `float`, *optional*, defaults to 10.0):
            Context duration in seconds for causal transformer.
        encoder_kwargs (`list[dict]`, *optional*):
            List of encoder module configurations. Each dict specifies a module type and its parameters.
        decoder_kwargs (`list[dict]`, *optional*):
            List of decoder module configurations in execution order.
        quantizer_type (`str`, *optional*, defaults to `"rlfq"`):
            Quantizer type. Options include `"rvq"`, `"spec_rvq"`, `"rlfq"`, `"random_prefix_rlfq"`.
        quantizer_kwargs (`dict`, *optional*):
            Configuration for the quantizer including `input_dim`, `rvq_dim`, `output_dim`, `num_quantizers`,
            `codebook_size`, and `codebook_dim`.
        """
        # Some checkpoints might include an incorrect/legacy `model_type` (e.g. "speech_tokenizer").
        # We drop it to avoid overriding the class-level `model_type`.
        kwargs.pop("model_type", None)

        # `version` is accepted for compatibility but not used in modeling.
        self.version = version
        self.sampling_rate = sampling_rate
        self.downsample_rate = downsample_rate
        self.causal_transformer_context_duration = causal_transformer_context_duration
        # Default encoder configuration
        if encoder_kwargs is None:
            encoder_kwargs = [
                {
                    "module_type": "PatchedPretransform",
                    "patch_size": 240,
                },
                {
                    "module_type": "Transformer",
                    "input_dimension": 240,
                    "output_dimension": 384,
                    "d_model": 768,
                    "num_heads": 12,
                    "num_layers": 12,
                    "dim_feedforward": 3072,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                },
                {
                    "module_type": "PatchedPretransform",
                    "patch_size": 2,
                },
                {
                    "module_type": "Transformer",
                    "input_dimension": 768,
                    "output_dimension": 384,
                    "d_model": 768,
                    "num_heads": 12,
                    "num_layers": 12,
                    "dim_feedforward": 3072,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                },
                {
                    "module_type": "PatchedPretransform",
                    "patch_size": 2,
                },
                {
                    "module_type": "Transformer",
                    "input_dimension": 768,
                    "output_dimension": 640,
                    "d_model": 768,
                    "num_heads": 12,
                    "num_layers": 12,
                    "dim_feedforward": 3072,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                },
                {
                    "module_type": "PatchedPretransform",
                    "patch_size": 2,
                },
                {
                    "module_type": "Transformer",
                    "input_dimension": 1280,
                    "output_dimension": 768,
                    "d_model": 1280,
                    "num_heads": 20,
                    "num_layers": 32,
                    "dim_feedforward": 5120,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                },
            ]
        self.encoder_kwargs = encoder_kwargs

        # Default decoder configuration (execution order)
        if decoder_kwargs is None:
            decoder_kwargs = [
                {
                    "module_type": "Transformer",
                    "input_dimension": 768,
                    "output_dimension": 1280,
                    "d_model": 1280,
                    "num_heads": 20,
                    "num_layers": 32,
                    "dim_feedforward": 5120,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                },
                {
                    "module_type": "PatchedPretransform",
                    "patch_size": 2,
                },
                {
                    "module_type": "Transformer",
                    "input_dimension": 640,
                    "output_dimension": 768,
                    "d_model": 768,
                    "num_heads": 12,
                    "num_layers": 12,
                    "dim_feedforward": 3072,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                },
                {
                    "module_type": "PatchedPretransform",
                    "patch_size": 2,
                },
                {
                    "module_type": "Transformer",
                    "input_dimension": 384,
                    "output_dimension": 768,
                    "d_model": 768,
                    "num_heads": 12,
                    "num_layers": 12,
                    "dim_feedforward": 3072,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                },
                {
                    "module_type": "PatchedPretransform",
                    "patch_size": 2,
                },
                {
                    "module_type": "Transformer",
                    "input_dimension": 384,
                    "output_dimension": 768,
                    "d_model": 768,
                    "num_heads": 12,
                    "num_layers": 12,
                    "dim_feedforward": 3072,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                },
                {
                    "module_type": "PatchedPretransform",
                    "patch_size": 2,
                },
                {
                    "module_type": "Transformer",
                    "input_dimension": 384,
                    "output_dimension": 240,
                    "d_model": 768,
                    "num_heads": 12,
                    "num_layers": 12,
                    "dim_feedforward": 3072,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                },
                {
                    "module_type": "PatchedPretransform",
                    "patch_size": 240,
                },
            ]
        self.decoder_kwargs = decoder_kwargs

        # Default quantizer configuration
        if quantizer_kwargs is None:
            quantizer_kwargs = {
                "input_dim": 768,
                "rvq_dim": 512,
                "output_dim": 768,
                "num_quantizers": 32,
                "codebook_size": 1024,
                "codebook_dim": 8,
                "quantizer_type": "rlfq",
            }

        # Handle quantizer_type from kwargs or config
        kw_qtype = quantizer_kwargs.get("quantizer_type")
        if kw_qtype is not None:
            self.quantizer_type = kw_qtype
        else:
            self.quantizer_type = quantizer_type
            quantizer_kwargs["quantizer_type"] = quantizer_type

        self.quantizer_kwargs = quantizer_kwargs

        super().__init__(**kwargs)

    @property
    def num_quantizers(self) -> int:
        """Return the number of quantizers from quantizer_kwargs."""
        return self.quantizer_kwargs.get("num_quantizers", 32)

    @property
    def codebook_size(self) -> int:
        """Return the codebook size from quantizer_kwargs."""
        return self.quantizer_kwargs.get("codebook_size", 4096)

    @property
    def frame_rate(self) -> float:
        """Return the frame rate (tokens per second)."""
        return self.sampling_rate / self.downsample_rate


__all__ = ["MossAudioTokenizerConfig"]
