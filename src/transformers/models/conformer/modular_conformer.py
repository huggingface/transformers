# Copyright 2026 HuggingFace Inc. team. All rights reserved.
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
from typing import overload

import torch
from huggingface_hub.dataclasses import strict
from tokenizers import AddedToken, decoders, pre_tokenizers
from torch import nn

from ...modeling_outputs import CausalLMOutput
from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import auto_docstring
from ..parakeet.configuration_parakeet import ParakeetCTCConfig, ParakeetEncoderConfig
from ..parakeet.feature_extraction_parakeet import ParakeetFeatureExtractor
from ..parakeet.modeling_parakeet import (
    ParakeetEncoder,
    ParakeetEncoderModelOutput,
    ParakeetForCTC,
    ParakeetGenerateOutput,
    ParakeetPreTrainedModel,
)
from ..parakeet.processing_parakeet import ParakeetProcessor


class ConformerEncoderModelOutput(ParakeetEncoderModelOutput):
    pass


class ConformerGenerateOutput(ParakeetGenerateOutput):
    pass


@overload
def _compute_conv_output_length(
    input_length: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> int: ...
@overload
def _compute_conv_output_length(
    input_length: torch.Tensor,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> torch.Tensor: ...
def _compute_conv_output_length(
    input_length: int | torch.Tensor,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> int | torch.Tensor:
    return ((input_length + (2 * padding) - (dilation * (kernel_size - 1)) - 1) // stride) + 1


class ConformerTokenizer(TokenizersBackend):
    def __init__(
        self,
        bos_token: str | AddedToken | None = None,
        eos_token: str | AddedToken | None = None,
        unk_token: str | AddedToken | None = None,
        pad_token: str | AddedToken = "<pad>",
        **kwargs,
    ):
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            **kwargs,
        )

        self.backend_tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
        self.backend_tokenizer.decoder = decoders.Sequence(
            [
                decoders.CTC(pad_token=str(pad_token)),
                decoders.Metaspace(),
            ]
        )


class ConformerFeatureExtractor(ParakeetFeatureExtractor):
    pass


class ConformerProcessor(ParakeetProcessor):
    pass


@auto_docstring(checkpoint="nvidia/stt_en_conformer_ctc_large")
@strict
class ConformerEncoderConfig(ParakeetEncoderConfig):
    r"""
    convolution_bias (`bool`, *optional*, defaults to `True`):
        Whether to use bias in convolutions of the conformer's convolution module.
    conv_kernel_size (`int`, *optional*, defaults to 31):
        The kernel size of the convolution layers in the Conformer block.
    subsampling_factor (`int`, *optional*, defaults to 4):
        The factor by which the input sequence is subsampled. This value must be a power of 2 greater than 1.
    subsampling_conv_channels (`int`, *optional*, defaults to 512):
        The number of channels in the subsampling convolution layers.
    num_mel_bins (`int`, *optional*, defaults to 80):
        Number of mel features.
    subsampling_conv_kernel_size (`int`, *optional*, defaults to 3):
        The kernel size of the subsampling convolution layers.
    subsampling_conv_stride (`int`, *optional*, defaults to 2):
        The stride of the subsampling convolution layers.
    dropout_positions (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the positions in the input sequence.
    scale_input (`bool`, *optional*, defaults to `True`):
        Whether to scale the input embeddings.

    Example:
        ```python
        >>> from transformers import ConformerEncoder, ConformerEncoderConfig

        >>> # Initializing a `ConformerEncoder` configuration
        >>> configuration = ConformerEncoderConfig()

        >>> # Initializing a model from the configuration
        >>> model = ConformerEncoder(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```

    This configuration class is based on the ConformerEncoder architecture from NVIDIA NeMo.
    You can find more details and the original NeMo checkpoint at
    [nvidia/stt_en_conformer_ctc_large](https://huggingface.co/nvidia/stt_en_conformer_ctc_large).
    """

    model_type = "conformer_encoder"

    hidden_size: int = 512
    num_hidden_layers: int = 18
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    conv_kernel_size: int = 31
    subsampling_factor: int = 4
    subsampling_conv_channels: int = 512


@auto_docstring(checkpoint="nvidia/stt_en_conformer_ctc_large")
@strict
class ConformerCTCConfig(ParakeetCTCConfig):
    model_type = "conformer_ctc"
    sub_configs = {"encoder_config": ConformerEncoderConfig}


class ConformerEncoderSubsamplingConv2D(nn.Module):
    def __init__(self, config: ConformerEncoderConfig):
        super().__init__()

        kernel_size = config.subsampling_conv_kernel_size
        stride = config.subsampling_conv_stride
        padding = (config.subsampling_conv_kernel_size - 1) // 2
        num_layers = int(math.log2(config.subsampling_factor))

        self.layers = nn.ModuleList()
        for index in range(num_layers):
            self.layers += [
                nn.Conv2d(
                    1 if index == 0 else config.subsampling_conv_channels,
                    config.subsampling_conv_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
                nn.ReLU(),
            ]

        feature_size = config.num_mel_bins
        for _ in range(num_layers):
            feature_size = _compute_conv_output_length(feature_size, kernel_size, stride, padding)

        self.linear = nn.Linear(config.subsampling_conv_channels * feature_size, config.hidden_size, bias=True)

    def forward(self, input_features: torch.Tensor, attention_mask: torch.Tensor | None = None):
        hidden_states = input_features[:, None, ...]  # (B, 1, T, F)
        output_lengths = attention_mask.sum(dim=-1) if attention_mask is not None else None

        for layer in self.layers:
            hidden_states = layer(hidden_states)  # (B, C_out, T_out, F_out)

            if isinstance(layer, nn.Conv2d) and (output_lengths is not None):
                output_lengths = self._compute_output_lengths(output_lengths, layer)
                mask = (
                    torch.arange(hidden_states.shape[2], device=output_lengths.device)[None, :]
                    < output_lengths[:, None]
                )
                hidden_states *= mask[:, None, :, None]

        hidden_states = hidden_states.transpose(1, 2)  # (B, T_out, C_out, F_out)
        hidden_states = hidden_states.reshape(*hidden_states.shape[:2], -1)  # (B, T_out, C_out * F_out)
        hidden_states = self.linear(hidden_states)  # (B, T_out, H)

        return hidden_states

    def _compute_output_lengths(self, input_lengths: torch.Tensor, conv: nn.Conv2d) -> torch.Tensor:
        assert isinstance(conv.padding, tuple)

        return _compute_conv_output_length(
            input_lengths,
            kernel_size=conv.kernel_size[0],
            stride=conv.stride[0],
            padding=conv.padding[0],
        )


@auto_docstring
class ConformerPreTrainedModel(ParakeetPreTrainedModel):
    pass


@auto_docstring(
    custom_intro="""
    Conformer model based on the architecture described in the [paper](https://huggingface.co/papers/2005.08100).
    """
)
class ConformerEncoder(ParakeetEncoder):
    def forward(self, **super_kwargs) -> ConformerEncoderModelOutput:
        r"""
        output_attention_mask (`bool`, *optional*, defaults to `True`):
            Whether to return the output attention mask. Only effective when `attention_mask` is provided.

        Example:

        ```python
        >>> from transformers import AutoProcessor, ConformerEncoder
        >>> from datasets import load_dataset, Audio

        >>> model_id = "nvidia/stt_en_conformer_ctc_large"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> encoder = ConformerEncoder.from_pretrained(model_id)

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

        >>> inputs = processor(ds[0]["audio"]["array"])
        >>> encoder_outputs = encoder(**inputs)

        >>> print(encoder_outputs.last_hidden_state.shape)
        ```
        """

        return super().forward(**super_kwargs)


@auto_docstring(
    custom_intro="""
    Conformer model with a Connectionist Temporal Classification (CTC) head.
    """
)
class ConformerForCTC(ParakeetForCTC):
    def forward(self, **super_kwargs) -> CausalLMOutput:
        r"""
        Example:

        ```python
        >>> from transformers import AutoProcessor, ConformerForCTC
        >>> from datasets import load_dataset, Audio

        >>> model_id = "nvidia/stt_en_conformer_ctc_large"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = ConformerForCTC.from_pretrained(model_id)

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

        >>> inputs = processor(ds[0]["audio"]["array"], text=ds[0]["text"])
        >>> outputs = model(**inputs)

        >>> print(outputs.loss)
        ```
        """

        return super().forward(**super_kwargs)

    def generate(self, **super_kwargs) -> ConformerGenerateOutput | torch.LongTensor:
        r"""
        Example:

        ```python
        >>> from transformers import AutoProcessor, ConformerForCTC
        >>> from datasets import load_dataset, Audio

        >>> model_id = "nvidia/stt_en_conformer_ctc_large"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = ConformerForCTC.from_pretrained(model_id)

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

        >>> inputs = processor(ds[0]["audio"]["array"], text=ds[0]["text"])
        >>> predicted_ids = model.generate(**inputs)
        >>> transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        >>> print(transcription)
        ```
        """

        return super().generate(**super_kwargs)


__all__ = [
    "ConformerCTCConfig",
    "ConformerEncoder",
    "ConformerEncoderConfig",
    "ConformerFeatureExtractor",
    "ConformerForCTC",
    "ConformerPreTrainedModel",
    "ConformerProcessor",
    "ConformerTokenizer",
]
