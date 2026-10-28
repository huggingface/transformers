# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass

import torch
from torch import nn

from ...generation import CompileConfig
from ...modeling_outputs import BaseModelOutputWithPooling, CausalLMOutput
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ..auto import AutoModel
from ..omniasr.modeling_omniasr import OmniASRPreTrainedModel
from ..parakeet.modeling_parakeet import ParakeetCTCGenerateOutput, ParakeetForCTC
from .configuration_omniasr_ctc import OmniASRCTCConfig


@auto_docstring
class OmniASRCTCPreTrainedModel(OmniASRPreTrainedModel):
    config: OmniASRCTCConfig

    def _get_subsampling_output_length(self, input_lengths: torch.LongTensor | int) -> torch.LongTensor | int:
        lengths = input_lengths
        for kernel_size, stride in zip(self.config.audio_config.conv_kernel, self.config.audio_config.conv_stride):
            lengths = torch.div(lengths - kernel_size, stride, rounding_mode="floor") + 1

        return lengths


@auto_docstring(custom_intro="""Outputs of OmniASR CTC model generation.""")
@dataclass
class OmniASRCTCGenerateOutput(ParakeetCTCGenerateOutput):
    pass


@auto_docstring(
    custom_intro="""
    The OmniASR CTC model, i.e. the speech encoder without the CTC head.
    """
)
class OmniASRCTCModel(OmniASRCTCPreTrainedModel):
    def __init__(self, config: OmniASRCTCConfig):
        super().__init__(config)
        self.audio_tower = AutoModel.from_config(config.audio_config)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Padding mask used to pad `input_values`.
        """
        return self.audio_tower(input_values=input_values, padding_mask=padding_mask, **kwargs)


class OmniASRForCTC(ParakeetForCTC):
    config: OmniASRCTCConfig

    def __init__(self, config: OmniASRCTCConfig):
        super().__init__(config)
        del self.encoder
        self.model = OmniASRCTCModel(config)
        self.ctc_head = nn.Linear(config.audio_config.hidden_size, config.vocab_size)

    # same as ParakeetForCTC but with `input_values` instead of `input_features` as we use audio values directly
    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutput:
        r"""
        padding_mask (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`):
            Padding mask used to pad `input_values`.

        Example:

        ```python
        >>> from transformers import AutoProcessor, OmniASRForCTC
        >>> from datasets import load_dataset, Audio

        >>> model_id = "bezzam/omniasr-ctc-300m-v2"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = OmniASRForCTC.from_pretrained(model_id)

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

        >>> inputs = processor(ds[0]["audio"]["array"], text=ds[0]["text"])
        >>> outputs = model(**inputs)

        >>> print(outputs.loss)
        ```"""

        if labels is not None:
            kwargs.setdefault("output_attention_mask", True)
        encoder_outputs = self.model(
            input_values=input_values,
            padding_mask=padding_mask,
            **kwargs,
        )

        hidden_states = encoder_outputs.last_hidden_state
        logits = self.ctc_head(hidden_states)

        loss = None
        if labels is not None:
            encoder_lengths = encoder_outputs.attention_mask.sum(-1)

            loss = self.loss_function(
                logits=logits,
                labels=labels,
                logit_lengths=encoder_lengths,
                blank_token_id=self.config.pad_token_id,
                reduction=self.config.ctc_loss_reduction,
                zero_infinity=self.config.ctc_zero_infinity,
                **kwargs,
            )

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    # same as ParakeetForCTC but with `input_values` instead of `input_features` as we use audio values directly
    def generate(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        return_dict_in_generate: bool = False,
        compile_config: CompileConfig | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> OmniASRCTCGenerateOutput | torch.LongTensor:
        r"""
        compile_config ([`~generation.CompileConfig`], *optional*):
            If provided, `torch.compile` will be applied to the forward calls in the decoding loop.

        Example:

        ```python
        >>> from transformers import AutoProcessor, OmniASRForCTC
        >>> from datasets import load_dataset, Audio

        >>> model_id = "bezzam/omniasr-ctc-300m-v2"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = OmniASRForCTC.from_pretrained(model_id)

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

        >>> inputs = processor(ds[0]["audio"]["array"], text=ds[0]["text"])
        >>> predicted_ids = model.generate(**inputs)
        >>> transcription = processor.decode(predicted_ids, skip_special_tokens=True)

        >>> print(transcription)
        ```
        """
        model_forward = self.get_compiled_call(compile_config) if compile_config is not None else self.__call__

        kwargs["return_dict"] = True
        outputs: CausalLMOutput = model_forward(
            input_values=input_values,
            padding_mask=padding_mask,
            **kwargs,
        )

        # greedy decoding
        sequences = outputs.logits.argmax(dim=-1)

        # mask out padded tokens
        if padding_mask is not None:
            output_mask = self._get_output_attention_mask(padding_mask, target_length=sequences.shape[1])
            sequences[~output_mask] = self.config.pad_token_id

        if return_dict_in_generate:
            return OmniASRCTCGenerateOutput(
                sequences=sequences,
                logits=outputs.logits,
                attentions=outputs.attentions,
                hidden_states=outputs.hidden_states,
            )

        return sequences


__all__ = ["OmniASRForCTC", "OmniASRCTCModel", "OmniASRCTCPreTrainedModel"]
