from dataclasses import dataclass
from typing import Any, Optional

import torch

from ...configuration_utils import PretrainedConfig
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..dac.configuration_dac import DacConfig
from ..pe_audio.configuration_pe_audio import PEAudioTransformerConfig
from ..pe_audio.modeling_pe_audio import (
    DacEncoderVAE,
    PEAudioTransformer,
)


class SamAudioJudgeTransformerConfig(PEAudioTransformerConfig): ...


class DACVAEConfig(DacConfig): ...


class SamAudioJudgeConfig(PretrainedConfig):
    """
    SamAudioJudgeConfig
    """

    sub_configs = {"text_model": AutoConfig}
    model_type = "sam_audio_judge"

    def __init__(
        self,
        dac_vae_encoder: Optional[dict] = None,
        transformer: Optional[dict] = None,
        finetune_transformer: Optional[dict] = None,
        text_model: Optional[dict] = None,
        nth_text_layer: Optional[int] = 22,
        bottleneck_dim: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        dac_vae_encoder = dac_vae_encoder or {}
        transformer = transformer or {}
        text_model = text_model or {}
        finetune_transformer = finetune_transformer or {}
        self.dac_vae_encoder = DACVAEConfig(**dac_vae_encoder)
        self.transformer = SamAudioJudgeTransformerConfig(**transformer)
        self.finetune_transformer = SamAudioJudgeTransformerConfig(**finetune_transformer)
        self.text_model = CONFIG_MAPPING[kwargs.get("model_type", "modernbert")](**text_model)
        self.nth_text_layer = nth_text_layer
        self.bottleneck_dim = bottleneck_dim


class SamAudioJudgeDacEncoderVAE(DacEncoderVAE): ...


class SamAudioJudgeTransformer(PEAudioTransformer): ...


class SamAudioJudgePretrainedModel(PreTrainedModel):
    config: SamAudioJudgeConfig
    base_model_prefix = "sam_audio_judge"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True


@dataclass
@auto_docstring
class SamAudioJudgeOutput(ModelOutput):
    r"""
    overall (torch.Tensor, optional): Overall score tensor of shape (batch_size, 1).
    recall (torch.Tensor, optional): Recall score tensor of shape (batch_size, 1).
    precision (torch.Tensor, optional): Precision score tensor of shape (batch_size, 1).
    faithfulness (torch.Tensor, optional): Faithfulness score tensor of shape (batch_size, 1).
    text_model_output (BaseModelOutputWithPooling): Output from the text model.
    audio_model_output (BaseModelOutputWithPooling): Output from the audio model.
    """

    overall: Optional[torch.Tensor] = None
    recall: Optional[torch.Tensor] = None
    precision: Optional[torch.Tensor] = None
    faithfulness: Optional[torch.Tensor] = None
    text_model_output: BaseModelOutputWithPooling = None
    audio_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> tuple[Any]:
        return tuple(self[k] if not k.endswith("model_output") else getattr(self, k).to_tuple() for k in self.keys())


class SamAudioJudgeModel(SamAudioJudgePretrainedModel):
    def __init__(self, config: SamAudioJudgeConfig):
        super().__init__(config)
        config.transformer._attn_implementation = config._attn_implementation
        config.finetune_transformer._attn_implementation = config._attn_implementation
        self.data_proj = torch.nn.Linear(config.dac_vae_encoder.codebook_dim, config.transformer.hidden_size)
        self.dac_vae_encoder = SamAudioJudgeDacEncoderVAE(config.dac_vae_encoder)
        self.transformer = SamAudioJudgeTransformer(config.transformer)
        self.finetune_transformer = SamAudioJudgeTransformer(config.finetune_transformer)
        self.text_model = AutoModel.from_config(config.text_model)
        self.cat_audio_proj = torch.nn.Linear(2 * config.transformer.hidden_size, config.bottleneck_dim)
        self.text_proj1 = torch.nn.Linear(
            in_features=config.text_model.hidden_size, out_features=config.transformer.hidden_size, bias=False
        )
        self.text_proj2 = torch.nn.Linear(
            in_features=config.transformer.hidden_size, out_features=config.bottleneck_dim
        )
        self.layer_norm = torch.nn.LayerNorm(config.bottleneck_dim)
        self.proj_audio_and_text = torch.nn.Linear(2 * config.bottleneck_dim, config.bottleneck_dim)
        self.finetune_data_proj = torch.nn.Linear(config.bottleneck_dim, config.finetune_transformer.hidden_size)
        self.head = torch.nn.Linear(config.finetune_transformer.hidden_size, 4, bias=False)
        self.mean = torch.nn.Parameter(torch.zeros(4, requires_grad=False))
        self.std = torch.nn.Parameter(torch.ones(4, requires_grad=False))

    def _get_text_output(self, input_ids, attention_mask):
        nth_layer = self.config.nth_text_layer
        output = self.text_model(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=nth_layer is not None
        )
        if nth_layer is None:
            text_model_output = output.last_hidden_state
        else:
            text_model_output = output.hidden_states[nth_layer]

        return BaseModelOutputWithPooling(last_hidden_state=text_model_output, pooler_output=text_model_output[:, 0])

    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor,  # tokenized text
        input_values: torch.Tensor,  # input audio waveform
        separated_values: torch.Tensor,  # separated audio waveform
        attention_mask: Optional[torch.Tensor] = None,  # text attention mask
        padding_mask: Optional[torch.Tensor] = None,  # audio padding mask
    ) -> SamAudioJudgeOutput:
        r"""
        Args:
            input_ids (torch.Tensor): Tokenized text input IDs.
            input_values (torch.Tensor): Audio waveform tensor for the input audio used in source separation.
            separated_values (torch.Tensor): Resulting waveform tensor from the separation model.
            attention_mask (Optional[torch.Tensor], optional): Attention mask for the text input. Defaults to None.
            padding_mask (Optional[torch.Tensor], optional): Padding mask for the audio input. Defaults to None.

        Examples:

        ```python
        from transformers import AutoProcessor, AutoModel

        model = AutoModel.from_pretrained("facebook/sam_audio_judge")
        processor = AutoProcessor.from_pretrained("facebook/sam_audio_judge")
        processed = processor(input_audio=[<"input mixture audio">], separated_audio=["<separated audio>"], text=["<prompt>"])
        result = model(**processed)
        ```

        """
        text_features = self.text_proj1(self._get_text_output(input_ids, attention_mask).pooler_output)
        stacked_audios = torch.cat([input_values, separated_values], dim=0)
        stacked_codec_features = self.dac_vae_encoder(stacked_audios)
        feature_padding_mask = None
        if padding_mask is not None:
            feature_padding_mask = padding_mask[:, :: self.dac_vae_encoder.config.hop_length]
        stacked_features = self.transformer(
            self.data_proj(stacked_codec_features.transpose(1, 2)), padding_mask=feature_padding_mask
        )
        input_features, hyp_features = stacked_features.last_hidden_state.chunk(2, 0)
        audio_features = self.cat_audio_proj(torch.cat([hyp_features, input_features], dim=2))
        expanded_text = self.layer_norm(self.text_proj2(text_features)).unsqueeze(1).expand_as(audio_features)
        audio_and_text = self.proj_audio_and_text(torch.cat([audio_features, expanded_text], dim=2))
        finetune_transformer_output = self.finetune_transformer(
            self.finetune_data_proj(audio_and_text), padding_mask=feature_padding_mask
        )
        result = self.head(finetune_transformer_output.last_hidden_state)
        if feature_padding_mask is not None:
            feature_padding_mask = feature_padding_mask.unsqueeze(-1)
        pooled = torch.masked.mean(result, mask=feature_padding_mask, dim=1)
        de_normalized = pooled * self.std + self.mean
        return SamAudioJudgeOutput(*de_normalized.chunk(4, dim=1))


__all__ = ["SamAudioJudgeConfig", "SamAudioJudgeModel"]
