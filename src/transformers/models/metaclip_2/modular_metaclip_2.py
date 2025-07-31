from typing import Optional

import torch
from torch import nn

from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, logging
from ..clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from ..clip.modeling_clip import (
    CLIPMLP,
    CLIPAttention,
    CLIPForImageClassification,
    CLIPModel,
    CLIPTextEmbeddings,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTextTransformer,
    CLIPVisionEmbeddings,
    CLIPVisionModel,
    CLIPVisionModelWithProjection,
)


logger = logging.get_logger(__name__)


class MetaCLIP2TextConfig(CLIPTextConfig):
    pass


class MetaCLIP2VisionConfig(CLIPVisionConfig):
    pass


class MetaCLIP2Config(CLIPConfig):
    pass


class MetaCLIP2TextEmbeddings(CLIPTextEmbeddings):
    pass


class MetaCLIP2VisionEmbeddings(CLIPVisionEmbeddings):
    pass


class MetaCLIP2Attention(CLIPAttention):
    pass


class MetaCLIP2MLP(CLIPMLP):
    pass


@auto_docstring
class MetaCLIP2PreTrainedModel(PreTrainedModel):
    config: MetaCLIP2Config
    base_model_prefix = "metaclip_2"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    # Copied from transformers.models.clip.modeling_clip.CLIPPreTrainedModel._init_weights with CLIP->MetaCLIP2
    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, MetaCLIP2TextEmbeddings):
            module.token_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
            module.position_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
        elif isinstance(module, MetaCLIP2VisionEmbeddings):
            factor = self.config.initializer_factor
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
        elif isinstance(module, MetaCLIP2Attention):
            factor = self.config.initializer_factor
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (module.embed_dim**-0.5) * factor
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        elif isinstance(module, MetaCLIP2MLP):
            factor = self.config.initializer_factor
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
        elif isinstance(module, MetaCLIP2Model):
            nn.init.normal_(
                module.text_projection.weight,
                std=module.text_embed_dim**-0.5 * self.config.initializer_factor,
            )
            nn.init.normal_(
                module.visual_projection.weight,
                std=module.vision_embed_dim**-0.5 * self.config.initializer_factor,
            )
        elif isinstance(module, MetaCLIP2VisionModelWithProjection):
            nn.init.normal_(
                module.visual_projection.weight,
                std=self.config.hidden_size**-0.5 * self.config.initializer_factor,
            )
        elif isinstance(module, MetaCLIP2TextModelWithProjection):
            nn.init.normal_(
                module.text_projection.weight,
                std=self.config.hidden_size**-0.5 * self.config.initializer_factor,
            )
        elif isinstance(module, MetaCLIP2ForImageClassification):
            nn.init.normal_(
                module.classifier.weight,
                std=self.config.vision_config.hidden_size**-0.5 * self.config.initializer_factor,
            )

        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class MetaCLIP2TextTransformer(CLIPTextTransformer):
    @auto_docstring
    def forward(
        self,
        input_ids,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutputWithPooling:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )

        # expand attention_mask
        if attention_mask is not None and not self._use_flash_attention_2:
            # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        index = (input_ids == 2).nonzero()
        pooled_output = last_hidden_state[index[:, 0], index[:, 1]]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class MetaCLIP2TextModel(CLIPTextModel):
    def __init__(self, config: MetaCLIP2TextConfig):
        super().__init__(config)
        self.text_model = MetaCLIP2TextTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()


class MetaCLIP2TextModelWithProjection(CLIPTextModelWithProjection):
    def __init__(self, config: MetaCLIP2TextConfig):
        super().__init__(config)

        text_model = MetaCLIP2TextModel._from_config(config)
        self.text_model = text_model.text_model

        self.text_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


class MetaCLIP2Model(CLIPModel):
    def __init__(self, config: MetaCLIP2Config):
        super().__init__(config)

        if not isinstance(config.text_config, MetaCLIP2TextConfig):
            raise TypeError(
                "config.text_config is expected to be of type MetaCLIP2TextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, MetaCLIP2VisionConfig):
            raise TypeError(
                "config.vision_config is expected to be of type MetaCLIP2VisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        text_model = MetaCLIP2TextModel._from_config(text_config)
        self.text_model = text_model.text_model

        vision_model = MetaCLIP2VisionModel._from_config(vision_config)
        self.vision_model = vision_model.vision_model

        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # Initialize weights and apply final processing
        self.post_init()


class MetaCLIP2VisionModel(CLIPVisionModel):
    pass


class MetaCLIP2VisionModelWithProjection(CLIPVisionModelWithProjection):
    pass


class MetaCLIP2ForImageClassification(CLIPForImageClassification):
    pass


__all__ = [
    "MetaCLIP2Config",
    "MetaCLIP2TextConfig",
    "MetaCLIP2VisionConfig",
    "MetaCLIP2Model",
    "MetaCLIP2PreTrainedModel",
    "MetaCLIP2TextModel",
    "MetaCLIP2TextModelWithProjection",
    "MetaCLIP2VisionModel",
    "MetaCLIP2VisionModelWithProjection",
    "MetaCLIP2ForImageClassification",
]
