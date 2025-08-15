from typing import Optional

import torch
from torch import nn

from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import check_model_inputs
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


class MetaClip2TextConfig(CLIPTextConfig):
    pass


class MetaClip2VisionConfig(CLIPVisionConfig):
    pass


class MetaClip2Config(CLIPConfig):
    pass


class MetaClip2TextEmbeddings(CLIPTextEmbeddings):
    pass


class MetaClip2VisionEmbeddings(CLIPVisionEmbeddings):
    pass


class MetaClip2Attention(CLIPAttention):
    pass


class MetaClip2MLP(CLIPMLP):
    pass


@auto_docstring
class MetaClip2PreTrainedModel(PreTrainedModel):
    config: MetaClip2Config
    base_model_prefix = "metaclip_2"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, MetaClip2TextEmbeddings):
            module.token_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
            module.position_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
        elif isinstance(module, MetaClip2VisionEmbeddings):
            factor = self.config.initializer_factor
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
        elif isinstance(module, MetaClip2Attention):
            factor = self.config.initializer_factor
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (module.embed_dim**-0.5) * factor
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        elif isinstance(module, MetaClip2MLP):
            factor = self.config.initializer_factor
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
        elif isinstance(module, MetaClip2Model):
            nn.init.normal_(
                module.text_projection.weight,
                std=module.text_embed_dim**-0.5 * self.config.initializer_factor,
            )
            nn.init.normal_(
                module.visual_projection.weight,
                std=module.vision_embed_dim**-0.5 * self.config.initializer_factor,
            )
        elif isinstance(module, MetaClip2VisionModelWithProjection):
            nn.init.normal_(
                module.visual_projection.weight,
                std=self.config.hidden_size**-0.5 * self.config.initializer_factor,
            )
        elif isinstance(module, MetaClip2TextModelWithProjection):
            nn.init.normal_(
                module.text_projection.weight,
                std=self.config.hidden_size**-0.5 * self.config.initializer_factor,
            )
        elif isinstance(module, MetaClip2ForImageClassification):
            nn.init.normal_(
                module.classifier.weight,
                std=self.config.vision_config.hidden_size**-0.5 * self.config.initializer_factor,
            )

        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class MetaClip2TextTransformer(CLIPTextTransformer):
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
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
            **kwargs,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # Use robust pooling like CLIP - finds the first EOS token position per sequence
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id).int().argmax(dim=-1),
        ]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class MetaClip2TextModel(CLIPTextModel):
    def __init__(self, config: MetaClip2TextConfig):
        super().__init__(config)
        self.text_model = MetaClip2TextTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()


class MetaClip2TextModelWithProjection(CLIPTextModelWithProjection):
    def __init__(self, config: MetaClip2TextConfig):
        super().__init__(config)

        text_model = MetaClip2TextModel._from_config(config)
        self.text_model = text_model.text_model

        self.text_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


class MetaClip2Model(CLIPModel):
    def __init__(self, config: MetaClip2Config):
        super().__init__(config)

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        text_model = MetaClip2TextModel._from_config(text_config)
        self.text_model = text_model.text_model

        vision_model = MetaClip2VisionModel._from_config(vision_config)
        self.vision_model = vision_model.vision_model

        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # Initialize weights and apply final processing
        self.post_init()


class MetaClip2VisionModel(CLIPVisionModel):
    pass


class MetaClip2VisionModelWithProjection(CLIPVisionModelWithProjection):
    pass


class MetaClip2ForImageClassification(CLIPForImageClassification):
    pass


__all__ = [
    "MetaClip2Config",
    "MetaClip2TextConfig",
    "MetaClip2VisionConfig",
    "MetaClip2Model",
    "MetaClip2PreTrainedModel",
    "MetaClip2TextModel",
    "MetaClip2TextModelWithProjection",
    "MetaClip2VisionModel",
    "MetaClip2VisionModelWithProjection",
    "MetaClip2ForImageClassification",
]
