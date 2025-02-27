import math
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

from ...activations import ACT2FN
from ...cache_utils import DynamicCache
from ...configuration_utils import PretrainedConfig
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...utils import (
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ..phi3.configuration_phi3 import Phi3Config
from ..phi3.modeling_phi3 import Phi3DecoderLayer, Phi3ForCausalLM, Phi3Model, Phi3RMSNorm
from ..siglip.configuration_siglip import SiglipVisionConfig
from ..siglip.modeling_siglip import (
    SiglipEncoder,
    SiglipEncoderLayer,
    SiglipMLP,
    SiglipMultiheadAttentionPoolingHead,
    SiglipPreTrainedModel,
    SiglipVisionEmbeddings,
    default_flax_embed_init,
    lecun_normal_,
)


logger = logging.get_logger(__name__)


class Phi4MultimodalVisionConfig(SiglipVisionConfig):
    model_type = "phi4_multimodal_vision"

    def __init__(
        self,
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_channels=3,
        image_size=448,
        patch_size=14,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        use_hd_transform: bool = True,
        crop_size: int = 448,
        hd_transform_order: str = "sub_glb",
        image_token_id: int = 200010,
        feature_layer: int = -2,
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_channels=num_channels,
            image_size=image_size,
            patch_size=patch_size,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            attention_dropout=attention_dropout,
            **kwargs,
        )
        self.use_hd_transform = use_hd_transform
        self.crop_size = crop_size
        self.hd_transform_order = hd_transform_order
        self.image_token_id = image_token_id
        self.feature_layer = feature_layer


class Phi4MultimodalAudioConfig(PretrainedConfig):
    model_type = "phi4_multimodal_audio"

    def __init__(
        self,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
        intermediate_size: int = 1536,
        activation: str = "swish",
        chunk_size: int = -1,
        left_chunk: int = 18,
        num_blocks: int = 24,
        dropout_rate: float = 0.0,
        causal: bool = True,
        batch_norm: bool = False,
        ext_pw_out_channel: int = 1024,
        ext_pw_kernel_size: int = 1,
        depthwise_seperable_out_channel: int = 1024,
        depthwise_multiplier: int = 1,
        kernel_size: int = 3,
        conv_activation: str = "swish",
        input_size: int = 80,
        conv_glu_type: str = "swish",
        bias_in_glu: bool = True,
        time_reduction: int = 8,
        bias_max_distance: int = 1000,
        bias_symmetric: bool = False,
        nemo_activation: str = "relu",
        nemo_conv_channels: int = 1024,
        downsample_rate: int = 1,
        audio_token_id: int = 200011,
        initializer_range: float = 0.02,
        feature_layer: int = -2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.activation = activation
        self.chunk_size = chunk_size
        self.left_chunk = left_chunk
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.causal = causal
        self.batch_norm = batch_norm
        self.ext_pw_out_channel = ext_pw_out_channel
        self.ext_pw_kernel_size = ext_pw_kernel_size
        self.depthwise_seperable_out_channel = depthwise_seperable_out_channel
        self.depthwise_multiplier = depthwise_multiplier
        self.kernel_size = kernel_size
        self.conv_activation = conv_activation
        self.input_size = input_size
        self.conv_glu_type = conv_glu_type
        self.bias_in_glu = bias_in_glu
        self.time_reduction = time_reduction
        self.bias_max_distance = bias_max_distance
        self.bias_symmetric = bias_symmetric
        self.nemo_activation = nemo_activation
        self.nemo_conv_channels = nemo_conv_channels
        self.downsample_rate = downsample_rate
        self.audio_token_id = audio_token_id
        self.initializer_range = initializer_range
        self.feature_layer = feature_layer


class Phi4MultimodalConfig(Phi3Config):
    sub_configs = {"audio_config": Phi4MultimodalAudioConfig, "vision_config": Phi4MultimodalVisionConfig}

    def __init__(
        self,
        vocab_size=200064,
        hidden_size=3072,
        intermediate_size=8192,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attention_dropout=0.0,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        partial_rotary_factor=1,
        bos_token_id=199999,
        eos_token_id=199999,
        pad_token_id=199999,
        original_max_position_embeddings=4096,
        sliding_window=None,
        vision_config=None,
        audio_config=None,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attention_dropout=attention_dropout,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            partial_rotary_factor=partial_rotary_factor,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            original_max_position_embeddings=original_max_position_embeddings,
            sliding_window=sliding_window,
            **kwargs,
        )

        if isinstance(vision_config, dict):
            vision_config = Phi4MultimodalVisionConfig(**vision_config)
        elif vision_config is None:
            Phi4MultimodalVisionConfig()
        self.vision_config = vision_config

        if isinstance(audio_config, dict):
            audio_config = Phi4MultimodalAudioConfig(**audio_config)
        elif vision_config is None:
            audio_config = Phi4MultimodalAudioConfig()
        self.audio_config = audio_config


# Special token ids
_COMPATIBLE_IMAGE_SPECIAL_TOKEN_ID_RANGE = [-9999, -1]  # For backward compatibility
_COMPATIBLE_AUDIO_SPECIAL_TOKEN_ID_RANGE = [float("-inf"), -10000]  # For backward compatibility


class Phi4MultimodalVisionMLP(SiglipMLP):
    pass


def simple_eager_attention_forward(
    module: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Phi4MultimodalVisionAttention(nn.Module):
    def __init__(self, config: Phi4MultimodalVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_interface: Callable = simple_eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights


class Phi4MultimodalVisionEncoderLayer(SiglipEncoderLayer):
    def __init__(self, config: Phi4MultimodalVisionConfig):
        super().__init__(config)
        self.self_attn = Phi4MultimodalVisionAttention(config)
        self.mlp = Phi4MultimodalVisionMLP(config)


class Phi4MultimodalVisionEncoder(SiglipEncoder):
    def __init__(self, config: Phi4MultimodalVisionConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [Phi4MultimodalVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )


class Phi4MultimodalVisionPreTrainedModel(SiglipPreTrainedModel):
    config_class = Phi4MultimodalVisionConfig
    base_model_prefix = "phi4_vision"
    supports_gradient_checkpointing = True

    _no_split_modules = ["Phi4MultimodalVisionEncoderLayer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, Phi4MultimodalVisionEmbeddings):
            width = (
                self.config.hidden_size
                if isinstance(self.config, Phi4MultimodalVisionConfig)
                else self.config.hidden_size
            )
            nn.init.normal_(module.position_embedding.weight, std=1 / np.sqrt(width))
        elif isinstance(module, nn.Embedding):
            default_flax_embed_init(module.weight)
        elif isinstance(module, Phi4MultimodalVisionAttention):
            nn.init.normal_(module.q_proj.weight)
            nn.init.normal_(module.k_proj.weight)
            nn.init.normal_(module.v_proj.weight)
            nn.init.normal_(module.out_proj.weight)
            nn.init.zeros_(module.q_proj.bias)
            nn.init.zeros_(module.k_proj.bias)
            nn.init.zeros_(module.v_proj.bias)
            nn.init.zeros_(module.out_proj.bias)
        elif isinstance(module, Phi4MultimodalVisionMLP):
            nn.init.normal_(module.fc1.weight)
            nn.init.normal_(module.fc2.weight)
            nn.init.normal_(module.fc1.bias, std=1e-6)
            nn.init.normal_(module.fc2.bias, std=1e-6)
        elif isinstance(module, Phi4MultimodalVisionMultiheadAttentionPoolingHead):
            nn.init.normal_(module.probe.data)
            nn.init.normal_(module.attention.in_proj_weight.data)
            nn.init.zeros_(module.attention.in_proj_bias.data)
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            lecun_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class Phi4MultimodalVisionEmbeddings(SiglipVisionEmbeddings, nn.Module):
    def __init__(self, config: Phi4MultimodalVisionConfig):
        nn.Module.__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.num_patches_per_side = config.image_size // self.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )
        self.position_embedding = nn.Embedding(self.num_patches_per_side**2, config.hidden_size)

    def forward(self, pixel_values: torch.FloatTensor, patch_attention_mask: torch.BoolTensor) -> torch.Tensor:
        batch_size = pixel_values.size(0)

        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        max_im_h, max_im_w = pixel_values.size(2), pixel_values.size(3)
        max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_size, max_im_w // self.patch_size
        boundaries = torch.arange(1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side)
        position_ids = torch.full((batch_size, max_nb_patches_h * max_nb_patches_w), fill_value=0)

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0].sum()

            fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
            fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

            pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids

        position_ids = position_ids.to(self.position_embedding.weight.device)

        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings


class Phi4MultimodalVisionMultiheadAttentionPoolingHead(SiglipMultiheadAttentionPoolingHead):
    def __init__(self, config: Phi4MultimodalVisionConfig):
        super().__init__(config)
        self.mlp = Phi4MultimodalVisionMLP(config)

    def forward(self, hidden_state, attention_mask):
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state = self.attention(
            query=probe, key=hidden_state, value=hidden_state, key_padding_mask=~attention_mask
        )[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


class Phi4MultimodalVisionModel(Phi4MultimodalVisionPreTrainedModel):
    config_class = Phi4MultimodalVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: Phi4MultimodalVisionConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = Phi4MultimodalVisionEmbeddings(config)
        self.encoder = Phi4MultimodalVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.head = Phi4MultimodalVisionMultiheadAttentionPoolingHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings.patch_embedding

    def forward(
        self,
        pixel_values,
        patch_attention_mask: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = pixel_values.size(0)
        if patch_attention_mask is None:
            patch_attention_mask = torch.ones(
                size=(
                    batch_size,
                    pixel_values.size(2) // self.config.patch_size,
                    pixel_values.size(3) // self.config.patch_size,
                ),
                dtype=torch.bool,
                device=pixel_values.device,
            )

        hidden_states = self.embeddings(pixel_values=pixel_values, patch_attention_mask=patch_attention_mask)

        patch_attention_mask = patch_attention_mask.view(batch_size, -1)
        # The call to `_upad_input` in `_flash_attention_forward` is expensive
        # So when the `patch_attention_mask` is full of 1s (i.e. attending to the whole sequence),
        # avoiding passing the attention_mask, which is equivalent to attending to the full sequence
        if not torch.any(~patch_attention_mask):
            attention_mask = None
        else:
            attention_mask = (
                _prepare_4d_attention_mask(patch_attention_mask, hidden_states.dtype)
                if not self.config._attn_implementation == "flash_attention_2"
                else patch_attention_mask
            )

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooled_output = self.head(
            hidden_state=last_hidden_state,
            attention_mask=patch_attention_mask,
        )

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class Phi4MultimodalImageEmbedding(nn.Module):
    """Image embedding."""

    def __init__(self, config: Phi4MultimodalConfig):
        super().__init__()
        self.config = config
        self.layer_idx = config.vision_config.feature_layer

        # n_embed or hidden_size
        hidden_size = config.hidden_size
        self.drop = nn.Dropout(config.embd_pdrop)

        self.img_processor = Phi4MultimodalVisionModel._from_config(config.vision_config)

        pe_weight = self.img_processor.embeddings.position_embedding.weight
        L, D = pe_weight.size()
        H = int(math.sqrt(L))
        assert H**2 == L
        if H % 2 != 0:
            self.img_processor_padding = nn.ReflectionPad2d((0, 1, 0, 1))
            H += 1
        self.num_img_tokens = (H // 2) ** 2

        self.image_dim_out = D
        self.image_attention_mask = None

        self.use_hd_transform = config.vision_config.use_hd_transform
        self.hd_transform_order = config.vision_config.hd_transform_order
        self.crop_size = config.vision_config.crop_size

        # image token compression
        self.image_token_compression = nn.AvgPool2d(kernel_size=2, stride=2)
        self.base_feat_height_reduction = 1

        if self.use_hd_transform:
            # 1024 * 4, merge spatial to channel dimension
            self.glb_GN = nn.Parameter(torch.zeros([1, 1, self.image_dim_out]))
            self.sub_GN = nn.Parameter(torch.zeros([1, 1, 1, self.image_dim_out]))

        dim_projection = hidden_size
        self.img_projection = nn.Sequential(
            nn.Linear(self.image_dim_out, dim_projection),
            nn.GELU(),
            nn.Linear(dim_projection, dim_projection),
        )

    def get_img_features(self, img_embeds: torch.FloatTensor, attention_mask=None) -> torch.FloatTensor:
        img_processor_output = self.img_processor(
            img_embeds, patch_attention_mask=attention_mask, output_hidden_states=True
        )
        img_feature = img_processor_output.hidden_states[self.layer_idx]

        patch_feature = img_feature
        # reshape to 2D tensor
        width = int(math.sqrt(patch_feature.size(1)))
        patch_feature = patch_feature.view(-1, width, width, patch_feature.size(-1))
        # convert to NCHW
        patch_feature = patch_feature.permute(0, 3, 1, 2)
        if getattr(self, "img_processor_padding", None) is not None:
            patch_feature = self.img_processor_padding(patch_feature)
        patch_feature = self.image_token_compression(patch_feature)
        # convert to NHWC
        patch_feature = patch_feature.permute(0, 2, 3, 1)
        patch_feature = patch_feature.view(-1, patch_feature.size(1) * patch_feature.size(2), patch_feature.size(-1))
        return patch_feature

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeds: torch.FloatTensor,
        image_sizes: Optional[torch.Tensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        wte: nn.Module = None,
    ) -> torch.FloatTensor:
        img_embeds = input_embeds
        if img_embeds is not None:
            img_embeds = img_embeds.to(self.img_processor.embeddings.patch_embedding.weight.dtype)

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        with torch.no_grad():
            positions = torch.nonzero(input_ids == self.config.vision_config.image_token_id, as_tuple=False)
            positions_tuple = torch.nonzero(input_ids == self.config.vision_config.image_token_id, as_tuple=True)

        select = False

        target_device = self.img_projection[0].bias.device
        target_dtype = self.img_projection[0].bias.dtype

        if len(positions.tolist()) > 0:
            select = True
            bs = img_embeds.shape[0]

            img_features = self.get_img_features(
                img_embeds.flatten(0, 1),
                attention_mask=image_attention_mask.type(torch.BoolTensor).flatten(0, 1).to(target_device),
            )

            base_resolution = self.crop_size
            base_feat_height = base_feat_width = int(np.sqrt(img_features.shape[1]))

            # bs x max_num_crops x (24x24) x C
            img_features = img_features.view(bs, -1, base_feat_height * base_feat_width, self.image_dim_out)
            C = self.image_dim_out
            H = base_feat_height

            output_imgs = []
            output_len = []
            # training is tensor, inference is list
            if isinstance(image_sizes, torch.Tensor):
                image_sizes = image_sizes.view(-1, 2)
            for _bs in range(bs):
                h, w = image_sizes[_bs]
                h = h // base_resolution
                w = w // base_resolution
                B_ = h * w

                # 1 x (24x24) x 1024
                global_img_feature = img_features[_bs, :1]

                # 1 x 12 x 12 x 4096
                glb_img = (
                    global_img_feature.reshape(1, H, 1, H, 1, C)
                    .contiguous()
                    .permute(0, 1, 3, 2, 4, 5)
                    .reshape(1, H, H, C)
                    .contiguous()
                )
                temp_glb_GN = self.sub_GN.repeat(1, H, 1, 1)

                # 1 x 156 x 4096
                glb_img = torch.cat([glb_img, temp_glb_GN], dim=2).reshape(1, -1, C)

                sub_img = img_features[_bs, 1:]
                sub_img = sub_img[:B_]

                sub_img = (
                    sub_img.reshape(B_, H, 1, H, 1, C)
                    .contiguous()
                    .permute(0, 1, 3, 2, 4, 5)
                    .reshape(B_, -1, C)
                    .contiguous()
                )
                sub_img = (
                    sub_img.reshape(1, h, w, base_feat_height, base_feat_width, -1)
                    .permute(0, 1, 3, 2, 4, 5)
                    .reshape(1, h * base_feat_height, w * base_feat_width, C)
                )

                if image_attention_mask is not None and len(image_attention_mask) > 0:
                    reshaped_image_attention_mask = (
                        image_attention_mask[_bs, 1 : B_ + 1, 0::2, 0::2]
                        .reshape(1, h, w, base_feat_height, base_feat_width)
                        .permute(0, 1, 3, 2, 4)
                        .reshape(1, h * base_feat_height, w * base_feat_width)
                    )
                    useful_height = int(reshaped_image_attention_mask[0, :, 0].sum().item())
                    useful_width = int(reshaped_image_attention_mask[0, 0, :].sum().item())
                    sub_img = sub_img[:, :useful_height, :useful_width]
                    temp_sub_GN = self.sub_GN.repeat(1, useful_height, 1, 1)
                    temp_len = (
                        int(image_attention_mask[_bs, : B_ + 1, 0::2, 0::2].sum().item())
                        + (useful_height + 1)
                        + base_feat_height
                    )
                else:
                    temp_sub_GN = self.sub_GN.repeat(1, h * base_feat_height, 1, 1)
                    temp_len = int((h * w + 1) * self.num_img_tokens + 1 + (h + 1) * base_feat_height)

                sub_img = torch.cat([sub_img, temp_sub_GN], dim=2).reshape(1, -1, C)

                # glb + sub
                if self.hd_transform_order == "glb_sub":
                    output_imgs.append(torch.cat([glb_img, self.glb_GN, sub_img], dim=1))
                elif self.hd_transform_order == "sub_glb":
                    output_imgs.append(torch.cat([sub_img, self.glb_GN, glb_img], dim=1))

                output_len.append(temp_len)

            img_set_tensor = []
            for _output_img in output_imgs:
                img_feature_proj = self.img_projection(_output_img.to(target_device).to(target_dtype))
                img_set_tensor.append(img_feature_proj)

        # we use the token embedding layer from the base model
        hidden_states = wte(input_ids)

        if select:
            merged_img_set_tensor = torch.cat(img_set_tensor, dim=1).squeeze(0)
            merged_img_set_tensor = merged_img_set_tensor.to(hidden_states.dtype).to(hidden_states.device)
            # Temporarily disable autocast to avoid issue on bf16 tensors
            # Ref: https://github.com/pytorch/pytorch/issues/132715
            with torch.autocast(device_type=hidden_states.device.type, enabled=False):
                new_hidden_states = hidden_states.index_put(
                    indices=positions_tuple, values=merged_img_set_tensor, accumulate=False
                )
            hidden_states = new_hidden_states

        if self.drop is not None:
            hidden_states = self.drop(hidden_states)

        return hidden_states


########################################################## AUDIO #############################################


class Phi4MultimodalAudioMLP(nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.act_fn = ACT2FN[config.activation]
        self.gate_up_proj = nn.Linear(config.hidden_size, config.intermediate_size * 2, config.bias_in_glu)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        up_states = self.gate_up_proj(hidden_states)
        up_states, gate = up_states.chunk(2, dim=-1)
        up_states = up_states * self.act_fn(gate)
        up_states = self.dropout(up_states)
        hidden_states = self.down_proj(up_states)
        out = self.dropout(hidden_states)

        return out


class Phi4MultimodalAudioAttention(nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__()
        self.config = config
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.dropout_rate
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        relative_attention_bias: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)
            if relative_attention_bias is not None:
                attention_mask = attention_mask + relative_attention_bias

        attention_interface: Callable = simple_eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class Phi4MultimodalAudioDepthWiseSeperableConv1d(nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig, padding: int = 0):
        super().__init__()
        self.dw_conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size * config.depthwise_multiplier,
            config.kernel_size,
            1,
            padding=padding,
            groups=config.hidden_size,
        )
        self.pw_conv = nn.Conv1d(
            config.hidden_size * config.depthwise_multiplier, config.depthwise_seperable_out_channel, 1, 1, 0
        )

    def forward(self, hidden_states):
        return self.pw_conv(self.dw_conv(hidden_states))


class Phi4MultimodalAudioGluPointWiseConv(nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__()
        self.config = config
        self.output_dim = config.ext_pw_out_channel
        kernel_size = config.ext_pw_kernel_size

        self.ext_pw_conv_1d = nn.Conv1d(
            config.hidden_size,
            config.ext_pw_out_channel * 2,
            kernel_size,
            1,
            padding=(kernel_size - 1) if config.causal else (kernel_size - 1) // 2,
        )

        self.glu_act = ACT2FN[config.conv_glu_type]

        if config.bias_in_glu:
            self.b1 = nn.Parameter(torch.zeros(1, config.ext_pw_out_channel, 1))
            self.b2 = nn.Parameter(torch.zeros(1, config.ext_pw_out_channel, 1))

    def forward(self, hidden_states):
        # to be consistent with GLULinear, we assume the input always has the #channel (#dim) in the last dimension of the
        # tensor, so need to switch the dimension first for 1D-Conv case
        hidden_states = hidden_states.permute([0, 2, 1])
        hidden_states = self.ext_pw_conv_1d(hidden_states)
        if self.config.bias_in_glu:
            hidden_states = (hidden_states[:, 0 : self.output_dim, :] + self.b1) * self.glu_act(
                hidden_states[:, self.output_dim : self.output_dim * 2, :] + self.b2
            )
        else:
            hidden_states = (hidden_states[:, 0 : self.output_dim, :]) * self.glu_act(
                hidden_states[:, self.output_dim : self.output_dim * 2, :]
            )

        out = hidden_states.permute([0, 2, 1])
        return out


class Phi4MultimodalAudioConvModule(nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__()
        self.config = config
        self.batch_norm = config.batch_norm
        self.kernel_size = config.kernel_size
        self.ext_pw_kernel_size = config.ext_pw_kernel_size

        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.glu = Phi4MultimodalAudioGluPointWiseConv(config)
        self.ln1 = (
            nn.Linear(config.ext_pw_out_channel, config.hidden_size)
            if config.hidden_size != config.ext_pw_out_channel
            else nn.Identity()
        )

        padding = config.kernel_size - 1 if config.causal else (config.kernel_size - 1) // 2
        self.dw_sep_conv_1d = Phi4MultimodalAudioDepthWiseSeperableConv1d(config, padding=padding)

        self.ln2 = (
            nn.Linear(config.depthwise_seperable_out_channel, config.hidden_size)
            if config.hidden_size != config.depthwise_seperable_out_channel
            else nn.Identity()
        )
        self.bn_layer = nn.BatchNorm1d(config.hidden_size) if config.batch_norm else nn.Identity()

        self.act = ACT2FN[config.conv_activation]

        self.ext_pw_conv_1d = nn.Conv1d(
            config.hidden_size,
            config.ext_pw_out_channel,
            config.ext_pw_kernel_size,
            1,
            padding=config.ext_pw_kernel_size - 1 if config.causal else (config.ext_pw_kernel_size - 1) // 2,
        )
        self.fix_len1 = True if config.causal and config.ext_pw_kernel_size > 1 else False
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.glu(self.layer_norm(hidden_states))
        if self.fix_len1:
            hidden_states = hidden_states[:, : -(self.ext_pw_kernel_size - 1), :]

        hidden_states = self.ln1(hidden_states)
        hidden_states = self.dw_sep_conv_1d(hidden_states.permute([0, 2, 1]))

        if self.config.causal and self.kernel_size > 1:
            hidden_states = hidden_states[:, :, : -(self.kernel_size - 1)]

        hidden_states = self.ln2(hidden_states.permute([0, 2, 1])).permute([0, 2, 1])
        hidden_states = self.bn_layer(hidden_states)
        hidden_states = self.act(hidden_states)

        hidden_states = self.ext_pw_conv_1d(hidden_states)
        if self.fix_len1:
            hidden_states = hidden_states[:, :, : -(self.ext_pw_kernel_size - 1)]

        hidden_states = self.ln1(hidden_states.permute([0, 2, 1]))
        out = self.dropout(hidden_states)
        return out


class Phi4MultimodalAudioConformerEncoderLayer(nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__()

        self.feed_forward_in = Phi4MultimodalAudioMLP(config)
        self.self_attn = Phi4MultimodalAudioAttention(config)
        self.conv = Phi4MultimodalAudioConvModule(config)
        self.feed_forward_out = Phi4MultimodalAudioMLP(config)
        self.layer_norm_att = nn.LayerNorm(config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor],
        relative_attention_bias: Optional[torch.Tensor] = None,
    ):
        hidden_states = hidden_states + 0.5 * self.feed_forward_in(hidden_states)
        hidden_states = self.layer_norm_att(hidden_states)

        hidden_states = hidden_states + self.self_attn(hidden_states, mask, relative_attention_bias)
        hidden_states = hidden_states + self.conv(hidden_states)
        hidden_states = hidden_states + 0.5 * self.feed_forward_out(hidden_states)

        out = self.layer_norm(hidden_states)

        return out


class Phi4MultimodalAudioNemoConvSubsampling(torch.nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__()
        self.subsampling_factor = config.time_reduction

        if self.subsampling_factor % 2 != 0:
            raise ValueError("Sampling factor should be a multiply of 2!")
        self.sampling_num = int(math.log(self.subsampling_factor, 2))

        self.act_fn = ACT2FN[config.nemo_activation]

        conv_channels = config.nemo_conv_channels
        layers = []

        layers.append(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=conv_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            )
        )
        layers.append(self.act_fn)

        for _ in range(self.sampling_num - 1):
            layers.append(
                torch.nn.Conv2d(
                    in_channels=conv_channels,
                    out_channels=conv_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=conv_channels,
                )
            )
            layers.append(
                torch.nn.Conv2d(
                    in_channels=conv_channels,
                    out_channels=conv_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                )
            )
            layers.append(self.act_fn)

        # Aggregate the layers
        self.conv = torch.nn.Sequential(*layers)

        in_length = torch.tensor(config.input_size, dtype=torch.float, device="cpu")
        out_length = calc_length(
            lengths=in_length,
            all_paddings=2,
            kernel_size=3,
            stride=2,
            ceil_mode=False,
            repeat_num=self.sampling_num,
        )
        self.out = torch.nn.Linear(conv_channels * int(out_length), config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, mask: Optional[torch.Tensor]):
        # Unsqueeze Channel Axis
        hidden_states = hidden_states.unsqueeze(1)

        hidden_states = self.conv(hidden_states)

        # Flatten Channel and Frequency Axes
        b, _, t, _ = hidden_states.size()
        hidden_states = self.out(hidden_states.transpose(1, 2).reshape(b, t, -1))

        if mask is None:
            return hidden_states, None

        max_audio_length = hidden_states.shape[1]
        feature_lens = mask.sum(1)
        padding_length = torch.ceil(feature_lens / self.subsampling_factor)
        arange_ = torch.arange(0, max_audio_length, device=hidden_states.device)
        pad_mask = arange_.expand(padding_length.size(0), -1) < padding_length.unsqueeze(1)
        return hidden_states, pad_mask.unsqueeze(1)


def calc_length(lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num=1):
    """Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
    add_pad: float = all_paddings - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)


class Phi4MultimodalAudioRelativeAttentionBias(nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__()

        self.max_distance = config.bias_max_distance
        self.symmetric = config.bias_symmetric
        self.num_buckets = self.max_distance
        if not config.bias_symmetric:
            self.num_buckets *= 2
        self.bias_values = nn.Embedding(self.num_buckets, config.num_attention_heads)

    def forward(self, x):
        # instantiate bias compatible with shape of x
        max_pos = x.size(1)
        context_position = torch.arange(max_pos, device=x.device, dtype=torch.long)[:, None]
        memory_position = torch.arange(max_pos, device=x.device, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        # clipping to a maximum distance using ops that play well with ONNX export
        relative_position = relative_position.masked_fill(relative_position < -self.max_distance, -self.max_distance)
        relative_position = relative_position.masked_fill(
            relative_position > self.max_distance - 1, self.max_distance - 1
        )

        # mapping from relative position to index in the bias parameter
        bias_idx = relative_position
        bias_idx = bias_idx.abs() if self.symmetric else bias_idx + self.num_buckets // 2

        att_bias = self.bias_values(bias_idx)  # [L, L, H]
        att_bias = att_bias.permute(2, 0, 1).unsqueeze(0)  # [1, H, L, L]

        return att_bias


class Phi4MultimodalAudioMeanVarianceNormLayer(nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__()
        self.register_buffer("global_mean", torch.zeros(config.input_size))
        self.register_buffer("global_invstd", torch.ones(config.input_size))

    def forward(self, x):
        return (x - self.global_mean) * self.global_invstd


class Phi4MultimodalAudioPreTrainedModel(PreTrainedModel):
    config_class = Phi4MultimodalAudioConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["Phi4MultimodalAudioConformerEncoderLayer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class Phi4MultimodalAudioModel(Phi4MultimodalAudioPreTrainedModel):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__(config)
        self.config = config

        self.encoder_embedding = Phi4MultimodalAudioMeanVarianceNormLayer(config)
        self.embed = Phi4MultimodalAudioNemoConvSubsampling(config)
        self.relative_attention_bias_layer = Phi4MultimodalAudioRelativeAttentionBias(config)
        self.encoders = nn.ModuleList(
            [Phi4MultimodalAudioConformerEncoderLayer(config) for _ in range(config.num_blocks)]
        )
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def _chunk_size_selection(self, chunk_size=None, left_chunk=None):
        """If chunk size is a list, we will randomly select a chunk size."""

        if chunk_size is None:
            chunk_size = self.chunk_size
        if left_chunk is None:
            left_chunk = self.left_chunk
        if isinstance(chunk_size, list):
            # Variable chunk size during training
            chunk_size_index = int(torch.randint(low=0, high=len(chunk_size), size=(1,)))
            chunk_size_train_eff = chunk_size[chunk_size_index]
            if not isinstance(left_chunk, list):
                raise ValueError("Since chunk_size is a list, left_chunk must be a list")
            if len(left_chunk) != len(chunk_size):
                raise ValueError("The length of left_chunk must be the same as length of chunk_size.")
            left_chunk_train_eff = left_chunk[chunk_size_index]
        else:
            chunk_size_train_eff = chunk_size
            left_chunk_train_eff = left_chunk

        return chunk_size_train_eff, left_chunk_train_eff

    def _streaming_mask(self, seq_len, batch_size, chunk_size, left_chunk):
        chunk_size_train_eff, left_chunk_train_eff = self._chunk_size_selection(chunk_size, left_chunk)

        # Create mask matrix for streaming
        # S stores start index. if chunksize is 18, s is [0,18,36,....]
        chunk_start_idx = np.arange(0, seq_len, chunk_size_train_eff)
        # avoid randomness when run evaluation or decoding
        if self.training and np.random.rand() > 0.5:
            # Either first or last chunk is not complete.
            # If only the last one is not complete, EOS is not effective
            chunk_start_idx = seq_len - chunk_start_idx
            chunk_start_idx = chunk_start_idx[::-1]
            chunk_start_idx = chunk_start_idx[:-1]
            chunk_start_idx = np.insert(chunk_start_idx, 0, 0)

        enc_streaming_mask = (
            adaptive_enc_mask(seq_len, chunk_start_idx, left_window=left_chunk_train_eff)
            .unsqueeze(0)
            .expand([batch_size, -1, -1])
        )
        return enc_streaming_mask

    def forward_embeddings(self, hidden_states, masks):
        """Forwarding the inputs through the top embedding layers"""
        seq_len = math.ceil(hidden_states.shape[1] / self.config.time_reduction)
        if seq_len <= 0:
            raise ValueError(
                f"""The squence length after time reduction is invalid: {seq_len}.
                Your input feature is too short. Consider filtering out the very
                short sentence from data loader""",
            )

        batch_size = hidden_states.shape[0]

        enc_streaming_mask = self._streaming_mask(seq_len, batch_size, self.config.chunk_size, self.config.left_chunk)
        enc_streaming_mask = enc_streaming_mask.to(hidden_states.device)

        hidden_states, masks = self.embed(hidden_states, masks)

        streaming_mask = enc_streaming_mask
        if streaming_mask is not None and masks is not None:
            hs_mask = masks & streaming_mask
        elif masks is not None:
            hs_mask = masks
        else:
            hs_mask = streaming_mask

        return hidden_states, hs_mask, masks

    def calculate_hs_mask(self, hidden_states, device, mask):
        max_audio_length = hidden_states.shape[1]
        batch_size = hidden_states.shape[0]
        enc_streaming_mask = self._streaming_mask(max_audio_length, batch_size, self.chunk_size, self.left_chunk)
        enc_streaming_mask = enc_streaming_mask.to(device)
        if mask is None:
            return enc_streaming_mask

        feature_lens = mask.sum(1)
        padding_length = feature_lens
        pad_mask = torch.arange(0, max_audio_length, device=device).expand(
            padding_length.size(0), -1
        ) < padding_length.unsqueeze(1)
        pad_mask = pad_mask.unsqueeze(1)
        pad_mask = pad_mask & enc_streaming_mask
        return pad_mask

    def forward(self, hidden_states: torch.Tensor, mask: Optional[torch.Tensor]):
        hidden_states = self.encoder_embedding(hidden_states)
        hidden_states, hs_mask, mask = self.forward_embeddings(hidden_states, mask)

        unfolded = False
        bs, seq_len, _ = hidden_states.shape
        max_seq_len = 500  # maxium position for absolute positional encoding
        if seq_len > max_seq_len:
            # audio sequence is longer than max_seq_len, unfold it into chunks of max_seq_len
            unfolded = True
            # the unfold op will drop residual frames, pad it to the multiple of max_seq_len
            if seq_len % max_seq_len > 0:
                chunk_pad_size = max_seq_len - (seq_len % max_seq_len)
            else:
                chunk_pad_size = 0
            if chunk_pad_size > 0:
                hidden_states_pad = F.pad(hidden_states, (0, 0, 0, chunk_pad_size), "constant", 0)
                hidden_states = hidden_states_pad.to(hidden_states.device)

            hidden_states = unfold_tensor(hidden_states, max_seq_len)
            masks_unfold = None
            if mask is not None:
                # revise hs_mask here because the previous calculated hs_mask did not consider extra pad
                subsampled_pad_mask = mask.squeeze(1)  # [bz, subsampled_unmask_seq_len]
                extra_padded_subsamlped_pad_mask = F.pad(
                    subsampled_pad_mask, (0, chunk_pad_size), "constant", False
                )  # extra padding to the pad mask
                extra_padded_subsamlped_pad_mask = extra_padded_subsamlped_pad_mask.unsqueeze(-1).float()
                masks_unfold = unfold_tensor(
                    extra_padded_subsamlped_pad_mask, max_seq_len
                )  # unfold the pad mask like we did to the input tensor
                masks_unfold = masks_unfold.squeeze(-1).bool()  # unfold op does not support bool tensor
            hs_mask = self.calculate_hs_mask(
                hidden_states, hidden_states.device, masks_unfold
            )  # calculate hs_mask based on the unfolded pad mask

        relative_attention_bias = self.relative_attention_bias_layer(hidden_states)

        for layer in self.encoders:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    layer.__call__, hidden_states, hs_mask, relative_attention_bias
                )
            else:
                hidden_states = layer(hidden_states, hs_mask, relative_attention_bias)

        if unfolded:
            embed_dim = hidden_states.shape[-1]
            hidden_states = hidden_states.reshape(bs, -1, embed_dim)
            # if we ever padded before unfolding, we need to remove the padding
            if chunk_pad_size > 0:
                hidden_states = hidden_states[:, :-chunk_pad_size, :]

        return hidden_states, mask


def unfold_tensor(xs_pad, max_seq_len):
    """
    For a given tensor with shape of (N, T, D), if sequence length T is longer than max_seq_len,
    this function unfold it to a (NT', max_seq_len, D) where T' is T // max_seq_len.
    Args:
        xs_pad: N, T, D
    """
    _, _, D = xs_pad.shape
    xs_pad = xs_pad.transpose(-1, -2)  # convert to N, D, T
    # N x D x 1 x T => N x (D x max_seq_len) x T'
    xs_pad = F.unfold(
        xs_pad[..., None, :],
        kernel_size=(1, max_seq_len),
        stride=(1, max_seq_len),
    )

    new_bsz, _, slen = xs_pad.shape
    # N x D x max_seq_len x T'
    xs_pad = xs_pad.view(new_bsz, -1, max_seq_len, slen)
    # N x T' x max_seq_len x D
    xs_pad = xs_pad.permute(0, 3, 2, 1).contiguous()
    # NT' x max_seq_len x D
    xs_pad = xs_pad.view(-1, max_seq_len, D)
    return xs_pad


def adaptive_enc_mask(x_len, chunk_start_idx, left_window=0, right_window=0):
    """
    The function is very important for Transformer Transducer Streaming mode
    Args:
        xs_len (int): sequence length
        chunk_start_idx (list): first idx of each chunk, such as [0,18,36,48]. It also supports adaptive chunk size [0,10,15,45]
        left_window (int): how many left chunks can be seen
        right_window (int): how many right chunks can be seen. It is used for chunk overlap model.
        Returns:
            mask (torch.Tensor): a mask tensor for streaming model
            Torch 1.0.1
            tensor([[1., 1., 0., 0.],
                    [0., 1., 1., 0.],
                    [0., 0., 1., 1.]])
            Torch 1.4.1
            tensor([[True., True., False., False.],
                    [False., True., True., False.],
                    [False., False., True., True.]])
    """
    chunk_start_idx = torch.Tensor(chunk_start_idx).long()  # first idx of each chunk, such as [0,18,36,48].
    start_pad = torch.nn.functional.pad(
        chunk_start_idx, (1, 0)
    )  # append 0 to the beginning, so it becomes [0, 0, 18, 36, 48]
    end_pad = torch.nn.functional.pad(
        chunk_start_idx, (0, 1), value=x_len
    )  # append x_len to the end, so it becomes [0,18,36,48, x_len]
    seq_range = torch.arange(0, x_len).unsqueeze(-1)  # seq_range size: [x_len, 1]
    idx = ((seq_range < end_pad) & (seq_range >= start_pad)).nonzero()[:, 1]  # idx size: [x_len]
    seq_range_expand = torch.arange(0, x_len).unsqueeze(0).expand(x_len, -1)  # seq_range_expand size [x_len, x_len]
    idx_left = idx - left_window
    idx_left[idx_left < 0] = 0
    boundary_left = start_pad[idx_left]
    mask_left = seq_range_expand >= boundary_left.unsqueeze(-1)
    idx_right = idx + right_window
    idx_right[idx_right > len(chunk_start_idx)] = len(chunk_start_idx)
    boundary_right = end_pad[idx_right]
    mask_right = seq_range_expand < boundary_right.unsqueeze(-1)
    return mask_left & mask_right


class Phi4MultimodalAudioEmbedding(nn.Module):
    def __init__(self, config: Phi4MultimodalConfig):
        super().__init__()
        self.config = config

        self.drop = nn.Dropout(config.embd_pdrop)
        self.encoder = Phi4MultimodalAudioModel._from_config(config.audio_config)

        self.layer_idx = config.audio_config.feature_layer
        self.audio_dim_out = config.audio_config.hidden_size
        self.audio_dim_in = config.audio_config.input_size
        self.downsample_rate = config.audio_config.downsample_rate
        dim_projection = config.hidden_size
        self.linear_downsample_rate = self.downsample_rate

        audio_projection_for_speech = nn.Sequential(
            nn.Linear(self.audio_dim_out * self.linear_downsample_rate, dim_projection),
            nn.GELU(),
            nn.Linear(dim_projection, dim_projection),
        )
        audio_projection_for_vision = nn.Sequential(
            nn.Linear(self.audio_dim_out * self.linear_downsample_rate, dim_projection),
            nn.GELU(),
            nn.Linear(dim_projection, dim_projection),
        )
        self.audio_projection = nn.ModuleDict(
            {"speech": audio_projection_for_speech, "vision": audio_projection_for_vision}
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeds: torch.FloatTensor,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        audio_projection_mode="speech",
        **kwargs,
    ) -> torch.FloatTensor:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        with torch.no_grad():
            positions = torch.nonzero(input_ids == self.config.audio_config.audio_token_id, as_tuple=False)
            positions_tuple = torch.nonzero(input_ids == self.config.audio_config.audio_token_id, as_tuple=True)

        target_device = self.audio_projection[audio_projection_mode][0].bias.device
        target_dtype = self.audio_projection[audio_projection_mode][0].bias.dtype

        if input_embeds is not None:
            input_embeds = input_embeds.to(device=target_device, dtype=target_dtype)

        if len(positions.tolist()) > 0:
            audio_features, _ = self.encoder(input_embeds, audio_attention_mask)
            audio_set_tensor = self.audio_projection[audio_projection_mode](audio_features)

        hidden_states = kwargs["wte"](input_ids)

        if len(positions.tolist()) > 0:
            merged_audio_set_tensor = torch.cat(
                [audio_set_tensor[i, : audio_embed_sizes[i], :] for i in range(len(audio_embed_sizes))], dim=0
            )
            merged_audio_set_tensor = merged_audio_set_tensor.to(hidden_states.dtype).to(hidden_states.device)
            # Temporarily disable autocast to avoid issue on bf16 tensors
            # Ref: https://github.com/pytorch/pytorch/issues/132715
            with torch.autocast(device_type=hidden_states.device.type, enabled=False):
                new_hidden_states = hidden_states.index_put(
                    indices=positions_tuple, values=merged_audio_set_tensor, accumulate=False
                )
            hidden_states = new_hidden_states

        if self.drop is not None:
            hidden_states = self.drop(hidden_states)

        return hidden_states


#################################################### TEXT ####################################################


class Phi4MultimodalRMSNorm(Phi3RMSNorm):
    pass


class Phi4MultimodalDecoderLayer(Phi3DecoderLayer):
    pass


class Phi4MultimodalFeatureEmbedding(nn.Module):
    """Image-audio embedding."""

    def __init__(self, config: Phi4MultimodalConfig) -> None:
        super().__init__()
        self.config = config
        self.image_embed = Phi4MultimodalImageEmbedding(config)
        self.audio_embed = Phi4MultimodalAudioEmbedding(config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_image_embeds: Optional[torch.FloatTensor] = None,
        input_audio_embeds: Optional[torch.FloatTensor] = None,
        image_sizes=None,
        image_attention_mask=None,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        wte=None,
    ) -> torch.FloatTensor:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        # backward compatibility
        with torch.no_grad():
            new_input_ids = input_ids.clone()
            new_input_ids[
                (input_ids >= _COMPATIBLE_IMAGE_SPECIAL_TOKEN_ID_RANGE[0])
                & (input_ids <= _COMPATIBLE_IMAGE_SPECIAL_TOKEN_ID_RANGE[1])
            ] = self.config.vision_config.image_token_id
            new_input_ids[
                (input_ids >= _COMPATIBLE_AUDIO_SPECIAL_TOKEN_ID_RANGE[0])
                & (input_ids <= _COMPATIBLE_AUDIO_SPECIAL_TOKEN_ID_RANGE[1])
            ] = self.config.audio_config.audio_token_id
            input_ids = new_input_ids

        with torch.no_grad():
            image_position_mask = (input_ids == self.config.vision_config.image_token_id).unsqueeze(-1)
            non_image_position_mask = ~image_position_mask

        if input_image_embeds is not None:
            image_hidden_states = self.image_embed(
                input_ids=input_ids,
                input_embeds=input_image_embeds,
                image_sizes=image_sizes,
                wte=wte,
                image_attention_mask=image_attention_mask,
            )
        if input_audio_embeds is not None:
            audio_projection_mode = "vision" if input_image_embeds is not None else "speech"
            audio_hidden_states = self.audio_embed(
                input_ids=input_ids,
                input_embeds=input_audio_embeds,
                audio_embed_sizes=audio_embed_sizes,
                audio_attention_mask=audio_attention_mask,
                wte=wte,
                audio_projection_mode=audio_projection_mode,
            )

        # merge image and audio hidden states
        if input_image_embeds is not None and input_audio_embeds is not None:
            hidden_states = image_hidden_states * image_position_mask + audio_hidden_states * non_image_position_mask
        elif input_image_embeds is not None:
            hidden_states = image_hidden_states
        elif input_audio_embeds is not None:
            hidden_states = audio_hidden_states
        else:
            hidden_states = wte(input_ids)

        return hidden_states


PHI4_MULTIMODAL_MODEL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding indices in `input_values`. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache`)`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.
            See our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        input_image_embeds (`torch.FloatTensor`, *optional*):
            If the input contains images, these correspond to the pixel values after transformations (as returned by
            the Processor)
        image_sizes (`torch.LongTensor`, *optional*):
            If the input contains images, these correspond to size of each image.
        image_attention_mask (`torch.LongTensor`, *optional*):
            Attention mask for the images.
        input_audio_embeds (`torch.FloatTensor`, *optional*):
            If the input contains audio samples, these correspond to the values after transformation (as returned by
            the Processor).
        audio_embed_sizes (`torch.Tensor`, *optional*):
            Size of the audio inputs.
        audio_attention_mask (`torch.Tensor, *optional*):
            Attention mask for the audio inputs.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


class Phi4MultimodalModel(Phi3Model, nn.Module):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Phi4MultimodalMMDecoderLayer`]
    Args:
        config: Phi4MultimodalMMConfig
    """

    def __init__(self, config: Phi4MultimodalConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_dropout = nn.Dropout(config.embd_pdrop)

        self.embed_tokens_extend = Phi4MultimodalFeatureEmbedding(config)

        self.layers = nn.ModuleList(
            [Phi4MultimodalDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Phi4MultimodalRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(PHI4_MULTIMODAL_MODEL_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        input_image_embeds: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.LongTensor] = None,
        image_attention_mask=None,
        input_audio_embeds: Optional[torch.FloatTensor] = None,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens_extend(
                input_ids=input_ids,
                input_image_embeds=input_image_embeds,
                input_audio_embeds=input_audio_embeds,
                image_sizes=image_sizes,
                image_attention_mask=image_attention_mask,
                audio_embed_sizes=audio_embed_sizes,
                audio_attention_mask=audio_attention_mask,
                wte=self.embed_tokens,
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()


class Phi4MultimodalForCausalLM(Phi3ForCausalLM, nn.Module):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Phi4MultimodalModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(PHI4_MULTIMODAL_MODEL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=Phi4MultimodalConfig)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        input_image_embeds: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.LongTensor] = None,
        image_attention_mask=None,
        input_audio_embeds: Optional[torch.FloatTensor] = None,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).
        Returns:

        Example:
        ```python
        >>> from transformers import AutoTokenizer, Phi4MultimodalForCausalLM
        >>> model = Phi4MultimodalForCausalLM.from_pretrained("TBA")
        >>> tokenizer = AutoTokenizer.from_pretrained("TBA")
        >>> prompt = "This is an example script ."
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        'This is an example script .\n Certainly! Below is a sample script that demonstrates a simple task, such as calculating the sum'
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            input_image_embeds=input_image_embeds,
            image_sizes=image_sizes,
            image_attention_mask=image_attention_mask,
            input_audio_embeds=input_audio_embeds,
            audio_embed_sizes=audio_embed_sizes,
            audio_attention_mask=audio_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        input_image_embeds=None,
        image_sizes=None,
        image_attention_mask=None,
        input_audio_embeds=None,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        logits_to_keep=0,
        **kwargs,
    ):
        # Overwritten -- this model may need to switch between short and long rope, invalidating the cache in the
        # process

        # When the first time input length reached long and short factor switching point, enforce re-compute cache
        # It will cause downside of slower at this single token position, however, better than current failure.
        if (
            past_key_values
            and self.config.rope_scaling
            and input_ids.shape[1] >= self.config.original_max_position_embeddings + 1
        ):
            past_length = cache_position[0]
            if past_length <= self.config.original_max_position_embeddings:
                past_key_values = None

        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            input_image_embeds=input_image_embeds,
            image_sizes=image_sizes,
            image_attention_mask=image_attention_mask,
            input_audio_embeds=input_audio_embeds,
            audio_embed_sizes=audio_embed_sizes,
            audio_attention_mask=audio_attention_mask,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        return model_inputs


__all__ = [
    "Phi4MultimodalPreTrainedModel",  # noqa
    "Phi4MultimodalModel",
    "Phi4MultimodalForCausalLM",
    "Phi4MultimodalVisionConfig",
    "Phi4MultimodalAudioConfig",
    "Phi4MultimodalConfig",
]
