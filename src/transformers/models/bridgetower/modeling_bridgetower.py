# coding=utf-8
# Copyright 2022 The Intel Labs Team Authors, The Microsoft Research Team Authors and HuggingFace Inc. team. All rights reserved.
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
"""PyTorch BridgeTower Model"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import math
import torch
import torch.utils.checkpoint
from torch import nn

from transformers import RobertaConfig, RobertaModel
from transformers.modeling_utils import PreTrainedModel, apply_chunking_to_forward

from ...activations import ACT2FN, QuickGELUActivation
from ...modeling_outputs import MaskedLMOutput, ModelOutput, SequenceClassifierOutput
from ...pytorch_utils import find_pruneable_heads_and_indices, is_torch_greater_or_equal_than_1_10, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_bridgetower import BridgeTowerConfig


logger = logging.get_logger(__name__)

if not is_torch_greater_or_equal_than_1_10:
    logger.warning(
        f"You are using torch=={torch.__version__}, but torch>=1.10.0 is required to use "
        "BridgeTowerModel. Please upgrade torch."
    )

_CONFIG_FOR_DOC = "BridgeTowerConfig"
_CHECKPOINT_FOR_DOC = "BridgeTower/bridgetower-base"

BRIDGETOWER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "BridgeTower/bridgetower-base",
    "BridgeTower/bridgetower-base-itm-mlm"
    # See all bridgetower models at https://huggingface.co/BridgeTower
]


BRIDGETOWER_START_DOCSTRING = r"""
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ subclass. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`BridgeTowerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

BRIDGETOWER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary. Indices can be obtained using [`BertTokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details. [What are input
            IDs?](../glossary#input-ids)

        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)

        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:
            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.
            [What are token type IDs?](../glossary#token-type-ids)

        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`BridgeTowerImageProcessor`]. See
            [`BridgeTowerImageProcessor.__call__`] for details.

        pixel_mask (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:

            - 1 for pixels that are real (i.e. **not masked**),
            - 0 for pixels that are padding (i.e. **masked**).
            `What are attention masks? <../glossary.html#attention-mask>`__

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.

        image_embeds (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`, *optional*):
            Optionally, instead of passing `pixel_values`, you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `pixel_values` into patch embeddings.

        image_token_type_idx (`int`, *optional*):
            - The token type ids for images.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class BridgeTowerLayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class BridgeTowerResidualAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = BridgeTowerLayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELUActivation()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = BridgeTowerLayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, x_mask: torch.Tensor):
        if x_mask is not None:
            x_mask = x_mask.to(dtype=torch.bool, device=x.device)
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask, key_padding_mask=x_mask)[0]

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        x = x + self.attention(self.ln_1(x), x_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class BridgeTowerTransformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        attn_mask: torch.Tensor = None,
        model_type: str = "bridgetower",
        stop_gradient: bool = False,
        vit_remove_last: bool = False,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        if vit_remove_last:
            self.resblocks = nn.Sequential(
                *[BridgeTowerResidualAttention(width, heads, attn_mask) for _ in range(layers - 1)]
            )
        else:
            self.resblocks = nn.Sequential(
                *[BridgeTowerResidualAttention(width, heads, attn_mask) for _ in range(layers)]
            )
        self.model_type = model_type
        self.stop_gradient = stop_gradient

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        xs = []
        for block in self.resblocks:
            x = block(x, x_mask)
            if self.model_type == "bridgetower":
                if self.stop_gradient:
                    xs.append(x.detach())
                else:
                    xs.append(x)
        if self.model_type == "bridgetower":
            return xs
        else:
            return x


class BridgeTowerVisualTransformer(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        resolution_after: int,
        model_type: str = "bridgetower",
        stop_gradient: bool = False,
        vit_layernorm_shared: bool = True,
        vit_remove_last: bool = False,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False
        )

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((resolution_after // patch_size) ** 2 + 1, width))
        self.ln_pre = BridgeTowerLayerNorm(width)

        self.transformer = BridgeTowerTransformer(
            width, layers, heads, model_type=model_type, stop_gradient=stop_gradient, vit_remove_last=vit_remove_last
        )
        self.ln_post = BridgeTowerLayerNorm(width)
        self.model_type = model_type
        self.vit_layernorm_shared = vit_layernorm_shared
        if not vit_layernorm_shared:
            self.ln_separate = nn.ModuleList([BridgeTowerLayerNorm(width) for _ in range(layers)])

    def forward(self, x: torch.Tensor, x_mask):
        # shape = [*, width, grid, grid]
        visual_output = self.conv1(x)
        # shape = [*, width, grid ** 2]
        visual_output = visual_output.reshape(
            visual_output.shape[0], visual_output.shape[1], -1)
        # shape = [*, grid ** 2, width]
        visual_output = visual_output.permute(0, 2, 1)
        t = self.class_embedding.to(visual_output.dtype) + torch.zeros(
            visual_output.shape[0], 1, visual_output.shape[-1], dtype=visual_output.dtype, device=visual_output.device
        )
        # shape = [*, grid ** 2 + 1, width]
        visual_output = torch.cat([t, visual_output], dim=1)
        visual_output = visual_output + self.positional_embedding.to(visual_output.dtype)
        visual_output = self.ln_pre(visual_output)
        # NLD -> LND
        visual_output = visual_output.permute(1, 0, 2)

        visual_outputs = self.transformer(visual_output, x_mask)
        # shape = [layers, width, *, grid ** 2]
        visual_outputs = torch.stack(visual_outputs, dim=0)
        # shape = [layers, *, width, grid ** 2]
        visual_outputs = visual_outputs.permute(0, 2, 1, 3)
        if self.vit_layernorm_shared:
            visual_outputs = self.ln_post(visual_outputs)
        else:
            visual_outputs_stack = []
            for visual_output, ln in zip(visual_outputs, self.ln_separate):
                visual_output = ln(visual_output)
                visual_outputs_stack.append(visual_output)
            # shape = [layers, *, width, grid ** 2]
            visual_outputs = torch.stack(visual_outputs_stack, dim=0)
        return visual_outputs

    def forward_pre(self, x: torch.Tensor):
        # shape = [*, width, grid, grid]
        visual_outputs_pre = self.conv1(x)
        # shape = [*, width, grid ** 2]
        visual_outputs_pre = visual_outputs_pre.reshape(
            visual_outputs_pre.shape[0], visual_outputs_pre.shape[1], -1)
        # shape = [*, grid ** 2, width]
        visual_outputs_pre = visual_outputs_pre.permute(0, 2, 1)
        embeddings_to = self.class_embedding.to(visual_outputs_pre.dtype) + torch.zeros(
            visual_outputs_pre.shape[0],
            1,
            visual_outputs_pre.shape[-1],
            dtype=visual_outputs_pre.dtype,
            device=visual_outputs_pre.device,)
        # shape = [*, grid ** 2 + 1, width]
        visual_outputs_pre = torch.cat([embeddings_to, visual_outputs_pre], dim=1)
        visual_outputs_pre = visual_outputs_pre + self.positional_embedding.to(visual_outputs_pre.dtype)
        visual_outputs_pre = self.ln_pre(visual_outputs_pre)
        # NLD -> LND
        visual_outputs_pre = visual_outputs_pre.permute(1, 0, 2)
        return visual_outputs_pre

    def forward_post(self, x: torch.Tensor):
        visual_output_post = x.permute(1, 0, 2)
        visual_output_post = self.ln_post(visual_output_post)
        return visual_output_post


class BridgeTowerCLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        # vision
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        transformer_width: int,
        resolution_after=224,
        model_type="bridgetower",
        stop_gradient=False,
        vit_layernorm_shared=True,
        vit_remove_last=False,
    ):
        super().__init__()

        vision_heads = vision_width // 64
        self.visual = BridgeTowerVisualTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            resolution_after=resolution_after,
            model_type=model_type,
            stop_gradient=stop_gradient,
            vit_layernorm_shared=vit_layernorm_shared,
            vit_remove_last=vit_remove_last,
        )

        self.ln_final = BridgeTowerLayerNorm(transformer_width)

        self.initialize_parameters()

    def initialize_parameters(self):
        proj_std = (self.visual.transformer.width**-0.5) * ((2 * self.visual.transformer.layers) ** -0.5)
        attn_std = self.visual.transformer.width**-0.5
        fc_std = (2 * self.visual.transformer.width) ** -0.5
        for block in self.visual.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def forward(self, image, image_mask=None):
        return self.visual(image.type(self.dtype), image_mask)


@dataclass
class BridgeTowerModelOutput(ModelOutput):
    """
    Output type of [`BridgeTowerModel`].

    Args:
        text_feats (`torch.FloatTensor` of shape `(batch_size, text_sequence_length, hidden_size)`):
            Sequence of hidden-states at the text output of the last layer of the model.
        image_feats (`torch.FloatTensor` of shape `(batch_size, image_sequence_length, hidden_size)`):
            Sequence of hidden-states at the image output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size x 2)`):
            Concatenation of last layer hidden-state of the first token of the text and image sequence (classification
            token), respectively, after further processing through layers used for auxiliary pretraining tasks.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    text_feats: torch.FloatTensor = None
    image_feats: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class BridgeTowerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BridgeTowerConfig
    base_model_prefix = "bridgetower"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BridgeTowerSelfAttention"]

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


@add_start_docstrings(
    (
        "The bare BridgeTower Model transformer outputting BridgeTowerModelOutput object without any specific head on"
        " top."
    ),
    BRIDGETOWER_START_DOCSTRING,
)
class BridgeTowerModel(BridgeTowerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.tokenizer_config = RobertaConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.hidden_size * config.mlp_ratio,
            max_position_embeddings=config.max_text_len,
            hidden_dropout_prob=config.drop_rate,
            attention_probs_dropout_prob=config.drop_rate,
        )

        if config.cross_modal_transform_shared:
            self.cross_modal_text_transform = nn.Linear(config.input_text_embed_size, config.hidden_size)
            self.cross_modal_image_transform = nn.Linear(config.input_image_embed_size, config.hidden_size)
        else:
            self.cross_modal_text_transform = nn.ModuleList(
                [nn.Linear(config.input_text_embed_size, config.hidden_size) for _ in range(config.num_hidden_layers)]
            )
            self.cross_modal_image_transform = nn.ModuleList(
                [nn.Linear(config.input_image_embed_size, config.hidden_size) for _ in range(config.num_hidden_layers)]
            )

        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)

        self.vit_model = BridgeTowerCLIP(
            config.vit_embed_dim,
            config.resolution_before,
            config.vit_num_hidden_layers,
            config.vit_hidden_size,
            config.vit_patch_size,
            config.vit_intermediate_size,
            resolution_after=config.image_size,
            stop_gradient=config.stop_gradient,
            vit_layernorm_shared=config.vit_layernorm_shared,
            vit_remove_last=config.vit_remove_last,
        )

        roberta_config = RobertaConfig.from_pretrained(config.tokenizer)
        self.text_transformer = RobertaModel(roberta_config)

        if not config.vit_layernorm_shared and config.vit_layernorm_init_from_vit:
            for ln in self.vit_model.visual.cross_modal_ln_separate:
                ln.weight.data = self.vit_model.visual.ln_post.weight.data
                ln.bias.data = self.vit_model.visual.ln_post.bias.data

        self.cross_modal_image_layers = nn.ModuleList(
            [BridgeTowerBertCrossLayer(self.tokenizer_config) for _ in range(config.num_hidden_layers)]
        )
        self.cross_modal_text_layers = nn.ModuleList(
            [BridgeTowerBertCrossLayer(self.tokenizer_config) for _ in range(config.num_hidden_layers)]
        )

        # Class token => Linear => Tanh
        self.cross_modal_image_pooler = BridgeTowerPooler(config)
        self.cross_modal_text_pooler = BridgeTowerPooler(config)

        # Initialize BridgeTower Components
        self.cross_modal_text_layernorm = nn.LayerNorm(config.hidden_size)
        self.cross_modal_image_layernorm = nn.LayerNorm(config.hidden_size)

        if config.link_tower_shared:
            self.cross_modal_text_link_tower = LinkTower(config)
            self.cross_modal_image_link_tower = LinkTower(config)
        else:
            self.cross_modal_text_link_tower = nn.ModuleList(
                [LinkTower(config) for _ in range(config.num_hidden_layers - 1)]
            )
            self.cross_modal_image_link_tower = nn.ModuleList(
                [LinkTower(config) for _ in range(config.num_hidden_layers - 1)]
            )

        self.post_init()

    @add_start_docstrings_to_model_forward(BRIDGETOWER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BridgeTowerModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        image_token_type_idx: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], BridgeTowerModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels are currently not supported.
        Returns:

        Examples:

        ```python
        >>> from transformers import BridgeTowerProcessor, BridgeTowerModel
        >>> from PIL import Image
        >>> import requests

        >>> # prepare image and text
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "hello world"
        >>> processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base")
        >>> model = BridgeTowerModel.from_pretrained("BridgeTower/bridgetower-base")

        >>> inputs = processor(image, text, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> outputs.keys()
        odict_keys(['text_feats', 'image_feats', 'pooler_output'])
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        image_token_type_idx = image_token_type_idx if image_token_type_idx else 1
        irtr_len = 0
        input_shape = input_ids.size()
        text_embeds = self.text_transformer.embeddings(input_ids=input_ids)

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, dtype=torch.long, device=self.device)
        extend_text_masks = self.text_transformer.get_extended_attention_mask(attention_mask, input_shape, self.device)

        split_index = len(self.text_transformer.encoder.layer) - self.config.num_hidden_layers + 1
        for layer in self.text_transformer.encoder.layer[:split_index]:
            text_embeds = layer(text_embeds, extend_text_masks)[0]

        image_embeds = self.vit_model.visual.forward_pre(pixel_values.type(self.vit_model.dtype))
        for block in self.vit_model.visual.transformer.resblocks[:split_index]:
            image_embeds = block(image_embeds)
        image_embeds_with_ln = self.vit_model.visual.forward_post(image_embeds.type(self.vit_model.dtype))

        # first layer
        cross_modal_text = self.cross_modal_text_transform(text_embeds)
        text_token_type_embeddings = self.token_type_embeddings(torch.zeros(1).long().to(self.device)).expand_as(
            cross_modal_text
        )
        cross_modal_text = self.cross_modal_text_layernorm(cross_modal_text + text_token_type_embeddings)

        image_embeds_with_ln = self.cross_modal_image_transform(image_embeds_with_ln)
        image_token_type_embeddings = self.token_type_embeddings(
            torch.zeros(1).long().to(self.device).fill_(image_token_type_idx)
        ).expand_as(image_embeds_with_ln)
        image_embeds_with_ln = image_embeds_with_ln + image_token_type_embeddings
        cross_modal_image = self.cross_modal_image_layernorm(image_embeds_with_ln)
        if irtr_len > 0:
            _bs, _L, _D = image_embeds_with_ln.size()
            cross_modal_image = cross_modal_image.unsqueeze(1).expand(_bs, irtr_len, _L, _D).reshape(-1, _L, _D)
        pixel_mask = torch.ones(
            (cross_modal_image.size(0), cross_modal_image.size(1)), dtype=torch.long, device=self.device
        )
        extend_image_masks = self.text_transformer.get_extended_attention_mask(
            pixel_mask, pixel_mask.size(), self.device
        )

        cross_text_feats = self.cross_modal_text_layers[0](
            cross_modal_text,
            cross_modal_image,
            attention_mask=extend_text_masks,
            encoder_attention_mask=extend_image_masks,
        )[0]
        cross_image_feats = self.cross_modal_image_layers[0](
            cross_modal_image,
            cross_modal_text,
            attention_mask=extend_image_masks,
            encoder_attention_mask=extend_text_masks,
        )[0]

        link_layer_index = 0

        # link tower fusion
        for i in range(split_index, len(self.text_transformer.encoder.layer)):
            text_embeds = self.text_transformer.encoder.layer[i](text_embeds, extend_text_masks)[0]
            image_embeds = self.vit_model.visual.transformer.resblocks[i](image_embeds).type(self.vit_model.dtype)
            image_embeds_with_ln = (
                self.cross_modal_image_transform(self.vit_model.visual.forward_post(image_embeds))
                + image_token_type_embeddings
            )

            text_link_tower = self.cross_modal_text_link_tower[link_layer_index]
            image_link_tower = self.cross_modal_image_link_tower[link_layer_index]

            cross_text_feats_ = text_link_tower(
                self.cross_modal_text_transform(text_embeds) + text_token_type_embeddings,
                cross_text_feats,
                extend_text_masks,
            )
            if irtr_len > 0:
                cross_image_feats_ = image_link_tower(
                    image_embeds_with_ln.unsqueeze(1).expand(_bs, irtr_len, _L, _D).reshape(-1, _L, _D),
                    cross_image_feats,
                    extend_image_masks,
                )
            else:
                cross_image_feats_ = image_link_tower(image_embeds_with_ln, cross_image_feats, extend_image_masks)

            cross_text_feats = self.cross_modal_text_layers[link_layer_index + 1](
                cross_text_feats_,
                cross_image_feats_,
                attention_mask=extend_text_masks,
                encoder_attention_mask=extend_image_masks,
            )[0]
            cross_image_feats = self.cross_modal_image_layers[link_layer_index + 1](
                cross_image_feats_,
                cross_text_feats_,
                attention_mask=extend_image_masks,
                encoder_attention_mask=extend_text_masks,
            )[0]

            link_layer_index += 1

        text_feats, image_feats = cross_text_feats, cross_image_feats
        cls_feats = self.get_cls_feats(text_feats, image_feats)

        if not return_dict:
            return tuple(v for v in [text_feats, image_feats, cls_feats] if v is not None)

        return BridgeTowerModelOutput(
            text_feats=text_feats,
            image_feats=image_feats,
            pooler_output=cls_feats,
        )

    def get_cls_feats(self, text_feats, image_feats):
        cls_feats_text = self.cross_modal_text_pooler(text_feats)
        cls_feats_image = self.cross_modal_image_pooler(image_feats)
        return torch.cat([cls_feats_text, cls_feats_image], dim=-1)


class LinkTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.link_tower_type = config.link_tower_type
        self.hidden_size = config.hidden_size
        if config.link_tower_type in [
            "add",
            "scaled_add",
            "interpolate",
        ]:
            if config.link_tower_type == "scaled_add":
                self.scaled_factor = nn.Parameter(torch.tensor(1.0))
            elif config.link_tower_type == "interpolate":
                self.beta = nn.Parameter(torch.tensor(0.5))
            self.LayerNorm = nn.LayerNorm(self.hidden_size)
        else:
            raise NotImplementedError(f"link_tower_type {config.link_tower_type} is not implemented")

    def forward(self, hidden_states, cross_modal_hidden_states, attention_mask):
        if self.link_tower_type == "add":
            return self.LayerNorm(hidden_states + cross_modal_hidden_states)
        elif self.link_tower_type == "scaled_add":
            return self.LayerNorm(hidden_states * self.scaled_factor + cross_modal_hidden_states)
        elif self.link_tower_type == "interpolate":
            return self.LayerNorm(hidden_states * (1 - self.beta) + cross_modal_hidden_states * self.beta)
        else:
            raise NotImplementedError(f"link_tower_type {self.link_tower_type} is not implemented")


class BridgeTowerBertCrossLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BridgeTowerAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        self.crossattention = BridgeTowerAttention(config)
        self.intermediate = BridgeTowerIntermediate(config)
        self.output = BridgeTowerOutput(config)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=None,
            output_attentions=output_attentions,
            past_key_value=None,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        # add self attentions if we output attention weights
        outputs = self_attention_outputs[1:]

        cross_attention_outputs = self.crossattention(
            attention_output,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )
        attention_output = cross_attention_outputs[0]
        # add cross attentions if we output attention weights
        outputs = outputs + cross_attention_outputs[1:-1]

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


@add_start_docstrings(
    """
    BridgeTower Model with a language modeling head on top as done during pretraining.
    """,
    BRIDGETOWER_START_DOCSTRING,
)
class BridgeTowerForMaskedLM(BridgeTowerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bridgetower = BridgeTowerModel(config)
        self.mlm_score = BridgeTowerMLMHead(config)

    def get_output_embeddings(self):
        return self.mlm_score.decoder

    def set_output_embeddings(self, new_embeddings):
        self.mlm_score.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(BRIDGETOWER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[MaskedLMOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels are currently not supported.
        Returns:

        Examples:

        ```python
        >>> from transformers import BridgeTowerProcessor, BridgeTowerForMaskedLM
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000360943.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        >>> text = "a <mask> looking out of the window"

        >>> processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
        >>> model = BridgeTowerForMaskedLM.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")

        >>> # prepare inputs
        >>> encoding = processor(image, text, return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(**encoding)

        >>> results = processor.decode(outputs.logits.argmax(dim=-1).squeeze(0).tolist())

        >>> print(results)
        .a cat looking out of the window.
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bridgetower(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        mlm_logits = self.mlm_score(outputs.text_feats if return_dict else outputs[0])

        if not return_dict:
            return tuple(mlm_logits)

        return MaskedLMOutput(logits=mlm_logits)


# Copied from transformers.models.vilt.modeling_vilt.ViltPredictionHeadTransform with Vilt->BridgeTower
class BridgeTowerPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BridgeTowerMLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.config = config
        self.transform = BridgeTowerPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        mlm_score = self.transform(x)
        mlm_score = self.decoder(mlm_score) + self.bias
        return mlm_score


class BridgeTowerITMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        itm_score = self.fc(x)
        return itm_score


@add_start_docstrings(
    """
    BridgeTower Model transformer with a classifier head on top (a linear layer on top of the final hidden state of the
    [CLS] token) for image-to-text matching.
    """,
    BRIDGETOWER_START_DOCSTRING,
)
class BridgeTowerForImageAndTextRetrieval(BridgeTowerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bridgetower = BridgeTowerModel(config)

        self.itm_score = BridgeTowerITMHead(config.hidden_size * 2)

    @add_start_docstrings_to_model_forward(BRIDGETOWER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels are currently not supported.
        Returns:

        Examples:

        ```python
        >>> from transformers import BridgeTowerProcessor, BridgeTowerForImageAndTextRetrieval
        >>> import requests
        >>> from PIL import Image

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> texts = ["An image of two cats chilling on a couch", "A football player scoring a goal"]

        >>> processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
        >>> model = BridgeTowerForImageAndTextRetrieval.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")

        >>> # forward pass
        >>> scores = dict()
        >>> for text in texts:
        ...     # prepare inputs
        ...     encoding = processor(image, text, return_tensors="pt")
        ...     outputs = model(**encoding)
        ...     scores[text] = outputs.logits[0, 1].item()
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bridgetower(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooler_output = outputs.pooler_output if return_dict else outputs[2]

        logits = self.itm_score(pooler_output)

        if not return_dict:
            return tuple(logits)

        return SequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->BridgeTower
class BridgeTowerSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->BridgeTower
class BridgeTowerSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->BridgeTower
class BridgeTowerAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = BridgeTowerSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BridgeTowerSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->BridgeTower
class BridgeTowerIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->BridgeTower
class BridgeTowerOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertPooler with Bert->BridgeTower
class BridgeTowerPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output