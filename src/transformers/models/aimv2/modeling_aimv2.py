# coding=utf-8
# Copyright 2024 Apple Inc., The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch AIMv2 model."""

import collections.abc
import math
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    MaskedImageModelingOutput,
)

from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
    torch_int,
)
from .configuration_aimv2 import AIMv2Config


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "AIMv2Config"

# Base docstring
_CHECKPOINT_FOR_DOC = "apple/aimv2-large"
_EXPECTED_OUTPUT_SHAPE = [1, 197, 768]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "google/aimv2-base-patch16-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "Egyptian cat"


class SinCosPosEmbed(nn.Module):
    def __init__(self, cls_token: bool = False):
        super().__init__()
        self.cls_token = cls_token

    def forward(self, h: int, w: int, embed_dim: int) -> torch.Tensor:
        assert embed_dim % 2 == 0, embed_dim

        grid_h = torch.arange(h).float()
        grid_w = torch.arange(w).float()
        grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
        grid = torch.stack(grid, dim=0)
        grid = grid.reshape([2, 1, h, w])

        emb_h = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        pos_embed = torch.concatenate([emb_h, emb_w], dim=1)  # (H*W, D)
        
        return pos_embed

    @staticmethod
    def _get_1d_sincos_pos_embed_from_grid(
        embed_dim: int, pos: torch.Tensor
    ) -> torch.Tensor:
        omega = torch.arange(embed_dim // 2).float()
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000**omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = pos[:, None] * omega[None, :]  # (M, D/2), outer product

        emb_sin = torch.sin(out)  # (M, D/2)
        emb_cos = torch.cos(out)  # (M, D/2)

        emb = torch.concatenate([emb_sin, emb_cos], dim=1)  # (M, D)
        return emb
    
class AIMv2PatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.norm = config.norm_layer(hidden_size) if config.norm_layer is not None else nn.Identity()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        embeddings = self.norm(embeddings)
        return embeddings
    
    
class Aimv2Embeddings(nn.Module):
    """
    Construct the CLS token, mask token, position and patch embeddings.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.patch_embeddings = Aimv2PatchEmbeddings(config)
        self.use_mask_token = config.use_mask_token
        num_patches = self.patch_embeddings.num_patches

        if config.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        else:
            self.cls_token = None

        if config.use_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        else:
            self.mask_token = None

        if config.use_pos_embed == "sincos":
             # Create an instance of SinCosPosEmbed class
             self.pos_embed = SinCosPosEmbed(cls_token = config.use_cls_token)

        elif config.use_pos_embed == "absolute":
            self.position_embeddings = nn.Parameter(
                torch.randn(1, num_patches + 1 if config.use_cls_token else num_patches, config.hidden_size) * config.initializer_range
            )

        else:
            self.pos_embed = None

        self.config = config

    def forward(
        self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        embeddings = self.patch_embeddings(pixel_values)
        batch_size, seq_len, _ = embeddings.shape

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        if bool_masked_pos is not None and self.mask_token is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # replace the masked visual tokens by mask_tokens
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1 - w) + mask_tokens * w

        if self.position_embeddings is not None:
            if self.config.use_pos_embed == "sincos":
                #The calculation in original repo is done using patch_size not image_size.
                h =  w = int(math.sqrt(seq_len))
                pos_embed = self.pos_embed(h, w, self.config.hidden_size).unsqueeze(0).to(embeddings.device)
                if self.config.use_cls_token:
                    pos_embed = torch.cat(
                            [torch.zeros([1, 1, self.config.hidden_size], device=embeddings.device), pos_embed], dim=1
                        )
            else:
                pos_embed = self.position_embeddings
                pos_embed = pos_embed.to(embeddings.device)

            embeddings = embeddings + pos_embed

        return embeddings


class Aimv2Attention(nn.Module):
    def __init__(self, config: Aimv2Config, use_bias = False) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.qkv = nn.Linear(config.hidden_size, self.all_head_size * 3, bias=config.qkv_bias)

        self.attn_drop = nn.Dropout(config.attention_probs_dropout_prob)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias = use_bias)

        self.proj_drop = nn.Dropout(config.hidden_dropout_prob)
        self.pruned_heads = set()

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        B, N, C = hidden_states.shape
        qkv = (
            self.qkv(hidden_states)
            .reshape(B, N, 3, self.num_attention_heads, C // self.num_attention_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        
        attn_weights = F.scaled_dot_product_attention(
            q, k, v
        )
        attn_weights = attn_weights.transpose(1, 2).reshape(B, N, C)
        attn_output = self.proj(attn_weights)
        attn_output = self.proj_drop(attn_output)

        return attn_output, attn_weights    





class Aimv2Mlp(nn.Module):
    def __init__(
        self,
        config: Aimv2Config,
        in_features: Optional[int] = None,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
    ):
        super().__init__()
        in_features = in_features if in_features is not None else config.hidden_size
        out_features = out_features if out_features is not None else config.hidden_size
        hidden_features = hidden_features if hidden_features is not None else config.intermediate_size

        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.fc2 = nn.Linear(hidden_features, in_features, bias=True)
        self.fc3 = nn.Linear(in_features, hidden_features, bias=True)
        self.norm_layer = config.norm_layer(hidden_features) if config.norm_layer is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.fc1(x)) * self.fc3(x)
        x = self.norm_layer(x)
        x = self.fc2(x)
        return x

class Aimv2EncoderLayer(nn.Module):
    """This corresponds to the Block class in the original implementation."""

    def __init__(self, config: Aimv2Config) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Aimv2Attention(config, use_bias=config.qkv_bias)
        self.layer_norm1 = config.norm_layer(self.embed_dim)
        self.mlp = Aimv2Mlp(config)
        self.layer_norm2 = config.norm_layer(self.embed_dim)
        

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class Aimv2Encoder(nn.Module):
    def __init__(self, config: Aimv2Config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([Aimv2EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )




#To handle initializations of various classes.
class Aimv2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Aimv2Config
    base_model_prefix = "aimv2"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Aimv2EncoderLayer"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, SinCosPosEmbed):
            pass
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Aimv2PatchEmbeddings):
            if isinstance(module.projection, nn.Conv2d):
                w = module.projection.weight.data
                nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, Aimv2Encoder):
            module.gradient_checkpointing = value

_CHECKPOINT_FOR_DOC = "TODO: Add a checkpoint"
_EXPECTED_OUTPUT_SHAPE = [1, 197, 768]

AIMV2_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`Aimv2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

AIMV2_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`BitImageProcessor.__call__`] for details.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

@add_start_docstrings(
    "The bare Aimv2 Model transformer outputting raw hidden-states without any specific head on top.",
    AIMV2_START_DOCSTRING,
)
class Aimv2Model(Aimv2PreTrainedModel):
    def __init__(self, config: Aimv2Config):
        super().__init__(config)
        self.config = config

        self.embeddings = Aimv2Embeddings(config)
        self.encoder = Aimv2Encoder(config)
        #final layer norm
        self.norm = config.norm_layer(config.hidden_size)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(AIMV2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

  
        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.norm(sequence_output)

        if not return_dict:
            return (sequence_output) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )




class Aimv2ClassificationHead(nn.Module):
    """
    Head for classification tasks.
    Args:
        config (`Aimv2Config`):
            The configuration of the preceding model.
        in_features (`int`):
            Number of input features.
        num_classes (`int`):
            Number of output classes.
        inner_dim (`int`, *optional*, defaults to `None`):
            Dimension of the penultimate layer. If `None`, defaults to `config.hidden_size`.
        use_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the classification layer.
        pool_mode (`str`, *optional*, defaults to `"avg"`):
            Pooling mode, choose from "avg" or "cls".
        drop_rate (`float`, *optional*, defaults to 0.0):
            Dropout rate.
    """

    def __init__(
        self,
        config,
        in_features: int,
        num_classes: int,
        inner_dim: Optional[int] = None,
        use_bias: bool = False,
        pool_mode: str = "avg",
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.config = config
        inner_dim = inner_dim if inner_dim is not None else config.hidden_size

        self.in_features = in_features
        self.num_classes = num_classes
        self.pool_mode = pool_mode
        self.drop_rate = drop_rate

        self.layers = nn.Sequential()

        if pool_mode == "avg":
            self.layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            self.layers.append(nn.Flatten())
        elif pool_mode == "cls":
            # Keep only the CLS token
            self.layers.append(nn.Identity())
        else:
            raise ValueError(f"Invalid pool_mode: {pool_mode}")

        if inner_dim != in_features:
            self.layers.append(nn.Linear(in_features, inner_dim, bias=False))
            self.layers.append(nn.Tanh())

        if self.drop_rate > 0:
            self.layers.append(nn.Dropout(self.drop_rate))

        self.layers.append(nn.Linear(inner_dim, num_classes, bias=use_bias))

        # initialize weights
        for m in self.layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=config.initializer_range)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool_mode == "avg":
            # If we use average pooling, we assume the input is a 4D tensor with shape (B, C, H, W)
            x = self.layers(x)
        else:
            # If we use a CLS token, we assume the input is a 3D tensor with shape (B, N, C)
            # where N is the sequence length and the CLS token is at index 0.
            x = self.layers(x[:, 0])

        return x

class Aimv2ForImageClassification(Aimv2PreTrainedModel):
    def __init__(self, config: Aimv2Config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.aimv2 = Aimv2Model(config)

        # Classifier head
        self.classifier = Aimv2ClassificationHead(
            config=config,
            in_features=config.hidden_size,
            num_classes=config.num_labels,
            pool_mode="avg",  # Assuming we're using average pooling
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(AIMV2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=ImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.aimv2(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        
        # Reshape from (B, N, C) to (B, C, H, W) for Aimv2ClassificationHead
        if self.config.use_cls_token:
            # Remove the cls token
            sequence_output = sequence_output[:, 1:, :]
        sequence_output = sequence_output.unflatten(1, (int(math.sqrt(sequence_output.shape[1])), int(math.sqrt(sequence_output.shape[1]))))
        sequence_output = sequence_output.permute(0, 2, 1).contiguous()
        batch_size, num_channels, sequence_length = sequence_output.shape
        height = width = int(sequence_length**0.5)
        sequence_output = sequence_output.reshape(batch_size, num_channels, height, width)

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
