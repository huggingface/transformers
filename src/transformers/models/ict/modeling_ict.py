# coding=utf-8
# Copyright 2023 Authors at City University of Hong Kong, Microsoft Cloud + AI,
# The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch ICT model."""


import math
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
import torchvision.models as models
from torch import nn

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, MaskedImageCompletionOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_ict import ICTConfig, ICTGuidedUpsamplerConfig, ICTTransformerConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "ICTConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "sheonhan/ict-imagenet-256"
_EXPECTED_OUTPUT_SHAPE = [3, 256, 256]


ICT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "sheonhan/ict-imagenet-256",
    "sheonhan/ict-ffhq-256",
    "sheonhan/ict-places-256",
    # See all ICT models at https://huggingface.co/models?filter=ict
]


# Copied from transformers.models.vit.modeling_vit.ViTSelfAttention with ViT->ICT
class ICTSelfAttention(nn.Module):
    def __init__(self, config: ICTTransformerConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.output = nn.Linear(config.hidden_size, config.hidden_size)

        self.attention_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.residual_dropout = nn.Dropout(config.residual_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.num_attention_heads, self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.query = prune_linear_layer(self.query, index)
        self.key = prune_linear_layer(self.key, index)
        self.value = prune_linear_layer(self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.num_attention_heads = self.num_attention_heads - len(heads)
        self.all_head_size = self.attention_head_size * self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self, hidden_states, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = self.output(context_layer)
        outputs = self.residual_dropout(outputs)

        return (outputs, attention_probs) if output_attentions else (outputs,)


class ICTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_embed = config.hidden_size
        intermediate_size = config.intermediate_size
        self.intermediate_act_fn = ACT2FN[config.activation_function]

        self.ln_1 = nn.LayerNorm(num_embed)
        self.attention = ICTSelfAttention(config)
        self.ln_2 = nn.LayerNorm(num_embed)
        self.mlp = nn.Sequential(
            nn.Linear(num_embed, intermediate_size),
            self.intermediate_act_fn,
            nn.Linear(intermediate_size, num_embed),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, hidden_states, output_attentions: bool = False):
        self_attention_outputs = self.attention(self.ln_1(hidden_states, output_attentions=output_attentions))
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))

        outputs = (hidden_states,) + outputs

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTPreTrainedModel with ViT->ICT,vit->ict
class ICTPretrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ICTConfig
    base_model_prefix = "ict"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = []

    def _init_weights(self, module: Union[nn.Linear, nn.Embedding, nn.LayerNorm, nn.Conv2d]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d)):
            module.weight.data = nn.init.normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if isinstance(module, (nn.Linear, nn.Conv2d)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value: bool = False) -> None:
        if isinstance(module, (ICTTransformerModel, ICTGuidedUpsampler)):
            module.gradient_checkpointing = value


ICT_TRANSFORMER_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ICTTransformerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


ICT_TRANSFORMER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, height * width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ICTImageProcessor.__call__`]
            for details.
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, height * width)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0). Generate random
            masks if not provided.
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
    "The transformer for outputting raw hidden-states without any specific head on top.",
    ICT_TRANSFORMER_START_DOCSTRING,
)
# Copied from transformers.models.vit.modeling_vit.ViTModel with VIT->ICT,ViT->ICT
class ICTTransformerModel(ICTPretrainedModel):
    config_class = ICTTransformerConfig
    main_input_name = "pixel_values"

    def __init__(self, config: ICTTransformerConfig):
        super().__init__(config)
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Parameter(torch.zeros(1, config.block_size, config.hidden_size))
        self.drop = nn.Dropout(config.residual_dropout_prob)

        self.gradient_checkpointing = False
        self.blocks = nn.ModuleList([ICTBlock(config) for _ in range(config.num_hidden_layers)])

        # Decoder head
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.token_embedding

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for block, heads in heads_to_prune.items():
            self.blocks[block].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(ICT_TRANSFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        pixel_values = pixel_values.to(torch.long)
        batch_size, num_pixel = pixel_values.size()

        inputs_embeds = self.token_embedding(pixel_values)

        if bool_masked_pos:
            masks = bool_masked_pos.unsqueeze(2).ong()
            inputs_embeds = inputs_embeds * (1 - masks.long())
        else:
            masks = torch.randint(low=0, high=2, size=(batch_size, num_pixel))
            inputs_embeds = inputs_embeds * (1 - masks)

        position_embeds = self.position_embedding[:, :num_pixel, :]
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        for _, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                )
            else:
                layer_outputs = block(hidden_states, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.utils.spectral_norm(
                nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=2, bias=False)
            ),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(
                nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=False)
            ),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out

class InpaintGenerator(nn.Module):
    def __init__(self, residual_blocks=8):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=7, padding=0),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
        )

        blocks = [ResnetBlock(256) for _ in range(residual_blocks)]

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = (torch.tanh(x) + 1) / 2

        return x


class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss https://arxiv.org/abs/1711.10337
    """

    def __init__(self, gan_loss_function="nsgan", target_real_label=1.0, target_fake_label=0.0):
        r"""
        gan_loss_function = nsgan | lsgan | hinge
        """
        super().__init__()

        self.gan_loss_function = gan_loss_function
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))

        if gan_loss_function == "nsgan":
            self.criterion = nn.BCELoss()

        elif gan_loss_function == "lsgan":
            self.criterion = nn.MSELoss()

        elif gan_loss_function == "hinge":
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_discriminator=None):
        if self.gan_loss_function == "hinge":
            if is_discriminator:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss


class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super().__init__()
        self.add_module("vgg", VGG19())
        self.criterion = torch.nn.L1Loss()

    def compute_gram_matrix(self, x):
        batch_size, channels, height, width = x.size()
        f = x.view(batch_size, channels, width * height)
        G = f.bmm(f.transpose(1, 2)) / (height * width * channels)

        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(
            self.compute_gram_matrix(x_vgg["relu2_2"]), self.compute_gram_matrix(y_vgg["relu2_2"])
        )
        style_loss += self.criterion(
            self.compute_gram_matrix(x_vgg["relu3_4"]), self.compute_gram_matrix(y_vgg["relu3_4"])
        )
        style_loss += self.criterion(
            self.compute_gram_matrix(x_vgg["relu4_4"]), self.compute_gram_matrix(y_vgg["relu4_4"])
        )
        style_loss += self.criterion(
            self.compute_gram_matrix(x_vgg["relu5_2"]), self.compute_gram_matrix(y_vgg["relu5_2"])
        )

        return style_loss


class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super().__init__()
        self.add_module("vgg", VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg["relu1_1"], y_vgg["relu1_1"])
        content_loss += self.weights[1] * self.criterion(x_vgg["relu2_1"], y_vgg["relu2_1"])
        content_loss += self.weights[2] * self.criterion(x_vgg["relu3_1"], y_vgg["relu3_1"])
        content_loss += self.weights[3] * self.criterion(x_vgg["relu4_1"], y_vgg["relu4_1"])
        content_loss += self.weights[4] * self.criterion(x_vgg["relu5_1"], y_vgg["relu5_1"])

        return content_loss


class VGG19(torch.nn.Module):
    def __init__(self):
        super().__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            "relu1_1": relu1_1,
            "relu1_2": relu1_2,
            "relu2_1": relu2_1,
            "relu2_2": relu2_2,
            "relu3_1": relu3_1,
            "relu3_2": relu3_2,
            "relu3_3": relu3_3,
            "relu3_4": relu3_4,
            "relu4_1": relu4_1,
            "relu4_2": relu4_2,
            "relu4_3": relu4_3,
            "relu4_4": relu4_4,
            "relu5_1": relu5_1,
            "relu5_2": relu5_2,
            "relu5_3": relu5_3,
            "relu5_4": relu5_4,
        }
        return out


ICT_GUIDED_UP_SAMPLER_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ICTGuidedUpsamplerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

ICT_GUIDED_UP_SAMPLER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ICTImageProcessor.__call__`]
            for details.
"""


@add_start_docstrings(
    "The guided sampler for outputting the completed images.",
    ICT_GUIDED_UP_SAMPLER_START_DOCSTRING,
)
class ICTGuidedUpsampler(ICTPretrainedModel):
    def __init__(self, config: ICTGuidedUpsamplerConfig):
        super().__init__(config)

        self.generator = InpaintGenerator(config.residual_blocks)
        
        self.post_init()

    def forward(self, images, edges, masks):
        images_masked = (images * (1 - masks).float()) + masks
        inputs = torch.cat((images_masked, edges), dim=1)
        outputs = self.generator(inputs)
        return outputs


ICT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ICTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

ICT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ICTImageProcessor.__call__`]
            for details.
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, height * width)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0). Generate random
            masks if not provided.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(ICT_START_DOCSTRING)
class ICTModel(ICTPretrainedModel):
    config_class = ICTConfig

    def __init__(self, config: ICTConfig):
        super().__init__(config)

        if not isinstance(config.transformer_config, ICTTransformerConfig):
            raise ValueError(
                "config.transformer_config is expected to be of type ICTTransformerConfig but is of type"
                f" {type(config.transformer_config)}."
            )

        if not isinstance(config.guided_upsampler_config, ICTGuidedUpsamplerConfig):
            raise ValueError(
                "config.guided_upsampler_config is expected to be of type ICTGuidedUpsamplerConfig but is of type"
                f" {type(config.guided_upsampler_config)}."
            )

        transformer_config = config.transformer_config
        guided_upsampler_config = config.guided_upsampler_config
        self.config = config
        self.transformer = ICTTransformerModel(transformer_config)
        self.guided_upsampler = ICTGuidedUpsampler(guided_upsampler_config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(ICT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MaskedImageCompletionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
                Returns:

                Example:
                ```python
                >>> import torch
                >>> import numpy as np
                >>> from PIL import Image
                >>> import requests

                >>> from transformers import AutoImageProcessor, ICTModel

                >>> processor = AutoImageProcessor.from_pretrained("sheonhan/ict-imagenet-256")
                >>> model = ICTModel.from_pretrained("sheonhan/ict-imagenet-256")

                >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
                >>> image = Image.open(requests.get(url, stream=True).raw)
                >>> pixel_values = processor(image, return_tensors="pt").pixel_values

                >>> # create random boolean mask of shape (batch_size, num_patches)
                >>> bool_masked_pos = torch.randint(
                ...     low=0, high=2, size=(pixel_values.shape[0], pixel_values.shape[1])
                ... ).bool()

                >>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
                ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        height, width = pixel_values.shape[2:]

        outputs = self.tranformer(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # TODO: Reshaping

        reconstructed_pixel_values = self.guided_upsampler(sequence_output)

        masked_image_completion_loss = None
        if bool_masked_pos is not None:
            size = self.config.image_size // self.config.patch_size
            bool_masked_pos = bool_masked_pos.reshape(-1, size, size)
            mask = (
                bool_masked_pos.repeat_interleave(self.config.patch_size, 1)
                .repeat_interleave(self.config.patch_size, 2)
                .unsqueeze(1)
                .contiguous()
            )
            reconstruction_loss = nn.functional.l1_loss(pixel_values, reconstructed_pixel_values, reduction="none")
            masked_image_completion_loss = (
                (reconstruction_loss * mask).sum() / (mask.sum() + 1e-5) / self.config.num_channels
            )

        if not return_dict:
            output = (reconstructed_pixel_values,) + outputs[1:]
            return ((masked_image_completion_loss,) + output) if masked_image_completion_loss is not None else output

        return MaskedImageCompletionOutput(
            loss=masked_image_completion_loss,
            reconstruction=reconstructed_pixel_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
