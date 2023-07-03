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
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.models as models
from torch import nn

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, MaskedImageModelingOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_ict import IctConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "IctConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "sheonhan/ict-imagenet-256"
_EXPECTED_OUTPUT_SHAPE = [3, 256, 256]


ICT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "sheonhan/ict-imagenet-256",
    "sheonhan/ict-ffhq-256",
    "sheonhan/ict-places-256",
    # See all ICT models at https://huggingface.co/models?filter=ict
]


class IctEmbeddings(nn.Module):
    """
    Construct the embeddings. Optionally, also the mask token.
    """

    def __init__(self, config, use_mask_token=False):
        super().__init__()

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.image_size * config.image_size, config.hidden_size)
        )
        self.dropout = nn.Dropout(config.embedding_dropout_prob)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None

    def forward(
        self, pixel_values: Optional[torch.FloatTensor], bool_masked_pos: Optional[torch.BoolTensor] = None
    ) -> Tuple[torch.Tensor]:
        batch_size, num_pixel = pixel_values.shape

        embeddings = self.token_embedding(pixel_values)

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # each position maps to a learnable vector
        position_embeds = self.position_embedding[:, :num_pixel, :]
        embeddings = embeddings + position_embeds
        embeddings = self.dropout(embeddings)

        return embeddings


class IctSelfAttention(nn.Module):
    def __init__(self, config: IctConfig) -> None:
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


class IctLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        self.intermediate_act_fn = ACT2FN[config.activation_function]

        self.layer_norm_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.attention = IctSelfAttention(config)
        self.layer_norm_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            self.intermediate_act_fn,
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(config.residual_dropout_prob),
        )

    def forward(self, hidden_states, output_attentions: bool = False):
        self_attention_outputs = self.attention(self.layer_norm_1(hidden_states), output_attentions=output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.mlp(self.layer_norm_2(hidden_states))

        outputs = (hidden_states,) + outputs

        return outputs


class IctEncoder(nn.Module):
    def __init__(self, config: IctConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([IctLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for _, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                )
            else:
                layer_outputs = layer(hidden_states, output_attentions)

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


class IctPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = IctConfig
    base_model_prefix = "ict"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = []

    def _init_weights(
        self, module: Union[nn.Linear, nn.Embedding, nn.LayerNorm, nn.Conv2d, nn.ConvTranspose2d]
    ) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d, nn.ConvTranspose2d)):
            module.weight.data = nn.init.normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value: bool = False) -> None:
        if isinstance(module, (IctEncoder)):
            module.gradient_checkpointing = value


class IctTransformerModel(IctPreTrainedModel):
    def __init__(self, config: IctConfig, use_mask_token: bool = False):
        super().__init__(config)
        self.config = config

        self.embeddings = IctEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = IctEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.token_embedding

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layers[layer].attention.prune_heads(heads)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

        encoder_outputs = self.encoder(
            embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class IctResnetBlock(nn.Module):
    """
    ResNet block without the final ReLU (https://torch.ch/blog/2016/02/04/resnets.html).
    """

    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=0, dilation=2),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=0, dilation=1),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class IctInpaintGenerator(nn.Module):
    def __init__(self, config):
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

        blocks = [IctResnetBlock() for _ in range(config.num_residual_blocks)]

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


class VGG19(nn.Module):
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


class IctAdversarialLoss(nn.Module):
    r"""
    ICT Adversarial loss https://arxiv.org/abs/1711.10337
    """

    def __init__(self, config):
        super().__init__()

        self.gan_loss_function = config.gan_loss_function
        self.real_label = torch.tensor(1.0)
        self.fake_label = torch.tensor(0.0)

        if self.gan_loss_function == "nsgan":
            self.criterion = nn.BCELoss()

        elif self.gan_loss_function == "lsgan":
            self.criterion = nn.MSELoss()

        elif self.gan_loss_function == "hinge":
            self.criterion = nn.ReLU()

        else:
            raise ValueError("`gan_loss_function` has to be `nsgan`, `lsgan`, or `hinge`.")

    def forward(self, outputs, is_real, is_discriminator=False):
        if self.gan_loss_function == "hinge":
            if is_discriminator:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
        loss = self.criterion(outputs, labels)
        return loss


class IctStyleLoss(nn.Module):
    r"""
    Style loss, VGG-based https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super().__init__()
        self.vgg = VGG19()
        self.criterion = torch.nn.L1Loss()

    def compute_gram_matrix(self, x):
        batch_size, channels, height, width = x.size()
        features = x.view(batch_size, channels, width * height)
        gram = features.bmm(features.transpose(1, 2)) / (height * width * channels)

        return gram

    def forward(self, x, y):
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


class IctPerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super().__init__()
        self.vgg = VGG19()
        self.criterion = torch.nn.L1Loss()
        self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]

    def forward(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg["relu1_1"], y_vgg["relu1_1"])
        content_loss += self.weights[1] * self.criterion(x_vgg["relu2_1"], y_vgg["relu2_1"])
        content_loss += self.weights[2] * self.criterion(x_vgg["relu3_1"], y_vgg["relu3_1"])
        content_loss += self.weights[3] * self.criterion(x_vgg["relu4_1"], y_vgg["relu4_1"])
        content_loss += self.weights[4] * self.criterion(x_vgg["relu5_1"], y_vgg["relu5_1"])

        return content_loss


class IctGuidedUpsampler(IctPreTrainedModel):
    def __init__(self, config: IctConfig):
        super().__init__(config)

        self.generator = IctInpaintGenerator(config)
        self.adversarial_loss = IctAdversarialLoss(config)
        self.l1_loss = nn.L1Loss()
        self.style_loss = IctStyleLoss()
        self.perceptual_loss = IctPerceptualLoss()
        self.output_image_size = config.output_image_size

        self.post_init()

    # modified from https://github.com/raywzy/ICT/blob/59dd12d374d47cdf0dce90923017ca3657e6aa0b/Guided_Upsample/src/dataset_my.py#L203-L209
    # and https://github.com/raywzy/ICT/blob/59dd12d374d47cdf0dce90923017ca3657e6aa0b/Guided_Upsample/src/dataset_my.py#L183-L186
    def resize(self, img: torch.Tensor, target_height: int, target_width: int):
        img = img.to(self.device)
        # If the image tensor is in the format (N, H, W, C), change it to (N, C, H, W)
        if img.dim() == 4 and img.shape[1] > img.shape[3]:
            img = img.permute(0, 3, 1, 2)

        # Handle boolean tensors
        if img.dim() == 3:
            img = img.unsqueeze(1)

        # Center crop for non-square images
        _, _, height, width = img.shape
        if height != width:
            side_length = min(height, width)
            height_offset = (height - side_length) // 2
            width_offset = (width - side_length) // 2
            img = img[:, :, height_offset : height_offset + side_length, width_offset : width_offset + side_length]

        img = img.float()
        img = F.interpolate(img, size=(target_height, target_width), mode="bicubic")

        return img

    # modified from https://github.com/raywzy/ICT/blob/59dd12d374d47cdf0dce90923017ca3657e6aa0b/Guided_Upsample/src/models.py#L165-L183
    def forward(self, images: List[torch.Tensor], appearance_priors: List[torch.Tensor], masks: List[torch.Tensor]):
        images = self.resize(images, self.output_image_size, self.output_image_size)
        appearance_priors = self.resize(appearance_priors, self.output_image_size, self.output_image_size)
        masks = self.resize(masks, self.output_image_size, self.output_image_size)

        images_masked = (images * (1 - masks).float()) + masks

        inputs = torch.cat((images_masked, appearance_priors), dim=1)
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
        config ([`IctConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

ICT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, height * width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`IctImageProcessor.__call__`]
            for details.
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, height * width)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0). Generate random
            masks if not provided.
        clusters (`np.ndarray`, of shape `(n_clusters, 3)`):
            Clusters used to quantize the image of shape `(n_clusters, 3)` before being fed to Guided Upsampler.
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
class IctModel(IctPreTrainedModel):
    config_class = IctConfig

    def __init__(self, config: IctConfig, use_mask_token: bool = True):
        super().__init__(config)

        self.config = config
        self.transformer = IctTransformerModel(config, use_mask_token=use_mask_token)
        self.guided_upsampler = IctGuidedUpsampler(config)
        self.clusters = config.clusters
        self.image_size = config.image_size

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embeddings.token_embedding

    def top_k_logits(self, logits, k):
        values, indices = torch.topk(logits, k)
        new_logits = torch.full_like(logits, -float("inf"))
        new_logits[:, indices[0]] = values
        return new_logits

    def sample_mask(self, pixel_values, logits, bool_masked_pos, temperature=1.0, top_k=50):
        logits = logits / temperature
        bool_masked_pos_expanded = bool_masked_pos.expand(logits.shape[0], logits.shape[1])

        logits = logits[bool_masked_pos_expanded].view(-1, logits.size(-1))
        logits = self.top_k_logits(logits, top_k)
        probs = nn.functional.softmax(logits, dim=-1)
        pred = torch.multinomial(probs, num_samples=1)

        output = torch.zeros_like(pixel_values)
        output[~bool_masked_pos_expanded] = pixel_values[~bool_masked_pos_expanded]
        output[bool_masked_pos_expanded] = pred.squeeze()

        return output

    @add_start_docstrings_to_model_forward(ICT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedImageModelingOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor],
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        clusters: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedImageModelingOutput]:
        r"""
        Returns:

        Example:
        ```python
        >>> import torch
        >>> import numpy as np
        >>> from PIL import Image
        >>> import requests

        >>> from transformers import AutoImageProcessor, IctModel

        >>> image_processor = image_AutoImageProcessor.from_pretrained("sheonhan/ict-imagenet-256")
        >>> model = IctModel.from_pretrained("sheonhan/ict-imagenet-256")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> pixel_values = image_processor(image, return_tensors="pt").pixel_values
        >>> clusters = image_processor.clusters

        >>> # create random boolean mask of shape (batch_size, num_patches)
        >>> bool_masked_pos = torch.randint(low=0, high=2, size=(pixel_values.shape[0] * pixel_values.shape[1])).bool()

        >>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos, clusters=clusters)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        if clusters is None:
            raise ValueError("You have to specify clusters")

        outputs = self.transformer(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs[1]
        batch_size, sequence_length, _ = logits.shape
        height = width = math.floor(sequence_length**0.5)

        original_images = clusters[pixel_values].view(batch_size, height, width, 3)

        recovered_pixel_values = self.sample_mask(
            pixel_values, logits, bool_masked_pos, temperature=self.config.temperature, top_k=self.config.top_k
        )
        recovered_images = clusters[recovered_pixel_values].view(batch_size, height, width, 3)

        if bool_masked_pos is None:
            reshaped_bool_masked_pos = torch.full((batch_size, height, width), 1)
        else:
            reshaped_bool_masked_pos = torch.tile(bool_masked_pos, (batch_size, 1, 1))

        reconstructed_pixel_values = self.guided_upsampler(original_images, recovered_images, reshaped_bool_masked_pos)

        loss = None
        if bool_masked_pos is not None:
            bool_masked_pos = bool_masked_pos.reshape(-1, self.image_size, self.image_size)
            bool_masked_pos.repeat_interleave(1, 1).repeat_interleave(1, 2).unsqueeze(1).contiguous()
            # nn.functional.l1_loss(pixel_values, reconstructed_pixel_values, reduction="none")
            # loss = (reconstruction_loss * mask).sum() / (mask.sum() + 1e-5) / self.config.num_channels

        if not return_dict:
            output = (reconstructed_pixel_values,) + outputs[2:]  # TODO
            return ((loss,) + output) if loss is not None else output

        return MaskedImageModelingOutput(
            loss=loss,
            reconstruction=reconstructed_pixel_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
