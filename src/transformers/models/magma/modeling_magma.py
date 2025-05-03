# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Magma model."""

import math
import re
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
import torch.distributed as dist
from ...modeling_utils import PreTrainedModel
from ...generation import GenerationMixin
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...utils import ModelOutput
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers import AutoConfig, AutoModelForCausalLM
from .configuration_magma import MagmaConfig
from .image_tower_magma import MagmaImageTower

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MagmaConfig"
    
@dataclass
class MagmaCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Magma causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.

            image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class MagmaMultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        projector_type = config.mm_projector_type
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            self.proj = nn.Sequential(*modules)

        if "mm_use_row_seperator" not in config or config.mm_use_row_seperator:
            # define a row seperator
            self.row_seperator = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        if 'mm_use_im_start_end' in config and config.mm_use_im_start_end:
            self.img_start_seperator = nn.Parameter(torch.zeros(1, config.hidden_size))
            self.img_end_seperator = nn.Parameter(torch.zeros(1, config.hidden_size))                        

    def forward(self, x):
        return self.proj(x)

MAGMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MagmaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    MAGMA_START_DOCSTRING,
)

class MagmaPreTrainedModel(PreTrainedModel):
    config_class = MagmaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MagmaImageTower"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self):
        """
        Retrieve language_model's attribute to check whether the model supports
        SDPA or not.
        """
        return self.language_model._supports_sdpa

MAGMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_crops, num_channels, image_size, image_size)`, *optional*):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`MagmaImageProcessor.__call__`] for details. [`MagmaProcessor`] uses
            [`MagmaImageProcessor`] for processing images.
        image_sizes (`torch.LongTensor` of shape `(batch_size, num_crops, 2)`, *optional*):
            The crop sizes of the images in the batch, being (num_crops_height, num_crops_width) for each image.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        vision_feature_layer (`str`, *optional*, defaults to `"clip_vis_dense"`):
            The key of the layer to select the vision feature.
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
"""

@add_start_docstrings(
    "The bare Magma Model outputting raw hidden-states without any specific head on top.",
    MAGMA_START_DOCSTRING,
)
class MagmaModel(MagmaPreTrainedModel):
    def __init__(self, config: MagmaConfig):
        super().__init__(config)
        self.vision_tower = MagmaImageTower(config.vision_config, require_pretrained=False)
        config.vision_config.mm_hidden_size = config.vision_config.mm_hidden_size \
            if 'mm_hidden_size' in config.vision_config else self.vision_tower.hidden_size
        config.vision_config.hidden_size = config.vision_config.hidden_size \
            if 'hidden_size' in config.vision_config else self.config.text_config.hidden_size
        self.multi_modal_projector = MagmaMultiModalProjector(config.vision_config)

        if hasattr(config.text_config, 'auto_map'):
            del config.text_config.auto_map

        try:
            self.language_model = AutoModelForCausalLM.from_config(
                config.text_config,
                attn_implementation=config._attn_implementation,
                trust_remote_code=True
            )
        except:
            self.language_model = AutoModelForCausalLM.from_pretrained(
                config.text_config._name_or_path,
                attn_implementation=config._attn_implementation,
                trust_remote_code=True
            )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: Union[torch.FloatTensor, List[torch.FloatTensor], List[List[torch.FloatTensor]]] = None,
        image_sizes: Union[torch.LongTensor, List[torch.LongTensor], List[List[torch.LongTensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MagmaCausalLMOutputWithPast]:
        # Implementation similar to MagmaForCausalLM's forward but without the language model head
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_config.vision_feature_layer
        )
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if isinstance(past_key_values, Cache):
            use_cache = True

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            if pixel_values is not None and input_ids.shape[1] != 1 and len(pixel_values) > 0:
                # Process images and merge embeddings similar to MagmaForCausalLM
                # ... (same image processing code as in MagmaForCausalLM)
                pass

        outputs = self.language_model.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs

@add_start_docstrings(
    """The Magma model which consists of a vision backbone and a language model.""",
    MAGMA_START_DOCSTRING,
)
class MagmaForCausalLM(MagmaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["language_model.lm_head.weight"]
    
    def __init__(self, config: MagmaConfig):
        super().__init__(config)

        self.vision_tower = MagmaImageTower(config.vision_config, require_pretrained=False)
        config.vision_config.mm_hidden_size = config.vision_config.mm_hidden_size \
            if 'mm_hidden_size' in config.vision_config else self.vision_tower.hidden_size
        config.vision_config.hidden_size = config.vision_config.hidden_size \
            if 'hidden_size' in config.vision_config else self.config.text_config.hidden_size
        self.multi_modal_projector = MagmaMultiModalProjector(config.vision_config)

        self.vocab_size = config.text_config.vocab_size
        if hasattr(config.text_config, 'auto_map'):
            del config.text_config.auto_map

        try:
            self.language_model = AutoModelForCausalLM.from_config(
                config.text_config, 
                attn_implementation=config._attn_implementation, 
                trust_remote_code=True
            )
        except:
            self.language_model = AutoModelForCausalLM.from_pretrained(
                config.text_config._name_or_path, 
                attn_implementation=config._attn_implementation, 
                trust_remote_code=True
            )
        
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self._padding_side = "left"  # set it to left by default, user can use setter to change padding_sides

        self.post_init()
    
    @property
    def padding_side(self):
        return self._padding_side

    @padding_side.setter
    def padding_side(self, padding_side: str):
        if padding_side not in ["left", "right"]:
            raise ValueError(f"{padding_side} is not `left` or `right`.")
        self._padding_side = padding_side

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def load_special_module_from_ckpt(self, ckpt_path, torch_dtype=None):
        from deepspeed.runtime.zero import Init
        from deepspeed import zero
        # Defer initialization for ZeRO-3 compatibility
        # with Init(data_parallel_group=None):
        #     # Initialize the special module
        #     self.vision_tower = MagmaImageTower(self.config.vision_config, require_pretrained=False)

        # Load checkpoint weights into the special module
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = {k.replace('visual.', ''): v for k, v in checkpoint.items() if 'visual.' in k}

        # Convert checkpoint weights to match model's parameter dtype
        if torch_dtype is None:
            model_dtype = next(self.vision_tower.clip_vision_model.parameters()).dtype
            for k, v in state_dict.items():
                state_dict[k] = v.to(model_dtype)
        else:
            for k, v in state_dict.items():
                state_dict[k] = v.to(torch_dtype)

        # Temporarily gather parameters for loading (if ZeRO-3 is active)
        with zero.GatheredParameters(list(self.vision_tower.parameters()), modifier_rank=0):
            # Load the state dictionary
            self.vision_tower.clip_vision_model.load_state_dict(state_dict, strict=False)
            # After loading, ensure the module is on the correct device
            for param in self.vision_tower.parameters():
                param.data = param.data.to(self.device).to(torch_dtype)

        # If using a DeepSpeed engine, attach the updated module
        if hasattr(self, "deepspeed_engine"):
            self.deepspeed_engine.module = self
        
    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def _merge_input_ids_with_image_features(
        self,
        image_features,
        inputs_embeds,
        input_ids,
        attention_mask,
        position_ids=None,
        labels=None,
        image_token_index=None,
        ignore_index=-100,
    ):
        """
        Merge input_ids with with image features into final embeddings

        Args:
            image_features (`torch.Tensor` of shape `(all_feature_lens, embed_dim)`):
                All vision vectors of all images in the batch
            feature_lens (`torch.LongTensor` of shape `(num_images)`):
                The length of visual embeddings of each image as stacked in `image_features`
            inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, embed_dim)`):
                Token embeddings before merging with visual embeddings
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input_ids of tokens, possibly filled with image token
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Mask to avoid performing attention on padding token indices.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.n_positions - 1]`.
            labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*)
                :abels need to be recalculated to support training (if provided)
            image_token_index (`int`, *optional*)
                Token id used to indicate the special "image" token. Defaults to `config.image_token_index`
            ignore_index (`int`, *optional*)
                Value that is used to pad `labels` and will be ignored when calculated loss. Default: -100.
        Returns:
            final_embedding, final_attention_mask, position_ids, final_labels

        Explanation:
            each image has variable length embeddings, with length specified by feature_lens
            image_features is concatenation of all visual embed vectors
            task: fill each <image> with the correct number of visual embeddings
            Example:
                X (5 patches), Y (3 patches), Z (8)
                X, Y are in the same sequence (in-context learning)
            if right padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    o p q r Z s t u v _ _ _ _ _ _
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    o p q r Z Z Z Z Z Z Z Z s t u v _ _ _ _ _
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    o p q r _ _ _ _ _ _ _ _ s t u v _ _ _ _ _
                ]
            elif left padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    _ _ _ _ _ _ o p q r Z s t u v
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    _ _ _ _ _ o p q r Z Z Z Z Z Z Z Z s t u v
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    _ _ _ _ _ o p q r _ _ _ _ _ _ _ _ s t u v
                ]
            Edge cases:
                * If tokens are same but image token sizes are different, then cannot infer left or right padding

                input_ids: [
                    a b c d X g h
                    i j Y k l m n
                ]
                where X is 3 tokens while Y is 5, this mean after merge
                if left-padding (batched generation)
                    input_ids should be: [
                        _ _ a b c d X X X g h
                        i j Y Y Y Y Y k l m n
                    ]
                elif (right padding) (training)
                    input_ids should be: [
                        a b c d X X X g h _ _
                        i j Y Y Y Y Y k l m n
                    ]
        """
        image_token_index = image_token_index if image_token_index is not None else self.config.image_token_index
        ignore_index = ignore_index if ignore_index is not None else self.config.ignore_index

        feature_lens = [elem.shape[0] for elem in image_features]
        image_features = torch.cat(image_features, 0)
        feature_lens = torch.tensor(feature_lens, dtype=torch.long, device=image_features.device)

        with torch.no_grad():
            num_images = feature_lens.size(0)
            num_image_features, embed_dim = image_features.shape
            if feature_lens.sum() != num_image_features:
                raise ValueError(f"{feature_lens=} / {feature_lens.sum()} != {image_features.shape=}")
            batch_size = input_ids.shape[0]
            _left_padding = torch.any(attention_mask[:, 0] == 0)
            _right_padding = torch.any(attention_mask[:, -1] == 0)

            left_padding = True
            if batch_size > 1:
                if _left_padding and not _right_padding:
                    left_padding = True
                elif not _left_padding and _right_padding:
                    left_padding = False
                elif not _left_padding and not _right_padding:
                    # both side is 1, so cannot tell
                    left_padding = self.padding_side == "left"
                else:
                    # invalid attention_mask
                    raise ValueError(f"both side of attention_mask has zero, invalid. {attention_mask}")

            # Whether to turn off right padding
            # 1. Create a mask to know where special image tokens are
            special_image_token_mask = input_ids == image_token_index
            # special_image_token_mask: [bsz, seqlen]
            num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
            # num_special_image_tokens: [bsz]
            # Reserve for padding of num_images
            total_num_special_image_tokens = torch.sum(special_image_token_mask)
            if total_num_special_image_tokens != num_images:
                raise ValueError(
                    f"Number of image tokens in input_ids ({total_num_special_image_tokens}) different from num_images ({num_images})."
                )
            # Compute the maximum embed dimension
            # max_image_feature_lens is max_feature_lens per batch
            feature_lens_batch = feature_lens.split(num_special_image_tokens.tolist(), dim=0)
            feature_lens_batch_sum = torch.tensor([x.sum() for x in feature_lens_batch], device=feature_lens.device)
            embed_sequence_lengths = (
                (attention_mask == 1).long().sum(-1) - num_special_image_tokens + feature_lens_batch_sum
            )
            max_embed_dim = embed_sequence_lengths.max()

            batch_indices, non_image_indices = torch.where((input_ids != image_token_index) & (attention_mask == 1))
            # 2. Compute the positions where text should be written
            # Calculate new positions for text tokens in merged image-text sequence.
            # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images` text tokens.
            # `torch.cumsum` computes how each image token shifts subsequent text token positions.
            # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
            # ! instead of special_image_token_mask * (num_image_patches - 1)
            #   special_image_token_mask * (num_feature_len - 1)
            special_image_token_mask = special_image_token_mask.long()
            special_image_token_mask[special_image_token_mask == 1] = feature_lens - 1
            new_token_positions = torch.cumsum((special_image_token_mask + 1), -1) - 1
            if left_padding:
                # shift right token positions so that they are ending at the same number
                # the below here was incorrect? new_token_positions += new_token_positions[:, -1].max() - new_token_positions[:, -1:]
                new_token_positions += max_embed_dim - 1 - new_token_positions[:, -1:]

            text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        final_labels = None
        if labels is not None:
            # NOTE: this is a bug in the original code!!!
            final_labels = torch.full_like(final_attention_mask.long(), ignore_index).to(torch.long)
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
        with torch.no_grad():
            image_to_overwrite = torch.full(
                (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
            )
            image_to_overwrite[batch_indices, text_to_overwrite] = False
            embed_indices = torch.arange(max_embed_dim).unsqueeze(0).to(target_device)
            embed_indices = embed_indices.expand(batch_size, max_embed_dim)
            embed_seq_lens = embed_sequence_lengths[:, None].to(target_device)

            if left_padding:
                # exclude padding on the left
                val = (max_embed_dim - embed_indices) <= embed_seq_lens
            else:
                # exclude padding on the right
                val = embed_indices < embed_seq_lens
            image_to_overwrite &= val

            if image_to_overwrite.sum() != num_image_features:
                raise ValueError(
                    f"{image_to_overwrite.sum()=} != {num_image_features=} The input provided to the model are wrong. "
                    f"The number of image tokens is {torch.sum(special_image_token_mask)} while"
                    f" the number of image given to the model is {num_images}. "
                    f"This prevents correct indexing and breaks batch generation."
                )
        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        return final_embedding, final_attention_mask, position_ids, final_labels

    @add_start_docstrings_to_model_forward(MAGMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MagmaCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: Union[torch.FloatTensor, List[torch.FloatTensor], List[List[torch.FloatTensor]]] = None,
        image_sizes: Union[torch.LongTensor, List[torch.LongTensor], List[List[torch.LongTensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, MagmaCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, MagmaForCausalLM

        >>> model = MagmaForCausalLM.from_pretrained("microsoft/magma-8b")
        >>> processor = AutoProcessor.from_pretrained("microsoft/magma-8b")

        >>> convs = [
        >>>     {"role": "system", "content": "You are agent that can see, talk and act."},            
        >>>     {"role": "user", "content": "<image_start><image><image_end>\nWhat is the letter on the robot?"},
        >>> ]
        >>> url = "https://microsoft.github.io/Magma/static/images/logo.png"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> prompt = processor.tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(images=[image], texts=prompt, return_tensors="pt")
        >>> inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
        >>> inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)     

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The letter on the robot is \"M\"."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_config.vision_feature_layer
        )
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if isinstance(past_key_values, Cache):
            use_cache = True

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1 and len(pixel_values) > 0:
                # ! infer image_num_patches from image_sizes
                if type(pixel_values) == list:
                    # nested list of pixel_values, each element is a list of pixel_values for each training instance, it could be multiple for video or interleaved setting
                    # e.g., pixel_values = [[img1, img2], [img1, img2, img3]]
                    n_imgs_per_sample = [len(pv) for pv in pixel_values]
                    pixels_values_list = sum(pixel_values, [])
                    image_sizes_list = sum(image_sizes, [])
                else:
                    image_num_patches = [(imsize[imsize.sum(1) > 0,0] * imsize[imsize.sum(1) > 0,1]).tolist() for imsize in image_sizes]       
                    # image_num_patches = [(imsize[:,0]*imsize[:,1]).tolist() for imsize in image_sizes]             
                    # figure out if pixel_values is concatenated or stacked
                    if pixel_values.dim() == 5:
                        # stacking when input is (batch_size, num_patches, num_channels, height, width)
                        _pixel_values_list = [
                            pix_val[:sum(num_patch)].split(num_patch, dim=0) for pix_val, num_patch in zip(pixel_values, image_num_patches)
                        ]
                        _image_sizes_list = [image_size[image_size.sum(-1) > 0].tolist() for image_size in image_sizes]
                    elif pixel_values.dim() != 4:
                        # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
                        raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")

                if self.config.vision_config.img_anyres_strategy == "global":
                    flattened_image_features = []
                    # NOTE: both _image_sizes_list and _pixel_values_list are lists of lists, each item represents an training instance with one or multiple images
                    for idx, (image_size_for_instance, pixel_values_for_instance) in enumerate(zip(_image_sizes_list, _pixel_values_list)):
                        assert len(image_size_for_instance) == len(pixel_values_for_instance), f"{len(image_size_for_instance)} != {len(pixel_values_for_instance)}"
                        for image_size, pixel_values_for_image in zip(image_size_for_instance, pixel_values_for_instance):
                            pixel_values_for_image = pixel_values_for_image.view(image_size[0], image_size[1], *pixel_values_for_image.shape[1:])
                            pixel_values_for_image = pixel_values_for_image.permute(2, 0, 3, 1, 4).flatten(3, 4).flatten(1, 2).unsqueeze(0)
                            image_features = self.vision_tower(pixel_values_for_image)
                            selected_image_feature = image_features[vision_feature_layer][0].permute(1, 2, 0)
                            selected_image_feature = self.multi_modal_projector(selected_image_feature)
                            if "mm_use_row_seperator" not in self.config.vision_config or self.config.vision_config.mm_use_row_seperator:
                                selected_image_feature = torch.cat((selected_image_feature, self.multi_modal_projector.row_seperator.repeat(selected_image_feature.shape[0],1,1)), dim=1)
                            flattened_image_features.append(selected_image_feature.flatten(0, 1))
                elif self.config.vision_config.img_anyres_strategy == "crop":
                    # calculate number of crops for each instance in the batch given _image_sizes_list
                    _image_sizes_list_temp = sum(_image_sizes_list, [])
                    # concate nate all images in _pixel_values_list
                    _pixel_values_list_temp = sum(_pixel_values_list, ())
                    _pixel_values_list_temp = torch.cat(_pixel_values_list_temp, dim=0)
                    image_features = self.vision_tower(_pixel_values_list_temp)[vision_feature_layer].permute(0, 2, 3, 1)
                    image_features = self.multi_modal_projector(image_features)

                    num_crops_list = [_image_size[0]*_image_size[1] for _image_size in _image_sizes_list_temp]
                    image_features_split = torch.split(image_features, num_crops_list, dim=0)
                    flattened_image_features = []
                    for image_feature, image_size in zip(image_features_split, _image_sizes_list_temp):
                        image_feature = image_feature.view(image_size[0], image_size[1], *image_feature.shape[1:])
                        image_feature = image_feature.permute(0, 2, 1, 3, 4).flatten(2, 3).flatten(0, 1)
                        if "mm_use_row_seperator" not in self.config.vision_config or self.config.vision_config.mm_use_row_seperator:
                            image_feature = torch.cat((image_feature, self.multi_modal_projector.row_seperator.repeat(image_feature.shape[0],1,1)), dim=1)
                        flattened_image_features.append(image_feature.flatten(0, 1))

                # NOTE we only support multimodal_patch_merge_type == "spatial_unpad"
                # inputs_embeds, attention_mask, position_ids, labels = self._merge_input_ids_with_image_features(
                #     flattened_image_features,
                #     inputs_embeds,
                #     input_ids,
                #     attention_mask,
                #     position_ids,
                #     labels=labels,
                # )

                inputs_embeds[input_ids == self.config.image_token_index] = torch.cat(flattened_image_features, dim=0)

            elif past_key_values is not None and pixel_values is None and input_ids.shape[1] == 1:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)

                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        outputs = self.language_model.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]

        loss = None

        if labels is not None and self.training:
            valid_mask = labels[..., 1:] != -100
            shift_logits = self.language_model.lm_head(hidden_states[:,:-1][valid_mask]).contiguous()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            logits = shift_logits # dummy logits
            shift_labels = labels[..., 1:][valid_mask].contiguous()
            shift_labels = shift_labels.to(shift_logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)
        else:
            logits = self.language_model.lm_head(hidden_states)
            logits = logits.float()

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return MagmaCausalLMOutputWithPast(
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
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        image_sizes=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs,
        )

        if cache_position[0] == 0:
            # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
            # Otherwise we need pixel values to be passed to model
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_sizes"] = image_sizes

        return model_inputs

    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)


__all__ = ["MagmaModel", "MagmaForCausalLM", "MagmaPreTrainedModel"]