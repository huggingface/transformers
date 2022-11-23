# coding=utf-8
# Copyright 2022 NAVER AI Labs and The HuggingFace Inc. team. All rights reserved.
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
""" TODO"""

import os
import pickle
import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    ModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from .import objectives
from .image_processing_bridgetower import build_model, adapt_position_encoding, swin_adapt_position_encoding
from .import swin_transformer as swin
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, is_torch_greater_or_equal_than_1_10, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_bridgetower import BridgeTowerConfig
from transformers import RobertaConfig, RobertaModel
from transformers import BertConfig, BertModel
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)

logger = logging.get_logger(__name__)

if not is_torch_greater_or_equal_than_1_10:
    logger.warning(
        f"You are using torch=={torch.__version__}, but torch>=1.10.0 is required to use "
        "BridgeTowerModel. Please upgrade torch."
    )

_CONFIG_FOR_DOC = "BridgeTowerConfig"
_CHECKPOINT_FOR_DOC = ""

BRIDGETOWER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "",
    # See all bridgetower models at https://huggingface.co/models?filter=bridgetower
]

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
    
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)#TODO:Add as self.config.initializer_range
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)#TODO: self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # TODO: Do we need this?
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BridgeTowerEncoder):
            module.gradient_checkpointing = value

class BridgeTowerModel(BridgeTowerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.is_clip= (not 'swin' in config.vit)

        #import ipdb; ipdb.set_trace()
        if 'roberta' in config.tokenizer:
            bert_config = RobertaConfig(
                vocab_size=config.vocab_size,
                hidden_size=config.hidden_size,
                num_hidden_layers=config.num_layers,
                num_attention_heads=config.num_heads,
                intermediate_size=config.hidden_size * config.mlp_ratio,
                max_position_embeddings=config.max_text_len,
                hidden_dropout_prob=config.drop_rate,
                attention_probs_dropout_prob=config.drop_rate
            )
        else:
            bert_config = BertConfig(
                vocab_size=config.vocab_size,
                hidden_size=config.hidden_size,
                num_hidden_layers=config.num_layers,
                num_attention_heads=config.num_heads,
                intermediate_size=config.hidden_size * config.mlp_ratio,
                max_position_embeddings=config.max_text_len,
                hidden_dropout_prob=config.drop_rate,
                attention_probs_dropout_prob=config.drop_rate
            )

        resolution_after=config.image_size

        if config.cross_modal_transform_shared:
            self.cross_modal_text_transform = nn.Linear(config.input_text_embed_size, config.hidden_size)
            self.cross_modal_image_transform = nn.Linear(config.input_image_embed_size, config.hidden_size)
        else:
            self.cross_modal_text_transform = nn.ModuleList([nn.Linear(config.input_text_embed_size, config.hidden_size) for _ in range(config.num_layers)])
            self.cross_modal_image_transform = nn.ModuleList([nn.Linear(config.input_image_embed_size, config.hidden_size) for _ in range(config.num_layers)])
        self.cross_modal_text_transform.apply(objectives.init_weights)
        self.cross_modal_image_transform.apply(objectives.init_weights)


        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.token_type_embeddings.apply(objectives.init_weights)

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                if self.is_clip:
                    build_model(config.vit, resolution_after=resolution_after, model_type=config.model_type, stop_gradient=config.stop_gradient, vit_layernorm_shared=config.vit_layernorm_shared, vit_remove_last=config.vit_remove_last)
                else:
                    getattr(swin, config.vit)(
                        pretrained=True, config=config,
                    )

                if 'roberta' in config.tokenizer:
                    RobertaModel.from_pretrained(config.tokenizer, cache_dir=f"{config.cache_dir}/{config.tokenizer}")
                else:
                    BertModel.from_pretrained(config.tokenizer, cache_dir=f"{config.cache_dir}/{config.tokenizer}")

            torch.distributed.barrier()

        if self.is_clip:
            self.vit_model = build_model(config.vit, resolution_after=resolution_after, model_type=config.model_type, stop_gradient=config.stop_gradient, vit_layernorm_shared=config.vit_layernorm_shared, vit_remove_last=config.vit_remove_last)
        else:
            self.vit_model = getattr(swin, config.vit)(
                pretrained=True, config=config,
            )
            self.avgpool = nn.AdaptiveAvgPool1d(1)

        if 'roberta' in config.tokenizer:
            roberta_config = RobertaConfig.from_pretrained(config.tokenizer)
            self.text_transformer = RobertaModel(roberta_config)
            #self.text_transformer = RobertaModel.from_pretrained(config.tokenizer, cache_dir=f"{config.cache_dir}/{config.tokenizer}")
        else:
            self.text_transformer = BertModel.from_pretrained(config.tokenizer, cache_dir=f"{config.cache_dir}/{config.tokenizer}")

        if not config.vit_layernorm_shared and config.vit_layernorm_init_from_vit:
            for ln in self.vit_model.visual.cross_modal_ln_separate:
                ln.weight.data = self.vit_model.visual.ln_post.weight.data
                ln.bias.data = self.vit_model.visual.ln_post.bias.data

        self.cross_modal_image_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config.num_layers)])
        self.cross_modal_image_layers.apply(objectives.init_weights)
        self.cross_modal_text_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config.num_layers)])
        self.cross_modal_text_layers.apply(objectives.init_weights)

        # Class token => Linear => Tanh
        self.cross_modal_image_pooler = Pooler(config.hidden_size)
        self.cross_modal_image_pooler.apply(objectives.init_weights)
        self.cross_modal_text_pooler = Pooler(config.hidden_size)
        self.cross_modal_text_pooler.apply(objectives.init_weights)

        if config.loss_names["mlm"] > 0:
            # MLM Head weights don't tie with BERT Embedding weights. Train from scratch.
            self.mlm_score = MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config.loss_names["itm"] > 0:
            self.itm_score = ITMHead(config.hidden_size * 2)
            self.itm_score.apply(objectives.init_weights)

        hs = config.hidden_size

        # ===================== Initialize LCI Components ===================== #
        # just for first layer
        self.cross_modal_text_layernorm = nn.LayerNorm(config.hidden_size)
        self.cross_modal_text_layernorm.apply(objectives.init_weights)
        self.cross_modal_image_layernorm = nn.LayerNorm(config.hidden_size)
        self.cross_modal_image_layernorm.apply(objectives.init_weights)

        if config.link_tower_shared:
            self.cross_modal_text_link_tower = LinkTower(config, bert_config)
            self.cross_modal_image_link_tower = LinkTower(config, bert_config)
        else:
            self.cross_modal_text_link_tower = nn.ModuleList([LinkTower(config, bert_config) for _ in range(config.num_layers - 1)])
            self.cross_modal_image_link_tower = nn.ModuleList([LinkTower(config, bert_config) for _ in range(config.num_layers - 1)])

        self.cross_modal_text_link_tower.apply(objectives.init_weights)
        self.cross_modal_image_link_tower.apply(objectives.init_weights)


        # ===================== Load Pretrained METER Weights ===================== 

        if (config.load_path != "" and not config.test_only):
            ckpt = torch.load(config.load_path, map_location="cpu")
            state_dict = ckpt["state_dict"]

            # with open('./original_model_dict.txt', 'w') as f:
            #     for name, param in state_dict.items():
            #         print(name, param.shape, param.type(), file=f)

            if config.only_load_cross_modal_from_meter:
                state_dict = {k: v for k, v in state_dict.items() if k.startswith("cross_modal")}
                # print(state_dict.keys())
            else:
                if self.is_clip:
                    state_dict = adapt_position_encoding(state_dict, after=resolution_after, patch_size=config.patch_size)
                else:
                    state_dict = swin_adapt_position_encoding(state_dict, after=resolution_after, before=config.resolution_before)
            
            if config.link_tower_type == 'add' and 'cross_modal_text_layernorm.5.weight' in state_dict.keys(): # For old version migration
                old_layernorm_dict = {}
                for name, param in state_dict.items():
                    if 'cross_modal_text_layernorm' in name or 'cross_modal_image_layernorm' in name:
                        old_layernorm_dict[name] = param
                for name in list(old_layernorm_dict.keys()):
                    if '0' in name:
                        state_dict.pop(name)
                        state_dict[name.replace('.0', '')] = old_layernorm_dict[name]
                    else:
                        state_dict.pop(name)
                        name_index = name.split('.')[1]
                        state_dict[name.replace('layernorm', 'link_tower').replace(f'{name_index}.', f'{str(int(name_index) - 1)}.LayerNorm.')] = old_layernorm_dict[name]

            self.load_state_dict(state_dict, strict=False)

        # ===================== Freeze specified modules ===================== #
        
        if config.freeze_ViT:
            self.vit_model.requires_grad_(False)
            if config.unfreeze_ViT_attention:
                for name, param in self.vit_model.named_parameters():
                    if 'attn' in name:
                        param.requires_grad_(True)
            if config.unfreeze_ViT_layernorm:
                for name, param in self.vit_model.named_parameters():
                    if 'ln_' in name:
                        param.requires_grad_(True)
        
        if config.freeze_RoBERTa:
            self.text_transformer.requires_grad_(False)
            
            if config.unfreeze_RoBERTa_embeddings:
                self.text_transformer.embeddings.requires_grad_(True)
            
            if config.unfreeze_RoBERTa_encoder:
                self.text_transformer.encoder.requires_grad_(True)
            
            if config.unfreeze_RoBERTa_attention:
                for name, param in self.text_transformer.named_parameters():
                    if 'attention' in name:
                        param.requires_grad_(True)

            if config.unfreeze_RoBERTa_layernorm:
                for name, param in self.text_transformer.named_parameters():
                    if 'LayerNorm' in name:
                        param.requires_grad_(True)
            
            
        if config.freeze_layer_count_roberta > 0:
            modules = [self.text_transformer.embeddings, *self.text_transformer.encoder.layer[:config.freeze_layer_count_roberta]] #Replace 5 by what you want
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False


        if config.freeze_layer_count_vit > 0:
            modules = [*self.vit_model.visual.transformer.resblocks[:config.freeze_layer_count_vit]]
            self.vit_model.visual.class_embedding.requires_grad = False
            for module in modules:
                # module.requires_grad_(False)
                for param in module.parameters():
                    param.requires_grad = False
                    
        # ===================== Downstream ===================== #
        
        if config.downstream_fusion:
            assert config.downstream_fusion_layers > 1
            self.cross_modal_downstream_fusion_head = FusionHead(config)
            self.cross_modal_downstream_fusion_head.apply(objectives.init_weights)

        hscale = config.head_hidden_scale
        dscale = 1 # downstream_fusion_scale
        if config.downstream_fusion:
            if config.downstream_fusion_method == 'concat':
                dscale = config.downstream_fusion_layers
            elif config.downstream_fusion_method in ['max_mean_pooling', 'lstm_bi']:
                dscale = 2
        if config.loss_names["vqa"] > 0:
            vs = config.vqav2_label_size
            if config.task_head_layers == 1:
                self.vqa_classifier = nn.Sequential(
                    nn.Linear(hs * 2 * dscale, vs),
                )
                if config.downstream_fusion_method == 'ensemble':
                    self.vqa_classifier = nn.ModuleList([nn.Sequential(
                        nn.Linear(hs * 2 * dscale, vs),
                    ) for _ in range(config.downstream_fusion_layers)])
            elif config.task_head_layers == 2:
                self.vqa_classifier = nn.Sequential(
                    nn.Linear(hs * 2 * dscale, hs * 2 * dscale * hscale),
                    nn.LayerNorm(hs * 2 * dscale * hscale),
                    nn.GELU(),
                    nn.Linear(hs * 2 * dscale * hscale, vs),
                )
                if config.downstream_fusion_method == 'ensemble':
                    self.vqa_classifier = nn.ModuleList([nn.Sequential(
                        nn.Linear(hs * 2 * dscale, hs * 2 * dscale * hscale),
                        nn.LayerNorm(hs * 2 * dscale * hscale),
                        nn.GELU(),
                        nn.Linear(hs * 2 * dscale * hscale, vs),
                    ) for _ in range(config.downstream_fusion_layers)])
            elif config.task_head_layers == 3:
                self.vqa_classifier = nn.Sequential(
                    nn.Linear(hs * 2 * dscale, hs * 2 * dscale * hscale),
                    nn.LayerNorm(hs * 2 * dscale * hscale),
                    nn.GELU(),
                    nn.Linear(hs * 2 * dscale * hscale, hs * 2 * dscale * hscale),
                    nn.LayerNorm(hs * 2 * dscale * hscale),
                    nn.GELU(),
                    nn.Linear(hs * 2 * dscale * hscale, vs),
                )
            self.vqa_classifier.apply(objectives.init_weights)

        if config.loss_names["nlvr2"] > 0:
            nlvr2_input_scale = 4 # 2 * 2
            if config.nlvr2_head_format == 'pair-biatten':
                self.nlvr2_biatten_head_attn1 = nn.MultiheadAttention(hs, config.num_heads, dropout=config.drop_rate, batch_first=True)
                self.nlvr2_biatten_head_attn1.apply(objectives.init_weights)
                self.nlvr2_biatten_head_attn2 = nn.MultiheadAttention(hs, config.num_heads, dropout=config.drop_rate, batch_first=True)
                self.nlvr2_biatten_head_attn2.apply(objectives.init_weights)
                self.nlvr2_biatten_head_fc = nn.Sequential(
                    nn.Linear(hs * 2, hs * 2), 
                    nn.LayerNorm(hs * 2), 
                    nn.GELU(), 
                    nn.Dropout(config.drop_rate)
                )
                self.nlvr2_biatten_head_fc.apply(objectives.init_weights)
                self.nlvr2_biatten_head_attn_pool = AttentionPool(hs * 2, config.drop_rate)
                self.nlvr2_biatten_head_attn_pool.apply(objectives.init_weights)
            elif config.nlvr2_head_format == 'triplet':
                nlvr2_input_scale = 3 # 1 + 1 + 1
            
            if config.task_head_layers == 1:
                self.nlvr2_classifier = nn.Sequential(
                    nn.Linear(hs * nlvr2_input_scale, 2),
                )
            elif config.task_head_layers == 2:
                self.nlvr2_classifier = nn.Sequential(
                    nn.Linear(hs * nlvr2_input_scale, int(hs * 2 * hscale)), # use int for triplet
                    nn.LayerNorm(int(hs * 2 * hscale)),
                    nn.GELU(),
                    nn.Linear(int(hs * 2 * hscale), 2),
                )
            self.nlvr2_classifier.apply(objectives.init_weights)
            self.nlvr2_classifier_dropout = nn.Dropout(config.classifier_drop_rate)
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, hs)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        if config.loss_names["snli"] > 0:
            if config.task_head_layers == 1:
                self.snli_classifier = nn.Sequential(
                    nn.Linear(hs * 2, 3),
                )
            elif config.task_head_layers == 2:
                self.snli_classifier = nn.Sequential(
                    nn.Linear(hs * 2, hs * 2 * hscale),
                    nn.LayerNorm(hs * 2 * hscale),
                    nn.GELU(),
                    nn.Linear(hs * 2 * hscale, 3),
                )
            self.snli_classifier.apply(objectives.init_weights)

        if config.loss_names["irtr"] > 0:
            self.rank_output = nn.Linear(hs * 2, 1)
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            for p in self.itm_score.parameters():
                p.requires_grad = False

        # ===================== load downstream (test_only) ======================

        if config.load_path != "" and config.test_only:
            ckpt = torch.load(config.load_path, map_location="cpu")
            state_dict = ckpt["state_dict"]
            if self.is_clip:
                state_dict = adapt_position_encoding(state_dict, after=resolution_after, patch_size=config.patch_size)
            else:
                state_dict = swin_adapt_position_encoding(state_dict, after=resolution_after, before=config.resolution_before)
            self.load_state_dict(state_dict, strict=False)

        #meter_utils.set_metrics(self)
        self.current_tasks = list()

    #def forward(
    #        self,
    #        batch,
    #        mask_text=False,
    #        mask_image=False,
    #        image_token_type_idx=1,
    #        img=None,
    #        irtr_len=0,
    #    ):
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
    ) -> Union[BaseModelOutputWithPooling, Tuple[torch.FloatTensor]]:
        #if img is None:
        #    if f"image_{image_token_type_idx - 1}" in batch:
        #        imgkey = f"image_{image_token_type_idx - 1}"
        #    else:
        #        imgkey = "image"
        #    img = batch[imgkey][0]
        
        #do_mlm = "_mlm" if mask_text else ""
        image_token_type_idx=1
        irtr_len=0
        text_ids = input_ids
        img = pixel_values.resize_(1,3,224,224)
        text_masks = attention_mask
	#text_labels = batch[f"text_labels{do_mlm}"]
        #text_masks = batch[f"text_masks"]
        input_shape = input_ids.size() 
        text_embeds = self.text_transformer.embeddings(input_ids=text_ids)
        
        #input_shape = text_masks.size()
        extend_text_masks = self.text_transformer.get_extended_attention_mask(text_masks, input_shape, self.device)
        
        split_index = len(self.text_transformer.encoder.layer) - self.config.num_layers + 1
        for layer in self.text_transformer.encoder.layer[:split_index]:
            text_embeds = layer(text_embeds, extend_text_masks)[0]
        
        image_embeds = self.vit_model.visual.forward_pre(img.type(self.vit_model.dtype))
        for block in self.vit_model.visual.transformer.resblocks[:split_index]:
            image_embeds = block(image_embeds)
        # image_embeds = checkpoint_sequential(self.vit_model.visual.transformer.resblocks[:split_index], 2, image_embeds)
        image_embeds_ = self.vit_model.visual.forward_post(image_embeds.type(self.vit_model.dtype))
        
        # first layer
        x = self.cross_modal_text_transform(text_embeds)
        text_token_type_embeddings = self.token_type_embeddings(torch.zeros(1).long().to(self.device)).expand_as(x)
        x = self.cross_modal_text_layernorm(x + text_token_type_embeddings)
        
        image_embeds_ = self.cross_modal_image_transform(image_embeds_)
        image_token_type_embeddings = self.token_type_embeddings(torch.zeros(1).long().to(self.device).fill_(image_token_type_idx)).expand_as(image_embeds_)
        image_embeds_ = image_embeds_ + image_token_type_embeddings
        y = self.cross_modal_image_layernorm(image_embeds_)
        if irtr_len > 0:
            _bs, _L, _D = image_embeds_.size()
            y = y.unsqueeze(1).expand(_bs, irtr_len, _L, _D).reshape(-1, _L, _D)
        image_masks = torch.ones((y.size(0), y.size(1)), dtype=torch.long, device=self.device)
        extend_image_masks = self.text_transformer.get_extended_attention_mask(image_masks, image_masks.size(), self.device)
        
        x1 = self.cross_modal_text_layers[0](x, y, extend_text_masks, extend_image_masks)[0]
        y1 = self.cross_modal_image_layers[0](y, x, extend_image_masks, extend_text_masks)[0]
        # assert batch['text_ids'].sum() * 0 == 0 
        link_layer_index = 0

        # link tower fusion
        for i in range(split_index, len(self.text_transformer.encoder.layer)):
            text_embeds = self.text_transformer.encoder.layer[i](text_embeds, extend_text_masks)[0]
            image_embeds = self.vit_model.visual.transformer.resblocks[i](image_embeds).type(self.vit_model.dtype)
            image_embeds_ = self.cross_modal_image_transform(self.vit_model.visual.forward_post(image_embeds)) + image_token_type_embeddings

            text_link_tower = self.cross_modal_text_link_tower[link_layer_index]
            image_link_tower = self.cross_modal_image_link_tower[link_layer_index]

            x1_ = text_link_tower(self.cross_modal_text_transform(text_embeds) + text_token_type_embeddings, x1, extend_text_masks)
            if irtr_len > 0:
                y1_ = image_link_tower(image_embeds_.unsqueeze(1).expand(_bs, irtr_len, _L, _D).reshape(-1, _L, _D), y1, extend_image_masks)
            else:
                y1_ = image_link_tower(image_embeds_, y1, extend_image_masks)
        
            x1 = self.cross_modal_text_layers[link_layer_index + 1](x1_, y1_, extend_text_masks, extend_image_masks)[0]
            y1 = self.cross_modal_image_layers[link_layer_index + 1](y1_, x1_, extend_image_masks, extend_text_masks)[0]

            link_layer_index += 1

        text_feats, image_feats = x1, y1
        cls_feats = self.get_cls_feats(text_feats, image_feats)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "text_ids": text_ids,
            "text_masks": text_masks,
        }
        return ret
    def get_cls_feats(self, text_feats, image_feats):
        cls_feats_text = self.cross_modal_text_pooler(text_feats)
        if self.is_clip:
            cls_feats_image = self.cross_modal_image_pooler(image_feats)
        else:
            avg_image_feats = self.avgpool(image_feats.transpose(1, 2)).view(image_feats.size(0), 1, -1)
            cls_feats_image = self.cross_modal_image_pooler(avg_image_feats)
        return torch.cat([cls_feats_text, cls_feats_image], dim=-1)


class LinkTower(nn.Module):
    def __init__(self, config, bert_config):
        super(LinkTower, self).__init__()
        self.link_tower_type = config.link_tower_type
        self.hidden_size = config.hidden_size
        if config.link_tower_type in ['add', 'scaled_add', 'scaled_add_tensor', 'interpolate', 'interpolate_tensor', 'gated_interpolate', 'elementwise_product', 'elementwise_product_residual', 'linear_add_fpn', 'linear_concat', 'linear_add', 'linear_concat_residual', 'linear_add_residual', 'mlp_concat', 'mlp_add', 'mlp_concat_residual', 'mlp_add_residual']:
            if config.link_tower_type == 'scaled_add':
                self.scaled_factor = nn.Parameter(torch.tensor(1.))
            elif config.link_tower_type == 'scaled_add_tensor':
                self.scaled_factor = nn.Parameter(torch.ones(self.hidden_size))
            elif config.link_tower_type == 'interpolate':
                self.beta = nn.Parameter(torch.tensor(0.5))
            elif config.link_tower_type == 'interpolate_tensor':
                self.beta = nn.Parameter(torch.ones(self.hidden_size))
                self.beta.data.fill_(0.5)
            elif config.link_tower_type == 'gated_interpolate':
                self.gate = nn.Sequential(
                    nn.Linear(self.hidden_size * 2, self.hidden_size),
                    nn.Sigmoid()
                )
            elif config.link_tower_type == 'linear_add_fpn':
                self.dense1 = nn.Linear(self.hidden_size, self.hidden_size)
                self.dense2 = nn.Linear(self.hidden_size, self.hidden_size)
            elif config.link_tower_type in ['linear_concat', 'linear_add', 'linear_concat_residual', 'linear_add_residual']:
                count = 2 if 'concat' in config.link_tower_type else 1
                self.dense = nn.Linear(self.hidden_size * count, self.hidden_size)
            elif config.link_tower_type in ['mlp_concat', 'mlp_add', 'mlp_concat_residual', 'mlp_add_residual']:
                count = 2 if 'concat' in config.link_tower_type else 1
                self.dense = nn.Sequential(
                    nn.Linear(self.hidden_size * count, self.hidden_size * count),
                    nn.LayerNorm(self.hidden_size * count),
                    nn.GELU(),
                    nn.Linear(self.hidden_size * count, self.hidden_size)
                )
            self.LayerNorm = nn.LayerNorm(self.hidden_size)
        elif config.link_tower_type in ['cross_attention', 'cross_attention_ffn']:
            self.dense = BertLinkLayer(bert_config, self.link_tower_type)
        else:
            raise NotImplementedError(f"link_tower_type {config.link_tower_type} is not implemented")

    def forward(self, hidden_states, cross_modal_hidden_states, attention_mask):
        if self.link_tower_type == 'add':
            return self.LayerNorm(hidden_states + cross_modal_hidden_states)
        elif self.link_tower_type in ['scaled_add', 'scaled_add_tensor']:
            return self.LayerNorm(hidden_states * self.scaled_factor + cross_modal_hidden_states)
        elif self.link_tower_type in ['interpolate', 'interpolate_tensor']:
            return self.LayerNorm(hidden_states * (1 - self.beta) + cross_modal_hidden_states * self.beta)
        elif self.link_tower_type == 'gated_interpolate':
            scaled_gate = self.gate(torch.cat([hidden_states, cross_modal_hidden_states], dim=-1))
            return self.LayerNorm(hidden_states * (1 - scaled_gate) + cross_modal_hidden_states * scaled_gate)
        elif self.link_tower_type in ['elementwise_product', 'elementwise_product_residual']:
            if 'residual' in self.link_tower_type:
                return self.LayerNorm(cross_modal_hidden_states + hidden_states * cross_modal_hidden_states)
            else:
                return self.LayerNorm(hidden_states * cross_modal_hidden_states)
        elif self.link_tower_type in ['linear_add_fpn']:
            return self.LayerNorm(self.dense1(hidden_states) + self.dense2(cross_modal_hidden_states))
        elif self.link_tower_type in ['linear_concat', 'linear_concat_residual', 'mlp_concat', 'mlp_concat_residual']:
            if 'residual' in self.link_tower_type:
                return self.LayerNorm(cross_modal_hidden_states + self.dense(torch.cat([hidden_states, cross_modal_hidden_states], dim=-1)))
            else:
                return self.LayerNorm(self.dense(torch.cat([hidden_states, cross_modal_hidden_states], dim=-1)))
        elif self.link_tower_type in ['linear_add', 'linear_add_residual', 'mlp_add', 'mlp_add_residual']:
            if 'residual' in self.link_tower_type:
                return self.LayerNorm(cross_modal_hidden_states + self.dense(hidden_states + cross_modal_hidden_states))
            else:
                return self.LayerNorm(self.dense(hidden_states + cross_modal_hidden_states))
        elif self.link_tower_type in ['cross_attention', 'cross_attention_ffn']:
            return self.dense(cross_modal_hidden_states, hidden_states, attention_mask)
        else:
            raise NotImplementedError(f"link_tower_type {self.link_tower_type} is not implemented")

class FusionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.downstream_fusion_layers = config.downstream_fusion_layers
        self.downstream_fusion_method = config.downstream_fusion_method
        self.hidden_size = config.hidden_size

        if self.downstream_fusion_method in ['max_mean_pooling', 'lstm_bi']:
            self.LayerNorm = nn.LayerNorm(self.hidden_size * 2 * 2)
        elif self.downstream_fusion_method in ['elmo', 'elmo_tensor', 'gated', 'gated_meter', 'max_pooling', 'mean_pooling', 'lstm_uni']:
            self.LayerNorm = nn.LayerNorm(self.hidden_size * 2)
        
        if self.downstream_fusion_method == 'elmo':
            self.learned_layer_weights = nn.Parameter(torch.zeros(self.downstream_fusion_layers,))
        elif self.downstream_fusion_method == 'elmo_tensor':
            self.learned_layer_weights = nn.Parameter(torch.zeros(self.downstream_fusion_layers, self.hidden_size * 2))
        elif self.downstream_fusion_method in ['gated', 'gated_meter']:
            self.gate = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)
        elif self.downstream_fusion_method in ['lstm_uni', 'lstm_bi']:
            if 'uni' in self.downstream_fusion_method:
                bidirectional = False
            elif 'bi' in self.downstream_fusion_method:
                bidirectional = True
            self.lstmpooler = nn.LSTM(self.hidden_size * 2, self.hidden_size * 2, batch_first=True, bidirectional=bidirectional)
            self.lstmpooler_dropout = nn.Dropout(config.drop_rate)

    def forward(self, feats):
        if self.downstream_fusion_method == 'concat':
            return torch.cat(feats, dim=-1)
        else:
            feats = torch.stack(feats, dim=0)

        if self.downstream_fusion_method == 'ensemble':
            return feats

        if self.downstream_fusion_method in ['elmo', 'elmo_tensor']:
            layer_weights = F.softmax(self.learned_layer_weights, dim=0)
            if self.downstream_fusion_method == 'elmo':
                feats = torch.sum(layer_weights.view(-1, 1, 1) * feats, dim=0)
            else:
                feats = torch.sum(layer_weights.view(-1, 1, self.hidden_size * 2) * feats, dim=0)
        elif self.downstream_fusion_method in ['gated', 'gated_meter']:
            if self.downstream_fusion_method == 'gated':
                layer_gate = F.softmax(self.gate(feats), dim=0)
                feats = torch.sum(layer_gate * feats, dim=0)
            else:
                layer_gate = F.softmax(self.gate(feats[:-1]), dim=0)
                feats = feats[-1] + torch.sum(layer_gate * feats[:-1], dim=0)
        elif self.downstream_fusion_method == 'max_pooling':
            feats = torch.max(feats, dim=0)[0]
        elif self.downstream_fusion_method == 'mean_pooling':
            feats = torch.mean(feats, dim=0)
        elif self.downstream_fusion_method == 'max_mean_pooling':
            max_feats = torch.max(feats, dim=0)[0]
            mean_feats = torch.mean(feats, dim=0)
            feats = torch.cat([max_feats, mean_feats], dim=-1)
        elif self.downstream_fusion_method in ['lstm_uni', 'lstm_bi']:
            feats = self.lstmpooler_dropout(self.lstmpooler(feats)[0][-1])
        
        return self.LayerNorm(feats)


class AttentionPool(nn.Module):
    """ attention pooling layer """
    def __init__(self, hidden_size, drop=0.0):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(hidden_size, 1), nn.GELU())
        self.dropout = nn.Dropout(drop)

    def forward(self, input_, mask=None):
        """input: [B, T, D], mask = [B, T]"""
        score = self.fc(input_).squeeze(-1)
        if mask is not None:
            mask = mask.to(dtype=input_.dtype) * -1e4
            score = score + mask
        norm_score = self.dropout(F.softmax(score, dim=1))
        output = norm_score.unsqueeze(1).matmul(input_).squeeze(1)
        return output

class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ITMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x


class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x



class BertLinkLayer(nn.Module):
    def __init__(self, config, link_tower_type):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.link_tower_type = link_tower_type
        # self.attention = BertAttention(config)
        # Like ViL-BERT and LXMERT, we don't use the self-attention and just use the cross-attention instead (optional with ffn). 
        self.crossattention = BertAttention(config)
        if self.link_tower_type == 'cross_attention_ffn':
            self.intermediate = BertIntermediate(config)
            self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        attention_mask=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        cross_attention_outputs = self.crossattention(
            hidden_states,
            attention_mask,
            None,
            encoder_hidden_states,
            encoder_attention_mask,
            None,
            output_attentions,
        )

        attention_output = cross_attention_outputs[0]
        outputs = cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

        if self.link_tower_type == 'cross_attention_ffn':
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
            )
            outputs = (layer_output,) + outputs

            return layer_output
        else:
            return attention_output

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertCrossLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        attention_mask=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = None #past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask=None,
            output_attentions=output_attentions,
            past_key_value=None,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        cross_attention_outputs = self.crossattention(
            attention_output,
            attention_mask,
            None,
            encoder_hidden_states,
            encoder_attention_mask,
            None,
            output_attentions,
        )
        attention_output = cross_attention_outputs[0]
        outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertPredictionHeadTransform(nn.Module):
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

class BertLinkLayer(nn.Module):
    def __init__(self, config, link_tower_type):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.link_tower_type = link_tower_type
        # self.attention = BertAttention(config)
        # Like ViL-BERT and LXMERT, we don't use the self-attention and just use the cross-attention instead (optional with ffn). 
        self.crossattention = BertAttention(config)
        if self.link_tower_type == 'cross_attention_ffn':
            self.intermediate = BertIntermediate(config)
            self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        attention_mask=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        cross_attention_outputs = self.crossattention(
            hidden_states,
            attention_mask,
            None,
            encoder_hidden_states,
            encoder_attention_mask,
            None,
            output_attentions,
        )

        attention_output = cross_attention_outputs[0]
        outputs = cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

        if self.link_tower_type == 'cross_attention_ffn':
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
            )
            outputs = (layer_output,) + outputs

            return layer_output
        else:
            return attention_output

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = BertSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutput(config)
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
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
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


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertSelfAttention(nn.Module):
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

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
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
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
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

        #if True:
        if False:
            self.save_attention_map(attention_probs)
            attention_probs.register_hook(self.save_attn_gradients) 

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


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
