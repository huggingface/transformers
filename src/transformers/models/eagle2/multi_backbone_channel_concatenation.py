# --------------------------------------------------------
# Eagle2
# Copyright (c) 2025 NVIDIA
# Licensed under The Apache License [see LICENSE for details]
# --------------------------------------------------------

import torch, os
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .siglip_vision_tower import SiglipVisionTower

import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from copy import deepcopy
import random
import math

class MultiBackboneChannelConcatenationVisionTower(nn.Module):
    def __init__(self,
                 vision_tower,
                 args,
                 grid_size=32,
                 convnext_img_size=1024,
                 normalize_type=None, raw_config=None):
        
        super().__init__()

        self.is_loaded = False
        self.grid_size = grid_size
        self.num_tokens = self.grid_size ** 2
        self.normalize_type = args.normalize_type
        self.moe_version_type = args.moe_version_type
        self.raw_config = raw_config
        print("moe_version_type: ", self.moe_version_type)
        assert self.moe_version_type in [None, 'all_tiling', 'seq_concat', 'feat_concat', 'convnext_512_siglip_448'], f"Unknown self.moe_version_type: {self.moe_version_type}"
        
        vision_tower_name_list = vision_tower.split(";")
        self.input_image_size = 1024
        self.convnext_img_size = convnext_img_size
        self.load_vision_towers(vision_tower_name_list, args)

      
    def load_vision_towers(self, vision_tower_name_list, args):
        self.vision_towers = nn.ModuleList()

        freeze_backbone_list = args.freeze_backbones # note this is a str
        if freeze_backbone_list is not None and len(freeze_backbone_list) > 0:
            print("The frozen backbones: ", freeze_backbone_list)
        else:
            # make it a blank str
            freeze_backbone_list = ""

        for name in vision_tower_name_list:
            
            ## ConvNeXt
            if name == 'convnext-1024':
                convnext_args = deepcopy(args)

                convnext_args.freeze_vision = False
                if 'convnext-1024' in freeze_backbone_list:
                    convnext_args.freeze_vision = True

                from .convnext_encoder import ConvNextVisionTower
                convnext_args.input_image_size = self.convnext_img_size
                convnext_vision_tower = args.vision_tower_convnext_path
                convnext_vision_tower = ConvNextVisionTower(convnext_vision_tower, 
                                                                convnext_args, delay_load=args.delay_load, normalize_type=self.normalize_type)
                convnext_vision_tower.load_model()      
                self.vision_towers.append(convnext_vision_tower)

            ## PaliSigLIP
            elif name == 'palisiglip':
                palisiglip_args = deepcopy(args)
                palisiglip_args.input_image_size = 448

                palisiglip_args.freeze_vision = False
                if 'palisiglip' in freeze_backbone_list:
                    palisiglip_args.freeze_vision = True

                palisiglip_vision_tower = SiglipVisionTower(args.vision_tower_siglip_path, palisiglip_args, delay_load=args.delay_load, raw_config=self.raw_config)     
   
                palisiglip_vision_tower.load_model()
                self.vision_towers.append(palisiglip_vision_tower)

        # Set the image processor
        self.image_processor = None
        self.is_loaded = True

    def load_model(self):
        assert self.is_loaded, "All the vision encoders should be loaded during initialization!"

    def forward(self, x):
        # x is a Tensor if moe_version_type is None or 'all_tiling'
        # else is a tuple(Tensor, Tensor)
        if self.moe_version_type in [None, 'all_tiling']:
            # The default pipeline
            features = []
            image_input_size = x.shape[2]
            assert x.shape[2] == x.shape[3], f"Image should be a square but size ({x.shape[2]} x {x.shape[3]})"
            for vision_tower in self.vision_towers:
        
                if vision_tower.input_image_size != image_input_size:
                    resized_x = F.interpolate(x.float(), 
                                            size=(vision_tower.input_image_size, vision_tower.input_image_size), 
                                            mode='bilinear', 
                                            align_corners=True).to(dtype=x.dtype)
                else:
                    resized_x = x
                
                feature = vision_tower(resized_x)
                
                if len(feature.shape) == 3: # b, n, c
                    b, n, c = feature.shape
                    if n == self.num_tokens:
                        features.append(feature)
                        continue
                    w = h = int(n**0.5)
                    feature = feature.transpose(1,2).reshape(b, c, h, w)
                else:
                    b, c, h, w = feature.shape

                if w != self.grid_size:
                    feature = F.interpolate(feature.float(), size=(self.grid_size, self.grid_size), mode='bilinear', align_corners=True).to(dtype=x.dtype)
                features.append(feature.flatten(2,3).transpose(1,2))
            
            features = torch.cat(features, dim=-1)
        elif self.moe_version_type == 'convnext_512_siglip_448':
            features = {}
            image_input_size = x.shape[2]
            assert x.shape[2] == x.shape[3], f"Image should be a square but size ({x.shape[2]} x {x.shape[3]})"
            for vision_tower in self.vision_towers:
        
                if vision_tower.input_image_size != image_input_size:
                    resized_x = F.interpolate(x.float(), 
                                            size=(vision_tower.input_image_size, vision_tower.input_image_size), 
                                            mode='bilinear', 
                                            align_corners=True).to(dtype=x.dtype)
                else:
                    resized_x = x
                
                feature = vision_tower(resized_x)
                
                # if len(feature.shape) == 3: # b, n, c
                #     b, n, c = feature.shape
                #     if n == self.num_tokens:
                #         features.append(feature)
                #         continue
                #     w = h = int(n**0.5)
                #     feature = feature.transpose(1,2).reshape(b, c, h, w)
                # else:
                #     b, c, h, w = feature.shape
                features[vision_tower.name] = feature

        else:
            assert isinstance(x, dict), "x is expected to be a dict but {}".format(type(x))
            pixel_values = x['pixel_values']
            num_patches = x['num_patches'] # num patch of paddings token in texts

            # calculated the real image patches
            if self.moe_version_type == 'seq_concat':
                image_in_num_patches = [i-1 for i in num_patches]
            else:
                image_in_num_patches = [i for i in num_patches]


            assert sum(image_in_num_patches) == pixel_values.size(0), "sum(image_in_num_patches) ({}) != pixel_values.size(0) ({})".format(sum(image_in_num_patches), pixel_values.size(0))

            # find the thubnail image id
            thumbnail_image_id = torch.cumsum(torch.tensor(image_in_num_patches).to(pixel_values.device), 0) - 1
            image_no_tiling = pixel_values[thumbnail_image_id]

            # By default, we use the 1st vision_tower for x, others for x_nt
            features = []
            for layer_id, vision_tower in enumerate(self.vision_towers):
                if layer_id == 0:
                    x = pixel_values
                else:
                    x = image_no_tiling

                if vision_tower.input_image_size != self.input_image_size:
                    resized_x = F.interpolate(x.float(), 
                                            size=(vision_tower.input_image_size, vision_tower.input_image_size), 
                                            mode='bilinear', 
                                            align_corners=True).to(dtype=x.dtype)
                else:
                    resized_x = x
                
                feature = vision_tower(resized_x)
                if len(feature.shape) == 3: # b, n, c
                    b, n, c = feature.shape
                    if n == self.num_tokens:
                        features.append(feature)
                        continue

                    w = h = int(n**0.5)
                    feature = feature.transpose(1,2).reshape(b, c, h, w)
                else:
                    b, c, h, w = feature.shape

                if w != self.grid_size:
                    feature = F.interpolate(feature.float(), size=(self.grid_size, self.grid_size), mode='bilinear', align_corners=True).to(dtype=x.dtype)
                features.append(feature.flatten(2,3).transpose(1,2))

            clip_embeds = features[0]
            if len(features) <= 1:
                no_tiling_embeds = None
            else:
                no_tiling_embeds = torch.cat(features[1:], dim=-1)

            if self.moe_version_type == 'feat_concat':
                # concat thumbnail images features together
                clip_thumbnail_embeds = clip_embeds[thumbnail_image_id]
                if no_tiling_embeds is not None:
                    no_tiling_embeds = torch.cat([clip_thumbnail_embeds, no_tiling_embeds], dim=-1)
                else:
                    no_tiling_embeds = clip_thumbnail_embeds

                # extra patch featureas
                clip_embeds_mask = ~torch.isin(torch.arange(clip_embeds.shape[0]).to(clip_embeds.device), thumbnail_image_id)
                clip_embeds = clip_embeds[clip_embeds_mask]
            

            features = {
                    'clip_embeds': clip_embeds, 
                    'no_tiling_embeds': no_tiling_embeds,
                    'num_patches': num_patches
                }

        # features is a Tensor if not clip_tiling_only

        return features
        
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.clip_vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.clip_vision_tower.parameters()).device

    @property
    def config(self):
        assert NotImplementedError
        pass

    @property
    def hidden_size(self):
        if self.moe_version_type == 'convnext_512_siglip_448':
            res = {}
            for vision_tower in self.vision_towers:
                res[vision_tower.name] = vision_tower.hidden_size
            return res
        else:
            return sum([_.hidden_size for _ in self.vision_towers])

    @property
    def num_patches(self):
        return self.num_tokens


class MultiBackboneChannelConcatenationVisionModel(nn.Module):

    """
    A vision model wrapper that concatenates channels from multiple backbones.
    Args:
        config (MultiBackboneChannelConcatenationVisionModelConfig): The configuration for the model.
    Attributes:
        vision_model (MultiBackboneChannelConcatenationVisionTower): The vision tower that performs the channel concatenation.
    Notes:
        **The class is not inherited from the PreTrainedModel in transformers**
    """

    config_class = MultiBackboneChannelConcatenationVisionModelConfig
    main_input_name = "pixel_values"

    def __init__(self, config: MultiBackboneChannelConcatenationVisionModelConfig, raw_config):
        super().__init__()

        self.vision_model = MultiBackboneChannelConcatenationVisionTower(
            vision_tower=config.vision_tower,
            args=config,
            grid_size=config.grid_size,
            convnext_img_size=config.convnext_img_size,
            normalize_type=config.normalize_type,
            raw_config=raw_config
        )


    def get_input_embeddings(self):
        # You might need to adjust this depending on how you want to handle input embeddings
        return self.vision_model.vision_towers[0].get_input_embeddings()

    def forward(
        self,
        pixel_values,
        return_dict: Optional[bool] = True,
        output_hidden_states: Optional[bool] = False,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        assert return_dict is True, "We only support return_dict"
        assert output_hidden_states is False, "We do not support output_hidden_states"

        features = self.vision_model(pixel_values)

        # We only supports features as model outputs
        return BaseModelOutputWithPooling(
            last_hidden_state=features,
            pooler_output=None,
            hidden_states=None,
            attentions=None,
        )

    @property
    def dummy_feature(self):
        return self.vision_model.dummy_feature

    @property
    def dtype(self):
        return self.vision_model.dtype

    @property
    def device(self):
        return self.vision_model.device

    @property
    def config(self):
        return self.vision_model.config

    @property
    def hidden_size(self):
        return self.vision_model.hidden_size

    @property
    def num_patches(self):
        return self.vision_model.num_patches