# --------------------------------------------------------
# Eagle2
# Copyright (c) 2025 NVIDIA
# Licensed under The Apache License [see LICENSE for details]
# --------------------------------------------------------

import warnings
from typing import Any, List, Optional, Tuple, Union

import torch.utils.checkpoint
import transformers
import deepcopy
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, SiglipVisionModel, Qwen2ForCausalLM, SiglipVisionConfig)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
from peft import LoraConfig, get_peft_model
from .configuration_eagle2 import Eagle2ChatConfig, MultiBackboneChannelConcatenationVisionModelConfig

logger = logging.get_logger(__name__)
torch.autograd.Function

def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))

class SiglipVisionTower(nn.Module):
    # We use the same wrapper as the default clip encoder. 
    # See `clip_encoder.py` in the same folder
    def __init__(self, vision_tower, args, delay_load=False, raw_config=None):
        super().__init__()

        self.is_loaded = False
        self.freeze_vision=args.freeze_vision
        self.input_image_size=args.input_image_size
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.name = 'siglip'
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.delay_load = delay_load
        self.raw_config = raw_config
        if not delay_load:
            self.load_model()
        else:
            if os.path.isfile(self.vision_tower_name):
                self.cfg_only = SiglipVisionConfig.from_pretrained(self.vision_tower_name, local_files_only=True)
            else:
                self.cfg_only = SiglipVisionConfig(**self.raw_config.vision_config.siglip_vision_config)


    def load_model(self):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        # self.image_processor = SiglipImageProcessor(size=1024)
        # self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name, local_files_only=True, torch_dtype=torch.bfloat16)
        if self.delay_load:
            # cfg = SiglipVisionConfig.from_pretrained(self.vision_tower_name, local_files_only=True)
            self.vision_tower = SiglipVisionModel(self.cfg_only)
        else:
            self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name, local_files_only=True)

        if self.freeze_vision:
            self.vision_tower.requires_grad_(False)

        self.vision_tower.vision_model.encoder.gradient_checkpointing = True
        self.is_loaded = True

    def forward(self, images):
        return self.vision_tower(
                pixel_values=images,
                output_hidden_states=False,
                return_dict=True).last_hidden_state


    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

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

                from .convnext import ConvNextVisionTower
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

class Eagle2ChatModel(PreTrainedModel):
    config_class = Eagle2ChatConfig
    main_input_name = 'pixel_values'
    _no_split_modules = ['LlamaDecoderLayer']

    def __init__(self, config: Eagle2ChatConfig, vision_model=None, language_model=None):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.2', 'ge')
        assert version_cmp(transformers.__version__, '4.39.2', 'le')
        image_size = config.force_image_size or config.vision_config.image_size
        if hasattr(config.vision_config, 'grid_size'):
            grid_size = config.vision_config.grid_size
            self.patch_size = 14
            self.num_image_token = int((grid_size * config.downsample_ratio) ** 2)
        else:
            patch_size = config.vision_config.patch_size
            self.patch_size = patch_size
            self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))

        self.select_layer = config.select_layer
        self.template = config.template

        self.downsample_ratio = config.downsample_ratio

        logger.info(f'num_image_token: {self.num_image_token}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            if config.vision_config.model_type == 'siglip_vision_model':
                self.vision_model = SiglipVisionModel(config.vision_config)
            elif config.vision_config.model_type.startswith("MOB"):
                self.vision_model = MultiBackboneChannelConcatenationVisionModel(config.vision_config, config)

        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        if vit_hidden_size == 'lazy_calculation':
            # a hack for Mixture of Backbones
            vit_hidden_size = self.vision_model.hidden_size
            print("The lazy calculated hidden_size: {} .. ".format(vit_hidden_size))
        llm_hidden_size = config.llm_config.hidden_size
        self.moe_version_type = getattr(config.vision_config, 'moe_version_type', None)
            
        if self.moe_version_type in ['seq_concat', 'feat_concat']:
            raise NotImplementedError
        elif self.moe_version_type == 'convnext_512_siglip_448':
            convnext_hidden_size = vit_hidden_size['convnext']
            siglip_hidden_size = vit_hidden_size['siglip']
            feature_concat_hidden_size = convnext_hidden_size + siglip_hidden_size * int(1 / self.downsample_ratio) ** 2
            self.mlp1 = nn.Sequential(
                nn.LayerNorm(feature_concat_hidden_size),
                nn.Linear(feature_concat_hidden_size, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size)
            )
        else:
            self.mlp1 = nn.Sequential(
                nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
                nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size)
            )
        self.img_context_token_id = None
        # self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message

        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)
    
    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):    
        lora_config = LoraConfig(
            r=r,
            target_modules=['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                            'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()


    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            num_patches_list: Optional[List[torch.Tensor]] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        
        if self.moe_version_type in ['seq_concat', 'feat_concat'] and not isinstance(pixel_values, dict):
            raise NotImplementedError
        vit_embeds = self.extract_feature(pixel_values)

        if not isinstance(image_flags, list):
            image_flags = image_flags.squeeze(-1)
            vit_embeds = vit_embeds[image_flags == 1]
        if isinstance(pixel_values, dict):
            # for MOE
            vit_batch_size = sum(pixel_values['num_patches'])
        else:
            vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

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

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        x = x.permute(0, 2, 1, 3).contiguous()
        return x
                
    def extract_feature(self, pixel_values):

        """
        """
        
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state # torch.Size([B, 1025, 1024])

        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        if type(self.vision_model) == SiglipVisionModel:
            pass
        elif type(self.vision_model) == MultiBackboneChannelConcatenationVisionModel:
            pass
        else:
            vit_embeds = vit_embeds[:, 1:, :] # torch.Size([B, 1024, 1024])

        if self.training and self.neftune_alpha is not None:
            vit_embeds = self.noised_embed(vit_embeds, self.neftune_alpha)

        if self.moe_version_type in ['feat_concat',  'seq_concat']:
            raise NotImplementedError
        elif self.moe_version_type == 'convnext_512_siglip_448':
            siglip_embeds = vit_embeds['siglip']
            convnext_embeds = vit_embeds['convnext']
            h = w = int(siglip_embeds.shape[1] ** 0.5)
            siglip_embeds = siglip_embeds.reshape(siglip_embeds.shape[0], h, w, -1)
            siglip_embeds = self.pixel_shuffle(siglip_embeds, scale_factor=self.downsample_ratio)
            siglip_embeds = siglip_embeds.reshape(siglip_embeds.shape[0], -1, siglip_embeds.shape[-1])
            vit_embeds = self.mlp1(torch.cat([siglip_embeds, convnext_embeds], dim=-1))
        else:
            h = w = int(vit_embeds.shape[1] ** 0.5)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)

            vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio) # torch.Size([B, 1024, 1024]) -> torch.Size([B, 16, 16, 4096])
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1]) # torch.Size([B, 16, 16, 4096]) -> torch.Size([B, 256, 4096])
            vit_embeds = self.mlp1(vit_embeds)#.to(pixel_values.device)

        return vit_embeds
    
    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep)[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False, llm_only=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            if llm_only:
                query = query.replace('<image>', '', 1)
            else:
                query = query.replace('<image>', image_tokens, 1)
        
        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id
        if self.moe_version_type is not None and self.moe_version_type != 'all_tiling' and self.moe_version_type != 'convnext_512_siglip_448':
            pixel_values = {
                'pixel_values': pixel_values,
                'num_patches': num_patches_list # num patch of each image.
            }
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)

            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs
        
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()
    
    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()