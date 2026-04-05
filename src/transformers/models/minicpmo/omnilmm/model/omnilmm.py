
import gc
import math
import timm
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import List, Optional, Tuple, Union

from transformers import AutoConfig, AutoModelForCausalLM
from transformers import MistralForCausalLM, MistralModel, MistralConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from omnilmm.model.utils import build_transform
from omnilmm.model.resampler import Resampler

DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


class OmniLMMConfig(MistralConfig):
    model_type = "omnilmm"


class Identity(torch.nn.Identity):
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return super().forward(input)


def create_vision_module(config):
    vision_tower = timm.create_model('eva02_enormous_patch14_clip_224.laion2b_plus',
                                     pretrained=False,
                                     num_classes=0,
                                     dynamic_img_size=True,
                                     dynamic_img_pad=True)

    if isinstance(vision_tower, timm.models.VisionTransformer):
        if vision_tower.attn_pool is not None:
            vision_tower.attn_pool = Identity()

    # use 2nd last layer's output
    vision_tower.blocks[-1] = Identity()

    embed_dim = config.hidden_size
    resampler = Resampler(
        grid_size=int(math.sqrt(config.num_query)),
        embed_dim=embed_dim,
        num_heads=embed_dim // 128,
        kv_dim=vision_tower.embed_dim,
    )
    return vision_tower, resampler


class OmniLMMModel(MistralModel):
    config_class = OmniLMMConfig

    def __init__(self, config: OmniLMMConfig, mm_vision_tower=None, mm_hidden_size=None, tune_clip=True):
        super(OmniLMMModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            vision_tower, resampler = create_vision_module(config)

            # print(__file__, 'skip loading vision tower weights')

            # HACK: for FSDP
            self.vision_tower = [vision_tower]
            self.resampler = resampler
            if tune_clip:
                self.vision_tower = self.vision_tower[0]

        self.vision_config = lambda x: None

    def initialize_vision_modules(self, vision_tower, no_randaug, num_query, image_size, tune_clip=False):
        self.config.mm_vision_tower = vision_tower
        self.config.use_mm_proj = True
        self.config.num_query = num_query
        self.config.image_size = image_size

        if not hasattr(self, 'vision_tower'):
            vision_tower, resampler = create_vision_module(self.config)
            state_dict = torch.load(
                '/tt/data/public/multimodal/multimodal_model_ckpts/timm/eva02_enormous_patch14_clip_224.laion2b_plus.pt')
            vision_tower.load_state_dict(state_dict, strict=False)
            del state_dict
            gc.collect()
        else:
            if isinstance(self.vision_tower, list):
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            resampler = self.resampler
        self.vision_tower = vision_tower if tune_clip else [vision_tower]
        self.resampler = resampler

        train_img_transform = build_transform(
            is_train=True, randaug=not no_randaug, input_size=self.config.image_size, std_mode='OPENAI_CLIP')
        eval_img_transform = build_transform(
            is_train=False, input_size=self.config.image_size, std_mode='OPENAI_CLIP')

        return dict(
            image_processor=(train_img_transform, eval_img_transform),
            image_token_len=num_query,
            vision_config=self.vision_config
        )

    def get_vision_embedding(self, pixel_values):
        if isinstance(self.vision_tower, list):
            vision_tower = self.vision_tower[0]  # HACK: for FSDP
        else:
            vision_tower = self.vision_tower

        dtype = vision_tower.pos_embed.data.dtype
        vision_embedding = vision_tower.forward_features(
            pixel_values.type(dtype))
        if hasattr(vision_tower, 'num_prefix_tokens') and vision_tower.num_prefix_tokens > 0:
            vision_embedding = vision_embedding[:,
                                                vision_tower.num_prefix_tokens:]
        res = self.resampler(vision_embedding)
        return res

    def get_vllm_embedding(self, data):

        if 'vision_hidden_states' not in data:
            pixel_values_list = data['pixel_values']
            vision_hidden_states = []
            for pixel_values in pixel_values_list:
                if len(pixel_values) > 0:
                    vision_hidden_states.append(self.get_vision_embedding(pixel_values.unsqueeze(0))[0])
                else:
                    vision_hidden_states.append([])
        else:
            vision_hidden_states = data['vision_hidden_states']

        #vllm_embedding = self.llm.model.embed_tokens(data['input_ids']) * self.llm.config.scale_emb
        inputs_embeds = self.embed_tokens(data['input_ids'])
        vision_hidden_states = [i.type(inputs_embeds.dtype) 
            if isinstance(i, torch.Tensor) else i for i in vision_hidden_states
        ]


        # HACK: replace back original embeddings for LLaVA pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)

        new_input_embeds = []
        cur_image_idx = 0
        for cur_input_ids, cur_input_embeds in zip(data['input_ids'], inputs_embeds):
            if (cur_input_ids == self.vision_config.im_patch_token).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = cur_input_embeds + (0. * dummy_image_features).sum()
                new_input_embeds.append(cur_input_embeds)
                continue

            if self.vision_config.use_im_start_end:
                cur_image_features = vision_hidden_states[cur_image_idx]
                num_patches = cur_image_features.shape[0]
                if (cur_input_ids == self.vision_config.im_start_token).sum() != (cur_input_ids == self.vision_config.im_end_token).sum():
                    raise ValueError(
                        "The number of image start tokens and image end tokens should be the same.")
                image_start_tokens = torch.where(
                    cur_input_ids == self.vision_config.im_start_token)[0]
                for image_start_token_pos in image_start_tokens:
                    cur_image_features = vision_hidden_states[cur_image_idx].to(
                        device=cur_input_embeds.device)
                    num_patches = cur_image_features.shape[0]
                    if cur_input_ids[image_start_token_pos + num_patches + 1] != self.vision_config.im_end_token:
                        raise ValueError(
                            "The image end token should follow the image start token.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:image_start_token_pos].detach(), cur_input_embeds[image_start_token_pos:image_start_token_pos+1], cur_image_features,
                                                         cur_input_embeds[image_start_token_pos + num_patches + 1:image_start_token_pos + num_patches + 2], cur_input_embeds[image_start_token_pos + num_patches + 2:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat(
                            (cur_input_embeds[:image_start_token_pos+1], cur_image_features, cur_input_embeds[image_start_token_pos + num_patches + 1:]), dim=0)
                    cur_image_idx += 1
                new_input_embeds.append(cur_new_input_embeds)
            else:
                raise NotImplementedError
        inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return inputs_embeds, vision_hidden_states

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # HACK: replace back original embeddings for LLaVA pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)

        if inputs_embeds is None and past_key_values is None:
          inputs_embeds = self.embed_tokens(input_ids)

          vision_tower = getattr(self, 'vision_tower', None)
          if vision_tower is not None and (input_ids.shape[1] != 1 or self.training) and images is not None:

            if type(images) is list:
                image_features = []
                for image in images:
                    image_forward_out = self.get_vision_embedding(image.unsqueeze(0))[
                        0]
                    image_features.append(image_forward_out)
            else:
                image_features = self.get_vision_embedding(images)

            dummy_image_features = torch.zeros(
                self.config.num_query,
                self.config.hidden_size,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype)

            new_input_embeds = []
            cur_image_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == self.vision_config.im_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + \
                        (0. * dummy_image_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    continue

                if self.vision_config.use_im_start_end:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (cur_input_ids == self.vision_config.im_start_token).sum() != (cur_input_ids == self.vision_config.im_end_token).sum():
                        raise ValueError(
                            "The number of image start tokens and image end tokens should be the same.")
                    image_start_tokens = torch.where(
                        cur_input_ids == self.vision_config.im_start_token)[0]
                    for image_start_token_pos in image_start_tokens:
                        cur_image_features = image_features[cur_image_idx].to(
                            device=cur_input_embeds.device)
                        num_patches = cur_image_features.shape[0]
                        if cur_input_ids[image_start_token_pos + num_patches + 1] != self.vision_config.im_end_token:
                            raise ValueError(
                                "The image end token should follow the image start token.")
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:image_start_token_pos].detach(), cur_input_embeds[image_start_token_pos:image_start_token_pos+1], cur_image_features,
                                                             cur_input_embeds[image_start_token_pos + num_patches + 1:image_start_token_pos + num_patches + 2], cur_input_embeds[image_start_token_pos + num_patches + 2:].detach()), dim=0)
                        else:
                            cur_new_input_embeds = torch.cat(
                                (cur_input_embeds[:image_start_token_pos+1], cur_image_features, cur_input_embeds[image_start_token_pos + num_patches + 1:]), dim=0)
                        cur_image_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    raise NotImplementedError
            inputs_embeds = torch.stack(new_input_embeds, dim=0)
            input_ids = None

        return super(OmniLMMModel, self).forward(
            input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )


class OmniLMMForCausalLM(MistralForCausalLM):
    config_class = OmniLMMConfig

    def __init__(self, config, mm_vision_tower=None, tune_clip=True):
        super(MistralForCausalLM, self).__init__(config)
        self.model = OmniLMMModel(
            config, mm_vision_tower=mm_vision_tower, tune_clip=tune_clip)

        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # print(f'@@@ At forward, labels: {labels.shape}-{labels}', flush=True)
        # print(f'@@@ At forward, input_ids: {input_ids.shape}-{input_ids}', flush=True)
        # print(f'@@@ At forward, input_ids: {attention_mask.shape}-{attention_mask}', flush=True)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            images=images,
            **kwargs
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
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

    # TODO could be removed for generate_vllm()
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

    def generate_vllm(
        self,
        input_ids: torch.LongTensor = None,
        images: Optional[torch.FloatTensor] = None,
        vision_hidden_states=None,
        return_vision_hidden_states=False,
        **kwargs
    ):
        model_inputs = {'input_ids': input_ids}
        if vision_hidden_states is None:
            model_inputs['pixel_values'] = images
        else:
            model_inputs['vision_hidden_states'] = vision_hidden_states

        with torch.inference_mode():
            inputs_embeds, vision_hidden_states = self.model.get_vllm_embedding(model_inputs)

            result = self.generate(
                inputs_embeds=inputs_embeds,
                **kwargs
            )

        if return_vision_hidden_states:
            return result, vision_hidden_states

        return result


    def initialize_vision_tokenizer(self, mm_use_im_start_end, tokenizer, device,
                                    tune_mm_mlp_adapter=False):
        self.model.vision_config.use_im_start_end = mm_use_im_start_end
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        if mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            self.model.vision_config.im_start_token, self.model.vision_config.im_end_token = tokenizer.convert_tokens_to_ids(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            # for new sft data
            num_new_tokens = tokenizer.add_tokens(
                ['<box>', '</box>', '<ref>', '</ref>', '<quad>', '</quad>'], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_mm_mlp_adapter:
                self.model.orig_embeds_params = [
                    self.get_input_embeddings().weight.data.clone().to(device=device)]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

        self.model.vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IMAGE_PATCH_TOKEN])[0]
        print(f'Tokenizer: {tokenizer}\n patch_token_id: {self.model.vision_config.im_patch_token}, visoin_config: {self.model.vision_config}', flush=True)
        # exit()


AutoConfig.register("omnilmm", OmniLMMConfig)
AutoModelForCausalLM.register(OmniLMMConfig, OmniLMMForCausalLM)
