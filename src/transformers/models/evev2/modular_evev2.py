from typing import List, Optional, Tuple, Union, Unpack

import torch
from torch import nn

from ...cache_utils import Cache
from ...modeling_outputs import CausalLMOutputWithPast
from ..clip.image_processing_clip import CLIPImageProcessor
from ..qwen2.configuration_qwen2 import Qwen2Config
from ..qwen2.modeling_qwen2 import KwargsForCausalLM, Qwen2ForCausalLM, Qwen2Model


class EVEV2Config(Qwen2Config):
    model_type = "evev2"

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        mm_hidden_size=1024,
        patch_stride=16,
        dense_stride=2,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            use_sliding_window=use_sliding_window,
            sliding_window=sliding_window,
            max_window_layers=max_window_layers,
            attention_dropout=attention_dropout,
            **kwargs,
        )

        self.mm_hidden_size = mm_hidden_size
        self.patch_stride = patch_stride
        self.dense_stride = dense_stride


class EVEV2ImageProcessor(CLIPImageProcessor):
    pass


class EVEV2VisionTokenizer(nn.Module):
    def __init__(self, config: EVEV2Config):
        super().__init__()

        input_size = config.mm_hidden_size
        output_size = config.hidden_size
        patch_stride = config.patch_stride
        dense_stride = config.dense_stride

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(3, input_size, kernel_size=patch_stride, stride=patch_stride),
            nn.GELU(),
            nn.Conv2d(input_size, output_size, kernel_size=dense_stride, stride=dense_stride),
        )
        self.class_embedding = nn.Parameter(torch.randn(output_size))
        self.split_embedding = nn.Parameter(torch.randn(output_size))

    def forward(self, pixel_values):
        patch_embeds = []
        for i in range(len(pixel_values)):
            pixel_value = pixel_values[i].to(dtype=self.dtype)
            patch_embed = self.patch_embedding(pixel_value.unsqueeze(0))[0]
            split_embed = self.split_embedding[:, None, None].repeat(1, patch_embed.shape[1], 1)
            patch_embed = torch.cat([patch_embed, split_embed.to(dtype=self.dtype)], dim=-1)

            class_embed = self.class_embedding[None, :].to(dtype=self.dtype)
            patch_embeds.append(torch.cat([class_embed, patch_embed.flatten(1).transpose(0, 1)], dim=0))

        return patch_embeds

    @property
    def dtype(self):
        return self.patch_embedding[0].weight.dtype

    @property
    def device(self):
        return self.patch_embedding[0].weight.device


class EVEV2Model(Qwen2Model):
    config_class = EVEV2Config

    def __init__(self, config: EVEV2Config):
        super().__init__(config)

        self.vision_tower = EVEV2VisionTokenizer(config)


class EVEV2ForCausalLM(Qwen2ForCausalLM):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        pixel_values: Optional[torch.FloatTensor] = None,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                visual_token_mask,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, pixel_values
            )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            visual_token_mask=visual_token_mask,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

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

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, attention_mask=None,
                                      **kwargs):
        pixel_values = kwargs.pop("pixel_values", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            **kwargs
        )
        if pixel_values is not None:
            _inputs['pixel_values'] = pixel_values
        return _inputs

    def get_transfer_tensor(self, tmp_tensor):
        return tmp_tensor.to(dtype=self.get_model().dtype, device=self.get_model().device)

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images
    ):
        IGNORE_INDEX = -100
        IMAGE_TOKEN_INDEX = -200
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[
                1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            visual_token_mask = self.get_transfer_tensor(torch.zeros_like(input_ids).unsqueeze(-1))
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, visual_token_mask

        image_features = self.get_vision_tower()(images)

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids_temp = input_ids  # points to the actual input_ids tensor

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in
                     zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        # -- TODO: better implementation?
        # replace IMAGE_TOKEN_INDEX(-200) with 0 to be compatible with repetition penalty
        input_ids_temp[input_ids_temp == IMAGE_TOKEN_INDEX] = 0

        new_labels = []
        new_input_embeds = []
        visual_token_mask = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                visual_token_mask.append(self.get_transfer_tensor(torch.zeros_like(cur_input_ids).unsqueeze(-1)))
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [
                cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_visual_mask = []
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                cur_visual_mask.append(
                    self.get_transfer_tensor(torch.zeros_like(cur_labels_noim[i]).unsqueeze(-1)))
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device,
                                   dtype=cur_labels.dtype))
                    cur_visual_mask.append(
                        self.get_transfer_tensor(torch.ones(cur_image_features.shape[0], 1)))

            new_input_embeds.append(torch.cat(cur_new_input_embeds))
            new_labels.append(torch.cat(cur_new_labels))
            visual_token_mask.append(torch.cat(cur_visual_mask))

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            visual_token_mask = [x[:tokenizer_model_max_length] for x in visual_token_mask]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype,
                                       device=new_labels[0].device)
        visual_token_mask_padded = self.get_transfer_tensor(torch.zeros_like(new_labels_padded).unsqueeze(-1))
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_label, cur_new_mask) in enumerate(
                zip(new_input_embeds, new_labels, visual_token_mask)):
            cur_len = cur_new_embed.shape[0]
            new_input_embeds_padded.append(torch.cat((
                cur_new_embed,
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device)
            ), dim=0))
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_label
                visual_token_mask_padded[i, :cur_len] = cur_new_mask
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                         device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, visual_token_mask_padded
