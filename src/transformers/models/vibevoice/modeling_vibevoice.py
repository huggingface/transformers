# coding=utf-8
# Copyright 2025 The Microsoft Team and The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


from .configuration_vibevoice import VibeVoiceDiffusionHeadConfig
from ...activations import ACT2FN

from ...modeling_outputs import BaseModelOutputWithPast, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...models.auto import AutoModel
from ...utils import logging, is_diffusers_available, auto_docstring, can_return_tuple
from ...utils.import_utils import requires_backends
from ..llama.modeling_llama import LlamaRMSNorm
from .configuration_vibevoice import VibeVoiceConfig
from .generation_vibevoice import VibeVoiceGenerationMixin
from ...modeling_outputs import ModelOutput

logger = logging.get_logger(__name__)

if is_diffusers_available():
    import diffusers


@dataclass
class VibeVoiceCausalLMOutputWithPast(ModelOutput):
    """
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    diffusion_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` for diffusion are provided):
        Diffusion head loss (for acoustic token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
        The hidden states at the last layer of the model.
    attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[torch.FloatTensor] = None
    diffusion_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


class TimestepEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_1 = nn.Linear(config.frequency_embedding_size, config.hidden_size, bias=False)
        self.act = ACT2FN[config.hidden_act]
        self.layer_2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.frequency_embedding_size = config.frequency_embedding_size

    def forward(self, timesteps):
        # TODO (ebezzam) using diffuser method like below instead of their custom method: https://github.com/pengzhiliang/transformers/blob/6e6e60fb95ca908feb0b039483adcc009809f579/src/transformers/models/vibevoice/modular_vibevoice_diffusion_head.py#L66
        requires_backends(self, ["diffusers"])
        t_freq = diffusers.models.embeddings.get_timestep_embedding(
            timesteps=timesteps,
            embedding_dim=self.frequency_embedding_size,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
            scale=1.0,
            max_period=10000,
        ).to(timesteps.dtype)   
        return self.layer_2(self.act(self.layer_1(t_freq)))


# TODO (ebezzam) modular from LlamaMLP
class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        hidden_act="silu",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.gate_proj = nn.Linear(self.embed_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(self.embed_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, self.embed_dim, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        gate = self.act_fn(gate)
        return self.down_proj(gate * up)


# TODO (ebezzam) modular instead of using LlamaRMSNorm
# TODO (ebezzam) Qwen 2.5 Omni has mode similar, but hardcoded fnn ratio: https://github.com/huggingface/transformers/blob/82451cbb30fde5ede89308ea2328f89c61d5a831/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py#L2927
class HeadLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffn_ratio = config.head_ffn_ratio
        ffn_dim = config.hidden_size * config.head_ffn_ratio
        self.ffn = FeedForwardNetwork(
            config.hidden_size,
            ffn_dim,
            hidden_act=config.hidden_act
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.act_fn = ACT2FN[config.hidden_act]
        self.linear = nn.Linear(config.hidden_size, ffn_dim, bias=False)

    def forward(self, hidden_states, condition):
        shift_ffn, scale_ffn, gate_ffn = self.linear(self.act_fn(condition)).chunk(self.ffn_ratio, dim=-1)
        modulated_hidden_states = self.norm(hidden_states) * (1 + scale_ffn) + shift_ffn
        hidden_states = hidden_states + gate_ffn * self.ffn(modulated_hidden_states)
        return hidden_states


class FinalLayer(nn.Module):
    def __init__(self, config, output_size, ffn_ratio=2):
        super().__init__()
        # Inline RMS normalization since there is no weight scaling
        self.norm_eps = config.rms_norm_eps
        self.ffn_ratio = ffn_ratio
        self.linear_1 = nn.Linear(config.hidden_size, ffn_ratio * config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.linear_2 = nn.Linear(config.hidden_size, output_size, bias=False)

    def forward(self, hidden_states, condition):
        shift, scale = self.linear_1(self.act_fn(condition)).chunk(self.ffn_ratio, dim=-1)
        hidden_states = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.norm_eps)
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


@auto_docstring
class VibeVoicePreTrainedModel(PreTrainedModel):
    # TODO (ebezzam) config or config_class?
    config_class = VibeVoiceConfig
    base_model_prefix = "model"
    # TODO (ebezzam) check below, probably from Qwen?
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True


# TODO (ebezzam) register in auto doc?
@auto_docstring(
    custom_intro="""
    Diffusion head for VibeVoice model, for predicting acoustic tokens.
    """
)
class VibeVoiceDiffusionHead(VibeVoicePreTrainedModel):
    # TODO (ebezzam) config or config_class?
    config_class = VibeVoiceDiffusionHeadConfig
    main_input_name = ["noisy_images", "timesteps", "condition"]

    def __init__(self, config):
        super().__init__(config)

        self.noisy_images_proj = nn.Linear(config.latent_size, config.hidden_size, bias=False)
        self.cond_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.timestep_embedder = TimestepEmbedder(config)
        self.layers = nn.ModuleList([HeadLayer(config) for _ in range(config.num_head_layers)])
        self.final_layer = FinalLayer(config, output_size=config.latent_size)

        self.post_init()

    def forward(
        self,
        noisy_images,
        timesteps,
        condition,
    ):
        """
        Forward pass of the prediction head.
        
        Args:
            noisy_images (`torch.Tensor`): Noisy images/latents to denoise
            timesteps (`torch.Tensor`): Timesteps for diffusion
            condition (`torch.Tensor`): Conditioning information
            
        Returns:
            `torch.Tensor`: The predicted noise/velocity
        """
        hidden_states = self.noisy_images_proj(noisy_images)
        embedded_timesteps = self.timestep_embedder(timesteps)
        condition = self.cond_proj(condition)
        condition = condition + embedded_timesteps

        for layer in self.layers:
            hidden_states = layer(hidden_states, condition)

        hidden_states = self.final_layer(hidden_states, condition)
        return hidden_states


# TODO (ebezzam) modular instead of using LlamaRMSNorm
# and maybe even modular for SpeechConnector itself? (Voxtral?)
class VibeVoiceSpeechConnector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.norm = LlamaRMSNorm(output_dim, eps=1e-6)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, features):
        x = self.fc1(features)
        x = self.norm(x)
        x = self.fc2(x)
        return x


@auto_docstring
class VibeVoiceModel(VibeVoicePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # # TODO (ebezzam) original would set dtype internally: https://github.com/pengzhiliang/transformers/blob/6e6e60fb95ca908feb0b039483adcc009809f579/src/transformers/models/vibevoice/modeling_vibevoice.py#L107
        # # is this something we do?
        # dtype = config.getattr("dtype", torch.float32)

        self.language_model = AutoModel.from_config(config.text_config)

        # Initialize speech components if needed
        # TODO (ebezzam) freeze tokenizer as mentioned in paper (p3)? Move to processor?
        self.acoustic_tokenizer = AutoModel.from_config(config.acoustic_tokenizer_config).eval()
        self.semantic_tokenizer = AutoModel.from_config(config.semantic_tokenizer_config).eval()

        self.acoustic_connector = VibeVoiceSpeechConnector(config.acoustic_hidden_size, config.text_config.hidden_size)
        self.semantic_connector = VibeVoiceSpeechConnector(config.semantic_hidden_size, config.text_config.hidden_size)

        # Register scaling factors as buffers - use 1D tensors for FSDP compatibility
        self.register_buffer('speech_scaling_factor', torch.tensor(float('nan')))
        self.register_buffer('speech_bias_factor', torch.tensor(float('nan')))

        # Initialize prediction head for speech generation
        self.diffusion_head = AutoModel.from_config(config.diffusion_head_config)
        requires_backends(self, ["diffusers"])
        self.noise_scheduler = diffusers.DPMSolverMultistepScheduler(
            num_train_timesteps=config.diffusion_head_config.ddpm_num_steps,
            beta_schedule=config.diffusion_head_config.ddpm_beta_schedule,
            prediction_type=config.diffusion_head_config.prediction_type
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    # TODO (ebezzam) move to processor since tokenizers are pretrained
    def get_speech_features(self, speech_tensors, speech_masks):
        """Process speech inputs through tokenizers and connectors."""
        # TODO (ebezzam) can remove unsqueeze since if we keep batch dim in processor?
        with torch.no_grad():
            # TODO (ebezzam) shifted no_grad from model def to when actually calling: https://github.com/pengzhiliang/transformers/blob/6e6e60fb95ca908feb0b039483adcc009809f579/src/transformers/models/vibevoice/modular_vibevoice_tokenizer.py#L1062
            acoustic_latents = self.acoustic_tokenizer.encode(speech_tensors.unsqueeze(1), sample=True).latents

        # Apply scaling and bias
        acoustic_features = (acoustic_latents + self.speech_bias_factor.to(acoustic_latents.device)) * self.speech_scaling_factor.to(acoustic_latents.device)

        # Connect to language model space
        acoustic_connected = self.acoustic_connector(acoustic_features)[speech_masks.cpu()]

        return acoustic_connected

    @can_return_tuple
    # @auto_docstring   # TODO (ebezzam) make work
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # TODO (ebezzam) determine correct usage for below
        speech_tensors: Optional[torch.FloatTensor] = None,
        speech_masks: Optional[torch.BoolTensor] = None,
        speech_input_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> Union[tuple, BaseModelOutputWithPast]:

        # TODO (ebezzam) copied from Llava but failing here
        # if (input_ids is None) ^ (inputs_embeds is not None):
        #     raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # TODO (ebezzam) second condition necessary? see Llava
        if speech_tensors is not None and speech_masks is not None:
            speech_embeds = self.get_speech_features(speech_tensors.to(self.dtype), speech_masks)
            if speech_input_mask is not None:
                inputs_embeds[speech_input_mask] = speech_embeds
        
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class VibeVoiceForConditionalGeneration(VibeVoicePreTrainedModel, VibeVoiceGenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}

    def __init__(self, config):
        super().__init__(config)

        # Initialize the base model
        self.model = VibeVoiceModel(config)

        # LM head for text generation
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        # inference configuration
        self.ddpm_inference_steps = config.diffusion_head_config.ddpm_num_inference_steps

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def noise_scheduler(self):
        return self.model.noise_scheduler

    @property
    def diffusion_head(self):
        return self.model.diffusion_head

    @property
    def speech_scaling_factor(self):
        return self.model.speech_scaling_factor

    @property
    def speech_bias_factor(self):
        return self.model.speech_bias_factor

    @property
    def acoustic_tokenizer(self):
        return self.model.acoustic_tokenizer

    @property
    def semantic_tokenizer(self):
        return self.model.semantic_tokenizer

    @property
    def acoustic_connector(self):
        return self.model.acoustic_connector

    @property
    def semantic_connector(self):
        return self.model.semantic_connector

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        """
        # Tie lm_head.weight to language_model.embed_tokens.weight
        if not getattr(self.config, 'tie_word_embeddings', False):
            return

        if hasattr(self, 'lm_head') and hasattr(self.model.language_model, 'embed_tokens'):
            self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_ddpm_inference_steps(self, num_steps=None):
        self.ddpm_inference_steps = num_steps or self.config.diffusion_head_config.ddpm_num_inference_steps

    # TODO (ebezzam) clean up this method like `get_speech_features`
    def forward_speech_features(
            self,
            speech_tensors=None,
            speech_masks=None,
        ):

        import pudb; pudb.set_trace()

        if speech_tensors is None:
            # Use config to get vae_dim instead of non-existent self.args
            vae_dim = self.config.acoustic_tokenizer_config.vae_dim
            audio_features = torch.zeros(1, 1, vae_dim).to(self.get_input_embeddings().weight)
            connect_features = self.model.acoustic_connector(audio_features)
            return audio_features, connect_features
        else:
            with torch.no_grad():
                frames = self.model.acoustic_tokenizer.encode(speech_tensors.unsqueeze(1))[0][0]
                # TODO (ebezzam) replaced line with below
                # audio_tokens = frames.sample(self.model.acoustic_tokenizer.std_dist_type)[0]
                audio_tokens = self.model.acoustic_tokenizer.sample(frames)[0]

                if torch.isnan(self.model.speech_scaling_factor) or torch.isnan(self.model.speech_bias_factor):
                    scaling_factor = 1. / audio_tokens[speech_masks].flatten().std()
                    bias_factor = -audio_tokens[speech_masks].flatten().mean()

                    # Only use distributed operations if the process group is initialized
                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(scaling_factor, op=dist.ReduceOp.SUM)
                        dist.all_reduce(bias_factor, op=dist.ReduceOp.SUM)
                        world_size = dist.get_world_size()
                        self.model.speech_scaling_factor.copy_(scaling_factor / world_size)
                        self.model.speech_bias_factor.copy_(bias_factor / world_size)
                        print(f"Speech scaling factor (distributed): {self.model.speech_scaling_factor}, bias factor: {self.model.speech_bias_factor}", flush=True)
                    else:
                        # Single process case
                        self.model.speech_scaling_factor.copy_(scaling_factor)
                        self.model.speech_bias_factor.copy_(bias_factor)
                        print(f"Speech scaling factor (single process): {self.model.speech_scaling_factor}, bias factor: {self.model.speech_bias_factor}", flush=True)

                audio_features = (audio_tokens + self.model.speech_bias_factor) * self.model.speech_scaling_factor

            connect_features = self.model.acoustic_connector(audio_features)
            return audio_features[speech_masks], connect_features[speech_masks]

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,   # TODO (ebezzam) seems to always be True?
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        speech_tensors: Optional[torch.FloatTensor] = None,
        speech_masks: Optional[torch.BoolTensor] = None,
        speech_input_mask: Optional[torch.BoolTensor] = None,
        logits_to_keep: Union[int, slice] = 0,
        acoustic_loss_mask: Optional[torch.BoolTensor] = None,
        ddpm_batch_mul: int = 1,
        **kwargs,
    ) -> Union[tuple, VibeVoiceCausalLMOutputWithPast]:
        """
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            speech_tensors (`torch.FloatTensor`, *optional*):
                Input speech waveforms for voice cloning or speech understanding.
            speech_masks (`torch.BoolTensor`, *optional*):
                Masks indicating valid speech frames.
            speech_input_mask (`torch.BoolTensor`, *optional*):
                Positions in the input sequence where speech embeddings should be inserted.
        
        Returns:
            `VibeVoiceCausalLMOutputWithPast` or tuple
        """

        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            speech_tensors=speech_tensors,
            speech_masks=speech_masks,
            speech_input_mask=speech_input_mask,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # Language model loss
        loss = None
        if labels is not None:
            # TODO (ebezzam) add loss according to original implementation
            # Their comment:
            # "The custom CE loss with masking is calculated in the training script.
            # We leave the standard loss calculation here as None."
            raise NotImplementedError("Loss computation is not implemented in this version.")

        # Diffusion loss
        diffusion_loss = None
        # TODO (ebezzam) for now, copy from original implementation: https://github.com/vibevoice-community/VibeVoice/blob/db4cad072368df79f628cb4a6a3cd7bf3d60b685/vibevoice/modular/modeling_vibevoice.py#L415
        if acoustic_loss_mask is not None:
            acoustic_loss_mask_sum = acoustic_loss_mask.sum().item()
            if speech_tensors is not None and acoustic_loss_mask_sum > 0:
                condition_features = hidden_states[acoustic_loss_mask]

                x = self.get_input_embeddings()(input_ids)

                speech_features, speech_connect_features = self.forward_speech_features(
                        speech_tensors=speech_tensors.type_as(x) if speech_tensors is not None else None,
                        speech_masks=speech_masks,
                    )
                if speech_tensors is not None:
                    x[speech_input_mask] = speech_connect_features
                speech_len, latent_size = speech_features.shape

                noise = torch.randn(
                    (speech_len * ddpm_batch_mul, latent_size),
                    device=hidden_states.device,
                    dtype=hidden_states.dtype
                )

                timesteps = torch.multinomial(
                    torch.ones(self.config.diffusion_head_config.ddpm_num_steps),
                    speech_len * ddpm_batch_mul,
                    replacement=True,
                ).to(hidden_states.device)

                speech_features_repeated = speech_features.repeat_interleave(ddpm_batch_mul, dim=0)
                condition_features_repeated = condition_features.repeat_interleave(ddpm_batch_mul, dim=0)

                noisy_speech_features = self.model.noise_scheduler.add_noise(
                    speech_features_repeated, noise, timesteps
                )

                model_output = self.model.diffusion_head(
                    noisy_speech_features,
                    timesteps.type_as(x),
                    condition_features_repeated
                )

                prediction_type = self.config.diffusion_head_config.prediction_type
                if prediction_type == "epsilon":
                    target_for_loss = noise
                elif prediction_type == "v_prediction":
                    target_for_loss = self.model.noise_scheduler.get_velocity(
                        speech_features_repeated, noise, timesteps
                    )
                else:
                    raise NotImplementedError(f"Prediction type {prediction_type} not implemented")

                diffusion_loss = F.mse_loss(model_output.float(), target_for_loss.float(), reduction='sum')
                if latent_size > 0 and ddpm_batch_mul > 0:
                    diffusion_loss = diffusion_loss / latent_size / ddpm_batch_mul
                else:
                    diffusion_loss = torch.tensor(0.0, device=diffusion_loss.device)
            else:
                # Dummy loss for DDP to work when there are no speech samples in a batch,
                # but we are in a speech context.
                diffusion_loss = sum(p.sum() for p in self.model.diffusion_head.parameters()) * 0.0
                diffusion_loss += sum(p.sum() for p in self.model.acoustic_connector.parameters()) * 0.0
                diffusion_loss += sum(p.sum() for p in self.model.semantic_connector.parameters()) * 0.0

        # TODO (ebezzam) use loss calculation from original `VibeVoiceForConditionalGeneration`? 
        return VibeVoiceCausalLMOutputWithPast(
            loss=loss,
            diffusion_loss=diffusion_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            last_hidden_state=hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "VibeVoiceModel",
    "VibeVoicePreTrainedModel",
    "VibeVoiceForConditionalGeneration",
    "VibeVoiceDiffusionHead"
]
