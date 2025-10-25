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
from ...utils import logging, is_diffusers_available, auto_docstring
from ...utils.import_utils import requires_backends
from ..llama.modeling_llama import LlamaRMSNorm
from .configuration_vibevoice import VibeVoiceConfig

logger = logging.get_logger(__name__)

if is_diffusers_available():
    import diffusers


@dataclass
class VibeVoiceCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    diffusion_loss: Optional[torch.FloatTensor] = None
    speech_token_num: Optional[int] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


@dataclass
class VibeVoiceGenerationOutput(ModelOutput):
    """
    Output type for VibeVoice generation.
    
    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. 
        speech_outputs (`List[torch.FloatTensor]`, *optional*):
            List of generated speech waveforms or latents for each speech segment.
    """
    sequences: torch.LongTensor = None
    speech_outputs: Optional[list[torch.FloatTensor]] = None


class TimestepEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_1 = nn.Linear(config.frequency_embedding_size, config.hidden_size, bias=False)
        self.act = ACT2FN[config.hidden_act]
        self.layer_2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.frequency_embedding_size = config.frequency_embedding_size

    def forward(self, timesteps):
        # TODO (ebezzam) use diffuser method like below instead of their custom method: https://github.com/pengzhiliang/transformers/blob/6e6e60fb95ca908feb0b039483adcc009809f579/src/transformers/models/vibevoice/modular_vibevoice_diffusion_head.py#L66
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
    # TODO (ebezzam) check below
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
        # TODO (ebezzam) freeze tokenizer as mentioned in paper (p3)?
        self.acoustic_tokenizer = AutoModel.from_config(config.acoustic_tokenizer_config).eval()
        self.semantic_tokenizer = AutoModel.from_config(config.semantic_tokenizer_config).eval()

        self.acoustic_connector = VibeVoiceSpeechConnector(config.acoustic_hidden_size, config.text_config.hidden_size)
        self.semantic_connector = VibeVoiceSpeechConnector(config.semantic_hidden_size, config.text_config.hidden_size)

        # Register scaling factors as buffers - use 1D tensors for FSDP compatibility
        self.register_buffer('speech_scaling_factor', torch.tensor(float('nan')))
        self.register_buffer('speech_bias_factor', torch.tensor(float('nan')))

        # Initialize prediction head for speech generation
        self.diffusion_head = AutoModel.from_config(config.diffusion_head_config)

        # Initialize noise scheduler
        requires_backends(self, ["diffusers"])
        self.noise_scheduler = diffusers.DPMSolverMultistepScheduler(
            num_train_timesteps=config.diffusion_head_config.ddpm_num_steps,
            beta_schedule=config.diffusion_head_config.ddpm_beta_schedule,
            prediction_type=config.diffusion_head_config.prediction_type
        )

        # Initialize weights and apply final processing
        self.post_init()


    def get_input_embeddings(self):
        if hasattr(self.language_model, 'embed_tokens'):
            # If the language model has an embed_tokens attribute, return it
            return self.language_model.embed_tokens

        for name, attr in self.language_model.fullmap.items(): # parallel by nnscaler, the name is changed
            if attr.orig_name == 'embed_tokens.weight':
                return getattr(self.language_model, name)
        assert False, 'should not arrive here'

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

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
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[tuple, BaseModelOutputWithPast]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward through language model
        outputs = self.language_model(
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
            **kwargs,
        )

        if not return_dict:
            return outputs

        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class VibeVoiceForConditionalGeneration(VibeVoicePreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}

    def __init__(self, config):
        super().__init__(config)
        self.model = VibeVoiceModel(config)
        self.vocab_size = config.decoder_config.vocab_size
        self.lm_head = nn.Linear(config.decoder_config.hidden_size, self.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_decoder(self, decoder):
        self.model.language_model = decoder

    def get_decoder(self):
        return self.model.language_model

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        """
        if getattr(self.config.decoder_config, 'tie_word_embeddings', False):
            # The standard PreTrainedModel method will handle the tying.
            # It typically does a simple parameter object assignment, which is
            # CORRECT to do BEFORE FSDP wraps the model.
            output_embeddings = self.get_output_embeddings()
            input_embeddings = self.get_input_embeddings()
            if hasattr(input_embeddings, 'weight'):
                output_embeddings.weight = input_embeddings.weight
            else:
                # maybe returned input_embeddings a tensor directly
                output_embeddings.weight = input_embeddings

            if getattr(output_embeddings, "bias", None) is not None:
                output_embeddings.bias.data = nn.functional.pad(
                    output_embeddings.bias.data,
                    (0, output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0]),
                    "constant",
                    0,
                )
            print("✅ Tied input and output embeddings using standard assignment.")
        else:
            print("ℹ️  tie_word_embeddings is False, not tying weights.")

    # Also, ensure set_output_embeddings is safe, though your implementation looks okay.
    # The key is to avoid calling it after accelerator.prepare().
    def set_output_embeddings(self, new_embeddings):
        # Your current implementation using data.copy_ is good practice,
        # but the best way is to not call this after prepare().
        self.lm_head = new_embeddings

    def forward_speech_features(
            self,
            speech_tensors=None,
            speech_masks=None,
            speech_type="audio",
            return_unmask=False
        ):
        if speech_tensors is None:
            # Use config to get vae_dim instead of non-existent self.args
            vae_dim = self.config.acoustic_tokenizer_config.vae_dim
            audio_features = torch.zeros(1, 1, vae_dim).to(self.get_input_embeddings().weight)
            connect_features = self.model.acoustic_connector(audio_features)
            return audio_features, connect_features
        else:
            with torch.no_grad():
                if speech_type == "audio":
                    with torch.no_grad():
                        frames = self.model.acoustic_tokenizer.encode(speech_tensors.unsqueeze(1))[0][0]
                    # TODO (ebezzam) replaced line with below
                    # audio_tokens = frames.sample(self.model.acoustic_tokenizer.std_dist_type)[0]
                    audio_tokens = self.model.acoustic_tokenizer.sample(frames)[0]

                elif speech_type == "vae":
                    # Use config to get vae_dim instead of non-existent self.args
                    vae_dim = self.config.acoustic_tokenizer_config.vae_dim
                    speech_mode = speech_tensors.reshape(speech_tensors.size(0), -1, vae_dim)

                    # gaussian sample from the speech_mode
                    batch_size = speech_mode.size(0)
                    value = self.model.acoustic_tokenizer.fix_std / 0.8
                    std = torch.randn(batch_size, dtype=speech_mode.dtype, device=speech_mode.device) * value
                    std = std.view(-1, *[1] * (speech_mode.dim() - 1))
                    audio_tokens = speech_mode + std * torch.randn(speech_mode.shape).to(speech_mode)
                else:
                    raise NotImplementedError(f"Speech type {speech_type} not implemented")

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
            if return_unmask:
                return audio_features, connect_features
            return audio_features[speech_masks], connect_features[speech_masks]

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # New arguments for speech processing and loss calculation
        speech_tensors: Optional[torch.FloatTensor] = None,
        speech_masks: Optional[torch.BoolTensor] = None,
        speeches_loss_input: Optional[torch.FloatTensor] = None,
        speech_semantic_tensors: Optional[torch.FloatTensor] = None,
        acoustic_input_mask: Optional[torch.BoolTensor] = None,
        acoustic_loss_mask: Optional[torch.BoolTensor] = None,
        ddpm_batch_mul: int = 1,
        **kwargs: Optional[dict[str, Union[torch.Tensor, str]]],
        ) -> Union[tuple, VibeVoiceCausalLMOutputWithPast]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        x = self.get_input_embeddings()(input_ids)

        semantic_speech_all_connect_features = self.model.semantic_connector(speech_semantic_tensors)
        if speeches_loss_input is not None:
            # only part audio need diffuse
            speech_all_features, speech_all_connect_features = self.forward_speech_features(
                    speech_tensors=speech_tensors.type_as(x) if speech_tensors is not None else None,
                    speech_masks=speech_masks,
                    speech_type=kwargs.get("speech_type", "audio"),
                    return_unmask=True
                )
            if speech_tensors is not None:
                if semantic_speech_all_connect_features is not None:
                    x[acoustic_input_mask] = speech_all_connect_features[speech_masks] + semantic_speech_all_connect_features[speech_masks]
                else:
                    x[acoustic_input_mask] = speech_all_connect_features[speech_masks]
                speech_features = speech_all_features[speeches_loss_input.unsqueeze(-1) & speech_masks] # only part audio need diffuse
                speech_connect_features = speech_all_connect_features[speeches_loss_input.unsqueeze(-1) & speech_masks]
        else:
            speech_features, speech_connect_features = self.forward_speech_features(
                    speech_tensors=speech_tensors.type_as(x) if speech_tensors is not None else None,
                    speech_masks=speech_masks,
                    speech_type=kwargs.get("speech_type", "audio"),
                )
            if speech_tensors is not None:
                x[acoustic_input_mask] = speech_connect_features

        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=x,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        # logits = logits.float()

        loss = None
        if labels is not None:
            # The custom CE loss with masking is calculated in the training script.
            # We leave the standard loss calculation here as None.
            pass

        # --- Diffusion Loss Calculation ---
        diffusion_loss = None
        # This block is executed only if we are in a context that involves speech.
        if speech_tensors is not None and acoustic_loss_mask.sum().item() > 0:
            condition_features = hidden_states[acoustic_loss_mask]

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
        # --- End Diffusion Loss Calculation ---

        if not return_dict:
            output = (logits, speech_len) + outputs.to_tuple()[1:]
            return (loss, diffusion_loss) + output

        return VibeVoiceCausalLMOutputWithPast(
            loss=loss,
            diffusion_loss=diffusion_loss,
            speech_token_num=speech_len if speech_tensors is not None else 0,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "VibeVoiceModel",
    "VibeVoicePreTrainedModel",
    "VibeVoiceForConditionalGeneration",
    "VibeVoiceDiffusionHead"
]
