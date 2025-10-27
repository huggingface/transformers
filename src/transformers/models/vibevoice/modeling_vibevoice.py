from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


from .configuration_vibevoice import VibeVoiceDiffusionHeadConfig
from ...activations import ACT2FN

from .audio_streamer import AsyncAudioStreamer, AudioStreamer
from ...generation import GenerationConfig, GenerationMixin, LogitsProcessor, LogitsProcessorList, StoppingCriteriaList
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
class VibeVoiceCausalLMOutputWithPast(BaseModelOutputWithPast):
    logits: Optional[torch.FloatTensor] = None
# @dataclass
# class VibeVoiceCausalLMOutputWithPast(ModelOutput):
#     loss: Optional[torch.FloatTensor] = None
#     diffusion_loss: Optional[torch.FloatTensor] = None
#     speech_token_num: Optional[int] = None
#     logits: torch.FloatTensor = None
#     past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None
#     hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
#     attentions: Optional[tuple[torch.FloatTensor, ...]] = None


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
    reach_max_step_sample: Optional[torch.BoolTensor] = None


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
    

class VibeVoiceTokenConstraintProcessor(LogitsProcessor):
    """Constrains token generation to only valid tokens during speech generation."""

    def __init__(self, valid_token_ids: list[int], device: torch.device = None):
        self.valid_token_ids = torch.tensor(valid_token_ids, dtype=torch.long, device=device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Create a mask for valid tokens
        mask = torch.full_like(scores, float('-inf'))
        mask[:, self.valid_token_ids] = 0

        # Apply mask to scores
        scores = scores + mask
        return scores


class VibeVoiceForConditionalGeneration(VibeVoicePreTrainedModel, GenerationMixin):
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

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_ddpm_inference_steps(self, num_steps=None):
        self.ddpm_inference_steps = num_steps or self.config.diffusion_head_config.ddpm_num_inference_steps

    def _process_speech_inputs(self, speech_tensors, speech_masks):
        """Process speech inputs through tokenizers and connectors."""
        # TODO (ebezzam) can remove unsqueeze since if we keep batch dim in processor?
        with torch.no_grad():
            # TODO (ebezzam) shifted no_grad from model def to when actually calling: https://github.com/pengzhiliang/transformers/blob/6e6e60fb95ca908feb0b039483adcc009809f579/src/transformers/models/vibevoice/modular_vibevoice_tokenizer.py#L1062
            acoustic_latents = self.model.acoustic_tokenizer.encode(speech_tensors.unsqueeze(1), sample=True).latents

        # Apply scaling and bias
        acoustic_features = (acoustic_latents + self.model.speech_bias_factor.to(acoustic_latents.device)) * self.model.speech_scaling_factor.to(acoustic_latents.device)

        # Connect to language model space
        acoustic_connected = self.model.acoustic_connector(acoustic_features)[speech_masks.cpu()]

        return acoustic_features, acoustic_connected

    # @can_return_tuple
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
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        speech_tensors: Optional[torch.FloatTensor] = None,
        speech_masks: Optional[torch.BoolTensor] = None,
        speech_input_mask: Optional[torch.BoolTensor] = None,
        logits_to_keep: Union[int, slice] = 0,
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

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        # Process speech inputs if provided
        if speech_tensors is not None and speech_masks is not None:
            acoustic_features, speech_embeds = self._process_speech_inputs(speech_tensors.to(self.dtype), speech_masks)
            if speech_input_mask is not None:
                inputs_embeds[speech_input_mask] = speech_embeds

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        if labels is not None:
            raise NotImplementedError("Loss computation is not implemented in this version.")

        # TODO (ebezzam) use loss calculation from original `VibeVoiceForConditionalGeneration`? 
        return VibeVoiceCausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            last_hidden_state=hidden_states,
            attentions=outputs.attentions,
        )

    def _build_generate_config_model_kwargs(self, generation_config, inputs, tokenizer, return_processors=False, **kwargs):
        if generation_config is None:
            generation_config = GenerationConfig(
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id = tokenizer.pad_token_id
            )
        else:
            generation_config = GenerationConfig(
                **generation_config,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id = tokenizer.pad_token_id
            )

        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config,
            True,
            speech_start_id=tokenizer.speech_start_id,
            speech_end_id=tokenizer.speech_end_id,
            speech_diffusion_id=tokenizer.speech_diffusion_id,
            **kwargs
        )
        generation_config.speech_start_id = tokenizer.speech_start_id
        generation_config.speech_end_id = tokenizer.speech_end_id
        generation_config.speech_diffusion_id = tokenizer.speech_diffusion_id

        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(inputs, generation_config.bos_token_id, model_kwargs)
        batch_size = inputs_tensor.shape[0]
        device = self.device

        self._prepare_special_tokens(generation_config, True, device=device)
        generation_config.use_cache = True
        model_kwargs["use_cache"] = True
        input_ids = inputs_tensor.to(self.device)

        input_ids_length = input_ids.shape[1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        max_cache_length = generation_config.max_length - 1
        self._prepare_cache_for_generation(generation_config, model_kwargs, None, batch_size, max_cache_length)
        model_kwargs['cache_position'] = torch.arange(input_ids_length, device=device, dtype=torch.long)
        for k, v in model_kwargs.items():
            if isinstance(v, torch.Tensor):
                model_kwargs[k] = v.to(device=device)

        if return_processors:
            logits_processor = self._get_logits_processor(
                generation_config=generation_config,
                input_ids_seq_length=input_ids_length,
                encoder_input_ids=inputs_tensor,
                prefix_allowed_tokens_fn=None,
                logits_processor=LogitsProcessorList(),
                device=inputs_tensor.device,
                model_kwargs=model_kwargs,
            )

            stopping_criteria = self._get_stopping_criteria(generation_config=generation_config, stopping_criteria=StoppingCriteriaList())

            return generation_config, model_kwargs, input_ids, logits_processor, stopping_criteria
        else:
            return generation_config, model_kwargs, input_ids

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        audio_streamer: Optional[Union[AudioStreamer, AsyncAudioStreamer]] = None,
        speech_tensors: Optional[torch.FloatTensor] = None,
        speech_masks: Optional[torch.BoolTensor] = None,        # TODO rename, this is to ignore padded parts
        speech_input_mask: Optional[torch.BoolTensor] = None,   # TODO rename, this is to know where is speech in script
        return_speech: bool = True,
        cfg_scale: float = 1.0,
        stop_check_fn: Optional[Callable[[], bool]] = None,
        **kwargs,
    ) -> Union[torch.LongTensor, VibeVoiceGenerationOutput]:
        """
        Generates sequences of token ids and optionally speech outputs.
        
        Args:
            All standard generation arguments from GenerationMixin
            negative_prompt_ids: Negative prompt for CFG in speech generation
            negative_prompt_attention_mask: Attention mask for negative prompt
            speech_tensors: Input speech for voice cloning
            speech_masks: Masks for speech tensors  
            speech_input_mask: Positions to insert speech embeddings
            return_speech: Whether to decode and return speech outputs
            cfg_scale: CFG scale for speech generation
            stop_check_fn: Optional callable that returns True if generation should stop
 
        Returns:
            Generated token sequences and optionally speech outputs
        """

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
        parsed_scripts = kwargs.pop("parsed_scripts", None)
        all_speakers_list = kwargs.pop("all_speakers_list", None)
        max_length_times = kwargs.pop("max_length_times", 2)

        if kwargs.get('max_new_tokens') is None:
            kwargs['max_new_tokens'] = self.config.decoder_config.max_position_embeddings - kwargs['input_ids'].shape[-1]

        generation_config, model_kwargs, input_ids, logits_processor, stopping_criteria = self._build_generate_config_model_kwargs(
            generation_config, inputs, tokenizer, return_processors=True, **kwargs
        )

        negative_kwargs = {
            'input_ids': torch.full((kwargs['input_ids'].shape[0], 1), tokenizer.speech_start_id, dtype=torch.long, device=kwargs['input_ids'].device),
            'attention_mask':  torch.ones((kwargs['input_ids'].shape[0], 1), dtype=torch.long, device=kwargs['input_ids'].device),
            'max_new_tokens': kwargs.get('max_new_tokens', 100)
        }
        negative_generation_config, negative_model_kwargs, negative_input_ids = self._build_generate_config_model_kwargs(
            None, None, tokenizer, return_processors=False, **negative_kwargs
        )

        acoustic_cache = None
        semantic_cache = None

        batch_size = input_ids.shape[0]
        device = input_ids.device
        finished_tags = torch.zeros(batch_size, dtype=torch.bool, device=device)
        correct_cnt = torch.zeros(batch_size, dtype=torch.long, device=device)
        is_prefill = True
        inputs_embeds = None
        verbose = kwargs.get("verbose", False)

        # Initialize audio chunks storage for each sample
        audio_chunks = [[] for _ in range(batch_size)]

        initial_length = input_ids.shape[-1]
        initial_length_per_sample = model_kwargs['attention_mask'].sum(dim=-1)

       # Define all valid tokens that can be generated
        valid_tokens = [
            generation_config.speech_start_id,
            generation_config.speech_end_id,
            generation_config.speech_diffusion_id,
            generation_config.eos_token_id
        ]
        # Add bos_token_id if it exists
        if hasattr(generation_config, 'bos_token_id') and generation_config.bos_token_id is not None:
            valid_tokens.append(generation_config.bos_token_id)

        # Add custom processor to constrain token generation
        token_constraint_processor = VibeVoiceTokenConstraintProcessor(valid_tokens, device=device)
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(token_constraint_processor)

        max_steps = min(generation_config.max_length - initial_length, int(max_length_times * initial_length))
        max_step_per_sample = torch.min(generation_config.max_length - initial_length_per_sample, (max_length_times * initial_length_per_sample).long())
        reach_max_step_sample = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Create progress iterator if verbose
        # TODO (ebezzam) remove from final?
        if kwargs.get("show_progress_bar", True):
            progress_bar = tqdm(range(max_steps), desc="Generating", leave=False)
        else:
            progress_bar = range(max_steps)

        for step in progress_bar:
            # Check for external stop signal
            if stop_check_fn is not None and stop_check_fn():
                if verbose:
                    print(f"Generation stopped externally at step {step + 1}")
                # End the audio streamer if it exists
                if audio_streamer is not None:
                    audio_streamer.end()
                break

            # Check if audio_streamer has been ended (stopped externally)
            if audio_streamer is not None and hasattr(audio_streamer, 'finished_flags'):
                if any(audio_streamer.finished_flags):
                    if verbose:
                        print(f"Audio generation stopped externally at step {step + 1}")
                    break

            if finished_tags.all():
                if hasattr(progress_bar, 'set_description'):
                    progress_bar.set_description("Generation complete")
                break

            if input_ids.shape[-1] >= generation_config.max_length:
                print(f"Reached maximum generation length {generation_config.max_length}, stopped it.")
                reached_samples = torch.arange(batch_size, device=device)[~finished_tags]
                if reached_samples.numel() > 0:
                    reach_max_step_sample[reached_samples] = True
                break

            # Update progress bar description with active samples
            if hasattr(progress_bar, 'set_description'):
                active_samples = (~finished_tags).sum().item()
                progress_bar.set_description(f"Generating (active: {active_samples}/{batch_size})")

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            if is_prefill:
                # we process the speech inputs only during the first generation step
                prefill_inputs = {
                    "speech_tensors": speech_tensors.to(device=device),
                    "speech_masks": speech_masks.to(device),
                    "speech_input_mask": speech_input_mask.to(device),
                }
                is_prefill = False
            else:
                _ = model_inputs.pop('inputs_embeds', None)
                prefill_inputs = {'inputs_embeds': inputs_embeds}

            # Forward pass through the model
            outputs = self(
                **model_inputs, **prefill_inputs, logits_to_keep=1, return_dict=True, output_attentions=False, output_hidden_states=False,
            )
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=False,
            )

            # Get logits and apply logits processor
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
            # next_token_logits = outputs.logits[:, -1, :].to(copy=True, device=input_ids.device)
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # token selection
            if generation_config.do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            next_tokens[finished_tags] = generation_config.eos_token_id
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            # reached end of generation
            if (next_tokens == generation_config.eos_token_id).any():
                eos_indices = (next_tokens == generation_config.eos_token_id).nonzero(as_tuple=False).squeeze(1)
                # Only print for samples that are newly finished (not already marked as finished)
                new_eos_indices = eos_indices[~finished_tags[eos_indices]]
                if new_eos_indices.numel() > 0:
                    finished_tags[new_eos_indices] = True
                    if verbose:
                        print(f"Samples {new_eos_indices.tolist()} reached EOS token at step {step + 1}.", flush=True)
                    if audio_streamer is not None:
                        audio_streamer.end(new_eos_indices)

            # Check if any sample reached its maximum generation length
            max_length_reached = step >= max_step_per_sample
            new_max_length_indices = torch.nonzero(max_length_reached & ~finished_tags, as_tuple=False).squeeze(1)
            if new_max_length_indices.numel() > 0:
                finished_tags[new_max_length_indices] = True
                reach_max_step_sample[new_max_length_indices] = True
                if verbose:
                    print(f"Samples {new_max_length_indices.tolist()} reached max generation length at step {step + 1}.", flush=True)
                if audio_streamer is not None:
                    audio_streamer.end(new_max_length_indices)

            # speech_end
            diffusion_end_indices = (next_tokens == generation_config.speech_end_id).nonzero(as_tuple=False).squeeze(1)
            if diffusion_end_indices.numel() > 0:
                # Clear tokenizer caches for samples that reached speech end
                if acoustic_cache is not None:
                    acoustic_cache.set_to_zero(diffusion_end_indices)
                if semantic_cache is not None:
                    semantic_cache.set_to_zero(diffusion_end_indices)

            # speech_begin
            diffusion_start_indices = torch.arange(batch_size, device=device)[~finished_tags & (next_tokens == generation_config.speech_start_id)]
            if diffusion_start_indices.numel() > 0 and kwargs.get('refresh_negative', True):
                # update attention mask
                for i, sample_idx in enumerate(diffusion_start_indices.tolist()):
                    negative_model_kwargs['attention_mask'][sample_idx, :] = 0
                    negative_model_kwargs['attention_mask'][sample_idx, -1] = 1
                # update past key values
                for layer_idx in range(len(negative_model_kwargs['past_key_values'])):
                    k_cache = negative_model_kwargs['past_key_values'].layers[layer_idx].keys
                    v_cache = negative_model_kwargs['past_key_values'].layers[layer_idx].values
                    # Process each non-diffusion sample
                    for sample_idx in diffusion_start_indices.tolist():
                        # Shift cache for this sample
                        k_cache[sample_idx, :, -1, :] = k_cache[sample_idx, :, 0, :].clone()
                        v_cache[sample_idx, :, -1, :] = v_cache[sample_idx, :, 0, :].clone()
                # update negative_input_ids
                for sample_idx in diffusion_start_indices.tolist():
                    negative_input_ids[sample_idx, -1] = generation_config.speech_start_id

            # Prepare inputs_embeds for next iteration
            # Initialize with default embeddings for all tokens
            next_inputs_embeds = self.model.get_input_embeddings()(next_tokens).unsqueeze(1)  # [batch_size, 1, hidden_size]

            # forward diffusion
            # Diffusion indices are those that are not finished and not special tokens
            diffusion_indices = torch.arange(batch_size, device=device)[~finished_tags & (next_tokens == generation_config.speech_diffusion_id)]

            if diffusion_indices.numel() > 0:
                negative_model_inputs = self.prepare_inputs_for_generation(negative_input_ids, **negative_model_kwargs)
                # Forward negative pass through the model
                if negative_model_inputs['inputs_embeds'] is None and inputs_embeds is not None:
                    negative_model_inputs['inputs_embeds'] = inputs_embeds
                    negative_model_inputs['input_ids'] = None

                negative_outputs = self(
                    **negative_model_inputs, logits_to_keep=0, return_dict=True, output_attentions=False, output_hidden_states=False,
                )
                negative_model_kwargs = self._update_model_kwargs_for_generation(
                    negative_outputs, negative_model_kwargs, is_encoder_decoder=False,
                )
                negative_input_ids = torch.cat([negative_input_ids, next_tokens[:, None]], dim=-1)

                # correct the non-diffusion indices
                # we forward all samples' negative outputs even if
                #   they are not in diffusion mode to keep the cache consistent
                # So we need to correct the kv cache of non-diffusion samples
                non_diffusion_mask = ~finished_tags & (next_tokens != generation_config.speech_diffusion_id)
                if non_diffusion_mask.any():
                    non_diffusion_indices = torch.arange(batch_size, device=device)[non_diffusion_mask]
                    start_indices = correct_cnt[non_diffusion_indices]

                    # 1. Update attention_mask - need to handle each sample separately
                    seq_len = negative_model_kwargs['attention_mask'].shape[1]
                    for i, (sample_idx, start_idx) in enumerate(zip(non_diffusion_indices.tolist(), start_indices.tolist())):
                        # Shift the attention mask for this sample
                        if start_idx + 1 < seq_len - 1:
                            negative_model_kwargs['attention_mask'][sample_idx, start_idx+1:] = \
                                negative_model_kwargs['attention_mask'][sample_idx, start_idx:-1].clone()
                        negative_model_kwargs['attention_mask'][sample_idx, start_idx] = 0

                    # 2. Update past_key_values
                    for layer_idx in range(len(negative_model_kwargs['past_key_values'])):
                        k_cache = negative_model_kwargs['past_key_values'].layers[layer_idx].keys
                        v_cache = negative_model_kwargs['past_key_values'].layers[layer_idx].values
                        # Process each non-diffusion sample
                        for sample_idx, start_idx in zip(non_diffusion_indices.tolist(), start_indices.tolist()):
                            if start_idx + 1 < k_cache.shape[2] - 1:
                                # Shift cache for this sample
                                k_cache[sample_idx, :, start_idx+1:, :] = k_cache[sample_idx, :, start_idx:-1, :].clone()
                                v_cache[sample_idx, :, start_idx+1:, :] = v_cache[sample_idx, :, start_idx:-1, :].clone()

                    # 3. Update negative_input_ids
                    for sample_idx, start_idx in zip(non_diffusion_indices.tolist(), start_indices.tolist()):
                        if start_idx + 1 < negative_input_ids.shape[1] - 1:
                            negative_input_ids[sample_idx, start_idx+1:] = \
                                negative_input_ids[sample_idx, start_idx:-1].clone()

                    correct_cnt[non_diffusion_indices] += 1

                positive_condition = outputs.last_hidden_state[diffusion_indices, -1, :]
                negative_condition = negative_outputs.last_hidden_state[diffusion_indices, -1, :]

                speech_latent = self.sample_speech_tokens(
                    positive_condition,
                    negative_condition,
                    cfg_scale=cfg_scale,
                ).unsqueeze(1)

                # Decode acoustic latent to audio using acoustic streaming cache
                scaled_latent = speech_latent / self.model.speech_scaling_factor.to(speech_latent.device) - self.model.speech_bias_factor.to(speech_latent.device)
                with torch.no_grad():
                    audio_output = self.model.acoustic_tokenizer.decode(
                        scaled_latent.to(self.model.acoustic_tokenizer.device),
                        past_conv_values=acoustic_cache,  # Use acoustic-specific cache
                        sample_indices=diffusion_indices.to(self.model.acoustic_tokenizer.device),
                        use_cache=True
                    )
                audio_chunk = audio_output.audio
                acoustic_cache = audio_output.past_conv_values

                # Store audio chunks for each sample
                for i, sample_idx in enumerate(diffusion_indices):
                    idx = sample_idx.item()
                    # Only append audio chunk if the sample is not finished
                    if not finished_tags[idx]:
                        audio_chunks[idx].append(audio_chunk[i])

                 # Add streaming support here
                if audio_streamer is not None:
                    # Stream the audio chunks immediately
                    audio_streamer.put(audio_chunk, diffusion_indices)

                # Encode audio to semantic features using semantic streaming cache
                with torch.no_grad():
                    semantic_outputs = self.model.semantic_tokenizer.encode(
                        audio_chunk,
                        past_conv_values=semantic_cache,  # Use semantic-specific cache
                        sample_indices=diffusion_indices,
                        use_cache=True
                    )
                semantic_features = semantic_outputs.latents
                semantic_cache = semantic_outputs.past_conv_values

                # Combine acoustic and semantic features for next input
                acoustic_embed = self.model.acoustic_connector(speech_latent)
                semantic_embed = self.model.semantic_connector(semantic_features)
                diffusion_embeds = acoustic_embed + semantic_embed

                # Update embeddings for diffusion indices
                next_inputs_embeds[diffusion_indices] = diffusion_embeds

            # Set inputs_embeds for next iteration
            inputs_embeds = next_inputs_embeds

        if audio_streamer is not None:
            audio_streamer.end()

        # Concatenate audio chunks for each sample
        final_audio_outputs = []
        for sample_chunks in audio_chunks:
            if sample_chunks:
                # Concatenate all chunks along the time dimension (assumed to be the last dimension)
                concatenated_audio = torch.cat(sample_chunks, dim=-1)
                final_audio_outputs.append(concatenated_audio)
            else:
                # If no audio was generated for this sample, append None
                final_audio_outputs.append(None)

        return VibeVoiceGenerationOutput(
            sequences=input_ids,
            speech_outputs=final_audio_outputs if return_speech else None,
            reach_max_step_sample=reach_max_step_sample,
        )

    @torch.no_grad()
    def sample_speech_tokens(self, condition, neg_condition, cfg_scale=3.0):
        self.model.noise_scheduler.set_timesteps(self.ddpm_inference_steps)
        condition = torch.cat([condition, neg_condition], dim=0).to(self.model.diffusion_head.device)
        speech = torch.randn(condition.shape[0], self.config.acoustic_hidden_size).to(condition)
        for t in self.model.noise_scheduler.timesteps:
            half = speech[: len(speech) // 2]
            combined = torch.cat([half, half], dim=0)
            eps = self.model.diffusion_head(combined, t.repeat(combined.shape[0]).to(combined), condition=condition)
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            speech = self.model.noise_scheduler.step(eps, t, speech).prev_sample
        return speech[: len(speech) // 2]


# # TODO (ebezzam) not used
# class VibeVoiceForConditionalGeneration(VibeVoicePreTrainedModel):
#     _tied_weights_keys = ["lm_head.weight"]
#     _tp_plan = {"lm_head": "colwise_rep"}

#     def __init__(self, config):
#         super().__init__(config)
#         self.model = VibeVoiceModel(config)
#         # TODO (ebezzam) change to text_config? (no error bc not used)
#         self.vocab_size = config.decoder_config.vocab_size
#         self.lm_head = nn.Linear(config.decoder_config.hidden_size, self.vocab_size, bias=False)

#         self.post_init()

#     def get_input_embeddings(self):
#         return self.model.get_input_embeddings()

#     def set_input_embeddings(self, value):
#         self.model.set_input_embeddings(value)

#     def get_output_embeddings(self):
#         return self.lm_head

#     def set_decoder(self, decoder):
#         self.model.language_model = decoder

#     def get_decoder(self):
#         return self.model.language_model

#     def tie_weights(self):
#         """
#         Tie the weights between the input embeddings and the output embeddings.
#         """
#         if getattr(self.config.decoder_config, 'tie_word_embeddings', False):
#             # The standard PreTrainedModel method will handle the tying.
#             # It typically does a simple parameter object assignment, which is
#             # CORRECT to do BEFORE FSDP wraps the model.
#             output_embeddings = self.get_output_embeddings()
#             input_embeddings = self.get_input_embeddings()
#             if hasattr(input_embeddings, 'weight'):
#                 output_embeddings.weight = input_embeddings.weight
#             else:
#                 # maybe returned input_embeddings a tensor directly
#                 output_embeddings.weight = input_embeddings

#             if getattr(output_embeddings, "bias", None) is not None:
#                 output_embeddings.bias.data = nn.functional.pad(
#                     output_embeddings.bias.data,
#                     (0, output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0]),
#                     "constant",
#                     0,
#                 )
#             print("✅ Tied input and output embeddings using standard assignment.")
#         else:
#             print("ℹ️  tie_word_embeddings is False, not tying weights.")

#     # Also, ensure set_output_embeddings is safe, though your implementation looks okay.
#     # The key is to avoid calling it after accelerator.prepare().
#     def set_output_embeddings(self, new_embeddings):
#         # Your current implementation using data.copy_ is good practice,
#         # but the best way is to not call this after prepare().
#         self.lm_head = new_embeddings

#     def forward_speech_features(
#             self,
#             speech_tensors=None,
#             speech_masks=None,
#             speech_type="audio",
#             return_unmask=False
#         ):
#         if speech_tensors is None:
#             # Use config to get vae_dim instead of non-existent self.args
#             vae_dim = self.config.acoustic_tokenizer_config.vae_dim
#             audio_features = torch.zeros(1, 1, vae_dim).to(self.get_input_embeddings().weight)
#             connect_features = self.model.acoustic_connector(audio_features)
#             return audio_features, connect_features
#         else:
#             with torch.no_grad():
#                 if speech_type == "audio":
#                     with torch.no_grad():
#                         frames = self.model.acoustic_tokenizer.encode(speech_tensors.unsqueeze(1))[0][0]
#                     # TODO (ebezzam) replaced line with below
#                     # audio_tokens = frames.sample(self.model.acoustic_tokenizer.std_dist_type)[0]
#                     audio_tokens = self.model.acoustic_tokenizer.sample(frames)[0]

#                 elif speech_type == "vae":
#                     # Use config to get vae_dim instead of non-existent self.args
#                     vae_dim = self.config.acoustic_tokenizer_config.vae_dim
#                     speech_mode = speech_tensors.reshape(speech_tensors.size(0), -1, vae_dim)

#                     # gaussian sample from the speech_mode
#                     batch_size = speech_mode.size(0)
#                     value = self.model.acoustic_tokenizer.fix_std / 0.8
#                     std = torch.randn(batch_size, dtype=speech_mode.dtype, device=speech_mode.device) * value
#                     std = std.view(-1, *[1] * (speech_mode.dim() - 1))
#                     audio_tokens = speech_mode + std * torch.randn(speech_mode.shape).to(speech_mode)
#                 else:
#                     raise NotImplementedError(f"Speech type {speech_type} not implemented")

#                 if torch.isnan(self.model.speech_scaling_factor) or torch.isnan(self.model.speech_bias_factor):
#                     scaling_factor = 1. / audio_tokens[speech_masks].flatten().std()
#                     bias_factor = -audio_tokens[speech_masks].flatten().mean()

#                     # Only use distributed operations if the process group is initialized
#                     if dist.is_available() and dist.is_initialized():
#                         dist.all_reduce(scaling_factor, op=dist.ReduceOp.SUM)
#                         dist.all_reduce(bias_factor, op=dist.ReduceOp.SUM)
#                         world_size = dist.get_world_size()
#                         self.model.speech_scaling_factor.copy_(scaling_factor / world_size)
#                         self.model.speech_bias_factor.copy_(bias_factor / world_size)
#                         print(f"Speech scaling factor (distributed): {self.model.speech_scaling_factor}, bias factor: {self.model.speech_bias_factor}", flush=True)
#                     else:
#                         # Single process case
#                         self.model.speech_scaling_factor.copy_(scaling_factor)
#                         self.model.speech_bias_factor.copy_(bias_factor)
#                         print(f"Speech scaling factor (single process): {self.model.speech_scaling_factor}, bias factor: {self.model.speech_bias_factor}", flush=True)

#                 audio_features = (audio_tokens + self.model.speech_bias_factor) * self.model.speech_scaling_factor

#             connect_features = self.model.acoustic_connector(audio_features)
#             if return_unmask:
#                 return audio_features, connect_features
#             return audio_features[speech_masks], connect_features[speech_masks]

#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[list[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = False,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         cache_position: Optional[torch.LongTensor] = None,
#         # New arguments for speech processing and loss calculation
#         speech_tensors: Optional[torch.FloatTensor] = None,
#         speech_masks: Optional[torch.BoolTensor] = None,
#         speeches_loss_input: Optional[torch.FloatTensor] = None,
#         speech_semantic_tensors: Optional[torch.FloatTensor] = None,
#         acoustic_input_mask: Optional[torch.BoolTensor] = None,
#         acoustic_loss_mask: Optional[torch.BoolTensor] = None,
#         ddpm_batch_mul: int = 1,
#         **kwargs: Optional[dict[str, Union[torch.Tensor, str]]],
#         ) -> Union[tuple, VibeVoiceCausalLMOutputWithPast]:

#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         x = self.get_input_embeddings()(input_ids)

#         semantic_speech_all_connect_features = self.model.semantic_connector(speech_semantic_tensors)
#         if speeches_loss_input is not None:
#             # only part audio need diffuse
#             speech_all_features, speech_all_connect_features = self.forward_speech_features(
#                     speech_tensors=speech_tensors.type_as(x) if speech_tensors is not None else None,
#                     speech_masks=speech_masks,
#                     speech_type=kwargs.get("speech_type", "audio"),
#                     return_unmask=True
#                 )
#             if speech_tensors is not None:
#                 if semantic_speech_all_connect_features is not None:
#                     x[acoustic_input_mask] = speech_all_connect_features[speech_masks] + semantic_speech_all_connect_features[speech_masks]
#                 else:
#                     x[acoustic_input_mask] = speech_all_connect_features[speech_masks]
#                 speech_features = speech_all_features[speeches_loss_input.unsqueeze(-1) & speech_masks] # only part audio need diffuse
#                 speech_connect_features = speech_all_connect_features[speeches_loss_input.unsqueeze(-1) & speech_masks]
#         else:
#             speech_features, speech_connect_features = self.forward_speech_features(
#                     speech_tensors=speech_tensors.type_as(x) if speech_tensors is not None else None,
#                     speech_masks=speech_masks,
#                     speech_type=kwargs.get("speech_type", "audio"),
#                 )
#             if speech_tensors is not None:
#                 x[acoustic_input_mask] = speech_connect_features

#         outputs = self.model(
#             input_ids=None,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=x,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=False,
#             return_dict=return_dict,
#             cache_position=cache_position,
#         )

#         hidden_states = outputs.last_hidden_state
#         logits = self.lm_head(hidden_states)
#         # logits = logits.float()

#         loss = None
#         if labels is not None:
#             # The custom CE loss with masking is calculated in the training script.
#             # We leave the standard loss calculation here as None.
#             pass

#         # --- Diffusion Loss Calculation ---
#         diffusion_loss = None
#         # This block is executed only if we are in a context that involves speech.
#         if speech_tensors is not None and acoustic_loss_mask.sum().item() > 0:
#             condition_features = hidden_states[acoustic_loss_mask]

#             speech_len, latent_size = speech_features.shape

#             noise = torch.randn(
#                 (speech_len * ddpm_batch_mul, latent_size),
#                 device=hidden_states.device,
#                 dtype=hidden_states.dtype
#             )

#             timesteps = torch.multinomial(
#                 torch.ones(self.config.diffusion_head_config.ddpm_num_steps),
#                 speech_len * ddpm_batch_mul,
#                 replacement=True,
#             ).to(hidden_states.device)

#             speech_features_repeated = speech_features.repeat_interleave(ddpm_batch_mul, dim=0)
#             condition_features_repeated = condition_features.repeat_interleave(ddpm_batch_mul, dim=0)

#             noisy_speech_features = self.model.noise_scheduler.add_noise(
#                 speech_features_repeated, noise, timesteps
#             )

#             model_output = self.model.diffusion_head(
#                 noisy_speech_features,
#                 timesteps.type_as(x),
#                 condition_features_repeated
#             )

#             prediction_type = self.config.diffusion_head_config.prediction_type
#             if prediction_type == "epsilon":
#                 target_for_loss = noise
#             elif prediction_type == "v_prediction":
#                 target_for_loss = self.model.noise_scheduler.get_velocity(
#                     speech_features_repeated, noise, timesteps
#                 )
#             else:
#                 raise NotImplementedError(f"Prediction type {prediction_type} not implemented")

#             diffusion_loss = F.mse_loss(model_output.float(), target_for_loss.float(), reduction='sum')
#             if latent_size > 0 and ddpm_batch_mul > 0:
#                 diffusion_loss = diffusion_loss / latent_size / ddpm_batch_mul
#             else:
#                 diffusion_loss = torch.tensor(0.0, device=diffusion_loss.device)

#         else:
#             # Dummy loss for DDP to work when there are no speech samples in a batch,
#             # but we are in a speech context.
#             diffusion_loss = sum(p.sum() for p in self.model.diffusion_head.parameters()) * 0.0
#             diffusion_loss += sum(p.sum() for p in self.model.acoustic_connector.parameters()) * 0.0
#             diffusion_loss += sum(p.sum() for p in self.model.semantic_connector.parameters()) * 0.0
#         # --- End Diffusion Loss Calculation ---

#         if not return_dict:
#             output = (logits, speech_len) + outputs.to_tuple()[1:]
#             return (loss, diffusion_loss) + output

#         return VibeVoiceCausalLMOutputWithPast(
#             loss=loss,
#             diffusion_loss=diffusion_loss,
#             speech_token_num=speech_len if speech_tensors is not None else 0,
#             logits=logits,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )


__all__ = [
    "VibeVoiceModel",
    "VibeVoicePreTrainedModel",
    "VibeVoiceForConditionalGeneration",
    "VibeVoiceDiffusionHead"
]
