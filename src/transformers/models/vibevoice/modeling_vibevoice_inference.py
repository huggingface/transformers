from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Callable
from tqdm import tqdm
import torch
import torch.nn as nn

from ..auto import AutoModel, AutoModelForCausalLM

from ...generation import GenerationMixin, GenerationConfig, LogitsProcessor, LogitsProcessorList, StoppingCriteriaList
from ...modeling_outputs import BaseModelOutputWithPast, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...utils import logging

from .configuration_vibevoice import VibeVoiceConfig
from .modular_vibevoice_tokenizer import VibeVoiceTokenizerStreamingCache, VibeVoiceTokenizerEncoderOutput
from .modular_vibevoice_diffusion_head import VibeVoiceDiffusionHead

from .modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizer, VibeVoiceTextTokenizerFast

from .modeling_vibevoice import VibeVoiceModel, VibeVoicePreTrainedModel
from .audio_streamer import AudioStreamer, AsyncAudioStreamer

logger = logging.get_logger(__name__)

# if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
#     modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]

@dataclass
class VibeVoiceCausalLMOutputWithPast(BaseModelOutputWithPast):
    logits: Optional[torch.FloatTensor] = None

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
    speech_outputs: Optional[List[torch.FloatTensor]] = None
    reach_max_step_sample: Optional[torch.BoolTensor] = None

class VibeVoiceTokenConstraintProcessor(LogitsProcessor):
    """Constrains token generation to only valid tokens during speech generation."""
    
    def __init__(self, valid_token_ids: List[int], device: torch.device = None):
        self.valid_token_ids = torch.tensor(valid_token_ids, dtype=torch.long, device=device)
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Create a mask for valid tokens
        mask = torch.full_like(scores, float('-inf'))
        mask[:, self.valid_token_ids] = 0
        
        # Apply mask to scores
        scores = scores + mask
        return scores
    
class VibeVoiceForConditionalGenerationInference(VibeVoicePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}

    def __init__(self, config):
        super().__init__(config)
        
        # Initialize the base model
        self.model = VibeVoiceModel(config)
        
        # LM head for text generation
        self.lm_head = nn.Linear(config.decoder_config.hidden_size, config.decoder_config.vocab_size, bias=False)
        
        # inference configuration
        self.ddpm_inference_steps = config.diffusion_head_config.ddpm_num_inference_steps

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def noise_scheduler(self):
        return self.model.noise_scheduler

    @property
    def prediction_head(self):
        return self.model.prediction_head
    
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
    
    def set_speech_tokenizers(self, acoustic_tokenizer=None, semantic_tokenizer=None):
        """Set the speech tokenizers used for encoding and decoding speech."""
        self.model.set_speech_tokenizers(acoustic_tokenizer, semantic_tokenizer)
    
    def set_ddpm_inference_steps(self, num_steps=None):
        self.ddpm_inference_steps = num_steps or self.config.diffusion_head_config.ddpm_num_inference_steps

    def _process_speech_inputs(self, speech_tensors, speech_masks, speech_type="audio"):
        """Process speech inputs through tokenizers and connectors."""
        encoder_output = self.model.acoustic_tokenizer.encode(speech_tensors.unsqueeze(1))
        acoustic_latents = encoder_output.sample(dist_type=self.model.acoustic_tokenizer.std_dist_type)[0]
        
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
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        speech_tensors: Optional[torch.FloatTensor] = None,
        speech_masks: Optional[torch.BoolTensor] = None,
        speech_input_mask: Optional[torch.BoolTensor] = None,
        logits_to_keep: Union[int, slice] = 0,
        **kwargs,
    ) -> Union[Tuple, VibeVoiceCausalLMOutputWithPast]:
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
        model_kwargs["use_cache"] = generation_config.use_cache
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
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        audio_streamer: Optional[Union[AudioStreamer, AsyncAudioStreamer]] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        speech_tensors: Optional[torch.FloatTensor] = None,
        speech_masks: Optional[torch.BoolTensor] = None,
        speech_input_mask: Optional[torch.BoolTensor] = None,
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

        if kwargs.get('max_new_tokens', None) is None:
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

        acoustic_cache = VibeVoiceTokenizerStreamingCache()
        semantic_cache = VibeVoiceTokenizerStreamingCache()
        
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
                acoustic_cache.set_to_zero(diffusion_end_indices)
                semantic_cache.set_to_zero(diffusion_end_indices)
            
            # speech_begin
            diffusion_start_indices = torch.arange(batch_size, device=device)[~finished_tags & (next_tokens == generation_config.speech_start_id)]
            if diffusion_start_indices.numel() > 0 and kwargs.get('refresh_negative', True):
                # update attention mask
                for i, sample_idx in enumerate(diffusion_start_indices.tolist()):
                    negative_model_kwargs['attention_mask'][sample_idx, :] = 0
                    negative_model_kwargs['attention_mask'][sample_idx, -1] = 1
                # update past key values
                for layer_idx, (k_cache, v_cache) in enumerate(zip(negative_model_kwargs['past_key_values'].key_cache, 
                                                                        negative_model_kwargs['past_key_values'].value_cache)):
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
                    for layer_idx, (k_cache, v_cache) in enumerate(zip(negative_model_kwargs['past_key_values'].key_cache, 
                                                                        negative_model_kwargs['past_key_values'].value_cache)):
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
                audio_chunk = self.model.acoustic_tokenizer.decode(
                    scaled_latent.to(self.model.acoustic_tokenizer.device),
                    cache=acoustic_cache,  # Use acoustic-specific cache
                    sample_indices=diffusion_indices.to(self.model.acoustic_tokenizer.device),
                    use_cache=True,
                    debug=False
                )
                
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
                semantic_features = self.model.semantic_tokenizer.encode(
                    audio_chunk,
                    cache=semantic_cache,  # Use semantic-specific cache
                    sample_indices=diffusion_indices,
                    use_cache=True,
                    debug=False
                ).mean # semantic tokenizer has no VAE.
                
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
        condition = torch.cat([condition, neg_condition], dim=0).to(self.model.prediction_head.device)
        speech = torch.randn(condition.shape[0], self.config.acoustic_vae_dim).to(condition)
        for t in self.model.noise_scheduler.timesteps:
            half = speech[: len(speech) // 2]
            combined = torch.cat([half, half], dim=0)
            eps = self.model.prediction_head(combined, t.repeat(combined.shape[0]).to(combined), condition=condition)
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            speech = self.model.noise_scheduler.step(eps, t, speech).prev_sample
        return speech[: len(speech) // 2]
    

AutoModelForCausalLM.register(VibeVoiceConfig, VibeVoiceForConditionalGenerationInference)

__all__ = [
    "VibeVoiceForConditionalGenerationInference",
]
