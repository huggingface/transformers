import torch, inspect
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

model_id = "openai/whisper-tiny.en"
model = WhisperForConditionalGeneration.from_pretrained(model_id)
processor = WhisperProcessor.from_pretrained(model_id)

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True)
audio_sample = ds[0]["audio"]
waveform = audio_sample["array"]
sampling_rate = audio_sample["sampling_rate"]

# Use the model and processor to transcribe the audio:
input_features = processor(
    waveform, sampling_rate=sampling_rate, return_tensors="pt"
).input_features


################################################################################################################################################################################
# FROM WHISPER GENERATION
# https://github.com/huggingface/transformers/blob/f7ef7cec6c6c162087421f36a17eabdbb223579d/src/transformers/models/whisper/generation_whisper.py#L483
kwargs = {}
input_stride = model.model.encoder.conv1.stride[0] * model.model.encoder.conv2.stride[0]
num_segment_frames = input_stride * model.config.max_source_positions
batch_size, total_input_frames = input_features.shape[0], input_features.shape[-1]
is_shortform = total_input_frames <= num_segment_frames
assert is_shortform

decoder_start_token_id = model.generation_config.decoder_start_token_id
eos_token_id = model.generation_config.eos_token_id

# FROM WHISPER GENERATION
# https://github.com/huggingface/transformers/blob/f7ef7cec6c6c162087421f36a17eabdbb223579d/src/transformers/models/whisper/generation_whisper.py#L533
# BOS TOKENS that start the decoder sequence
init_tokens = model._retrieve_init_tokens(
    input_features,
    generation_config=model.generation_config,
    config=model.config,
    num_segment_frames=num_segment_frames,
    kwargs=kwargs,
)
begin_index = len(init_tokens)

# FROM WHISPER GENERATION
# https://github.com/huggingface/transformers/blob/f7ef7cec6c6c162087421f36a17eabdbb223579d/src/transformers/models/whisper/generation_whisper.py#L548
model.generation_config.no_speech_threshold = None
logits_processor = model._retrieve_logit_processors(
    generation_config=model.generation_config,
    logits_processor=None,
    begin_index=begin_index,
    is_shortform=is_shortform,
    num_beams=1,
)

one_tensor = torch.ones((batch_size, 1), device=model.device, dtype=torch.long)
decoder_input_ids = torch.cat([t * one_tensor for t in init_tokens], dim=-1)
################################################################################################################################################################################

assert model.model.main_input_name == model.model.encoder.main_input_name
model_input_name = model.model.main_input_name
inputs_tensor = input_features
model_kwargs = {
    "decoder_input_ids": decoder_input_ids
}

batch_size = inputs_tensor.shape[0]
# FROM GENERATION UTILS
# https://github.com/huggingface/transformers/blob/f7ef7cec6c6c162087421f36a17eabdbb223579d/src/transformers/generation/utils.py#L469
model_kwargs["attention_mask"] = model._prepare_attention_mask_for_generation(
    inputs_tensor, model.generation_config.pad_token_id, model.generation_config.eos_token_id
)
model_kwargs["output_attentions"] = model.generation_config.output_attentions
model_kwargs["output_hidden_states"] = model.generation_config.output_hidden_states
model_kwargs["use_cache"] = model.generation_config.use_cache

# RUN THROUGH THE ENCODER
# https://github.com/huggingface/transformers/blob/f7ef7cec6c6c162087421f36a17eabdbb223579d/src/transformers/generation/utils.py#L487
# model._prepare_encoder_decoder_kwargs_for_generation()
irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
encoder_kwargs = {
    argument: value
    for argument, value in model_kwargs.items()
    if not any(argument.startswith(p) for p in irrelevant_prefix)
}
encoder_kwargs["return_dict"] = True
encoder_kwargs[model_input_name] = inputs_tensor
model_kwargs["encoder_outputs"] = model.model.encoder(**encoder_kwargs)

# RUN THROUGH DECODER
input_ids = model_kwargs.pop("decoder_input_ids")
input_ids_length = input_ids.shape[-1]

logits_processor = model._get_logits_processor(
    generation_config=model.generation_config,
    input_ids_seq_length=input_ids_length,
    encoder_input_ids=inputs_tensor,
    prefix_allowed_tokens_fn=None,
    logits_processor=logits_processor,
    model_kwargs=model_kwargs,
    negative_prompt_ids=None,
    negative_prompt_attention_mask=None,
)

def prepare_inputs_for_generation(
    decoder_input_ids,
    past_key_values=None,
    use_cache=None,
    encoder_outputs=None,
    attention_mask=None,
    decoder_attention_mask=None,
    **kwargs,
):  
    assert decoder_attention_mask is None
    decoder_position_ids = None
    
    if past_key_values is not None:
        past_length = past_key_values[0][0].shape[2]

        # Some generation methods already pass only the last input ID
        if decoder_input_ids.shape[1] > past_length:
            remove_prefix_length = past_length
        else:
            assert False
            # Default to old behavior: keep only final ID
            remove_prefix_length = decoder_input_ids.shape[1] - 1

        # just get the last token
        decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]
    
    return {
        "encoder_outputs": encoder_outputs,
        "past_key_values": past_key_values,
        "decoder_input_ids": decoder_input_ids,
        "use_cache": use_cache,
        "decoder_attention_mask": decoder_attention_mask,
        "decoder_position_ids": decoder_position_ids,
    }

# decoding loop:
TOKENS_TO_GENERATE = 100
for _ in range(TOKENS_TO_GENERATE):
    model_inputs = prepare_inputs_for_generation(input_ids, **model_kwargs)

    outputs = model(
        **model_inputs,
        return_dict=True,
        output_attentions=model.generation_config.output_attentions,
        output_hidden_states=model.generation_config.output_hidden_states,
    )

    next_token_logits = outputs.logits[:, -1, :]
    next_tokens_scores = logits_processor(input_ids, next_token_logits)
    next_tokens = torch.argmax(next_tokens_scores, dim=-1)
    
    if next_tokens == eos_token_id:
        break

    # Update inputs for next round of generation.
    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
    model_kwargs["past_key_values"] = outputs.past_key_values
    
transcription = processor.batch_decode(input_ids, skip_special_tokens=True)[0]
print(transcription)