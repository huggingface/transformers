"""
LLaSa training repo: https://github.com/zhenye234/LLaSA_training
- training script: https://github.com/zhenye234/LLaSA_training/blob/main/train_tts.py
- training config: https://github.com/zhenye234/LLaSA_training/blob/main/config.json

Process
- Text is encoded by the tokenizer from Llama
- Unique tokens for speech/text are added
- Speech tokens are generated auto-regressively
- Speech tokens are decoded to audio waveform by X-codec2

LLaSa models on Hugging Face, and corresponding Llama models
1. 1B config: https://huggingface.co/HKUSTAudio/Llasa-1B/blob/main/config.json
- same as https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/blob/main/config.json
2. 3B config: https://huggingface.co/HKUSTAudio/Llasa-3B/blob/main/config.json
- same as https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/blob/main/config.json
3. 8B config: https://huggingface.co/HKUSTAudio/Llasa-8B/blob/main/config.json
- same as https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/config.json
"""

from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import LlamaForCausalLM, LlamaModel, LlamaPreTrainedModel


# TODO use OrderedDict? Should only make difference for Python < 3.7
TTS_TOKENS_DICT = {
    "text_generation_start": "<|TEXT_GENERATION_START|>",
    "text_generation_end": "<|TEXT_GENERATION_END|>",
    "text_understanding_start": "<|TEXT_UNDERSTANDING_START|>",
    "text_understanding_end": "<|TEXT_UNDERSTANDING_END|>",
    "speech_generation_start": "<|SPEECH_GENERATION_START|>",
    "speech_generation_end": "<|SPEECH_GENERATION_END|>",
    "speech_understanding_start": "<|SPEECH_UNDERSTANDING_START|>",
    "speech_understanding_end": "<|SPEECH_UNDERSTANDING_END|>",
}


# Tokenizer for LLaSA as created in their training script:
# https://github.com/zhenye234/LLaSA_training/blob/main/train_tts.py#L214-L237
# - setting model_max_length, padding_side, pad_token, and more tokens (for speech and llasa)
# From `Llama-3.2-1B-Instruct` config, seems to be `PreTrainedTokenizerFast`:
# https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/blob/main/tokenizer_config.json#L2061
# Model max length:
# - config: https://github.com/zhenye234/LLaSA_training/blob/ef5c2605927190ba40656d09b3a9e10df6631149/config.json#L23
# - training script: https://github.com/zhenye234/LLaSA_training/blob/ef5c2605927190ba40656d09b3a9e10df6631149/train_tts.py#L216
# For codebook size, see https://github.com/zhenye234/LLaSA_training/blob/ef5c2605927190ba40656d09b3a9e10df6631149/train_tts.py#L235C58-L235C63
# and paper (Table 1, last row): https://arxiv.org/abs/2502.04128
# NOTE: would normally make sense to use `LlamaTokenizerFast` but it seems to add an extra token in original vocabulary
# -- moreover `Llama-3.2-1B-Instruct` uses that tokenizer class instead of `LlamaTokenizerFast`: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/blob/main/tokenizer_config.json#L2061
class LlasaTokenizer(PreTrainedTokenizerFast):
    padding_side = "right"

    def __init__(self, model_max_length=2048, codebook_size=65536, llasa_start_end_tokens=None, *args, **kwargs):
        super().__init__(model_max_length=model_max_length, *args, **kwargs)

        # TODO: correct way to overwrite variables?
        self.pad_token = self.eos_token

        # codebook indices
        if llasa_start_end_tokens is None:
            llasa_start_end_tokens = TTS_TOKENS_DICT
        self.llasa_token = llasa_start_end_tokens
        self.codebook_size = codebook_size
        self.speech_tokens = [f"<|s_{i}|>" for i in range(codebook_size)]

    @classmethod
    def from_pretrained_llm(cls, *args, **kwargs):
        """
        Load the tokenizer from a pre-trained LLM model, and add relevant speech and Llasa tokens.
        """
        tokenizer = super().from_pretrained(*args, **kwargs)
        tokenizer.add_tokens(list(tokenizer.llasa_token.values()) + tokenizer.speech_tokens)
        return tokenizer


# Very similar to Llama https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/blob/main/config.json
# TODO: Default to 1B parameters?
# Differences:
# - `eos_token_id` corresponds to ID of "speech_generation_end"
# - added `codebook_size` and `llasa_start_end_tokens`, which increases vocab size
class LlasaConfig(LlamaConfig):
    model_type = "llasa"

    def __init__(
        self,
        vocab_size=128256,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=128261,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
        codebook_size=65536,
        llasa_start_end_tokens=None,
        sampling_rate=16000,
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
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pretraining_tp=pretraining_tp,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            mlp_bias=mlp_bias,
            head_dim=head_dim,
            **kwargs,
        )

        # Adjust vocab size to accommodate the new tokenizer
        # https://github.com/zhenye234/LLaSA_training/blob/main/train_tts.py#L249-L251
        self.sampling_rate = sampling_rate
        self.codebook_size = codebook_size
        if llasa_start_end_tokens is None:
            llasa_start_end_tokens = TTS_TOKENS_DICT
        self.llasa_start_end_tokens = llasa_start_end_tokens

    @classmethod
    def from_pretrained_llm(cls, *args, **kwargs):
        """
        Load LLM config and add relevant Llasa tokens.
        """
        config = super().from_pretrained(*args, **kwargs)
        config.vocab_size += config.codebook_size + len(config.llasa_start_end_tokens)
        return config


class LlasaPreTrainedModel(LlamaPreTrainedModel):
    pass


class LlasaModel(LlamaModel):
    pass


class LlasaForCausalLM(LlamaForCausalLM):
    pass

    # """
    # TODO: how to overwrite generate method?
    # Not necessary but could be nice to check max_length < 2048 (what model was trained on)
    # I get the following error (I think because `generate` isn't method of LlamaForCausalLM but its parent):
    # ```
    # File "/home/eric_bezzam/transformers/utils/modular_model_converter.py", line 355, in replace_super_calls
    # original_modeling_method_body = self.original_modeling_methods[func_name].body.body
    # KeyError: 'generate'
    # ```
    # """
    # @torch.no_grad()
    # def generate(
    #     inputs,
    #     max_length=2048,
    #     **kwargs,
    # ):
    #     """
    #     Set specific parameters from Llasa processor output
    #     """
    #     if max_length > 2048:
    #         raise ValueError("Max length should be less than or equal to 2048.")

    #     # Call the parent class's generate method
    #     return super().generate(
    #         inputs,
    #         max_length=inputs["max_length"],
    #         **kwargs
    #     )


__all__ = ["LlasaForCausalLM", "LlasaModel", "LlasaPreTrainedModel", "LlasaConfig", "LlasaTokenizer"]
