"""

LLaSa training code: https://github.com/zhenye234/LLaSA_training

Text sequence is encoded by the text tokenizer from Llama:
https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
- Config: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/blob/main/config.json

Speech_sequence is extracted through X-codec2: https://github.com/zhenye234/X-Codec-2.0
Value of speech tokens is changed by adding:
len(text tokenizer) +8 special tokens
thereby forming a unified tokenizer that encompasses both speech and text.

Special tokens: https://github.com/zhenye234/LLaSA_training/blob/1d65cf3e34c0d5b508404d67ff41b3b6fb1ecab7/train_tts.py#L67

Training script: https://github.com/zhenye234/LLaSA_training/blob/main/train_tts.py
Training config: https://github.com/zhenye234/LLaSA_training/blob/main/config.json



1B config: https://huggingface.co/HKUSTAudio/Llasa-1B/blob/main/config.json
- same as https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/blob/main/config.json
3B config: https://huggingface.co/HKUSTAudio/Llasa-3B/blob/main/config.json
- same as https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/blob/main/config.json
8B config: https://huggingface.co/HKUSTAudio/Llasa-8B/blob/main/config.json
- same as https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/config.json


vocab_size=193800 as speech and llasa tokens are added to the original tokenizer
"""

from typing import Optional, Union

from transformers.utils import TensorType

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

        # correct way to overwrite variables?
        self.pad_token = self.eos_token

        # codebook indices
        if llasa_start_end_tokens is None:
            llasa_start_end_tokens = TTS_TOKENS_DICT
        self.llasa_token = llasa_start_end_tokens
        self.codebook_size = codebook_size
        self.speech_tokens = [f"<|s_{i}|>" for i in range(codebook_size)]
        self.add_tokens(list(self.llasa_token.values()) + self.speech_tokens)

    def custom_prepare(
        self,
        text: str,
        return_tensors: Optional[Union[str, TensorType]] = "pt",
        continue_final_message: bool = True,
        **kwargs,
    ) -> list[int]:
        """
        Prepare text for the model by adding start and end tokens with chat template.
        https://github.com/zhenye234/LLaSA_training/blob/ef5c2605927190ba40656d09b3a9e10df6631149/train_tts.py#L114
        Easier to see in their example on their model card: https://huggingface.co/HKUSTAudio/Llasa-1B#how-to-use
        """
        formatted_text = (
            f"{self.llasa_token['text_understanding_start']}{text}{self.llasa_token['text_understanding_end']}"
        )
        chat = [
            {"role": "user", "content": "Convert the text to speech:" + formatted_text},
            {"role": "assistant", "content": self.llasa_token["speech_generation_start"]},
        ]
        input_ids = self.apply_chat_template(
            chat, return_tensors=return_tensors, continue_final_message=continue_final_message, **kwargs
        )
        return input_ids


# Same as Llama https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/blob/main/config.json
# TODO: Default to 1B parameters?
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
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
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
        self.llm_vocab_size = vocab_size
        self.codebook_size = codebook_size
        if llasa_start_end_tokens is None:
            llasa_start_end_tokens = TTS_TOKENS_DICT
        self.llasa_start_end_tokens = llasa_start_end_tokens
        self.vocab_size += codebook_size + len(llasa_start_end_tokens)


class LlasaPreTrainedModel(LlamaPreTrainedModel):
    pass


class LlasaModel(LlamaModel):
    pass


class LlasaForCausalLM(LlamaForCausalLM):
    def extract_speech_ids(speech_tokens_str):
        speech_ids = []
        for token_str in speech_tokens_str:
            if token_str.startswith("<|s_") and token_str.endswith("|>"):
                # TODO fix hardcoded integers
                num_str = token_str[4:-2]
                num = int(num_str)
                speech_ids.append(num)
            else:
                raise ValueError(f"Unexpected token: {token_str}")
        return speech_ids

    # TODO: how to overwrite generate method?
    # def generate(
    #     self,
    #     max_length: Optional[int] = 2048,
    #     eos_token_id: Optional[int] = None,
    #     **kwargs,
    # ) -> Union[GenerateOutput, torch.LongTensor]:
    #     """
    #     Set specific parameters.

    #     TODO max length and eos token ID in model config?
    #     """

    #     # Call the parent class's generate method
    #     return super().generate(
    #         max_length=max_length,
    #         eos_token_id=eos_token_id,
    #         **kwargs
    #     )


__all__ = ["LlasaForCausalLM", "LlasaModel", "LlasaPreTrainedModel", "LlasaConfig", "LlasaTokenizer"]
