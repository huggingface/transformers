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

Run following command to creating Llasa transformer files:
```
python utils/modular_model_converter.py --files-to-parse src/transformers/models/llasa/modular_llasa.py
```

"""

from typing import Optional, Union

import torch
from torch import nn

from ...cache_utils import Cache
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...processing_utils import Unpack
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
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
# - adding `from_pretrained_llm` to use load from existing LLM and use `codebook_size` and `llasa_start_end_tokens` to increase vocab size
# - adding `sampling_rate` for audio processing and overwriting docstrings
class LlasaConfig(LlamaConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlasaModel`]. It is used to instantiate an Llasa
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of
    [HKUSTAudio/Llasa-1B](https://huggingface.co/HKUSTAudio/Llasa-1B), which is similar to
    [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) but with additional
    tokens for speech generation and understanding.


    Args:
        vocab_size (`int`, *optional*, defaults to 128256):
            Vocabulary size of the Llasa model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LlasaModel`]
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 8192):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 128261):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/main/perf_train_gpu_many#tensor-parallelism) to
            understand more about it. This value is necessary to ensure exact reproducibility of the pretraining
            results. Please refer to [this issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llasa3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llasa3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llasa3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llasa3'. Scaling factor applied to high frequency components of the RoPE
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.
        head_dim (`int`, *optional*):
            The attention head dimension. If None, it will default to hidden_size // num_attention_heads
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate of the audio data used for training the model.
    """

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
        self.sampling_rate = sampling_rate

    @classmethod
    def from_pretrained_llm(cls, codebook_size=65536, llasa_start_end_tokens=None, *args, **kwargs):
        """
        Load LLM config and add relevant Llasa tokens.

        See https://github.com/zhenye234/LLaSA_training/blob/main/train_tts.py#L249-L251

        Args:
            codebook_size (int): Size of the codebook for speech tokens. Default is 65536.
            llasa_start_end_tokens (dict, optional): Dictionary containing start and end tokens for Llasa processing.
                Defaults to 8 tokens for text and speech generation/understanding: `<|TEXT_GENERATION_START|>`,
                `<|TEXT_GENERATION_END|>`, `<|TEXT_UNDERSTANDING_START|>`, `<|TEXT_UNDERSTANDING_END|>`,
                `<|SPEECH_GENERATION_START|>`, `<|SPEECH_GENERATION_END|>`, `<|SPEECH_UNDERSTANDING_START|>`,
                `<|SPEECH_UNDERSTANDING_END|>`.

        Example:
        ```python
        >>> from transformers import LlasaForCausalLM, LlasaConfig

        >>> # Initializing Llasa with `meta-llama/Llama-3.2-1B-Instruct`
        >>> configuration = LlasaConfig.from_pretrained_llm(
        >>>     pretrained_model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
        >>>     codebook_size=65536
        >>> )

        >>> # Initializing a model from the llasa-7b style configuration
        >>> model = LlasaForCausalLM(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```
        """
        config = super().from_pretrained(*args, **kwargs)
        if llasa_start_end_tokens is None:
            llasa_start_end_tokens = TTS_TOKENS_DICT
        config.vocab_size += codebook_size + len(llasa_start_end_tokens)
        return config


@auto_docstring(
    custom_intro="""
    Bare Llasa model that outputs raw hidden-states without any specific head on top.
    """
)
class LlasaPreTrainedModel(LlamaPreTrainedModel):
    pass


@auto_docstring(
    custom_intro="""
    Bare Llasa model that outputs raw hidden-states without any specific head on top.
    """
)
class LlasaModel(LlamaModel):
    pass


@auto_docstring(
    custom_intro="""
    The Llasa model consists of a Llama model with a modified configuration to support the generation of speech tokens.
    """
)
# TODO: copying over but really just want to overwrite docstrings from `forward`
class LlasaForCausalLM(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlasaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Forward pass for the Llasa model.
        """
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # TODO: how to overwrite generate method?
    # Not necessary but could be nice to check max_length <= 2048 (what model was trained on)
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
