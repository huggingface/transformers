import math
import re
import base64
import io
import librosa
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import soundfile as sf
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple, Union, Callable
from urllib.parse import urlparse

from transformers.configuration_utils import PretrainedConfig
from transformers.audio_utils import AudioInput
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import TextInput

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    MoeCausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.utils import auto_docstring, can_return_tuple
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import TransformersKwargs, check_model_inputs
from ..qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig, Qwen3OmniMoeTextConfig, Qwen3OmniMoeThinkerConfig,
    Qwen3OmniMoeConfig
)
from ..qwen3_omni_moe.processing_qwen3_omni_moe import (
    _get_feat_extract_output_lengths, Qwen3OmniMoeProcessor
)
from ..qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerTextRMSNorm, rotate_half, repeat_kv, apply_rotary_pos_emb,
    eager_attention_forward, Qwen3OmniMoeThinkerTextAttention, Qwen3OmniMoeThinkerTextMLP, 
    Qwen3OmniMoeThinkerTextDecoderLayer, _get_feat_extract_output_lengths, 
    Qwen3OmniMoePreTrainedModelForConditionalGeneration, Qwen3OmniMoeAudioAttention,
    SinusoidsPositionEmbedding, Qwen3OmniMoeAudioEncoderLayer, Qwen3OmniMoeAudioEncoder,
    Qwen3OmniMoeThinkerTextRotaryEmbedding, Qwen3OmniMoeThinkerTextMLP, 
    Qwen3OmniMoeThinkerTextRMSNorm, Qwen3OmniMoeThinkerTextModel, 
    Qwen3OmniMoeThinkerForConditionalGeneration
)
from ..audioflamingo3.processing_audioflamingo3 import AudioFlamingo3Processor

class Qwen3ASRAudioEncoderConfig(Qwen3OmniMoeAudioEncoderConfig):
    pass

class Qwen3ASRTextConfig(Qwen3OmniMoeTextConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3ASRTextModel`]. It is used to instantiate a
    Qwen3-ASR model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen3-ASR-1.7B [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the model.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `32`.

        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 128000):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*):
            End of stream token id.

    ```python
    >>> from transformers import Qwen3ASRTextModel, Qwen3ASRTextConfig

    >>> # Initializing a configuration
    >>> configuration = Qwen3ASRTextConfig()

    >>> # Initializing a model with random weights
    >>> model = Qwen3ASRTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=128000,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        rope_parameters=None,
        attention_bias=False,
        sliding_window=None,
        attention_dropout=0.0,
        pad_token_id=None,
        bos_token_id= None,
        eos_token_id=None,
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
            rope_parameters=rope_parameters,
            attention_bias=attention_bias,
            sliding_window=sliding_window,
            attention_dropout=attention_dropout,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        del self.decoder_sparse_step
        del self.moe_intermediate_size
        del self.num_experts_per_tok
        del self.num_experts
        del self.norm_topk_prob
        del self.output_router_logits
        del self.router_aux_loss_coef
        del self.mlp_only_layers

# TODO: cannot inherit from Qwen3OmniMoeThinkerConfig due to vision_config block
class Qwen3ASRThinkerConfig(Qwen3OmniMoeThinkerConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3ASRThinker`]. It is used to instantiate a
    Qwen3-ASR-Thinker model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the thinker component of the Qwen3-Omni
    architecture.

    e.g. [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        audio_config (`dict`, *optional*):
            The config dictionary of the audio backbone.
        text_config (`dict`, *optional*):
            The config dictionary of the text backbone.
        audio_token_id (`int`, *optional*, defaults to 151646):
            The audio token id to encode the audio prompt.
        audio_start_token_id (`int`, *optional*, defaults to 151647):
            The audio start token id to encode the audio prompt.
        user_token_id (`int`, *optional*, defaults to 872):
            The user token id to encode the user token.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:

    ```python
    >>> from transformers import Qwen3ASRThinkerModel, Qwen3ASRThinkerConfig

    >>> # Initializing a default Qwen3ASRThinkerConfig
    >>> configuration = Qwen3ASRThinkerConfig()

    >>> # Initializing a model (with random weights) from the default configuration
    >>> model = Qwen3ASRThinkerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    sub_configs = {
        "audio_config": Qwen3ASRAudioEncoderConfig,
        "text_config": Qwen3ASRTextConfig,
    }

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        audio_token_id=151646,
        audio_start_token_id=151647,
        user_token_id=872,
        initializer_range=0.02,
        attn_implementation=None,
        **kwargs,
    ):
        PreTrainedConfig.__init__(**kwargs)

        self.user_token_id = user_token_id
        self.audio_start_token_id = audio_start_token_id
        self.initializer_range = initializer_range

        if isinstance(audio_config, dict):
            audio_config = Qwen3ASRAudioEncoderConfig(**audio_config)
        elif audio_config is None:
            audio_config = Qwen3ASRAudioEncoderConfig()
        self.audio_config = audio_config

        if isinstance(text_config, dict):
            text_config = Qwen3ASRTextConfig(**text_config)
        elif text_config is None:
            text_config = Qwen3ASRTextConfig()
        self.text_config = text_config
        self.audio_token_id = audio_token_id

class Qwen3ASRConfig(Qwen3OmniMoeConfig):
    """
    This is the configuration class to store the configuration of a [`Qwen3ASRForConditionalGeneration`]. It is used to instantiate a Qwen3ASR
    model according to the specified sub-models configurations, defining the model architecture.

    Instantiating a configuration with the defaults will yield a similar configuration to that of the
    [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        thinker_config (`dict`, *optional*): Configuration of the underlying thinker sub-model.
        support_languages (`List[str]`, *optional*): The languages supported by the model.

    Example:

    ```python
    >>> from transformers import (
    ...     Qwen3ASRThinkerConfig,
    ...     Qwen3ASRForConditionalGeneration,
    ...     Qwen3ASRConfig,
    ... )

    >>> # Initializing a Qwen3ASR style configuration
    >>> configuration = Qwen3ASRConfig()

    >>> # Initializing a model from the configuration
    >>> model = Qwen3ASRForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    sub_configs = {
        "thinker_config": Qwen3ASRThinkerConfig,
    }

    def __init__(
        self,
        thinker_config=None,
        talker_config=None,
        code2wav_config=None,
        support_languages=None,
        attn_implementation=None,
        **kwargs,
    ):
        super().__init__(
            thinker_config=thinker_config,
            support_languages=support_languages,
            attn_implementation=attn_implementation,
            **kwargs,
        )
        self.support_languages = support_languages
        self._attn_implementation = attn_implementation
        del self.talker_config
        del self.code2wav_config
        del self.initializer_range
        del self.enable_audio_output
        del self.enable_audio_output
        del self.im_start_token_id
        del self.im_end_token_id
        del self.tts_pad_token_id
        del self.tts_bos_token_id
        del self.tts_eos_token_id 
        del self.system_token_id
        del self.user_token_id
        del self.assistant_token_id

    @property
    def num_attention_heads(self):
        return self.thinker_config.text_config.num_attention_heads

    @property
    def hidden_size(self):
        return self.thinker_config.text_config.hidden_size

    @property
    def vocab_size(self):
        return self.thinker_config.text_config.vocab_size

    @vocab_size.setter
    def vocab_size(self, value):
        self.thinker_config.text_config.vocab_size = value

class Qwen3ASRProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "padding_side": "left",
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": True,
            "return_attention_mask": True,
        },
    }

class Qwen3ASRProcessor(AudioFlamingo3Processor):
    r"""
    Constructs a Qwen3ASR processor.
    [`Qwen3ASRProcessor`] offers all the functionalities of [`WhisperFeatureExtractor`], and [`Qwen2TokenizerFast`]. See the
    [`~Qwen3ASRProcessor.__call__`] and [`~Qwen3ASRProcessor.decode`] for more information.

    Args:
        feature_extractor ([`WhisperFeatureExtractor`], *optional*):
            The audio feature extractor.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The text tokenizer.
        chat_template (`Optional[str]`, *optional*):
            The Jinja template to use for formatting the conversation. If not provided, the default chat template is used.
    """

    attributes = ["tokenizer", "feature_extractor"]
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(
        self,
        feature_extractor=None, 
        tokenizer=None, 
        chat_template=None
    ):  
        super().__init__(feature_extractor,tokenizer,chat_template)
        del self.audio_token
        del self.audio_token_id
        del self.default_transcription_prompt
        del self.max_audio_len
        self.audio_token = self.tokenizer.audio_token
        self.audio_bos_token = self.tokenizer.audio_bos_token
        self.audio_eos_token = self.tokenizer.audio_eos_token

    def _get_audio_token_length(self, audio_lengths: "torch.Tensor") -> "torch.Tensor":
        raise ValueError("Not needed.")

    def __call__(
        self,
        text: TextInput = None,
        audio: AudioInput = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the audio(s), this method forwards the `audio` and `kwargs` arguments to
        WhisperFeatureExtractor's [`~WhisperFeatureExtractor.__call__`] if `audio` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            audio (`np.ndarray`, `List[np.ndarray]`):
                The audio or batch of audio to be prepared. Each audio can be a NumPy array.
        """

        if text is None:
            raise ValueError("You need to specify either a `text` input to process.")

        output_kwargs = self._merge_kwargs(
            Qwen3ASRProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if audio is not None:
            output_kwargs["audio_kwargs"]["padding"] = True
            output_kwargs["audio_kwargs"]["truncation"] = False
            audio_inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
            audio_inputs["feature_attention_mask"] = audio_inputs.pop(
                "attention_mask"
            )  # rename feature_attention_mask to prevent conflicts later on
            audio_inputs["input_features"] = audio_inputs.pop(
                "input_features"
            )  # rename input_features to prevent conflicts later on
            audio_lengths = iter(_get_feat_extract_output_lengths(audio_inputs["feature_attention_mask"].sum(-1)))
        else:
            audio_inputs = {}
            audio_lengths = iter([])

        if not isinstance(text, list):
            text = [text]

        text = self.replace_multimodal_special_tokens(
            text,
            audio_lengths,
        )

        texts_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(
            data={**texts_inputs, **audio_inputs},
            tensor_type=kwargs.get("return_tensors"),
        )

    def apply_transcription_request(
        self,
        audio: Union[str, list[str], AudioInput],
        prompt: Optional[Union[str, list[str]]] = None,
        **kwargs: Unpack[Qwen3ASRProcessorKwargs],
    ) -> BatchFeature:
        raise ValueError("Not needed.")

    def batch_decode(self, *args, strip_prefix=False, **kwargs):
        raise ValueError("Not needed.")

    def _strip_assistant_prefix_and_quotes(self, text: str) -> str:
        raise ValueError("Not needed.")

    def get_chunked_index(self, token_indices: np.ndarray, tokens_per_chunk: int) -> list[tuple[int, int]]:
        """
        Splits token index list into chunks based on token value ranges.

        Given a list of token indices, returns a list of (start, end) index tuples representing
        slices of the list where the token values fall within successive ranges of `t_ntoken_per_chunk`.

        For example, if `t_ntoken_per_chunk` is 1000, the function will create chunks such that:
        - the first chunk contains token values < 1000,
        - the second chunk contains values >= 1000 and < 2000, and so on.

        Parameters:
            token_indices (`np.ndarray`): A monotonically increasing list of token index values.
            t_ntoken_per_chunk (`int`): Number of tokens per chunk (used as the chunk size threshold).

        Returns:
            `list[tuple[int, int]]`: A list of tuples, each representing the start (inclusive)
                                and end (exclusive) indices of a chunk in `token_indices`.
        """

        def _iter():
            i, start_idx = 0, 0  # skip bos token
            current_chunk = 1
            while i < len(token_indices):  # skip eos token
                if token_indices[i] >= current_chunk * tokens_per_chunk:
                    yield (start_idx, i)
                    start_idx = i
                    current_chunk += 1
                i += 1
            yield (start_idx, len(token_indices))

        return list(_iter())

    def apply_chat_template(self, conversations, chat_template=None, **kwargs):
        return ProcessorMixin.apply_chat_template(conversations, chat_template, **kwargs)

    def replace_multimodal_special_tokens(
        self,
        text,
        audio_lengths,
    ):
        processed_text = []
        for sample in text:
            positions = []
            special_tokens = [re.escape(tok) for tok in [self.audio_token]]
            pattern = "|".join(special_tokens)
            positions = sorted([(match.start(), match.group()) for match in re.finditer(pattern, sample)])
            positions.sort(key=lambda x: x[0])

            for _, special_token in positions:
                if special_token == self.audio_token:
                    sample = sample.replace(self.audio_token, "<|audio_placeholder|>" * next(audio_lengths), 1)

            sample = sample.replace("<|audio_placeholder|>", self.audio_token)
            processed_text.append(sample)
        return processed_text

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(
            dict.fromkeys(
                tokenizer_input_names
                + feature_extractor_input_names
                + ["feature_attention_mask"]
            )
        )

class Qwen3ASRTextRMSNorm(Qwen3OmniMoeThinkerTextRMSNorm):
    pass

class Qwen3ASRTextAttention(Qwen3OmniMoeThinkerTextAttention):
    pass

class Qwen3ASRTextMLP(Qwen3OmniMoeThinkerTextMLP):
    pass

class Qwen3ASRThinkerTextDecoderLayer(Qwen3OmniMoeThinkerTextDecoderLayer):
    def __init__(self, config: Qwen3ASRConfig, layer_idx: int):
        GradientCheckpointingLayer.__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3ASRTextAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3ASRTextMLP(config)
        self.input_layernorm = Qwen3ASRTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3ASRTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

@auto_docstring
class Qwen3ASRPreTrainedModel(PreTrainedModel):
    config: Qwen3ASRConfig
    base_model_prefix = "model"
    input_modalities = ("audio", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3ASRThinkerTextDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "attentions": Qwen3ASRTextAttention,
    }

@dataclass
class Qwen3ASRThinkerCausalLMOutputWithPast(MoeCausalLMOutputWithPast):
    r"""
    Args:
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    rope_deltas: Optional[torch.LongTensor] = None

class Qwen3ASRPreTrainedModelForConditionalGeneration(Qwen3OmniMoePreTrainedModelForConditionalGeneration):
    def _prepare_4d_causal_attention_mask_with_cache_position(
        self,
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        config=None,
        past_key_values=None,
        device: torch.device = None,
        min_dtype: float = None,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            min_dtype (`float`):
                The minimum value representable with the dtype `dtype`.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        ###
        device = device or attention_mask.device
        min_dtype = min_dtype if min_dtype is not None else torch.finfo(dtype).min
        ###
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


    def get_rope_index(
        self,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the rope index in LLM.

        Explanation:
            Each embedding sequence contains text embedding.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            audio_seqlens (`torch.LongTensor` of shape `(num_audios)`, *optional*):
                The length of feature shape of each audio in LLM.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        mrope_position_deltas = []

        position_ids = attention_mask.float().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)

        return position_ids, mrope_position_deltas

class Qwen3ASRAudioAttention(Qwen3OmniMoeAudioAttention):
    pass

class Qwen3ASRAudioEncoderLayer(Qwen3OmniMoeAudioEncoderLayer):
    pass

@auto_docstring(
    custom_intro="""
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`Qwen3ASRAudioEncoderLayer`].
    """
)
class Qwen3ASRAudioEncoder(Qwen3OmniMoeAudioEncoder):
    pass

class Qwen3ASRThinkerTextRotaryEmbedding(Qwen3OmniMoeThinkerTextRotaryEmbedding):
    def __init__(self, config: Qwen3ASRConfig, device=None):
        super().__init__()
        self.rope_type = config.rope_scaling.get("rope_type", "linear")
        self.mrope_section = config.rope_scaling.get("mrope_section", [24, 20, 20])

class Qwen3ASRThinkerTextMLP(Qwen3OmniMoeThinkerTextMLP):
    pass

class Qwen3ASRThinkerTextRMSNorm(Qwen3OmniMoeThinkerTextRMSNorm):
    pass

class Qwen3ASRThinkerTextAttention(Qwen3OmniMoeThinkerTextAttention):
    pass

@auto_docstring(
    custom_intro=(
        "Text part of Qwen3ASRThinker, "
    )
)
class Qwen3ASRThinkerTextModel(Qwen3OmniMoeThinkerTextModel):
    _can_record_outputs = {
        "hidden_states": Qwen3ASRThinkerTextDecoderLayer,
        "attentions": Qwen3ASRTextAttention,
    }

    def __init__(self, config: Qwen3ASRConfig):
        super().__init__(config)

    @check_model_inputs()
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )
    
    def _deepstack_process(
        self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
    ):
        raise ValueError("Not needed.")


@auto_docstring(
    custom_intro="""
    The Qwen3ASRThinker model which consists of a audio backbone and a language model.
    """
)
class Qwen3ASRThinkerForConditionalGeneration(Qwen3OmniMoeThinkerForConditionalGeneration):
    _can_record_outputs = {
        "hidden_states": Qwen3ASRThinkerTextDecoderLayer,
        "attentions": Qwen3ASRTextAttention,
    }

    def __init__(self, config):
        super().__init__(config)
        if "forced_aligner" in config.model_type:
            self.lm_head = nn.Linear(config.text_config.hidden_size, config.classify_num, bias=False)
        ###
        if getattr(config.text_config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.get_input_embeddings().weight
        ###
        self.pad_token_id = (
            self.config.text_config.pad_token_id
            if self.config.text_config.pad_token_id is not None
            else -1
        )        
        self.post_init()
        del self.visual
        del self.spatial_merge_size 
        del self.num_experts
        del self.num_experts_per_tok 
        del self.router_aux_loss_coef

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: Optional[torch.LongTensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
    ):
        """
        Encodes audios into continuous embeddings that can be forwarded to the language model.

        Args:
            input_features (`torch.FloatTensor`):
                The tensors corresponding to the input audios.
            feature_attention_mask (`torch.LongTensor`, *optional*):
                Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:
            audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
                The length of feature shape of each audio in LLM.
        """
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None
        feature_lens = audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
    
        # audio encoder do not support batch inference to keep precision
        audio_features = []
        for input_feature, feature_len in zip(input_features, feature_lens):
            audio_output = self.audio_tower(
                input_feature[:, :feature_len],
                feature_lens=feature_len.unsqueeze(0),
            )
            audio_feature = audio_output.last_hidden_state
            audio_features.append(audio_feature)
        audio_features = torch.cat(audio_features, dim=0)

        return audio_features

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithDeepstackFeatures:
        raise ValueError("Not needed.")

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithDeepstackFeatures:
        raise ValueError("Not needed.")

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_audio_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(self.config.audio_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            ).all(-1)
        else:
            special_audio_mask = input_ids == self.config.audio_token_id

        special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        return special_audio_mask

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids=None,
        input_features=None,
        attention_mask=None,
        feature_attention_mask=None,
        audio_feature_lengths=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        rope_deltas=None,
        labels=None,
        use_cache=None,
        cache_position=None,
        **kwargs,
    ) -> Union[tuple, Qwen3ASRThinkerCausalLMOutputWithPast]:
        r"""
        feature_attention_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`, *optional*):
            Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
            The length of feature shape of each audio in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # 2. Merge text, audios
        if input_features is not None:
            audio_features = self.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
            )
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None

        ### Changed the following in order to pass test_generate_from_inputs_embeds_with_static_cache
        ### old
        #if attention_mask is not None and position_ids is None:
        #    if (
        #        cache_position is None
        #        or (cache_position is not None and cache_position[0] == 0)
        #        or self.rope_deltas is None
        #    ):
        #        delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
        #        position_ids, rope_deltas = self.get_rope_index(
        #            attention_mask,
        #        )
        #        rope_deltas = rope_deltas - delta0
        #        self.rope_deltas = rope_deltas
        #    else:
        #        batch_size, seq_length = input_ids.shape
        #        delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
        #        position_ids = torch.arange(seq_length, device=input_ids.device)
        #        position_ids = position_ids.view(1, -1).expand(batch_size, -1)
        #        position_ids = position_ids.add(delta)
        #        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        ### new
        # Determine batch and sequence length early
        batch_size, seq_length = inputs_embeds.shape[:2]

        # -------------------------------------------------
        # 1. Build cache_position if missing
        # -------------------------------------------------
        if cache_position is None:
            past_seen = (
                past_key_values.get_seq_length()
                if past_key_values is not None
                else 0
            )
            cache_position = torch.arange(
                past_seen,
                past_seen + seq_length,
                device=inputs_embeds.device,
            )

        # -------------------------------------------------
        # 2. Build position_ids only if not provided
        # -------------------------------------------------
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(
                3, batch_size, -1
            )

        # -------------------------------------------------
        # 3. Compute rope_deltas ONLY during prefill
        # -------------------------------------------------
        if (
            self.rope_deltas is None
            and attention_mask is not None
            and attention_mask.dim() == 2
            and cache_position is not None
            and cache_position[0] == 0
        ):
            max_position = cache_position[-1]
            valid_tokens = attention_mask.sum(dim=-1)
            rope_deltas = (max_position + 1 - valid_tokens).unsqueeze(-1)
            self.rope_deltas = rope_deltas

        # -------------------------------------------------
        # 4. Apply rope delta if it exists
        # -------------------------------------------------
        if self.rope_deltas is not None:
            position_ids = position_ids + self.rope_deltas.unsqueeze(0)
        ###
        
        batch_size, seq_length = inputs_embeds.shape[:2]

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.get_text_config().vocab_size
            )

        return Qwen3ASRThinkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        input_features=None,
        feature_attention_mask=None,
        **kwargs,
    ):
        model_inputs = GenerationMixin.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            **kwargs,
        )

        model_inputs["position_ids"] = None

        if cache_position is not None and cache_position[0] != 0:
            model_inputs["input_features"] = None

        return model_inputs


@auto_docstring
class Qwen3ASRThinkerTextPreTrainedModel(PreTrainedModel):
    config = Qwen3ASRConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3ASRThinkerTextDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = False  # MoE models don't work with torch.compile (`torch.where(condition)` not supported)
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen3ASRThinkerTextDecoderLayer,
        "attentions": Qwen3ASRTextAttention,
    }
    config_class = Qwen3ASRConfig


class Qwen3ASRForConditionalGeneration(Qwen3ASRPreTrainedModel, GenerationMixin):
    config_class = Qwen3ASRConfig
    base_model_prefix = "thinker"

    def __init__(self, config: Qwen3ASRConfig):
        super().__init__(config)
        self.config = config

        self.thinker = Qwen3ASRThinkerForConditionalGeneration._from_config(config.thinker_config)
        self.post_init()
    
    def get_support_languages(self):
        return self.config.support_languages

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: int = 4096,
        eos_token_id: int | list[int] = [151645, 151643],
        **kwargs,
    ):
        shared_kwargs = {}
        thinker_kwargs = {
            "max_new_tokens": max_new_tokens,
            "eos_token_id": eos_token_id,
        }

        for key, value in kwargs.items():
            # Process special input values
            if key == "feature_attention_mask":
                thinker_kwargs[key] = value
            elif key in ("input_features", "attention_mask"):
                thinker_kwargs[key] = value
            # Put other key to shared kwargs
            else:
                shared_kwargs[key] = value

        # Merge kwargs
        for key, value in shared_kwargs.items():
            if key not in thinker_kwargs:
                thinker_kwargs[key] = value
            
        thinker_result = self.thinker.generate(input_ids=input_ids, **thinker_kwargs)

        return thinker_result

    ### added the following in order to pass tests
    @property
    def base_model(self):
        return getattr(self, self.base_model_prefix)

    def get_input_embeddings(self):
        return self.thinker.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.thinker.set_input_embeddings(value)

    def forward(
        self,
        input_ids=None,
        input_features=None,
        attention_mask=None,
        feature_attention_mask=None,
        audio_feature_lengths=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        rope_deltas=None,
        labels=None,
        use_cache=None,
        cache_position=None,
        **kwargs,
    ):
        return self.thinker(
            input_ids=input_ids,
            input_features=input_features,
            attention_mask=attention_mask,
            feature_attention_mask=feature_attention_mask,
            audio_feature_lengths=audio_feature_lengths,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            rope_deltas=rope_deltas,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
    ###


__all__ = [
    "Qwen3ASRAudioEncoderConfig",
    "Qwen3ASRThinkerConfig",
    "Qwen3ASRConfig",
    "Qwen3ASRProcessor",
    "Qwen3ASRForConditionalGeneration",
    "Qwen3ASRThinkerTextModel",
    "Qwen3ASRThinkerForConditionalGeneration",
    "Qwen3ASRPreTrainedModel",
    "Qwen3ASRPreTrainedModelForConditionalGeneration",
    "Qwen3ASRThinkerTextPreTrainedModel",
]