"""AudioFlamingo3 model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


# Model Constants
IGNORE_INDEX = -100
SENTINEL_TOKEN = "<vila/sentinel>"

MEDIA_TOKENS = {
    "sound": "<sound>",
}

NUM_EXTRA_TOKENS = 10


class LlavaConfig(PretrainedConfig):
    model_type = "llava"

    def __init__(
        self,
        llm_cfg=None,
        sound_tower_cfg=None,
        sound_mm_projector_cfg=None,
        architectures=None,
        resume_path=None,
        hidden_size=None,
        sound_hidden_size=None,
        sound_encoder: str = '{"_target_": "llava.encoders.BasicSoundEncoder"}',
        **kwargs,
    ):
        super().__init__()
        self.architectures = architectures
        self.llm_cfg = llm_cfg
        self.sound_tower_cfg = sound_tower_cfg
        self.sound_mm_projector_cfg = sound_mm_projector_cfg
        self.resume_path = resume_path
        self.hidden_size = hidden_size
        self.sound_hidden_size = sound_hidden_size
        self.sound_encoder = '{"_target_": "llava.encoders.BasicSoundEncoder"}'


class SoundMultimodalProjectorConfig(PretrainedConfig):
    model_type = "sound_mm_projector"

    def __init__(self, sound_mm_projector_type: str = None, **kwargs):
        super().__init__()
        self.sound_mm_projector_type = sound_mm_projector_type

# -------------------------------------------------------------------------------------------------

class AudioFlamingo3EncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`AudioFlamingo3Encoder`]. It is used to instantiate a
    AudioFlamingo3 audio encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the audio encoder of the AudioFlamingo3
    architecture.

    e.g. [NVIDIA/AudioFlamingo3](https://huggingface.co/NVIDIA/AudioFlamingo3)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_mel_bins (`int`, *optional*, defaults to 128):
            Number of mel features used per input features. Should correspond to the value used in the
            `AudioFlamingo3Processor` class.
        encoder_layers (`int`, *optional*, defaults to 32):
            Number of encoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads for each attention layer in the Transformer encoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 5120):
            Dimensionality of the "intermediate" (often named feed-forward) layer in encoder.
        encoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://huggingface.co/papers/1909.11556)
            for more details.
        d_model (`int`, *optional*, defaults to 1280):
            Dimensionality of the layers.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Scale embeddings by diving by sqrt(d_model).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        max_source_positions (`int`, *optional*, defaults to 1500):
            The maximum sequence length of log-mel filter-bank features that this model might ever be used with.

    Example:

    ```python
    >>> from transformers import AudioFlamingo3EncoderConfig, AudioFlamingo3Encoder

    >>> # Initializing a AudioFlamingo3EncoderConfig
    >>> configuration = AudioFlamingo3EncoderConfig()

    >>> # Initializing a AudioFlamingo3Encoder (with random weights)
    >>> model = AudioFlamingo3Encoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "audioflamingo3_encoder"

    def __init__(
        self,
        num_mel_bins=128,
        encoder_layers=32,
        encoder_attention_heads=20,
        encoder_ffn_dim=5120,
        encoder_layerdrop=0.0,
        d_model=1280,
        dropout=0.0,
        attention_dropout=0.0,
        activation_function="gelu",
        activation_dropout=0.0,
        scale_embedding=False,
        initializer_range=0.02,
        max_source_positions=1500,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.encoder_layerdrop = encoder_layerdrop
        self.num_hidden_layers = encoder_layers
        self.initializer_range = initializer_range
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.max_source_positions = max_source_positions


class AudioFlamingo3Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`AudioFlamingo3ForConditionalGeneration`]. It is used to instantiate an
    AudioFlamingo3 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the AudioFlamingo3.

    e.g. [NVIDIA/AudioFlamingo3-7B](https://huggingface.co/NVIDIA/AudioFlamingo3-7B)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        audio_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `CLIPVisionConfig`):
            The config object or dictionary of the audio backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
            The config object or dictionary of the text backbone.
        audio_token_index (`int`, *optional*, defaults to 151646):
            The image token index to encode the image prompt.

    Example:

    ```python
    >>> from transformers import AudioFlamingo3ForConditionalGeneration, AudioFlamingo3Config, AudioFlamingo3EncoderConfig

    >>> # Initializing a AudioFlamingo3Encoder config
    >>> audio_config = AudioFlamingo3EncoderConfig()

    >>> # Initializing a AudioFlamingo3 configuration
    >>> configuration = AudioFlamingo3Config(audio_config, text_config)

    >>> # Initializing a model from the AudioFlamingo3 style configuration
    >>> model = AudioFlamingo3ForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "audioflamingo3"
    attribute_map = {
        "audio_token_id": "audio_token_index",
    }
    sub_configs = {"text_config": AutoConfig, "audio_config": AutoConfig}

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        audio_token_index=151646,
        **kwargs,
    ):
        self.audio_token_index = audio_token_index

        if isinstance(audio_config, dict):
            audio_config["model_type"] = audio_config.get("model_type", "audioflamingo3_encoder")
            audio_config = CONFIG_MAPPING[audio_config["model_type"]](**audio_config)
        elif audio_config is None:
            audio_config = CONFIG_MAPPING["audioflamingo3_encoder"](
                d_model=1280,
                encoder_attention_heads=20,
                encoder_ffn_dim=5120,
                encoder_layerdrop=0.0,
                encoder_layers=32,
                num_mel_bins=128,
                max_source_positions=1500,
                scale_embedding=False,
                activation_function="gelu",
            )

        self.audio_config = audio_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "qwen2")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["qwen2"]()

        self.text_config = text_config

        super().__init__(**kwargs)


__all__ = ["AudioFlamingo3Config", "AudioFlamingo3EncoderConfig"]
