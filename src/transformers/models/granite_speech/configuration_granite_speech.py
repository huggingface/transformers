from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING, AutoConfig


class GraniteSpeechEncoderConfig(PretrainedConfig):
    model_type = "granite_speech_encoder"

    def __init__(
        self,
        input_dim=160,
        num_layers=10,
        hidden_dim=1024,
        feedforward_mult=4,
        num_heads=8,
        dim_head=128,
        output_dim=42,
        context_size=200,
        dropout=0.1,
        conv_kernel_size=15,
        conv_expansion_factor=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.feedforward_mult = feedforward_mult
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.output_dim = output_dim
        self.context_size = context_size
        self.dropout = dropout
        self.conv_kernel_size = conv_kernel_size
        self.conv_expansion_factor = conv_expansion_factor


## adapted from transformers.models.blip.configuration_blip_2.Blip2VisionConfig
class GraniteSpeechProjectorConfig(PretrainedConfig):
    model_type = "granite_speech_qformer"

    def __init__(
        self,
        llm_dim=4096,
        downsample_rate=5,
        window_size=15,
        hidden_size=1024,
        num_attention_heads=16,
        intermediate_size=4096,
        num_hidden_layers=2,
        encoder_hidden_size=1024,
        cross_attention_frequency=1,
        max_position_embeddings=2048,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_qformer_text_input=False,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.cross_attention_frequency = cross_attention_frequency
        self.encoder_hidden_size = encoder_hidden_size
        self.use_qformer_text_input = use_qformer_text_input
        self.downsample_rate = downsample_rate
        self.window_size = window_size
        self.llm_dim = llm_dim


class GraniteSpeechConfig(PretrainedConfig):
    model_type = "granite_speech"
    sub_configs = {
        "text_config": AutoConfig,
        "encoder_config": GraniteSpeechEncoderConfig,
        "projector_config": GraniteSpeechProjectorConfig,
    }

    def __init__(
        self,
        encoder_config=None,
        text_config=None,
        projector_config=None,
        audio_token_index=49155,
        initializer_range=0.02,
        has_lora_adapter=True,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "granite"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["granite"]()

        if isinstance(projector_config, dict):
            # TODO - In the future, we should make this generic.
            projector_config = GraniteSpeechProjectorConfig(**projector_config)
        elif projector_config is None:
            projector_config = GraniteSpeechProjectorConfig()

        if not isinstance(encoder_config, GraniteSpeechEncoderConfig):
            encoder_config = {} if encoder_config is None else encoder_config
            encoder_config = GraniteSpeechEncoderConfig(**encoder_config)

        self.text_config = text_config
        self.encoder_config = encoder_config
        self.projector_config = projector_config
        self.audio_token_index = audio_token_index
        self.initializer_range = initializer_range
        self.has_lora_adapter = has_lora_adapter
        super().__init__(**kwargs)


__all__ = ["GraniteSpeechEncoderConfig", "GraniteSpeechProjectorConfig", "GraniteSpeechConfig"]
