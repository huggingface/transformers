from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import AutoConfig
from transformers.models.blip_2.configuration_blip_2 import Blip2QFormerConfig
from ..auto import CONFIG_MAPPING, AutoConfig

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


class GraniteSpeechProjectorConfig(Blip2QFormerConfig):
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
        use_qformer_text_input=False,
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            encoder_hidden_size=encoder_hidden_size,
            cross_attention_frequency=cross_attention_frequency,
            max_position_embeddings=max_position_embeddings,
            use_qformer_text_input=use_qformer_text_input,
            **kwargs,
        )

        self.downsample_rate = downsample_rate
        self.window_size = window_size
        self.llm_dim = llm_dim


class GraniteSpeechConfig(PretrainedConfig):
    model_type = "granite_speech"
    # TODO - Probably should consolidate these into a single config
    sub_configs = {
        "llm_config": AutoConfig,
        "encoder_config": GraniteSpeechEncoderConfig,
        "projector_config": GraniteSpeechProjectorConfig,
    }

    def __init__(
        self,
        encoder_config=None,
        llm_config=None,
        projector_config=None,
        audio_token_index=49155,
        tie_word_embeddings=True,
        **kwargs,
    ):
        if isinstance(llm_config, dict):
            llm_config["model_type"] = llm_config["model_type"] if "model_type" in llm_config else "granite"
            llm_config = CONFIG_MAPPING[llm_config["model_type"]](**llm_config)
        elif llm_config is None:
            llm_config = CONFIG_MAPPING["granite"]()

        if isinstance(projector_config, dict):
            # TODO - Make this generic after blip2qformer is moved out to its own model dir.
            if projector_config["model_type"] != "blip_2_qformer":
                raise ValueError("Granite speech currently requires blip2 qformer as its encoder!")
            projector_config = Blip2QFormerConfig(**projector_config)
        elif projector_config is None:
            projector_config = Blip2QFormerConfig()

        if not isinstance(encoder_config, GraniteSpeechEncoderConfig):
            encoder_config = dict() if encoder_config is None else encoder_config
            encoder_config = GraniteSpeechEncoderConfig(**encoder_config)

        self.llm_config = llm_config
        self.encoder_config = encoder_config
        self.projector_config = projector_config
        self.audio_token_index = audio_token_index
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

__all__ = ["GraniteSpeechEncoderConfig", "GraniteSpeechProjectorConfig", "GraniteSpeechConfig"]