from transformers import PretrainedConfig


class HiFiGANConfig(PretrainedConfig):
    model_type = "hifigan"

    def __init__(
        self,
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[8, 8, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[16, 16, 4, 4],
        model_in_dim=80,
        sampling_rate=22050,
        initializer_range=0.02,
        leaky_relu_slope=0.1,
        **kwargs,
    ):
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.model_in_dim = model_in_dim
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.sampling_rate = sampling_rate
        self.initializer_range = initializer_range
        self.leaky_relu_slope = leaky_relu_slope
        super().__init__(**kwargs)


class CodeHiFiGANConfig(PretrainedConfig):
    def __init__(
        self,
        upsample_rates=[5, 4, 4, 2, 2],
        upsample_kernel_sizes=[11, 8, 8, 4, 4],
        upsample_initial_channel=512,
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        num_embeddings=1000,
        embedding_dim=128,
        model_in_dim=128,
        f0=False,
        f0_quant_num_bin=0,
        duration_predictor=True,
        encoder_embed_dim=128,
        variance_predictor_hidden_dim=128,
        variance_predictor_kernel_size=3,
        variance_predictor_dropout=0.5,
        sampling_rate=16000,
        multispeaker=False,
        num_speakers=200,
        speaker_embedding=False,
        speaker_embedding_dim=256,
        initializer_range=0.02,
        duration_predictor_activation="relu",
        leaky_relu_slope=0.1,
        **kwargs,
    ):
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.model_in_dim = model_in_dim
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.sampling_rate = sampling_rate
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.f0 = f0
        self.f0_quant_num_bin = f0_quant_num_bin
        self.duration_predictor = duration_predictor
        self.encoder_embed_dim = encoder_embed_dim
        self.variance_predictor_hidden_dim = variance_predictor_hidden_dim
        self.variance_predictor_kernel_size = variance_predictor_kernel_size
        self.variance_predictor_dropout = variance_predictor_dropout
        self.multispeaker = multispeaker
        self.num_speakers = num_speakers
        self.speaker_embedding = speaker_embedding
        self.speaker_embedding_dim = speaker_embedding_dim
        self.initializer_range = initializer_range
        self.duration_predictor_activation = duration_predictor_activation
        self.leaky_relu_slope = leaky_relu_slope
        super().__init__(**kwargs)
