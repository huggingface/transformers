from transformers import PretrainedConfig


class HiFiGANConfig(PretrainedConfig):
    model_type = "hifigan"

    def __init__(
        self,
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[4, 4, 4, 4],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[8, 8, 8, 8],
        model_in_dim=80,
        sampling_rate=16000,
        initializer_range=0.01,
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
