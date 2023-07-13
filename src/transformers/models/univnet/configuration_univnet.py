from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


# TODO: Add UNIVNET_PRETRAINED_CONFIG_ARCHIVE_MAP?
# Should be a dict mapping model ids to huggingface hub config.json file


class UnivNetGanConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`UnivNetGan`]. It is used to instantiate
    a UnivNet vocoder model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the UnivNet
    [microsoft/speecht5_hifigan](https://huggingface.co/microsoft/speecht5_hifigan) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        model_in_dim (`int`, *optional*, defaults to 80):
            The number of frequency bins in the input log-mel spectrogram.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the output audio will be generated, expressed in hertz (Hz).
        upsample_initial_channel (`int`, *optional*, defaults to 512):
            The number of input channels into the upsampling network.
        upsample_rates (`Tuple[int]` or `List[int]`, *optional*, defaults to `[4, 4, 4, 4]`):
            A tuple of integers defining the stride of each 1D convolutional layer in the upsampling network. The
            length of *upsample_rates* defines the number of convolutional layers and has to match the length of
            *upsample_kernel_sizes*.
        upsample_kernel_sizes (`Tuple[int]` or `List[int]`, *optional*, defaults to `[8, 8, 8, 8]`):
            A tuple of integers defining the kernel size of each 1D convolutional layer in the upsampling network. The
            length of *upsample_kernel_sizes* defines the number of convolutional layers and has to match the length of
            *upsample_rates*.
        resblock_kernel_sizes (`Tuple[int]` or `List[int]`, *optional*, defaults to `[3, 7, 11]`):
            A tuple of integers defining the kernel sizes of the 1D convolutional layers in the multi-receptive field
            fusion (MRF) module.
        resblock_dilation_sizes (`Tuple[Tuple[int]]` or `List[List[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`):
            A nested tuple of integers defining the dilation rates of the dilated 1D convolutional layers in the
            multi-receptive field fusion (MRF) module.
        initializer_range (`float`, *optional*, defaults to 0.01):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        leaky_relu_slope (`float`, *optional*, defaults to 0.1):
            The angle of the negative slope used by the leaky ReLU activation.
        normalize_before (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the spectrogram before vocoding using the vocoder's learned mean and variance.
    """
    model_type = "univnetgan"

    def __init__(
        self,
        model_in_dim=64,
        channel_size=32,
        num_mel_channels=100,
        resblock_kernel_sizes=[3, 3, 3],
        resblock_stride_sizes = [8, 8, 4],
        resblock_dilation_sizes=[[1, 3, 9, 27], [1, 3, 9, 27], [1, 3, 9, 27]],
        kernel_predictor_hidden_channels=64,
        kernel_predictor_conv_size=3,
        kernel_predictor_dropout=0.0,
        initializer_range=0.01,
        leaky_relu_slope=0.2,
        **kwargs,
    ):
        self.model_in_dim = model_in_dim
        self.channel_size = channel_size
        self.num_mel_channels = num_mel_channels
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_stride_sizes = resblock_stride_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.kernel_predictor_hidden_channels = kernel_predictor_hidden_channels
        self.kernel_predictor_conv_size = kernel_predictor_conv_size
        self.kernel_predictor_dropout = kernel_predictor_dropout
        self.initializer_range = initializer_range
        self.leaky_relu_slope = leaky_relu_slope
        super().__init__(**kwargs)