from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


UNIVNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "dg845/univnet-dev": "https://huggingface.co/dg845/univnet-dev/resolve/main/config.json",
}


class UnivNetGanConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`UnivNetGan`]. It is used to instantiate a UnivNet
    vocoder model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the UnivNet
    [dg845/univnet-dev](https://huggingface.co/dg845/univnet-dev) architecture, which corresponds to the 'c32'
    architecture in [mindslab-ai/univnet](https://github.com/mindslab-ai/univnet/blob/master/config/default_c32.yaml).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        model_in_channels (`int`, *optional*, defaults to 64):
            The number of input channels for the UnivNet residual network. This corresponds to dimension 1 of the input
            noise sequence to the vocoder, which should have shape (batch_size, model_in_dim, seq_length).
        model_hidden_channels ('int', *optional*, defaults to 32):
            The number of hidden channels of each residual block in the UnivNet residual network.
        num_mel_channels ('int', *optional*, defaults to 100):
            The number of frequency bins in the conditioning log-mel spectrogram.
        upsample_rates (`Tuple[int]` or `List[int]`, *optional*, defaults to `[1]`):
            The UnivNet vocoder doesn't upsample the input spectrogram and this is set to a dummy value of `[1]` to be
            compatible with vocoders which do upsample the input spectrogram (e.g. [`SpeechT5HifiGan`]).
        resblock_kernel_sizes (`Tuple[int]` or `List[int]`, *optional*, defaults to `[3, 3, 3]`):
            A tuple of integers defining the kernel sizes of the 1D convolutional layers in the UnivNet residual
            network. The length of `resblock_kernel_sizes` defines the number of resnet blocks and should match that
            of `resblock_stride_sizes` and `resblock_dilation_sizes`.
        resblock_stride_sizes (`Tuple[int]` or `List[int]`, *optional*, defaults to `[8, 8, 4]`):
            A tuple of integers defining the stride sizes of the 1D convolutional layers in the UnivNet residual
            network. The length of `resblock_stride_sizes` should match that of `resblock_kernel_sizes` and
            `resblock_dilation_sizes`.
        resblock_dilation_sizes (`Tuple[Tuple[int]]` or `List[List[int]]`, *optional*, defaults to `[[1, 3, 9, 27], [1, 3, 9, 27], [1, 3, 9, 27]]`):
            A nested tuple of integers defining the dilation rates of the dilated 1D convolutional layers in the
            UnivNet residual network. The length of `resblock_dilation_sizes` should match that of
            `resblock_kernel_sizes` and `resblock_stride_sizes`. The length of each nested list in
            `resblock_dilation_sizes` defines the number of convolutional layers per resnet block.
        kernel_predictor_num_blocks (`int`, *optional*, defaults to 3):
            The number of residual blocks in the kernel predictor network, which calculates the kernel and bias for
            each location variable convolution layer in the UnivNet residual network.
        kernel_predictor_hidden_channels (`int`, *optional*, defaults to 64):
            The number of hidden channels for each residual block in the kernel predictor network.
        kernel_predictor_conv_size (`int`, *optional*, defaults to 3):
            The kernel size of each 1D convolutional layer in the kernel predictor network.
        kernel_predictor_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for each residual block in the kernel predictor network.
        initializer_range (`float`, *optional*, defaults to 0.01):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        leaky_relu_slope (`float`, *optional*, defaults to 0.2):
            The angle of the negative slope used by the leaky ReLU activation.

    Example:

    ```python
    >>> from transformers import UnivNetHifiGan, UnivNetGanConfig

    >>> # Initializing a Tortoise TTS style configuration
    >>> configuration = UnivNetGanConfig()

    >>> # Initializing a model (with random weights) from the Tortoise TTS style configuration
    >>> model = UnivNetGan(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """
    model_type = "univnet"

    def __init__(
        self,
        model_in_channels=64,
        model_hidden_channels=32,
        num_mel_channels=100,
        upsample_rates=[1],
        resblock_kernel_sizes=[3, 3, 3],
        resblock_stride_sizes=[8, 8, 4],
        resblock_dilation_sizes=[[1, 3, 9, 27], [1, 3, 9, 27], [1, 3, 9, 27]],
        kernel_predictor_num_blocks=3,
        kernel_predictor_hidden_channels=64,
        kernel_predictor_conv_size=3,
        kernel_predictor_dropout=0.0,
        initializer_range=0.01,
        leaky_relu_slope=0.2,
        **kwargs,
    ):
        self.model_in_channels = model_in_channels
        self.model_hidden_channels = model_hidden_channels
        self.num_mel_channels = num_mel_channels
        self.upsample_rates = upsample_rates
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_stride_sizes = resblock_stride_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.kernel_predictor_num_blocks = kernel_predictor_num_blocks
        self.kernel_predictor_hidden_channels = kernel_predictor_hidden_channels
        self.kernel_predictor_conv_size = kernel_predictor_conv_size
        self.kernel_predictor_dropout = kernel_predictor_dropout
        self.initializer_range = initializer_range
        self.leaky_relu_slope = leaky_relu_slope
        super().__init__(**kwargs)
