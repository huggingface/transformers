from collections import OrderedDict
from collections.abc import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig


class FirefliesConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`FirefliesModel`]. It is used to instantiate a
    Fireflies model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read more in
    the [documentation](https://huggingface.co/docs/transformers/main_classes/configuration).

    Example:

    ```python
    >>> from transformers import FirefliesConfig, FirefliesModel

    >>> config = FirefliesConfig()
    >>> model = FirefliesModel(config)
    ```

    [Arynz/Fireflies300M](https://huggingface.co/Arynz/Fireflies300M)
    """

    model_type = "fireflies"

    def __init__(
        self,
        vocab_size=32000,
        d_model=1024,
        n_heads=8,
        num_layers=12,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers


class FirefliesOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            {
                "input_ids": {0: "batch", 1: "sequence"},
            }
        )


__all__ = ["FirefliesConfig", "FirefliesOnnxConfig"]
