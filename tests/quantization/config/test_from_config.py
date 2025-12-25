import pytest

from transformers import AutoConfig, AutoModel


def test_quantization_from_config_raises():
    config = AutoConfig.from_pretrained("gpt2")
    config.quantization_config = {"quant_method": "fp8"}

    with pytest.raises(
        NotImplementedError,
        match="Quantization via",
    ):
        AutoModel.from_config(config)
