from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from collections import OrderedDict
from collections.abc import Mapping

class FirefliesConfig(PretrainedConfig):
    model_type = "fireflies"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
        hidden_act="gelu",
        max_position_embeddings=512,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
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
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

class FirefliesOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict({
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
        })
