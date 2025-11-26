import unittest

from transformers import LwDetrViTConfig, is_torch_available
from transformers.testing_utils import require_torch

from ...test_backbone_common import BackboneTesterMixin
from ...test_modeling_common import floats_tensor, ids_tensor


if is_torch_available():
    from transformers import LwDetrViTBackbone


class LwDetrVitModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        num_labels=3,
        num_channels=3,
        use_labels=True,
        is_training=True,
        image_size=256,
        hidden_size=16,
        num_hidden_layers=6,
        num_attention_heads=2,
        window_block_indices=[0, 2],
        out_indices=[1, 3, 5],
        num_windows=16,
        dropout_prob=0.0,
        attn_implementation="eager",
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.num_channels = num_channels
        self.use_labels = use_labels
        self.image_size = image_size

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.window_block_indices = window_block_indices
        self.out_indices = out_indices
        self.num_windows = num_windows
        self.dropout_prob = dropout_prob
        self.attn_implementation = attn_implementation

        self.is_training = is_training

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.num_labels)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return LwDetrViTConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            window_block_indices=self.window_block_indices,
            out_indices=self.out_indices,
            num_windows=self.num_windows,
            hidden_dropout_prob=self.dropout_prob,
            attention_probs_dropout_prob=self.dropout_prob,
            attn_implementation=self.attn_implementation,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class LwDetrViTBackboneTest(BackboneTesterMixin, unittest.TestCase):
    all_model_classes = (LwDetrViTBackbone,) if is_torch_available() else ()
    has_attentions = False
    config_class = LwDetrViTConfig

    def setUp(self):
        self.model_tester = LwDetrVitModelTester(self)
