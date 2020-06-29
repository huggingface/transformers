"""Documentation string."""


import unittest

from transformers import PegasusConfig, is_tf_available

from .test_configuration_common import ConfigTester
from .test_modeling_tf_common import TFModelTesterMixin, ids_tensor
from .utils import require_tf, slow


if is_tf_available():
    import tensorflow as tf
    from transformers import TFPegasusModel


class TFPegasusModelTester(object):
    def __init__(
            self,
            parent,
            batch_size=13,
            is_training=True,
            vocab_size=99,
            seq_length=14,
            hidden_size=32,
            ffn_dim=128,
            num_heads=4,
            num_hidden_layers=5,
            dropout=0.1,
            pad_token_id=0,
            eos_token_id=1,
            scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        config = PegasusConfig(
            vocab_size=self.vocab_size,
            max_input_len=self.seq_length,
            max_target_len=self.seq_length,
            max_decode_len=self.seq_length,
            hidden_size=self.hidden_size,
            ffn_dim=self.ffn_dim,
            num_heads=self.num_heads,
            num_encoder_layers=self.num_hidden_layers,
            num_decoder_layers=self.num_hidden_layers,
            dropout=self.dropout,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )

        return (config, input_ids)

    def create_and_check_pegasus_model(self, config, input_ids):
        model = TFPegasusModel(config=config)
        inputs = {
            "inputs": input_ids,
        }
        decoder_output, decoder_past, encoder_output = model(inputs)

        result = {
            "encoder_output": encoder_output.numpy(),
            "decoder_past": decoder_past,
            "decoder_output": decoder_output.numpy(),
        }
        self.parent.assertListEqual(
            list(result["encoder_output"].shape), [self.batch_size, self.seq_length, self.hidden_size]
        )
        self.parent.assertListEqual(
            list(result["decoder_output"].shape), [self.batch_size, self.seq_length, self.hidden_size]
        )
        # self.parent.assertEqual(len(decoder_past), 2)
        # # decoder_past[0] should correspond to encoder output
        # self.parent.assertTrue(tf.reduce_all(tf.math.equal(decoder_past[0][0], encoder_output)))
        # # There should be `num_layers` key value embeddings stored in decoder_past[1]
        # self.parent.assertEqual(len(decoder_past[1]), config.num_layers)
        # # There should be a self attn key, a self attn value, a cross attn key and a cross attn value stored in each decoder_past[1] tuple
        # self.parent.assertEqual(len(decoder_past[1][0]), 4)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids) = config_and_inputs
        inputs_dict = {
            "inputs": input_ids,
        }
        return config, inputs_dict


@require_tf
class TFPegasusModelTest(TFModelTesterMixin, unittest.TestCase):

    is_encoder_decoder = True
    all_model_classes = (TFPegasusModel,) if is_tf_available() else ()

    def setUp(self):
        self.model_tester = TFPegasusModelTest.TFPegasusModelTester(self)
        self.config_tester = ConfigTester(self, config_class=PegasusConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_pegasus_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_pegasus_model(*config_and_inputs)

