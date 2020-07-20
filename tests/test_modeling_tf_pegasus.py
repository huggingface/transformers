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


@require_tf
class TFPegasusModelTest(unittest.TestCase):

    is_encoder_decoder = True
    all_model_classes = (TFPegasusModel,) if is_tf_available() else ()

    def setUp(self):
        pass

    # TODO: load checkpoint and test
    # TODO: refactor to follow transformers' standard testing pipeline
    def test_pegasus_model(self, model_dir, spm_model):
        self.assertTrue(tf.compat.v1.train.checkpoint_exists(model_dir))
        vocab_size = 96000 + 103
        hidden_size = 1024
        filter_size = 4096
        num_heads = 16
        num_encoder_layers = 16
        num_decoder_layers = 16
        label_smoothing = 0.0
        dropout = 0.1
        beam_size = 1

        with tf.Graph().as_default() as graph:
            with tf.compat.v1.Session() as sess:
                model = transformer.TransformerEncoderDecoderModel(
                    vocab_size=vocab_size,
                    hidden_size=hidden_size,
                    filter_size=filter_size,
                    num_heads=num_heads,
                    num_encoder_layers=num_encoder_layers,
                    num_decoder_layers=num_decoder_layers,
                    label_smoothing=label_smoothing,
                    dropout=dropout
                )

                # run the model to build all variables (but not initialized yet)
                loss, outputs = model(
                    {
                        "inputs": tf.ones((2, 7), tf.int64),
                        "targets": tf.ones((2, 5), tf.int64)
                    }, True)
                self.assertEqual(loss.shape, [])
                self.assertEqual(outputs["logits"].shape, [2, 5, vocab_size])

                # create assignment map
                ignore_name = ["Adafactor", "global_step"]
                var_list = tf.compat.v1.global_variables(scope=None)
                ckpt_var_list = tf.train.list_variables(model_dir)
                ckpt_var_list = [var for var in ckpt_var_list if not any(ign in var[0] for ign in ignore_name)]
                new_var_name_dict = {var.name: var for var in var_list}
                assignment_map = {}
                for var in ckpt_var_list:
                    old_var_name = var[0]
                    new_var_name = var[0] + ":0"
                    assert new_var_name in new_var_name_dict
                    assignment_map[old_var_name] = new_var_name_dict[new_var_name]

                # define the initialization (but not intialized until global_variables_initializer is called)
                tf.compat.v1.train.init_from_checkpoint(
                    model_dir, assignment_map
                )

                # check running
                raw_input_str = ("To ensure a smooth flow of bank resolutions to the necessary signatories, "
                                 "I am requesting that Enron Treasury first route the bank resolutions to Angela Davis "
                                 "(EWS Legal) to be initialed before being routed to John Lavorato or Louise Kitchen.\n"
                                 "If you have any questions please call me at 3-6544."
                                 "Thank you for your attention to this matter.")
                raw_target_str = ("Treasury Bank Resolutions")

                input_str = tf.compat.v1.placeholder(tf.string, shape=[1, ], name=None)
                target_str = tf.compat.v1.placeholder(tf.string, shape=[1, ], name=None)

                # tokenization
                input_ids = public_parsing_ops.encode(input_str, 512, spm_model, encoder_type="sentencepiece")
                target_ids = public_parsing_ops.encode(target_str, 32, spm_model, encoder_type="sentencepiece")

                input_ids = tf.reshape(input_ids, [1, 512])
                target_ids = tf.reshape(target_ids, [1, 32])

                output_ids = model.predict(
                    {
                        "inputs": input_ids,
                        "targets": target_ids,
                    }, 32, beam_size)
                self.assertEqual(output_ids["outputs"].shape, [1, 32])

                # decode to str
                output_str = public_parsing_ops.decode(output_ids["outputs"], spm_model, encoder_type="sentencepiece")

                sess.run(tf.compat.v1.global_variables_initializer())
                print(sess.run(output_str, feed_dict={input_str: [raw_input_str], target_str: [raw_target_str]}))
