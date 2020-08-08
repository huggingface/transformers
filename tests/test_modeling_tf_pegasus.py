"""Documentation string."""


import os
import unittest

from transformers import BartForConditionalGeneration, PegasusConfig, PegasusTokenizer, is_tf_available
from transformers.file_utils import cached_property
from transformers.testing_utils import require_tf, slow, torch_device

from .test_configuration_common import ConfigTester
from .test_modeling_tf_common import TFModelTesterMixin, ids_tensor


BANK_SNIPPET = (
    "To ensure a smooth flow of bank resolutions to the necessary signatories, "
    "I am requesting that Enron Treasury first route the bank resolutions to Angela Davis "
    "(EWS Legal) to be initialed before being routed to John Lavorato or Louise Kitchen.\n"
    "If you have any questions please call me at 3-6544."
    "Thank you for your attention to this matter."
)
raw_target_str = "Treasury Bank Resolutions"  # or something close
if is_tf_available():
    import tensorflow as tf
    from transformers import TFPegasusLegacyModel, TFPegasusPretrainedModel
    from transformers.modeling_tf_pegasus import encode, decode
    from .test_modeling_bart import ModelTester
    from transformers.configuration_pegasus import PegasusConfig
    from .test_modeling_tf_common import TFModelTesterMixin, ids_tensor


class TFPegasusModelTester:
    def __init__(
        self, parent,
    ):
        self.parent = parent
        self.batch_size = 13
        self.seq_length = 7
        self.is_training = True
        self.use_labels = False
        self.vocab_size = 99
        self.hidden_size = 16
        self.num_hidden_layers = 2
        self.num_attention_heads = 4
        self.intermediate_size = 4
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 20
        self.eos_token_id = 2
        self.pad_token_id = 1
        self.bos_token_id = 0

    def prepare_config_and_inputs_for_common(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)  # .clamp(3,)

        input_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

        config = PegasusConfig(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            eos_token_id=1,
            bos_token_id=None,
            pad_token_id=0,
        )
        # inputs_dict = prepare_bart_inputs_dict(config, input_ids)
        # input_ids[:, -1] = config.eos_token_id
        return config, {"inputs": input_ids, "attention_mask": input_mask, "training": True}


class TFPegasusModelTest(TFModelTesterMixin, unittest.TestCase):
    is_encoder_decoder = True
    all_model_classes = (TFPegasusPretrainedModel,) if is_tf_available() else ()

    def setUp(self):
        self.model_tester = TFPegasusModelTester(self)
        self.config_tester = ConfigTester(self, config_class=PegasusConfig, d_model=32)


@require_tf
class IntegrationTest(unittest.TestCase):
    @cached_property
    def model(self):
        raise NotImplementedError("no s3 yet")

    # TODO: refactor to follow transformers' standard testing pipeline
    # @slow
    def test_legacy_generate(self):
        model_dir = "../pegasus/ckpt/aeslc"
        spm_model = "../pegasus/ckpt/c4.unigram.newline.10pct.96000.model"
        assert os.path.exists(model_dir)

        self.assertTrue(tf.compat.v1.train.checkpoint_exists(model_dir))
        config = PegasusConfig(vocab_size=96000 + 103, d_model=1024, num_beams=2)

        # #hidden_size = 1024
        # #filter_size = 4096
        # num_heads = 16
        # num_encoder_layers = 16
        # num_decoder_layers = 16
        # label_smoothing = 0.0
        # dropout = 0.1
        # beam_size = 8
        # beam_alpha = 0.6

        with tf.Graph().as_default() as graph:
            with tf.compat.v1.Session() as sess:
                model = TFPegasusLegacyModel(config)

                # run the model to build all variables (but not initialized yet)
                dummy_inputs = {"inputs": tf.ones((2, 7), tf.int64), "targets": tf.ones((2, 5), tf.int64)}
                loss, outputs = model(dummy_inputs, True)
                self.assertEqual(loss.shape, [])
                self.assertEqual(outputs["logits"].shape, [2, 5, config.vocab_size])

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
                tf.compat.v1.train.init_from_checkpoint(model_dir, assignment_map)

                input_str = tf.compat.v1.placeholder(tf.string, shape=[1,], name=None)
                target_str = tf.compat.v1.placeholder(tf.string, shape=[1,], name=None)

                # tokenization
                input_ids = encode(input_str, 512, spm_model, encoder_type="sentencepiece")

                target_ids = encode(target_str, 32, spm_model, encoder_type="sentencepiece")

                input_ids = tf.reshape(input_ids, [1, 512])
                target_ids = tf.reshape(target_ids, [1, 32])

                generate_outputs: dict = model.predict(
                    {"inputs": input_ids, "targets": target_ids}, 10, config.num_beams, beam_alpha=0.6
                )
                self.assertEqual(generate_outputs["outputs"].shape, [1, 10])
                self.assertEqual(generate_outputs["hypos"].shape, [1, 10])
                # decode to str
                output_str = decode(generate_outputs["outputs"], spm_model, encoder_type="sentencepiece")
                sess.run(tf.compat.v1.global_variables_initializer())
                # Run it
                feed_dict = {input_str: [BANK_SNIPPET], target_str: [raw_target_str]}
                results, emb = sess.run([output_str, model.embedded_inputs], feed_dict=feed_dict)

                (
                    after_time,
                    after_stack,
                    logits,
                    enc_input,
                    encoder_states,
                    debug_history0,
                    debug_history1,
                    attn_history0,
                ) = sess.run(
                    [
                        model.signalled,
                        model.decoder_output,
                        model.logits,
                        model.encoder_layer_input,
                        model.encoder_states,
                        model._encoder_layers[0].debug_history,
                        model._encoder_layers[1].debug_history,
                        model._encoder_layers[0]._self_attn_layer.debug_history,
                    ],
                    feed_dict=feed_dict,
                )

        print(f"Summary: {results}")
        return
        print_tensor("1. embedded", emb)
        print_tensor("2. after pos", after_time)
        print_tensor("3. 2-1", after_time - emb)
        print_tensor("4. encoder layer 0 input", enc_input)
        for k, v in attn_history0.items():
            msg = "layer0" + k
            for tensor in v:
                if tensor.shape[0] != 1:
                    print(f"skipping dummy input result")
                    continue
                print_tensor(msg, tensor)
        print_tensor("5. encoder layer 1 input", encoder_states[0])
        print_tensor("6. encoder layer 2 input", encoder_states[1])

        print_tensor("7. encoder last layer output", encoder_states[-1])

        # import ipdb; ipdb.set_trace()
        # print(f'after_time: {after_time}')
        # print(f'after_stack: {after_stack}')
        # print(f'logits: {logits}')

    def test_legacy_forward(self):
        model_dir = "../pegasus/ckpt/aeslc"
        spm_model = "../pegasus/ckpt/c4.unigram.newline.10pct.96000.model"
        assert os.path.exists(model_dir)

        self.assertTrue(tf.compat.v1.train.checkpoint_exists(model_dir))
        config = PegasusConfig(vocab_size=96000 + 103, d_model=1024, num_beams=8)

        # #hidden_size = 1024
        # #filter_size = 4096
        # num_heads = 16
        # num_encoder_layers = 16
        # num_decoder_layers = 16
        # label_smoothing = 0.0
        # dropout = 0.1
        # beam_size = 8
        # beam_alpha = 0.6

        with tf.Graph().as_default() as graph:
            with tf.compat.v1.Session() as sess:
                model = TFPegasusLegacyModel(config)

                # run the model to build all variables (but not initialized yet)
                dummy_inputs = {"inputs": tf.ones((2, 7), tf.int64), "targets": tf.ones((2, 5), tf.int64)}
                loss, outputs = model(dummy_inputs, True)
                self.assertEqual(loss.shape, [])
                self.assertEqual(outputs["logits"].shape, [2, 5, config.vocab_size])

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
                tf.compat.v1.train.init_from_checkpoint(model_dir, assignment_map)

                # check running
                raw_input_str = BANK_SNIPPET
                raw_target_str = "Treasury Bank Resolutions"  # or something close

                input_str = tf.compat.v1.placeholder(tf.string, shape=[1,], name=None)
                target_str = tf.compat.v1.placeholder(tf.string, shape=[1,], name=None)
                src_len = 512
                tgt_len = 32
                # tokenization
                input_ids = encode(input_str, src_len, spm_model, encoder_type="sentencepiece")
                target_ids = encode(target_str, tgt_len, spm_model, encoder_type="sentencepiece")

                input_ids = tf.reshape(input_ids, [1, src_len])
                target_ids = tf.reshape(target_ids, [1, tgt_len])

                model_outputs = model(
                    {"inputs": input_ids, "targets": target_ids}, False  # 32, config.num_beams, beam_alpha=0.6
                )
                # self.assertEqual(output_ids["outputs"].shape, [1, 32])
                # decode to str
                # output_str = decode(output_ids["outputs"], spm_model, encoder_type="sentencepiece")
                sess.run(tf.compat.v1.global_variables_initializer())
                # Run it
                feed_dict = {input_str: [raw_input_str], target_str: [raw_target_str]}

                (
                    results,
                    target_ids,
                    emb,
                    after_pad,
                    after_time,
                    dec_last_layer_out,
                    logits,
                    enc_input,
                    encoder_states,
                    decoder_states,
                    dec_input,
                    enc_output,
                ) = sess.run(
                    [
                        model_outputs,
                        target_ids,
                        model._emb,
                        model.after_pad,
                        model.after_time_signal,
                        model.decoder_output,
                        model.logits,
                        model.encoder_layer_input,
                        model.encoder_states,
                        model.decoder_states,
                        # model._encoder_layers[0].debug_history, model._encoder_layers[1].debug_history, model._encoder_layers[0]._self_attn_layer.debug_history,
                        model.decoder_input,
                        model.memory_context,
                    ],
                    feed_dict=feed_dict,
                )

        # print(f"Summary: {results}")
        print(f"loss: {results[0]}")
        print_tensor("1. encoder layer 0 input", enc_input)
        print_tensor("2. encoder last layer output", encoder_states[-1])
        print_tensor("3. enc output", enc_output)
        print_tensor("target_ids", target_ids)
        print_tensor("decoder after emb layer", emb)
        print_tensor("decoder after pad", after_pad)
        print_tensor("decoder after time signal", after_time)
        print_tensor("dec layer 0 input", dec_input)
        print_tensor("dec layer 1 input", decoder_states[0])
        print_tensor("decoder last layer output before layernorm", decoder_states[-1])
        print_tensor("decoder output", dec_last_layer_out)
        print_tensor("final logits", results[1]["logits"])

        # print_tensor("1. embedded", emb)
        # print_tensor("2. after pos", after_time)
        # print_tensor("3. 2-1", after_time - emb)

        # for k,v in attn_history0.items():
        #     msg = 'layer0' + k
        #     for tensor in v:
        #         if tensor.shape[0] != 1:
        #             print(f'skipping dummy input result')
        #             continue
        #         print_tensor(msg, tensor)
        # print_tensor("5. encoder layer 1 input", encoder_states[0])
        # print_tensor("6. encoder layer 2 input", encoder_states[1])

        # import ipdb; ipdb.set_trace()
        # print(f'after_time: {after_time}')
        # print(f'after_stack: {after_stack}')
        # print(f'logits: {logits}')

    def test_eager_pegasus(self):
        model_dir = "../pegasus/ckpt/aeslc"
        spm_model = "../pegasus/ckpt/c4.unigram.newline.10pct.96000.model"
        assert os.path.exists(model_dir)

        self.assertTrue(tf.compat.v1.train.checkpoint_exists(model_dir))
        config = PegasusConfig(vocab_size=96000 + 103, d_model=1024, num_beams=8)

        # #hidden_size = 1024
        # #filter_size = 4096
        # num_heads = 16
        # num_encoder_layers = 16
        # num_decoder_layers = 16
        # label_smoothing = 0.0
        # dropout = 0.1
        # beam_size = 8
        # beam_alpha = 0.6

        # run the model to build all variables (but not initialized yet)
        model = TFPegasusPretrainedModel(config)
        dummy_inputs = {"inputs": tf.ones((2, 7), tf.int64), "targets": tf.ones((2, 5), tf.int64)}
        loss, outputs = model(dummy_inputs, True)
        self.assertEqual(loss.shape, [])
        self.assertEqual(outputs["logits"].shape, [2, 5, config.vocab_size])
        model.restore

        # create assignment map
        ignore_name = ["Adafactor", "global_step"]
        # var_list = tf.compat.v1.global_variables(scope=None)
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
        tf.compat.v1.train.init_from_checkpoint(model_dir, assignment_map)

        # check running
        raw_input_str = (
            "To ensure a smooth flow of bank resolutions to the necessary signatories, "
            "I am requesting that Enron Treasury first route the bank resolutions to Angela Davis "
            "(EWS Legal) to be initialed before being routed to John Lavorato or Louise Kitchen.\n"
            "If you have any questions please call me at 3-6544."
            "Thank you for your attention to this matter."
        )
        raw_target_str = "Treasury Bank Resolutions"  # or something close

        # input_str = tf.compat.v1.placeholder(tf.string, shape=[1, ], name=None)
        # target_str = tf.compat.v1.placeholder(tf.string, shape=[1, ], name=None)

        # tokenization
        input_ids = encode(raw_input_str, 512, spm_model, encoder_type="sentencepiece")
        target_ids = encode(raw_target_str, 32, spm_model, encoder_type="sentencepiece")

        input_ids = tf.reshape(input_ids, [1, 512])
        target_ids = tf.reshape(target_ids, [1, 32])

        output_ids = model.predict({"inputs": input_ids, "targets": target_ids,}, 32, config.num_beams, beam_alpha=0.6)
        self.assertEqual(output_ids["outputs"].shape, [1, 32])

        # decode to str
        output_str = decode(output_ids["outputs"], spm_model, encoder_type="sentencepiece")
        print(output_str)
        #
        # sess.run(tf.compat.v1.global_variables_initializer())
        #
        # print(sess.run(output_str,
        #                feed_dict={input_str: [raw_input_str], target_str: [raw_target_str]}))

    def test_bart_logits(self):
        tok = PegasusTokenizer.from_pretrained("sshleifer/pegasus")
        model = BartForConditionalGeneration.from_pretrained(
            "peg_aeslc_bart_transposed",  # "sshleifer/pegasus/aeslc",
            scale_embedding=True,
            num_beams=1,  # activation='relu',
        ).to(torch_device)
        assert model.config.activation_function == "relu"
        assert model.model.decoder.embed_tokens.padding_idx == tok.pad_token_id
        batch = tok([BANK_SNIPPET], return_tensors="pt").to(torch_device)
        decoder_ids = tok([raw_target_str], return_tensors="pt").to(torch_device)
        print_tensor("target_ids", decoder_ids.input_ids)
        output = model.forward(
            batch.input_ids,
            attention_mask=batch.attention_mask,
            decoder_input_ids=decoder_ids.input_ids,
            use_cache=False,
        )

        # summary = tok.batch_decode(model.generate(batch.input_ids), skip_special_tokens=False)[0]
        # self.assertEqual(summary, "Bank Resolutions")

    def test_bart_generate(self):
        tok = PegasusTokenizer.from_pretrained("sshleifer/pegasus")
        model = BartForConditionalGeneration.from_pretrained(
            "peg_aeslc_bart_transposed_v2",  # "sshleifer/pegasus/aeslc",
            scale_embedding=True,
            num_beams=1,  # activation='relu',
        ).to(torch_device)
        assert model.config.activation_function == "relu"
        batch = tok([BANK_SNIPPET], return_tensors="pt").to(torch_device)
        import torch

        assert model.model.shared(torch.zeros_like(batch.input_ids)).max().item() == 0
        generated_ids = model.generate(batch.input_ids, early_stopping=False, max_length=10)
        summary = tok.batch_decode(generated_ids, skip_special_tokens=False)[0]
        self.assertEqual(summary, "Bank Resolutions")

    def test_pegasus_config(self):
        ckpt = "peg_aeslc_bart_transposed"
        config = PegasusConfig.from_pretrained(ckpt)
        assert config.activation_function == "relu"
        assert config.scale_embedding
        assert config.dropout > 0
        assert config.activation_dropout == config.dropout
        assert config.attention_dropout == config.dropout
        assert config.normalize_before
        assert not config.normalize_embedding
        assert config.num_beams == 8
        # assert co


def print_tensor(msg, t):
    # assert t.shape
    ndim = len(t.shape)
    if ndim == 1:
        slice = t[:3]
    elif ndim == 2:
        slice = t[:3, :3]
    elif ndim == 3:
        slice = t[:3, :3, :3]
    elif ndim == 4:
        slice = t[:3, :3, :3, :3]
    print(f"{msg}: shape: {t.shape}, slice: {slice}")
