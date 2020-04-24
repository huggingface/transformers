# coding=utf-8 # Copyright 2020 Huggingface
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unittest

import gin
import numpy as np

# trax imports - to be deleted later
import trax
from transformers import is_torch_available
from trax.shapes import ShapeDtype as trax_ShapeDtype

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from .utils import CACHE_DIR, require_torch, slow, torch_device


if is_torch_available():
    from transformers import ReformerConfig, ReformerAttention, ReformerModel, ReformerModelWithLMHead

    #    from transformers.modeling_reformer import REFORMER_PRETRAINED_MODEL_ARCHIVE_MAP
    import torch


@require_torch
class ReformerLocalAttnModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (ReformerModel, ReformerModelWithLMHead) if is_torch_available() else ()
    test_pruning = False
    test_headmasking = False
    test_torchscript = False

    class ReformerLocalAttnModelTester(object):
        def __init__(
            self,
            parent,
            batch_size=13,
            seq_length=16,
            is_training=True,
            is_decoder=False,
            use_input_mask=True,
            vocab_size=32,
            attention_head_size=16,
            hidden_size=32,
            num_attention_heads=2,
            local_attn_chunk_length=4,
            num_chunks_before=1,
            num_chunks_after=0,
            chunk_size_lm_head=0,
            chunk_size_feed_forward=0,
            feed_forward_size=32,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            initializer_range=0.02,
            axial_norm_std=1.0,
            layer_norm_eps=1e-12,
            axial_pos_embds=True,
            axial_pos_shape=[4, 4],
            axial_pos_embds_dim=[16, 16],
            attn_layers=["local", "local", "local", "local"],
            pad_token_id=0,
            eos_token_id=2,
            scope=None,
        ):
            self.parent = parent
            self.batch_size = batch_size
            self.seq_length = seq_length
            self.is_training = is_training
            self.is_decoder = is_decoder
            self.use_input_mask = use_input_mask
            self.vocab_size = vocab_size
            self.attention_head_size = attention_head_size
            self.hidden_size = hidden_size
            self.num_attention_heads = num_attention_heads
            self.num_hidden_layers = len(attn_layers)
            self.local_attn_chunk_length = local_attn_chunk_length
            self.num_chunks_after = num_chunks_after
            self.num_chunks_before = num_chunks_before
            self.hidden_act = hidden_act
            self.feed_forward_size = feed_forward_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
            self.axial_pos_embds = axial_pos_embds
            self.axial_pos_shape = tuple(axial_pos_shape)
            self.axial_pos_embds_dim = tuple(axial_pos_embds_dim)
            self.axial_norm_std = axial_norm_std
            self.chunk_size_lm_head = chunk_size_lm_head
            self.chunk_size_feed_forward = chunk_size_feed_forward
            self.scope = scope
            self.attn_layers = attn_layers
            self.pad_token_id = pad_token_id

            self.encoder_seq_length = seq_length // local_attn_chunk_length + (
                self.seq_length % local_attn_chunk_length != 0
            )
            self.key_length = (self.num_chunks_before + self.num_chunks_after + 1) * local_attn_chunk_length
            self.chunk_length = local_attn_chunk_length

        def prepare_config_and_inputs(self):
            input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

            input_mask = None
            if self.use_input_mask:
                input_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

            config = ReformerConfig(
                vocab_size=self.vocab_size,
                hidden_size=self.hidden_size,
                num_hidden_layers=self.num_hidden_layers,
                num_attention_heads=self.num_attention_heads,
                feed_forward_size=self.feed_forward_size,
                hidden_act=self.hidden_act,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                max_position_embeddings=self.max_position_embeddings,
                is_decoder=self.is_decoder,
                axial_pos_embds=self.axial_pos_embds,
                axial_pos_shape=self.axial_pos_shape,
                axial_pos_embds_dim=self.axial_pos_embds_dim,
                local_attn_chunk_length=self.local_attn_chunk_length,
                num_chunks_after=self.num_chunks_after,
                num_chunks_before=self.num_chunks_before,
                attn_layers=self.attn_layers,
                pad_token_id=self.pad_token_id,
            )

            return (
                config,
                input_ids,
                input_mask,
            )

        def check_loss_output(self, result):
            self.parent.assertListEqual(list(result["loss"].size()), [])

        def create_and_check_reformer_model(
            self, config, input_ids, input_mask,
        ):
            model = ReformerModel(config=config)
            model.to(torch_device)
            model.eval()
            (sequence_output,) = model(input_ids, attention_mask=input_mask)
            (sequence_output,) = model(input_ids)

            result = {
                "sequence_output": sequence_output,
            }
            # 2 * hidden_size because we use reversible resnet layers
            self.parent.assertListEqual(
                list(result["sequence_output"].size()), [self.batch_size, self.seq_length, 2 * self.hidden_size]
            )

        def create_and_check_reformer_with_lm(
            self, config, input_ids, input_mask,
        ):
            model = ReformerModelWithLMHead(config=config)
            model.to(torch_device)
            model.eval()
            loss, prediction_scores = model(input_ids, attention_mask=input_mask, lm_labels=input_ids)
            result = {
                "loss": loss,
                "prediction_scores": prediction_scores,
            }
            self.parent.assertListEqual(
                list(result["prediction_scores"].size()), [self.batch_size, self.seq_length, self.vocab_size]
            )
            self.check_loss_output(result)

        def create_and_check_reformer_model_with_attn_mask(
            self, config, input_ids, input_mask, is_decoder
        ):
            # no special position embeddings
            config.axial_pos_embds = False
            config.is_decoder = is_decoder

            model = ReformerModel(config=config)
            model.to(torch_device)
            model.eval()
            # set all position encodings to zero so that postions don't matter
            with torch.no_grad():
                embedding = model.embeddings.position_embeddings.embedding
                embedding.weight = torch.nn.Parameter(torch.zeros(embedding.weight.shape))
                embedding.weight.requires_grad = False

            half_seq_len = self.seq_length // 2
            roll = self.local_attn_chunk_length
            roll = self.local_attn_chunk_length
            half_input_ids = input_ids[:, :half_seq_len]

            # normal padded
            attn_mask = torch.cat([torch.ones_like(half_input_ids), torch.zeros_like(half_input_ids)], dim=-1)
            input_ids_padded = torch.cat([half_input_ids, ids_tensor((self.batch_size, half_seq_len), self.vocab_size)], dim=-1)

            # shifted padded
            input_ids_roll = torch.cat([half_input_ids, ids_tensor((self.batch_size, half_seq_len), self.vocab_size)], dim=-1)
            input_ids_roll = torch.roll(input_ids_roll, roll, dims=-1)
            attn_mask_roll = torch.roll(attn_mask, roll, dims=-1)

#            input_ids_padded_begin = torch.cat([torch.full_like(input_ids[:, :half_seq_len], self.pad_token_id), input_ids[:, :half_seq_len],], dim=-1)

            output_padded = model(input_ids_padded, attention_mask=attn_mask)[0][:, :half_seq_len]
            output_padded_rolled = model(input_ids_roll, attention_mask=attn_mask_roll)[0][:, roll: half_seq_len + roll]

            self.parent.assertTrue(torch.allclose(output_padded, output_padded_rolled, atol=1e-3))

        def prepare_config_and_inputs_for_common(self):
            config_and_inputs = self.prepare_config_and_inputs()
            (config, input_ids, input_mask,) = config_and_inputs
            inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
            return config, inputs_dict

    def setUp(self):
        self.model_tester = ReformerLocalAttnModelTest.ReformerLocalAttnModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ReformerConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_reformer_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_model(*config_and_inputs)

    def test_reformer_model_attn_masking(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_model_with_attn_mask(*config_and_inputs, True)
        self.model_tester.create_and_check_reformer_model_with_attn_mask(*config_and_inputs, False)

    def test_for_reformer_with_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_with_lm(*config_and_inputs)


@require_torch
class ReformerLSHAttnModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (ReformerModel, ReformerModelWithLMHead) if is_torch_available() else ()
    test_pruning = False
    test_headmasking = False
    test_torchscript = False

    class ReformerLSHAttnModelTester(object):
        def __init__(
            self,
            parent,
            batch_size=13,
            seq_length=13,
            use_input_mask=True,
            is_training=False,
            is_decoder=False,
            vocab_size=32,
            attention_head_size=16,
            hidden_size=64,
            num_attention_heads=2,
            num_buckets=2,
            num_hashes=4,
            lsh_attn_chunk_length=4,
            num_chunks_before=2,
            num_chunks_after=3,
            chunk_size_lm_head=5,
            chunk_size_feed_forward=6,
            feed_forward_size=32,
            hidden_act="relu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            sinusoidal_pos_embds=True,
            attn_layers=["lsh", "lsh", "lsh", "lsh"],
            pad_token_id=0,
            eos_token_id=2,
            scope=None,
            hash_seed=0,
        ):
            self.parent = parent
            self.batch_size = batch_size
            self.seq_length = seq_length
            self.is_training = is_training
            self.is_decoder = is_decoder
            self.use_input_mask = use_input_mask
            self.vocab_size = vocab_size
            self.attention_head_size = attention_head_size
            self.hidden_size = hidden_size
            self.num_attention_heads = num_attention_heads
            self.num_hashes = num_hashes
            self.num_hidden_layers = len(attn_layers)
            self.num_buckets = tuple(num_buckets) if isinstance(num_buckets, list) else num_buckets
            self.lsh_attn_chunk_length = lsh_attn_chunk_length
            self.num_chunks_after = num_chunks_after
            self.num_chunks_before = num_chunks_before
            self.hidden_act = hidden_act
            self.feed_forward_size = feed_forward_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
            self.sinusoidal_pos_embds = sinusoidal_pos_embds
            self.chunk_size_lm_head = chunk_size_lm_head
            self.chunk_size_feed_forward = chunk_size_feed_forward
            self.scope = scope
            self.attn_layers = attn_layers
            self.hash_seed = hash_seed
            self.pad_token_id = pad_token_id

            self.encoder_seq_length = seq_length // lsh_attn_chunk_length + (seq_length % lsh_attn_chunk_length != 0)
            self.key_length = (self.num_chunks_before + self.num_chunks_after + 1) * lsh_attn_chunk_length
            self.chunk_length = lsh_attn_chunk_length

        def prepare_config_and_inputs(self):
            input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

            input_mask = None
            if self.use_input_mask:
                input_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

            config = ReformerConfig(
                vocab_size=self.vocab_size,
                hidden_size=self.hidden_size,
                num_hidden_layers=self.num_hidden_layers,
                num_attention_heads=self.num_attention_heads,
                feed_forward_size=self.feed_forward_size,
                hidden_act=self.hidden_act,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                max_position_embeddings=self.max_position_embeddings,
                is_decoder=self.is_decoder,
                sinusoidal_pos_embds=self.sinusoidal_pos_embds,
                num_hashes=self.num_hashes,
                num_buckets=self.num_buckets,
                lsh_attn_chunk_length=self.lsh_attn_chunk_length,
                num_chunks_after=self.num_chunks_after,
                num_chunks_before=self.num_chunks_before,
                attn_layers=self.attn_layers,
                hash_seed=self.hash_seed,
                pad_token_id=self.pad_token_id,
            )

            return (
                config,
                input_ids,
                input_mask,
            )

        def check_loss_output(self, result):
            self.parent.assertListEqual(list(result["loss"].size()), [])

        def create_and_check_reformer_model(
            self, config, input_ids, input_mask,
        ):
            model = ReformerModel(config=config)
            model.to(torch_device)
            model.eval()
            (sequence_output,) = model(input_ids, attention_mask=input_mask)
            (sequence_output,) = model(input_ids)

            result = {
                "sequence_output": sequence_output,
            }
            # 2 * hidden_size because we use reversible resnet layers
            self.parent.assertListEqual(
                list(result["sequence_output"].size()), [self.batch_size, self.seq_length, 2 * self.hidden_size]
            )

        def create_and_check_reformer_with_lm(
            self, config, input_ids, input_mask,
        ):
            model = ReformerModelWithLMHead(config=config)
            model.to(torch_device)
            model.eval()
            loss, prediction_scores = model(input_ids, attention_mask=input_mask, lm_labels=input_ids)
            result = {
                "loss": loss,
                "prediction_scores": prediction_scores,
            }
            self.parent.assertListEqual(
                list(result["prediction_scores"].size()), [self.batch_size, self.seq_length, self.vocab_size]
            )
            self.check_loss_output(result)

        def create_and_check_reformer_model_with_attn_mask(
            self, config, input_ids, input_mask, is_decoder
        ):
            # no special position embeddings
            config.axial_pos_embds = False
            config.is_decoder = is_decoder

            # need to set chunk length equal sequence length to be certain that chunking works
            config.lsh_attn_chunk_length = self.seq_length

            model = ReformerModel(config=config)
            model.to(torch_device)
            model.eval()
            # set all position encodings to zero so that postions don't matter
            with torch.no_grad():
                embedding = model.embeddings.position_embeddings.embedding
                embedding.weight = torch.nn.Parameter(torch.zeros(embedding.weight.shape))
                embedding.weight.requires_grad = False

            half_seq_len = self.seq_length // 2
            roll = self.lsh_attn_chunk_length
            roll = half_seq_len
            half_input_ids = input_ids[:, :half_seq_len]

            # normal padded
            attn_mask = torch.cat([torch.ones_like(half_input_ids), torch.zeros_like(half_input_ids)], dim=-1)
            input_ids_padded = torch.cat([half_input_ids, ids_tensor((self.batch_size, half_seq_len), self.vocab_size)], dim=-1)

            # shifted padded
            input_ids_roll = torch.cat([half_input_ids, ids_tensor((self.batch_size, half_seq_len), self.vocab_size)], dim=-1)
            input_ids_roll = torch.roll(input_ids_roll, roll, dims=-1)
            attn_mask_roll = torch.roll(attn_mask, roll, dims=-1)

#            input_ids_padded_begin = torch.cat([torch.full_like(input_ids[:, :half_seq_len], self.pad_token_id), input_ids[:, :half_seq_len],], dim=-1)

            output_padded = model(input_ids_padded, attention_mask=attn_mask)[0][:, :half_seq_len]
            output_padded_rolled = model(input_ids_roll, attention_mask=attn_mask_roll)[0][:, roll: half_seq_len + roll]

            self.parent.assertTrue(torch.allclose(output_padded, output_padded_rolled, atol=1e-3))

        def prepare_config_and_inputs_for_common(self):
            config_and_inputs = self.prepare_config_and_inputs()
            (config, input_ids, input_mask,) = config_and_inputs
            inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
            return config, inputs_dict

    def setUp(self):
        self.model_tester = ReformerLSHAttnModelTest.ReformerLSHAttnModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ReformerConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_reformer_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_model(*config_and_inputs)

    def test_reformer_model_attn_masking(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_model_with_attn_mask(*config_and_inputs, True)
        self.model_tester.create_and_check_reformer_model_with_attn_mask(*config_and_inputs, False)

    def test_for_reformer_with_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_with_lm(*config_and_inputs)


@require_torch
class ReformerIntegrationTests(unittest.TestCase):
    def test_lsh_layer(self):
        config = ReformerConfig()
        shape = (1, 64, config.hidden_size)  # Batch x SeqLen x hiddenSize
        np_input = np.random.rand(*shape)

        trax_layer = self.load_lsh_layer(config)
        input_signature = trax_ShapeDtype(shape, np.float32)
        trax_weights, trax_state = trax_layer.init(input_signature)

        mask = np.ones(shape[:-1], dtype=np.int32)

        trax_output = trax_layer(np_input, weights=trax_weights, state=trax_state)

        trax_torch_output = torch.tensor(np.asarray(trax_output))

        hf_input = torch.tensor(np_input, dtype=torch.float)
        config.attn_layers = ["lsh"]
        hf_layer = ReformerAttention(config)
        self._set_layer_weights_in_torch_lsh(trax_weights, hf_layer, config.hidden_size)
        hf_layer.eval()

        hf_attention_all_heads = hf_layer.self_attention(hf_input, attention_mask=torch.tensor(mask))[0]
        hf_output = hf_layer.output(hf_attention_all_heads)

        self.assertTrue(torch.allclose(hf_output, trax_torch_output, atol=1e-3))

    def test_local_layer(self):
        config = ReformerConfig()
        shape = (1, 64, config.hidden_size)  # Batch x SeqLen x hiddenSize
        np_input = np.random.rand(*shape)

        trax_layer = self.load_local_layer(config)
        input_signature = trax_ShapeDtype(shape, np.float32)
        trax_weights, trax_state = trax_layer.init(input_signature)
        mask = np.ones(shape[:-1], dtype=np.int32)

        trax_output = trax_layer(np_input, weights=trax_weights, state=trax_state)

        hf_input = torch.tensor(np_input, dtype=torch.float)
        config.attn_layers = ["local"]
        hf_layer = ReformerAttention(config)
        self._set_layer_weights_in_torch_local(trax_weights, hf_layer, config.hidden_size)
        hf_layer.eval()

        hf_attention_all_heads = hf_layer.self_attention(hf_input, attention_mask=torch.tensor(mask))[0]
        hf_output = hf_layer.output(hf_attention_all_heads)

        trax_torch_output = torch.tensor(np.asarray(trax_output))
        self.assertTrue(torch.allclose(hf_output, trax_torch_output, atol=1e-3))

    def test_reformer_lm_model(self):
        config = ReformerConfig(sinusoidal_pos_embds=True, hash_seed=0, is_decoder=True)

        shape = (1, 192)  # Batch x SeqLen x ModelDimPerHead
        np_input = np.random.randint(0, config.vocab_size, size=shape)
        np_zeros = np.zeros((shape[0], 1), dtype=np.int)

        mode = "eval"
        trax_model = self.load_reformer_lm_model(config, mode=mode)

        assert (
            config.is_decoder is True
        ), "trax can only test casaul mask for ReformerLM. Use tests for layers to test non-casaul mask"
        input_signature = trax_ShapeDtype(shape, np.int32)
        trax_weights, trax_state = trax_model.init(input_signature)
        trax_output = trax_model(np_input, weights=trax_weights, state=trax_state)

        trax_torch_output = torch.tensor(np.asarray(trax_output[0]))

        if mode != "predict":
            hf_input = torch.cat([torch.tensor(np_zeros), torch.tensor(np_input[:, :-1])], dim=-1)
        else:
            hf_input = torch.tensor(np_input)

        hf_model = ReformerModelWithLMHead(config)
        self._set_model_weights_in_torch(trax_weights, hf_model, config.hidden_size)
        hf_model.eval()

        hf_output = hf_model(hf_input)
        log_softmax_output = torch.nn.functional.log_softmax(hf_output[0], dim=-1)

        self.assertTrue(torch.allclose(log_softmax_output, trax_torch_output, atol=1e-3))

    def test_backprop_lm_model(self):
        config = ReformerConfig()

        shape = (1, 192)  # Batch x SeqLen x ModelDimPerHead
        input_ids = torch.tensor(
            np.random.randint(0, config.vocab_size, size=shape), dtype=torch.long, device=torch_device
        )

        model = ReformerModelWithLMHead(config)
        loss = model(input_ids, lm_labels=input_ids)[0]
        loss.backward()

    def test_pretrained_crime_and_punishment_lm_model(self):
        hf_model = ReformerModelWithLMHead.from_pretrained("patrickvonplaten/reformer-crime-and-punish")
        config = hf_model.config

        trax_model_path = "/home/patrick/hugging_face/models/trained_reformer_colab/model.pkl"

        shape = (1, 128)
        np_input = np.random.randint(0, config.vocab_size, size=shape)

        hf_input = torch.tensor(np_input)

        input_signature = trax_ShapeDtype(shape, np.int32)
        trax_model = self.load_crime_and_punishment_model(trax_model_path, input_signature)

        hf_output = hf_model(hf_input)
        log_softmax_output = torch.nn.functional.log_softmax(hf_output[0], dim=-1)

        trax_output = trax_model(np_input)
        trax_torch_output = torch.tensor(np.asarray(trax_output[0]))

        self.assertTrue(torch.allclose(log_softmax_output, trax_torch_output, atol=1e-3))

    def load_lsh_layer(self, config, mode="eval"):
        gin_config = """
            import trax.layers

            # Parameters for LSHSelfAttention:
            # ==============================================================================
            LSHSelfAttention.n_heads = {}
            LSHSelfAttention.d_qk = {}
            LSHSelfAttention.d_v = {}
            LSHSelfAttention.chunk_len = {}
            LSHSelfAttention.n_chunks_before = {}
            LSHSelfAttention.n_chunks_after = {}
            LSHSelfAttention.n_hashes = {}
            LSHSelfAttention.n_buckets = {}
            LSHSelfAttention.attention_dropout = {}
            LSHSelfAttention.output_dropout = {}
            LSHSelfAttention.lsh_seed = {}
            LSHSelfAttention.causal= {}
            LSHSelfAttention.use_reference_code = True
            """.format(
            config.num_attention_heads,
            config.attention_head_size,
            config.attention_head_size,
            config.lsh_attn_chunk_length,
            config.num_chunks_before,
            config.num_chunks_after,
            config.num_hashes,
            config.num_buckets,
            config.attention_probs_dropout_prob,
            config.hidden_dropout_prob,
            config.hash_seed,
            config.is_decoder,
        )
        gin.parse_config(gin_config)
        layer = trax.layers.LSHSelfAttention(mode=mode)
        return layer

    def load_local_layer(self, config, mask=False, mode="eval"):
        gin_config = """
            import trax.layers

            # Parameters for SelfAttention:
            # ==============================================================================
            SelfAttention.n_heads = {}
            SelfAttention.d_qk = {}
            SelfAttention.d_v = {}
            SelfAttention.chunk_len = {}
            SelfAttention.n_chunks_before = {}
            SelfAttention.n_chunks_after = {}
            SelfAttention.attention_dropout = {}
            SelfAttention.output_dropout = {}
            SelfAttention.causal = {}
            SelfAttention.masked= {}
            SelfAttention.use_reference_code = True
            """.format(
            config.num_attention_heads,
            config.attention_head_size,
            config.attention_head_size,
            config.local_attn_chunk_length,
            config.num_chunks_before,
            config.num_chunks_after,
            config.attention_probs_dropout_prob,
            config.hidden_dropout_prob,
            config.is_decoder,
            mask,
        )
        gin.parse_config(gin_config)
        layer = trax.layers.SelfAttention(mode=mode)
        return layer

    def load_reformer_lm_model(self, config, mode="eval"):
        if config.hidden_act == "gelu":
            hidden_act = "Gelu"
        elif config.hidden_act == "relu":
            hidden_act = "Relu"
        else:
            raise ValueError()
        attn_type = config.attn_layers[0]
        if attn_type == "lsh":
            attn_type = "LSHSelfAttention"
        elif attn_type == "local":
            attn_type = "SelfAttention"
        else:
            raise ValueError()
        if config.sinusoidal_pos_embds is True:
            axial_pos_shape = ()
            d_axial_pos_embs = None
        else:
            axial_pos_shape = config.axial_pos_shape
            d_axial_pos_embs = config.axial_pos_embds_dim

        gin_config = """
            import trax.layers
            import trax.models

            # Parameters for LSHSelfAttention:
            # ==============================================================================
            LSHSelfAttention.chunk_len = {}
            LSHSelfAttention.predict_mem_len = {}
            LSHSelfAttention.predict_drop_len = {}
            LSHSelfAttention.n_chunks_before = {}
            LSHSelfAttention.n_chunks_after = {}
            LSHSelfAttention.n_hashes = {}
            LSHSelfAttention.n_buckets = {}
            LSHSelfAttention.lsh_seed = {}
            LSHSelfAttention.causal= {}
            LSHSelfAttention.use_reference_code = True

            # Parameters for SelfAttention:
            # ==============================================================================
            SelfAttention.chunk_len = {}
            SelfAttention.n_chunks_before = {}
            SelfAttention.n_chunks_after = {}
            SelfAttention.causal= {}
            SelfAttention.use_reference_code = True

            # Parameters for ReformerLM:
            # ==============================================================================
            ReformerLM.vocab_size = {}
            ReformerLM.d_model = {}
            ReformerLM.d_ff = {}
            ReformerLM.d_attention_key = {}
            ReformerLM.d_attention_value = {}
            ReformerLM.n_layers = {}
            ReformerLM.n_heads = {}
            ReformerLM.max_len = {}
            ReformerLM.axial_pos_shape = {}
            ReformerLM.d_axial_pos_embs = {}
            ReformerLM.ff_chunk_size = {}
            ReformerLM.ff_activation = @trax.layers.{}
            ReformerLM.attention_type = @trax.layers.{}

            ReformerLM.n_chunks = 0
            ReformerLM.n_attention_chunks = None
            ReformerLM.share_qk = False
            ReformerLM.ff_use_sru = 0
            """.format(
            config.lsh_attn_chunk_length,
            config.lsh_attn_chunk_length,
            config.lsh_attn_chunk_length // 2,
            config.num_chunks_before,
            config.num_chunks_after,
            config.num_hashes,
            config.num_buckets,
            config.hash_seed,
            config.is_decoder,
            config.local_attn_chunk_length,
            config.num_chunks_before,
            config.num_chunks_after,
            config.is_decoder,
            config.vocab_size,
            config.hidden_size,
            config.feed_forward_size,
            config.attention_head_size,
            config.attention_head_size,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.max_position_embeddings,
            axial_pos_shape,
            d_axial_pos_embs,
            config.chunk_size_feed_forward,
            hidden_act,
            attn_type,
        )
        gin.parse_config(gin_config)
        model = trax.models.ReformerLM(mode=mode)
        return model

    def load_crime_and_punishment_model(self, trax_model_path, input_signature, mode="predict"):
        gin.parse_config(
            """
            import trax.layers
            import trax.models
            import trax.optimizers
            import trax.supervised.inputs
            import trax.supervised.trainer_lib

            # Parameters that will vary between experiments:
            # ==============================================================================
            train.model = @trax.models.ReformerLM
            # Our model will have 6 layers, alternating between the LSH attention proposed
            # in the Reformer paper and local attention within a certain context window.
            n_layers = 6
            attn_type = [
              @SelfAttention,
              @LSHSelfAttention,
              @SelfAttention,
              @LSHSelfAttention,
              @SelfAttention,
              @LSHSelfAttention,
              ]
            share_qk = False  # LSH attention ignores this flag and always shares q & k
            n_heads = 2
            attn_kv = 64
            dropout = 0.05
            n_tokens = 524288

            # Parameters for SelfAttention:
            # ==============================================================================
            SelfAttention.chunk_len = 64
            SelfAttention.n_chunks_before = 1
            SelfAttention.n_parallel_heads = 1

            # Parameters for LSHSelfAttention:
            # ==============================================================================
            LSHSelfAttention.chunk_len = 64
            LSHSelfAttention.n_buckets = [64, 128]
            LSHSelfAttention.n_chunks_after = 0
            LSHSelfAttention.n_chunks_before = 1
            LSHSelfAttention.n_hashes = 1
            LSHSelfAttention.n_parallel_heads = 1
            LSHSelfAttention.predict_drop_len = 32 # different from original to make code equal
            LSHSelfAttention.predict_mem_len = 64 # different from original to make code equal
            LSHSelfAttention.lsh_seed = 0

            # Parameters for ReformerLM:
            # ==============================================================================
            ReformerLM.attention_type = %attn_type
            ReformerLM.d_attention_key = %attn_kv
            ReformerLM.d_attention_value = %attn_kv
            ReformerLM.d_model = 256
            ReformerLM.d_ff = 512
            ReformerLM.dropout = %dropout
            ReformerLM.ff_activation = @trax.layers.Relu
            ReformerLM.max_len = %n_tokens
            ReformerLM.mode = 'train'
            ReformerLM.n_heads = %n_heads
            ReformerLM.n_layers = %n_layers
            ReformerLM.vocab_size = 320
            ReformerLM.share_qk = %share_qk
            ReformerLM.axial_pos_shape = (512, 1024)
            ReformerLM.d_axial_pos_embs= (64, 192)
            """
        )
        trax_model = trax.models.ReformerLM(mode=mode)
        trax_model.init(input_signature)
        trax_model.init_from_file(trax_model_path, weights_only=True)
        return trax_model

    def _set_param(self, torch_layer, weight, bias=None):
        with torch.no_grad():
            assert torch_layer.weight.shape == weight.shape, "{} layer.weight does not match".format(torch_layer)
            torch_layer.weight = torch.nn.Parameter(weight)
            if bias is not None:
                assert torch_layer.bias.shape == bias.shape, "{} layer.bias does not match".format(torch_layer)
                torch_layer.bias = torch.nn.Parameter(bias)

    def _set_layer_weights_in_torch_lsh(self, weights, torch_layer, hidden_size):
        # set torch weights for 1-to-1 comparison
        np_query_key = np.asarray(weights[0])
        np_value = np.asarray(weights[1])
        np_dense = np.asarray(weights[2])

        self._set_param(
            torch_layer.self_attention.query_key,
            torch.tensor(np_query_key).transpose(1, 2).contiguous().view(-1, hidden_size),
        )
        self._set_param(
            torch_layer.self_attention.value,
            torch.tensor(np_value).transpose(1, 2).contiguous().view(-1, hidden_size),
        )
        self._set_param(
            torch_layer.output.dense, torch.tensor(np_dense).view(-1, hidden_size).contiguous().transpose(0, 1),
        )

    def _set_layer_weights_in_torch_local(self, weights, torch_layer, hidden_size):
        # set torch weights for 1-to-1 comparison
        np_query = np.asarray(weights[0])
        np_key = np.asarray(weights[1])
        np_value = np.asarray(weights[2])
        np_dense = np.asarray(weights[3])

        self._set_param(
            torch_layer.self_attention.query,
            torch.tensor(np_query).transpose(1, 2).contiguous().view(-1, hidden_size),
        )
        self._set_param(
            torch_layer.self_attention.key, torch.tensor(np_key).transpose(1, 2).contiguous().view(-1, hidden_size),
        )
        self._set_param(
            torch_layer.self_attention.value,
            torch.tensor(np_value).transpose(1, 2).contiguous().view(-1, hidden_size),
        )
        self._set_param(
            torch_layer.output.dense, torch.tensor(np_dense).view(-1, hidden_size).contiguous().transpose(0, 1),
        )

    def _set_block_weights_in_torch(self, weights, torch_block, hidden_size):
        # layernorm 1
        layer_norm_1 = weights[0][0][0]
        layer_norm_1_weight = np.asarray(layer_norm_1[0])
        layer_norm_1_bias = np.asarray(layer_norm_1[1])
        self._set_param(
            torch_block.attention.layer_norm, torch.tensor(layer_norm_1_weight), torch.tensor(layer_norm_1_bias),
        )

        # lsh weights + output
        attn_weights = weights[0][1]
        if len(attn_weights) < 4:
            self._set_layer_weights_in_torch_lsh(attn_weights, torch_block.attention, hidden_size)
        else:
            self._set_layer_weights_in_torch_local(attn_weights, torch_block.attention, hidden_size)

        # intermediate weighs
        intermediate_weights = weights[2][0][2][2]

        # Chunked Feed Forward
        if len(intermediate_weights) == 4:
            intermediate_weights = intermediate_weights[2]

        # layernorm 2
        layer_norm_2_weight = np.asarray(intermediate_weights[0][0])
        layer_norm_2_bias = np.asarray(intermediate_weights[0][1])
        self._set_param(
            torch_block.feed_forward.layer_norm, torch.tensor(layer_norm_2_weight), torch.tensor(layer_norm_2_bias),
        )

        # intermediate dense
        inter_dense_weight = np.asarray(intermediate_weights[1][0])
        inter_dense_bias = np.asarray(intermediate_weights[1][1])
        self._set_param(
            torch_block.feed_forward.dense.dense,
            torch.tensor(inter_dense_weight).transpose(0, 1).contiguous(),
            torch.tensor(inter_dense_bias),
        )

        # intermediate out
        out_dense_weight = np.asarray(intermediate_weights[4][0])
        out_dense_bias = np.asarray(intermediate_weights[4][1])
        self._set_param(
            torch_block.feed_forward.output.dense,
            torch.tensor(out_dense_weight).transpose(0, 1).contiguous(),
            torch.tensor(out_dense_bias),
        )

    def _set_model_weights_in_torch(self, weights, torch_model, hidden_size):
        # reformer model
        torch_model_reformer = torch_model.reformer

        # word embeds
        word_embeddings = np.asarray(weights[1])
        self._set_param(
            torch_model_reformer.embeddings.word_embeddings, torch.tensor(word_embeddings),
        )

        if isinstance(weights[3], tuple):
            position_embeddings = torch_model_reformer.embeddings.position_embeddings
            for emb_idx in range(len(position_embeddings.weights)):
                emb_weights = np.asarray(weights[3][emb_idx][0])
                assert position_embeddings.weights[emb_idx].shape == emb_weights.shape, "{} emb does not match".format(
                    position_embeddings[emb_idx]
                )
                position_embeddings.weights[emb_idx] = torch.nn.Parameter(torch.tensor(emb_weights))

        trax_layer_weights = weights[5]
        assert len(torch_model_reformer.encoder.layers) * 4 + 1 == len(
            trax_layer_weights
        ), "HF and trax model do not have the same number of layers"
        for layer_idx, layer in enumerate(torch_model_reformer.encoder.layers):
            block_weights = trax_layer_weights[4 * layer_idx : 4 * (layer_idx + 1)]
            self._set_block_weights_in_torch(block_weights, layer, hidden_size)

        # output weights
        out_weights = weights[6]

        # output layer norm
        layer_norm_out_weight = np.asarray(out_weights[0][0])
        layer_norm_out_bias = np.asarray(out_weights[0][1])
        self._set_param(
            torch_model_reformer.encoder.layer_norm,
            torch.tensor(layer_norm_out_weight),
            torch.tensor(layer_norm_out_bias),
        )

        # output embeddings
        output_embed_weights = np.asarray(out_weights[2][0])
        output_embed_bias = np.asarray(out_weights[2][1])
        self._set_param(
            torch_model.lm_head.decoder,
            torch.tensor(output_embed_weights).transpose(0, 1).contiguous(),
            torch.tensor(output_embed_bias),
        )
