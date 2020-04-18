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
from transformers import is_torch_available  # noqa: F401

from trax.shapes import ShapeDtype as trax_ShapeDtype

from .utils import require_torch, torch_device  # noqa: F401


if is_torch_available():
    import torch
    from transformers import ReformerAttention, ReformerConfig, ReformerModelWithLMHead


@require_torch
class ReformerIntegrationTests(unittest.TestCase):
    def test_lsh_layer(self):
        config = ReformerConfig()
        shape = (2, 192, config.hidden_size)  # Batch x SeqLen x hiddenSize
        np_input = np.random.rand(*shape)

        trax_layer = self.load_lsh_layer(config)
        input_signature = trax_ShapeDtype(shape, np.float32)
        trax_weights, trax_state = trax_layer.init(input_signature)

        trax_output = trax_layer(np_input, weights=trax_weights, state=trax_state)

        trax_torch_output = torch.tensor(np.asarray(trax_output))

        hf_input = torch.tensor(np_input, dtype=torch.float)
        config.attn_type = "lsh"
        hf_layer = ReformerAttention(config)
        self._set_layer_weights_in_torch_lsh(trax_weights, hf_layer, config.hidden_size)
        hf_layer.eval()

        hf_attention_all_heads = hf_layer.self_attention(hf_input)[0]
        hf_output = hf_layer.output(hf_attention_all_heads)

        self.assertTrue(torch.allclose(hf_output, trax_torch_output, atol=1e-3))

    def test_local_layer(self):
        config = ReformerConfig()
        shape = (1, 64, config.hidden_size)  # Batch x SeqLen x hiddenSize
        np_input = np.random.rand(*shape)

        trax_layer = self.load_local_layer(config)
        input_signature = trax_ShapeDtype(shape, np.float32)
        trax_weights, trax_state = trax_layer.init(input_signature)
        trax_output = trax_layer(np_input, weights=trax_weights, state=trax_state)

        hf_input = torch.tensor(np_input, dtype=torch.float)
        config.attn_type = "local"
        hf_layer = ReformerAttention(config)
        self._set_layer_weights_in_torch_local(trax_weights, hf_layer, config.hidden_size)
        hf_layer.eval()

        hf_attention_all_heads = hf_layer.self_attention(hf_input)[0]
        hf_output = hf_layer.output(hf_attention_all_heads)

        trax_torch_output = torch.tensor(np.asarray(trax_output))
        self.assertTrue(torch.allclose(hf_output, trax_torch_output, atol=1e-3))

    def test_reformer_lm_model(self):
        config = ReformerConfig()

        shape = (1, 192)  # Batch x SeqLen x ModelDimPerHead
        np_input = np.random.randint(0, config.vocab_size, size=shape)
        np_zeros = np.zeros((shape[0], 1), dtype=np.int)

        mode = "predict"
        trax_model = self.load_reformer_lm_model(config, mode=mode)

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
            config.chunk_length,
            config.num_chunks_before,
            config.num_chunks_after,
            config.num_hashes,
            config.num_buckets,
            config.attention_probs_dropout_prob,
            config.hidden_dropout_prob,
            config.seed,
            config.is_decoder,
        )
        gin.parse_config(gin_config)
        layer = trax.layers.LSHSelfAttention(mode=mode)
        return layer

    def load_local_layer(self, config, mode="eval"):
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
            SelfAttention.use_reference_code = True
            """.format(
            config.num_attention_heads,
            config.attention_head_size,
            config.attention_head_size,
            config.chunk_length,
            config.num_chunks_before,
            config.num_chunks_after,
            config.attention_probs_dropout_prob,
            config.hidden_dropout_prob,
            config.is_decoder,
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
        if config.attn_type == "lsh":
            attn_type = "LSHSelfAttention"
        elif config.attn_type == "local":
            attn_type = "SelfAttention"
        else:
            raise ValueError()

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
            config.chunk_length,
            config.chunk_length,
            config.chunk_length // 2,
            config.num_chunks_before,
            config.num_chunks_after,
            config.num_hashes,
            config.num_buckets,
            config.seed,
            config.is_decoder,
            config.chunk_length,
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
            config.axial_pos_shape,
            config.axial_pos_embds_dim,
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
