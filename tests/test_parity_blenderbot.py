import unittest

import torch

from parlai.agents.transformer.modules import (
    MultiHeadAttention,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerGeneratorModel,
)
from transformers import BlenderbotConfig, BlenderbotForConditionalGeneration
from transformers.modeling_bart import SelfAttention, fill_with_neg_inf, invert_mask
from transformers.testing_utils import require_torch, torch_device

from .test_modeling_bart import assert_tensors_close


@require_torch
class BlenderbotParityTests(unittest.TestCase):
    """Show that small randomly initialized layers with same parameters produce same outputs. Requires PARLAI"""

    variant = "prelayernorm"

    def get_config_and_data(self):

        torch.manual_seed(0)
        input_ids = torch.tensor(
            [
                [64, 61, 14, 42, 96, 32, 82, 7, 64, 61, 14, 42, 96, 32, 82, 7, 2],
                [94, 23, 54, 10, 10, 41, 90, 48, 94, 23, 54, 10, 10, 41, 90, 4, 2],
                # parity tests break with following padding case
                [94, 23, 54, 10, 10, 41, 90, 48, 94, 23, 54, 10, 10, 41, 90, 2, 1],
            ],
            dtype=torch.long,
            device=torch_device,
        )

        config = BlenderbotConfig(
            d_model=16,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            vocab_size=100,
            encoder_layers=1,
            decoder_layers=1,
            encoder_ffn_dim=8,
            decoder_ffn_dim=8,
            max_position_embeddings=24,
            eos_token_id=2,
            pad_token_id=1,
            bos_token_id=0,
            normalize_before=True,
            variant=self.variant,
            normalize_embedding=True,
        )
        mask = (input_ids != config.pad_token_id).to(torch.long)
        return config, input_ids, mask

    @torch.no_grad()
    def _set_param(self, blender_layer, parlai_layer, bias=None):
        assert blender_layer.weight.shape == parlai_layer.weight.shape, "{} layer.weight does not match"
        blender_layer.weight = torch.nn.Parameter(parlai_layer.weight)
        if bias is not None:
            assert blender_layer.bias.shape == parlai_layer.bias.shape, "{} layer.bias does not match"
            blender_layer.bias = torch.nn.Parameter(parlai_layer.bias)

    def copy_attn_weights(self, blender_layer, parlai_layer):
        self._set_param(blender_layer.q_proj, parlai_layer.q_lin, bias=True)
        self._set_param(blender_layer.v_proj, parlai_layer.v_lin, bias=True)
        self._set_param(blender_layer.k_proj, parlai_layer.k_lin, bias=True)
        self._set_param(blender_layer.out_proj, parlai_layer.out_lin, bias=True)

    def _copy_layer_weights_in_blender_encoder_layer(self, blender_layer, parlai_layer):
        self.copy_attn_weights(blender_layer.self_attn, parlai_layer.attention)
        self._set_param(blender_layer.self_attn_layer_norm, parlai_layer.norm1, bias=True)
        self._set_param(blender_layer.fc1, parlai_layer.ffn.lin1, bias=True)
        self._set_param(blender_layer.fc2, parlai_layer.ffn.lin2, bias=True)
        self._set_param(blender_layer.final_layer_norm, parlai_layer.norm2, bias=True)

    def _copy_layer_weights_in_blender_decoder_layer(self, blender_layer, parlai_layer):
        self.copy_attn_weights(blender_layer.self_attn, parlai_layer.self_attention)
        self.copy_attn_weights(blender_layer.encoder_attn, parlai_layer.encoder_attention)
        self._set_param(blender_layer.self_attn_layer_norm, parlai_layer.norm1, bias=True)
        self._set_param(blender_layer.fc1, parlai_layer.ffn.lin1, bias=True)
        self._set_param(blender_layer.fc2, parlai_layer.ffn.lin2, bias=True)
        self._set_param(blender_layer.encoder_attn_layer_norm, parlai_layer.norm2, bias=True)
        self._set_param(blender_layer.final_layer_norm, parlai_layer.norm3, bias=True)

    def _copy_layer_weights_in_blender_encoder(self, blender_encoder, parlai_encoder, num_layers):
        for i in range(num_layers):
            self._copy_layer_weights_in_blender_encoder_layer(blender_encoder.layers[i], parlai_encoder.layers[i])
        self._set_param(blender_encoder.layernorm_embedding, parlai_encoder.norm_embeddings, bias=True)
        self._set_param(blender_encoder.embed_positions, parlai_encoder.position_embeddings)
        self._set_param(blender_encoder.embed_tokens, parlai_encoder.embeddings)

    @torch.no_grad()
    def test_self_attention_parity(self):
        """If all we send is query, bart attention is the same iff query is transposed and key_padding_mask has 1s at blocked positions"""
        config, input_ids, mask = self.get_config_and_data()
        bs, seq_len = 3, 5  # odd numbers for less confusion
        hidden_states = torch.rand(bs, seq_len, config.d_model)
        mask = torch.ones(hidden_states.shape[:2])
        mask[-1, -1] = 0
        blenderbot_model = BlenderbotForConditionalGeneration(config).to(torch_device).eval()
        self_attn = blenderbot_model.encoder.layers[0].self_attn
        parlai_attn = MultiHeadAttention(config.encoder_attention_heads, config.d_model, dropout=config.dropout).eval()

        self.copy_attn_weights(self_attn, parlai_attn)
        expected_output = parlai_attn(query=hidden_states, mask=mask, key=None)[0]
        bart_mask = invert_mask(mask)
        bart_output = self_attn(query=hidden_states.transpose(0, 1), key_padding_mask=bart_mask, key=None)[
            0
        ].transpose(0, 1)
        assert_tensors_close(expected_output, bart_output, atol=1e-8)

    @torch.no_grad()
    def test_cross_attention_parity(self):
        """If all we send is query, bart attention is the same iff query is transposed and key_padding_mask has
        1s at blocked positions"""
        config, input_ids, mask = self.get_config_and_data()
        bs, seq_len = 3, 5  # odd numbers for less confusion
        hidden_states = torch.rand(bs, seq_len, config.d_model)
        # d_model = 16,
        # encoder_attention_heads = 2,
        # decoder_attention_heads = 2,
        model = BlenderbotForConditionalGeneration(config).to(torch_device).eval()
        cross_attn: SelfAttention = model.decoder.layers[0].encoder_attn
        parlai_attn = MultiHeadAttention(config.encoder_attention_heads, config.d_model).eval()

        self.copy_attn_weights(cross_attn, parlai_attn)
        dummy_encoder_output = torch.rand(bs, 11, config.d_model)
        mask = torch.ones(dummy_encoder_output.shape[:2])
        # This passed
        # parlai_way = cross_attn.prepare_head(dummy_encoder_output)
        # hf_way = cross_attn._shape(dummy_encoder_output.transpose(0, 1), 11, bs)
        # assert_tensors_close(parlai_way, hf_way, atol=1e-6)

        mask[-1, -1] = 0
        bart_mask = invert_mask(mask)
        expected_output = parlai_attn.forward(
            query=hidden_states, mask=mask, key=dummy_encoder_output, static_kv=True
        )[0]

        self.assertTrue(cross_attn.encoder_decoder_attention)
        bart_output = cross_attn.forward(
            query=hidden_states.transpose(0, 1), key_padding_mask=bart_mask, key=dummy_encoder_output.transpose(0, 1)
        )[0].transpose(0, 1)
        assert_tensors_close(expected_output, bart_output, atol=1e-8)
        #

    @torch.no_grad()
    def test_encoder_layer_parity(self):
        config, input_ids, mask = self.get_config_and_data()

        # hidden_states = torch.tensor(2 * [5 * [config.d_model * [0.1]]])
        bs, seq_len = 3, 5  # odd numbers for less confusion
        hidden_states = torch.rand(bs, seq_len, config.d_model)
        mask = torch.ones(hidden_states.shape[:2])
        mask[-1, -1] = 0
        bart_mask = invert_mask(mask)
        blenderbot_model = BlenderbotForConditionalGeneration(config).to(torch_device).eval()
        bart_encoder_layer = blenderbot_model.encoder.layers[0]

        embeddings = torch.nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        parlai_encoder = TransformerEncoder(
            config.encoder_attention_heads,
            config.encoder_layers,
            config.d_model,
            config.encoder_ffn_dim,
            config.vocab_size,
            learn_positional_embeddings=True,
            variant=self.variant,
            n_positions=config.max_position_embeddings,
            activation=config.activation_function,
            dropout=config.dropout,
            embedding=embeddings,
            reduction_type=None,
        ).eval()
        parlai_encoder_layer = parlai_encoder.layers[0]
        self._copy_layer_weights_in_blender_encoder_layer(bart_encoder_layer, parlai_encoder_layer)
        expected_output = parlai_encoder_layer(hidden_states, mask)
        blender_output = bart_encoder_layer(hidden_states.transpose(1, 0), bart_mask)[0].transpose(1, 0)
        assert_tensors_close(expected_output, blender_output, atol=1e-6)

    @torch.no_grad()
    def test_encoder_parity(self):
        config, input_ids, mask = self.get_config_and_data()
        bs, seq_len = input_ids.shape  # odd numbers for less confusion
        bart_mask = input_ids != config.pad_token_id  # gets inverted by code
        # mask[-1, -1] = 0
        # bart_mask = invert_mask(mask)

        torch.manual_seed(0)

        blenderbot_model = BlenderbotForConditionalGeneration(config).to(torch_device).eval()
        bart_encoder = blenderbot_model.encoder

        parlai_encoder = TransformerEncoder(
            config.encoder_attention_heads,
            config.encoder_layers,
            config.d_model,
            config.encoder_ffn_dim,
            config.vocab_size,
            learn_positional_embeddings=True,
            variant=config.variant,
            n_positions=config.max_position_embeddings,
            activation=config.activation_function,
            dropout=config.dropout,
            embedding=bart_encoder.embed_tokens,
            reduction_type=None,
            padding_idx=config.pad_token_id,
        )
        parlai_encoder.eval()

        self._copy_layer_weights_in_blender_encoder(bart_encoder, parlai_encoder, config.encoder_layers)

        expected_output = parlai_encoder.forward(input_ids)[0]  # makes own mask
        blender_output = bart_encoder(input_ids, attention_mask=bart_mask)[0]
        assert_tensors_close(expected_output[:-1], blender_output[:-1], atol=1e-4)
        assert_tensors_close(expected_output[-1, :-1], blender_output[-1, :-1], atol=1e-4)
        # Only difference is at pad position, known Issue
        # with self.assertRaises(AssertionError):
        assert_tensors_close(expected_output[-1, -1], blender_output[-1, -1], atol=1e-4)

    @torch.no_grad()
    def test_decoder_layer_parity(self):
        config, input_ids, mask = self.get_config_and_data()

        torch.manual_seed(0)
        embeddings = torch.nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)

        blenderbot_model = BlenderbotForConditionalGeneration(config).to(torch_device).eval()
        bart_dec_layer = blenderbot_model.decoder.layers[0]

        parlai_decoder_layer = TransformerDecoderLayer(
            config.encoder_attention_heads,
            config.d_model,
            config.encoder_ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            variant=config.variant,
        ).eval()

        blender_encoder = blenderbot_model.encoder
        # blender_encoder.embed_tokens = embeddings

        parlai_encoder = TransformerEncoder(
            config.encoder_attention_heads,
            config.encoder_layers,
            config.d_model,
            config.encoder_ffn_dim,
            config.vocab_size,
            learn_positional_embeddings=True,
            variant=config.variant,
            n_positions=config.max_position_embeddings,
            activation=config.activation_function,
            dropout=config.dropout,
            embedding=embeddings,
            reduction_type=None,
            padding_idx=config.pad_token_id,
        ).eval()

        self._copy_layer_weights_in_blender_encoder(blender_encoder, parlai_encoder, config.encoder_layers)

        expected_encoder_output = parlai_encoder(input_ids)[0]
        bart_encoder_output = blender_encoder(input_ids, attention_mask=mask)[0]
        assert_tensors_close(expected_encoder_output[:, :-1], bart_encoder_output[:, :-1], atol=1e-4)

        self._copy_layer_weights_in_blender_decoder_layer(bart_dec_layer, parlai_decoder_layer)
        tensor = embeddings(input_ids)
        # decoder_padding
        expected_decoder_layer_output, *_ = parlai_decoder_layer.forward(
            tensor, encoder_output=expected_encoder_output, encoder_mask=mask,
        )
        causal_mask = parlai_decoder_layer._create_selfattn_mask(tensor)

        tgt_len = causal_mask.shape[1]
        causal_mask = torch.triu(fill_with_neg_inf(torch.zeros(tgt_len, tgt_len)), 1).to(
            dtype=tensor.dtype, device=tensor.device
        )
        # import ipdb; ipdb.set_trace()

        bart_mask = invert_mask(mask)
        blender_decoder_layer_output = bart_dec_layer.forward(
            tensor.transpose(1, 0),
            expected_encoder_output.transpose(1, 0),
            encoder_attn_mask=bart_mask,
            causal_mask=causal_mask,
        )[0]
        assert_tensors_close(expected_decoder_layer_output, blender_decoder_layer_output.transpose(0, 1), atol=1e-4)
        if self.variant == "xlm":
            return
        expected_slice = torch.tensor(
            [
                -1.7712,
                0.2689,
                -1.8851,
                -2.8012,
                1.3736,
                -0.7266,
                3.1182,
                1.1434,
                -1.4304,
                1.2469,
                -0.3417,
                0.3943,
                3.1211,
                3.3170,
                2.1522,
                -2.1234,
            ],
            device=torch_device,
        )
        assert_tensors_close(expected_slice, blender_decoder_layer_output[0, 0], atol=1e-4)
        # self.assertTrue(torch.allclose(expected_output, blender_output, atol=1e-4))


class Blenderbot90Tests(BlenderbotParityTests):
    """Show that small randomly initialized layers with same parameters produce same outputs. Requires PARLAI"""

    variant = "xlm"

    def get_config_and_data(self):
        torch.manual_seed(0)
        input_ids = torch.tensor(
            [
                [64, 61, 14, 42, 96, 32, 82, 7, 64, 61, 14, 42, 96, 32, 82, 7, 2],
                [94, 23, 54, 10, 10, 41, 90, 48, 94, 23, 54, 10, 10, 41, 90, 4, 2],
                # parity tests break with following padding case
                [94, 23, 54, 10, 10, 41, 90, 48, 94, 23, 54, 10, 10, 41, 90, 2, 1],
            ],
            dtype=torch.long,
            device=torch_device,
        )

        config = BlenderbotConfig(
            d_model=16,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            vocab_size=100,
            encoder_layers=1,
            decoder_layers=1,
            encoder_ffn_dim=8,
            decoder_ffn_dim=8,
            max_position_embeddings=24,
            eos_token_id=2,
            pad_token_id=1,
            bos_token_id=0,
            normalize_before=True,
            variant="xlm",
            normalize_embedding=True,
        )
        mask = (input_ids != config.pad_token_id).to(torch.long)
        return config, input_ids, mask
