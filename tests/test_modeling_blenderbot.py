import tempfile
import unittest

from transformers import is_torch_available
from transformers.file_utils import cached_property
from transformers.modeling_bart import _prepare_bart_decoder_inputs

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, ids_tensor
#from .utils import require_torch, slow, torch_device
from transformers.testing_utils import require_torch, slow, torch_device


#parlai import to test Blenderbot outputs agains parlai (will be removed at the end)
from parlai.agents.transformer.modules import MultiHeadAttention, TransformerEncoderLayer, TransformerDecoderLayer, TransformerEncoder, TransformerDecoder

if is_torch_available():
    import torch
    from transformers import (
        AutoModel,
        AutoTokenizer,
        BlenderbotTokenizer,
        BlenderbotConfig,
        BlenderbotForConditionalGeneration,
        BLENDERBOT_PRETRAINED_MODEL_ARCHIVE_LIST,
    )


weights_path = "sshleifer/blenderbot-3B"


@require_torch
class BlenderbotModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_len=10,
        vocab_size=100,
        hidden_size=16,
        is_training=False,
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_dropout_prob=0.2,
        max_position_embeddings=50,
        eos_token_id=2,
        bos_token_id=1,
        pad_token_id=0,
        use_labels=True,
        ffn_size=4,
        attention_dropout=0.2,
        activation="gelu",
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.is_training = is_training
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.max_position_embeddings = max_position_embeddings
        self.use_labels = use_labels
        self.ffn_size = ffn_size
        self.activation = activation
        self.attention_dropout = attention_dropout
        torch.manual_seed(0)

    def prepare_config_and_inputs_for_common(self):
        input_ids = ids_tensor([self.batch_size, self.seq_len], self.vocab_size)

        config = BlenderbotConfig(
            d_model=self.hidden_size,
            dropout=self.hidden_dropout_prob,
            vocab_size=self.vocab_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            attention_dropout=self.attention_dropout,
            encoder_ffn_dim=self.ffn_size,
            decoder_ffn_dim=self.ffn_size,
            max_position_embeddings=self.max_position_embeddings,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )
        attention_mask = ids_tensor([self.batch_size, self.seq_len], vocab_size=2)
        inputs_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        return config, inputs_dict


@require_torch
class BlenderbotTesterMixin(ModelTesterMixin, unittest.TestCase):
    all_generative_model_classes = (BlenderbotForConditionalGeneration,) if is_torch_available else ()

    is_encoder_decoder = True
    test_head_masking = False
    test_pruning = False
    test_torchscript = False
    test_resize_embeddings = True
    test_missing_keys = False

    def setUp(self):
        self.model_tester = BlenderbotModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BlenderbotConfig)

    def test_inputs_embeds(self):
        pass

    def test_initialization_module(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = BlenderbotForConditionalGeneration(config)
        model.to(torch_device)
        model.eval()
        self.assertTrue((model.encoder.embed_tokens.weight == model.shared.weight).all().item())
        self.assertAlmostEqual(torch.std(model.encoder.embed_tokens.weight).item(), config.init_std, 2)
        self.assertAlmostEqual(torch.std(model.encoder.embed_positions.weight).item(), config.init_std, 2)

    def test_embed_pos_shape(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = BlenderbotForConditionalGeneration(config)
        expected_shape = (config.max_position_embeddings, config.d_model)
        self.assertEqual(model.encoder.embed_positions.weight.shape, expected_shape)
        self.assertEqual(model.decoder.embed_positions.weight.shape, expected_shape)
        


@require_torch
class BlenderbotIntegrationTests(unittest.TestCase):
    def get_config_and_data(self):
        input_ids = torch.tensor(
            [
                [64, 61, 14, 42, 96, 32, 82, 7, 64, 61, 14, 42, 96, 32, 82, 7, 2],
                [94, 23, 54, 10, 10, 41, 90, 48, 94, 23, 54, 10, 10, 41, 90, 48, 2]
            ],
            dtype=torch.long,
            device=torch_device,
        )
        

        config = BlenderbotConfig(
            d_model=16,
            vocab_size=100,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=8,
            decoder_ffn_dim=8,
            max_position_embeddings=24,
            eos_token_id=2,
            pad_token_id=0,
            bos_token_id=1,
        )
        mask = (input_ids != config.pad_token_id).to(torch.long)
        return config, input_ids, mask

    @cached_property
    def model(self):
        model = BlenderbotForConditionalGeneration.from_pretrained(weights_path).to(torch_device)
        if torch_device == "cuda":
            model = model.half()
        return model
    
    def _set_param(self, blender_layer, parlai_layer, bias=None):
        with torch.no_grad():
            assert blender_layer.weight.shape == parlai_layer.weight.shape, "{} layer.weight does not match".format(layer)
            blender_layer.weight = torch.nn.Parameter(parlai_layer.weight)
            if bias is not None:
                assert blender_layer.bias.shape == parlai_layer.bias.shape, "{} layer.bias does not match".format(layer)
                blender_layer.bias = torch.nn.Parameter(parlai_layer.bias)
        return True
    
    def _copy_layer_weights_in_blender_self_attn(self, blender_layer, parlai_layer):
        self._set_param(blender_layer.q_proj, parlai_layer.q_lin, bias=True)
        self._set_param(blender_layer.v_proj, parlai_layer.v_lin, bias=True)
        self._set_param(blender_layer.k_proj, parlai_layer.k_lin, bias=True)
        self._set_param(blender_layer.out_proj, parlai_layer.out_lin, bias=True)
    
    def _copy_layer_weights_in_blender_encoder_layer(self, blender_layer, parlai_layer):
        self._copy_layer_weights_in_blender_self_attn(blender_layer.self_attn, parlai_layer.attention)
        self._set_param(blender_layer.self_attn_layer_norm, parlai_layer.norm1, bias=True)
        self._set_param(blender_layer.fc1, parlai_layer.ffn.lin1, bias=True)
        self._set_param(blender_layer.fc2, parlai_layer.ffn.lin2, bias=True)
        self._set_param(blender_layer.final_layer_norm, parlai_layer.norm2, bias=True)
        
    def _copy_layer_weights_in_blender_decoder_layer(self, blender_layer, parlai_layer):
        self._copy_layer_weights_in_blender_self_attn(blender_layer.self_attn, parlai_layer.self_attention)
        self._copy_layer_weights_in_blender_self_attn(blender_layer.encoder_attn, parlai_layer.encoder_attention)
        self._set_param(blender_layer.self_attn_layer_norm, parlai_layer.norm1, bias=True)
        self._set_param(blender_layer.fc1, parlai_layer.ffn.lin1, bias=True)
        self._set_param(blender_layer.fc2, parlai_layer.ffn.lin2, bias=True)
        self._set_param(blender_layer.final_layer_norm, parlai_layer.norm3, bias=True)
        self._set_param(blender_layer.encoder_attn_layer_norm, parlai_layer.norm2, bias=True)
        
    def _copy_layer_weights_in_blender_encoder(self, blender_encoder, parlai_encoder, num_layers):
        for i in range(num_layers):
            self._copy_layer_weights_in_blender_encoder_layer(blender_encoder.layers[i], parlai_encoder.layers[i])
        self._set_param(blender_encoder.layer_norm, parlai_encoder.norm_embeddings, bias=True)
        self._set_param(blender_encoder.embed_positions, parlai_encoder.position_embeddings)
        self._set_param(blender_encoder.embed_tokens, parlai_encoder.embeddings)
    
    def test_blenderbot_encoder_selfAttention_forward(self):
        config, input_ids, mask = self.get_config_and_data()
        
        hidden_states = torch.tensor(2*[5*[config.d_model*[0.1]]])
        mask = torch.ones(hidden_states.shape[:2])
        
        torch.manual_seed(0)
        
        blenderbot_model = BlenderbotForConditionalGeneration(config).to(torch_device)
        self_attn = blenderbot_model.encoder.layers[0].self_attn
        self_attn.eval()
        
        parlai_attn = MultiHeadAttention(config.encoder_attention_heads, config.d_model, dropout=config.dropout)
        parlai_attn.eval()
        
        self._copy_layer_weights_in_blender_self_attn(self_attn, parlai_attn)
        expected_output = parlai_attn(query=hidden_states, mask=mask, key=None)[0]
        blender_output = self_attn(query=hidden_states.transpose(1,0), key_padding_mask=mask, key=None)[0].transpose(1,0)
        self.assertTrue(torch.allclose(expected_output, blender_output, atol=1e-3))
        
    def test_blenderbot_encoder_layer_forward(self):
        config, input_ids, mask = self.get_config_and_data()
        
        hidden_states = torch.tensor(2*[5*[config.d_model*[0.1]]])
        mask = torch.ones(hidden_states.shape[:2])
        
        torch.manual_seed(0)
        
        blenderbot_model = BlenderbotForConditionalGeneration(config).to(torch_device)
        blender_encoder_layer = blenderbot_model.encoder.layers[0]
        blender_encoder_layer.eval()
        
        parlai_encoder_layer = TransformerEncoderLayer(config.encoder_attention_heads, config.d_model, config.encoder_ffn_dim, 
                                                       dropout=config.dropout, activation='gelu', variant='prelayernorm')
        parlai_encoder_layer.eval()
        
        self._copy_layer_weights_in_blender_encoder_layer(blender_encoder_layer, parlai_encoder_layer)
        expected_output = parlai_encoder_layer(hidden_states, mask)
        blender_output = blender_encoder_layer(hidden_states.transpose(1,0), encoder_padding_mask=mask)[0].transpose(1,0)
        self.assertTrue(torch.allclose(expected_output, blender_output, atol=1e-4))
        
    def test_blenderbot_encoder_forward(self):
        config, input_ids, mask = self.get_config_and_data()
        embeddings = torch.nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        
        torch.manual_seed(0)
        
        blenderbot_model = BlenderbotForConditionalGeneration(config).to(torch_device)
        blender_encoder = blenderbot_model.encoder
        #blender_encoder.embed_tokens = embeddings
        blender_encoder.eval()
        
        parlai_encoder = TransformerEncoder(config.encoder_attention_heads, config.encoder_layers, config.d_model, config.encoder_ffn_dim,
                                            config.vocab_size, learn_positional_embeddings=True, variant='prelayernorm',n_positions=config.max_position_embeddings,
                                            activation=config.activation_function, dropout=config.dropout, embedding=embeddings, reduction_type=None)
        parlai_encoder.eval()
        
        self._copy_layer_weights_in_blender_encoder(blender_encoder, parlai_encoder, config.encoder_layers)
        
        expected_output = parlai_encoder(input_ids)[0]
        blender_output = blender_encoder(input_ids, attention_mask=mask)[0]
        self.assertTrue(torch.allclose(expected_output, blender_output, atol=1e-3))
        
    def test_blenderbot_decoder_layer_forward(self):
        config, input_ids, mask = self.get_config_and_data()
        
        torch.manual_seed(0)
        embeddings = torch.nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        
        blenderbot_model = BlenderbotForConditionalGeneration(config).to(torch_device)
        blender_decoder_layer = blenderbot_model.decoder.layers[0]
        blender_decoder_layer.eval()
        
        parlai_decoder_layer = TransformerDecoderLayer(config.encoder_attention_heads, config.d_model, config.encoder_ffn_dim, 
                                                       dropout=config.dropout, activation='gelu', variant='prelayernorm')
        parlai_decoder_layer.eval()
        
        blender_encoder = blenderbot_model.encoder
        #blender_encoder.embed_tokens = embeddings
        blender_encoder.eval()
        
        parlai_encoder = TransformerEncoder(config.encoder_attention_heads, config.encoder_layers, config.d_model, config.encoder_ffn_dim,
                                            config.vocab_size, learn_positional_embeddings=True, variant='prelayernorm',n_positions=config.max_position_embeddings,
                                            activation=config.activation_function, dropout=config.dropout, embedding=embeddings, reduction_type=None)
        parlai_encoder.eval()
        
        self._copy_layer_weights_in_blender_encoder(blender_encoder, parlai_encoder, config.encoder_layers)
        
        expected_encoder_output = parlai_encoder(input_ids)[0]
        blender_encoder_output = blender_encoder(input_ids, attention_mask=mask)[0]
        self.assertTrue(torch.allclose(expected_encoder_output, blender_encoder_output, atol=1e-3))
        
        self._copy_layer_weights_in_blender_decoder_layer(blender_decoder_layer, parlai_decoder_layer)
        tensor = embeddings(input_ids)
        expected_decoder_layer_output = parlai_decoder_layer(tensor, encoder_output=expected_encoder_output, encoder_mask=mask)
        
        blender_decoder_layer_output = blender_decoder_layer(tensor.transpose(1,0), encoder_hidden_states=blender_encoder_output, encoder_attn_mask=mask)[0] 
        print(blender_decoder_layer_output)
        # self.assertTrue(torch.allclose(expected_output, blender_output, atol=1e-4))

    @slow
    def test_inference(self):
        config, input_ids, mask, batch_size = self.get_config_data()
        inputs_dict = {"input_ids": input_ids, "attention_mask": mask}
        with torch.no_grad():
            output = self.model(**inputs_dict)[0]
        expected_shape = torch.Size((batch_size, input_ids.size(1), self.model.config.vocab_size))
        self.assertEqual(output.size(), expected_shape)
