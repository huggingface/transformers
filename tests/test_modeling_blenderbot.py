import unittest
import tempfile
from transformers import is_torch_available

if is_torch_available():
    import torch
    from transformers import (
        AutoModel,
        AutoTokenizer,
        BlenderbotTokenizer,
        BlenderbotConfig,
        BlenderbotConditionalGeneration,
        BLENDERBOT_PRETRAINED_MODEL_ARCHIVE_LIST
    )
from .utils import require_torch, slow, torch_device
from .test_modeling_common import ids_tensor, ModelTesterMixin
from .test_configuration_common import ConfigTester
from transformers.modeling_bart import _prepare_bart_decoder_inputs

weights_path = "/home/mariama/PycharmProjects/blender/blender_90M"  # won't be needed  once the pretrained weights uploaded on S3

@require_torch
class BlenderbotModelTester:
    def __init__(self,
                 parent,
                 batch_size=4,
                 seq_len=17,
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
                 activation='gelu'):
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
        
        
        
        config = BlenderbotConfig(d_model=self.hidden_size,
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
                                  pad_token_id=self.pad_token_id
                                  )
        attention_mask = ids_tensor([self.batch_size, self.seq_len], vocab_size=2)
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        return config, inputs_dict
        
@require_torch       
class BlenderbotTesterMixin(ModelTesterMixin, unittest.TestCase):
    all_generative_model_classes  = (BlenderbotConditionalGeneration,) if is_torch_available else ()
    
    is_encoder_decoder = True
    test_head_masking = False
    test_pruning = False
    test_torchscript = False
    test_resize_embeddings = True
    test_missing_keys = False
    
    def setUp(self):
        self.model_tester  = BlenderbotModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BlenderbotConfig)
        
    def test_inputs_embeds(self):
        pass
    
    def test_model_from_pretrained(self):
        for model_name in BLENDERBOT_PRETRAINED_MODEL_ARCHIVE_LIST:
            model = BlenderbotConditionalGeneration.from_pretrained(weights_path) # will change weight_path to model_name once the pretrained weights uploaded on S3
            self.assertIsNotNone(model)
          
    def test_initialization_module(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = BlenderbotConditionalGeneration(config)
        model.to(torch_device)
        model.eval()
        self.assertTrue((model.encoder.embed_tokens.weight == model.shared.weight).all().item())
        self.assertAlmostEqual(torch.std(model.encoder.embed_tokens.weight).item(), config.init_std, 2)
        self.assertAlmostEqual(torch.std(model.encoder.embed_positions.weight).item(), config.init_std, 2)   

@require_torch
class BlenderbotIntegrationTests(unittest.TestCase):
    vocab_size = 100
    @property
    def model(self):
        model = BlenderbotConditionalGeneration.from_pretrained(weights_path)
        return model.to(torch_device)
    
    def get_config_data(self):
        input_ids = torch.tensor([[ 64,  61,  14,  42,  96,  32,  82,   7, 64,  61,  14,  42,  96,  32,  82,   7, 2],
                    [ 94,  23,  54,  10,  10,  41,  90,  48, 94,  23,  54,  10,  10,  41,  90,  48, 2],
                    [ 63,  98,  72,   8,  37, 99,  54,  93,  63,  98,  72,   8,  37, 99,  54,  93,  2],
                    [ 98,  82,   1,  58,  74,  88, 99,  12, 98,  82,   1,  58,  74,  88, 99,  12,   2]], dtype=torch.long, device=torch_device)
        mask = torch.tensor([[0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1],
                    [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1 ],
                    [1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1]], dtype=torch.long, device=torch_device)
        
        batch_size = input_ids.size(0)
        config = BlenderbotConfig(hidden_size=16,vocab_size=self.vocab_size, encoder_layers=2, decoder_layers=2,
                                  encoder_attention_heads=2,
                                  decoder_attention_heads=2,
                                  encoder_ffn_dim=8,
                                  decoder_ffn_dim=8,
                                  max_position_embeddings=24,
                                  eos_token_id=2,
                                  pad_token_id=0,
                                  bos_token_id=1)
        return config, input_ids, mask, batch_size
    def test_generation_with_labels_forward(self):
        config, input_ids, mask, batch_size = self.get_config_data()
        labels = input_ids
        model = BlenderbotConditionalGeneration(config)
        model.to(torch_device)
        outputs = model(input_ids=input_ids,
                        attention_mask=mask,
                        decoder_input_ids=input_ids,
                        labels=labels)
        loss = outputs[0]
        self.assertIsInstance(loss.item(), float)   
        scores = outputs[1]
        
        expected_shape = (batch_size, input_ids.size(1), self.vocab_size)
        self.assertEqual(scores.size(),expected_shape)
        
    def test_generate_beam_search(self):
        config, input_ids, mask, batch_size = self.get_config_data()
       
        model = BlenderbotConditionalGeneration(config)
        model.eval()
        max_seq_len = 10
        generated_ids = model.generate(
                input_ids.clone(),
                do_sample=True,
                num_return_sequences=1,
                num_beams=2,
                no_repeat_ngram_size=3,
                max_length=max_seq_len
        )
        self.assertTrue(generated_ids.size(), (batch_size, max_seq_len))    
     
    @unittest.skipIf(torch_device == "cpu", "half precision can't be used on cpu")    
    def test_model_fp16_forward(self):
        config, input_ids, mask, batch_size = self.get_config_data()
        model = BlenderbotConditionalGeneration(config=config)
        model.to(torch_device)
        model.half()
        model.eval()
        output = model(input_ids, attention_mask=mask)[0]
        self.assertFalse(torch.isnan(output).any().item())

    @unittest.skipIf(torch_device == "cpu", "half precision can't be used on cpu")
    def test_model_fp16_generate(self, config, input_ids, input_mask):
        config, input_ids, mask, batch_size = self.get_config_data()
        model = BlenderbotConditionalGeneration(config=config)
        model.to(torch_device)
        model.half()
        model.eval()
        output = model.generate(input_ids, attention_mask=mask, do_sample=False)
        self.assertFalse(torch.isnan(output).any().item())
        
    @slow
    def test_inference(self):
        model = BlenderbotConditionalGeneration.from_pretrained(weights_path).to(device)
        config, input_ids, mask, batch_size = self.get_config_data()
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": mask
        }
        with torch.no_grad():
            output = model(**inputs_dict)[0]
        expected_shape = torch.Size((batch_size, input_ids.size(1), model.config.vocab_size))
        self.assertEqual(output.size(), expected_shape)
        
