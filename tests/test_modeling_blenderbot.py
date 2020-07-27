import unittest

from transformers import BlenderbotConfig, BlenderbotTokenizer, is_torch_available
from transformers.file_utils import cached_property
from transformers.testing_utils import require_torch, slow, torch_device
from transformers.tokenization_blenderbot import BlenderbotSmallTokenizer

from .test_configuration_common import ConfigTester
from .test_modeling_bart import _long_tensor, assert_tensors_close
from .test_modeling_common import ModelTesterMixin, ids_tensor


if is_torch_available():
 import torch
 from transformers import BlenderbotForConditionalGeneration


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
    bos_token_id=0,
    pad_token_id=1,
    use_labels=True,
    ffn_size=4,
    attention_dropout=0.2,
    activation="gelu",
    variant='xlm',
    scale_embedding=True
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
    self.variant = variant
    self.scale_embedding = scale_embedding
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
    variant=self.variant,
    scale_embedding=self.scale_embedding,
    bos_token_id=self.bos_token_id,
    eos_token_id=self.eos_token_id,
    pad_token_id=self.pad_token_id,
    num_beams=1,
    min_length=3,
    max_length=10,
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
 test_torchscript = True
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
class AbstractBlenderBotIntegrationTests(unittest.TestCase):
 checkpoint_name = "sshleifer/blenderbot-3B"
 tokenizer_cls = BlenderbotTokenizer

 @cached_property
 def model(self):
  model = BlenderbotForConditionalGeneration.from_pretrained(self.checkpoint_name).to(torch_device)
  if torch_device == "cuda":
   model = model.half()
  return model

 @cached_property
 def tokenizer(self):
  return self.tokenizer_cls.from_pretrained(self.checkpoint_name)


#@unittest.skipUnless(torch_device != "cpu", "3b test are very slow on CPU.")
@require_torch
class Blenderbot3BIntegrationTests(AbstractBlenderBotIntegrationTests):
 @torch.no_grad()
 @slow
 def test_generation_same_as_parlai_3B(self):
  src_text = [
   "sam",
  ]

  model_inputs = self.tokenizer(src_text, return_tensors="pt").to(torch_device)
  generated_utterances = self.model.generate(**model_inputs)
  tgt_text = ['Sam is a great name for a boy or a girl. It means "Sun" in Gaelic.']
  """
  Batch[  0] Beam[  0]: (-6.57): Sam is a great name. It means "sunshine" in Gaelic.
  Batch[  0] Beam[  0]: tokens: tensor([   1, 5502,  315,  265,  848, 1356,   21,  452,
 1361,  472,   90,  415,
   803,  556, 9,  302,  485,   72,  491,  317,   21, 2],
    device='cuda:0')
  """
  generated_txt = self.tokenizer.batch_decode(generated_utterances)
  self.assertListEqual(tgt_text, generated_txt)

 @torch.no_grad()
 @slow
 def test_loss_same_as_parlai_3B(self):
  input_ids = _long_tensor([[268, 343, 2]])  # sam

  generated_ids = self.model.generate(input_ids).tolist()[0]
  expected_ids = [
   1,
   5502,
   315,
   265,
   848,
   1356,
   21,
   452,
   1361,
   472,
   90,
   415,
   803,
   556,
   9,
   302,
   485,
   72,
   491,
   317,
   21,
   2,
  ]
  self.assertListEqual(expected_ids, generated_ids)


@require_torch
class Blenderbot90MIntegrationTests(AbstractBlenderBotIntegrationTests):
  checkpoint_name = "sshleifer/blenderbot-90M"
  tokenizer_cls = BlenderbotSmallTokenizer

  def test_tokenization_same_as_parlai(self):
    tok = self.tokenizer
    self.assertListEqual(tok("sam").input_ids, [1384])     

  @torch.no_grad()
  @slow
  def test_generation_same_as_parlai_90(self):
    src_text = [
    "Social anxiety\nWow, I am never shy. Do you have anxiety?\nYes. I end up sweating and blushing and feel like i'm going to throw up.\nand why is that?"
    ]
    tgt_text = ["__start__ i ' m not sure . i just feel like i ' m going to throw up . __end__"]  

    
    model_inputs = self.tokenizer(src_text, return_tensors="pt").to(torch_device)
    generated_utterances = self.model.generate(**model_inputs, min_length=15, length_penalty=0.65, max_length=128, early_stopping=True)
   
    self.assertListEqual(tgt_text, self.tokenizer.batch_decode(generated_utterances))

  @torch.no_grad()
  @slow
  def test_generation_same_as_parlai_90_short_input(self):
    input_ids = _long_tensor([[1384]])  # sam
    assert self.model.config.variant == "xlm"

    encoder_output = self.model.encoder(input_ids)[0]
    assert encoder_output.shape == (1, 1, 512)
    generated_utterances = self.model.generate(
    input_ids, min_length=20, length_penalty=1.0, max_length=128, early_stopping=True
    ).tolist()
    expected_tokens = [
    1,
    49,
    15,
    286,
    474,
    10,
    1384,
    5186,
    20,
    21,
    8,
    17,
    50,
    241,
    1789,
    6,
    6299,
    6,
    9,
    2147,
    5,
    ]  # FIXME, there should be a 2 here

    # PARLAI
    """
    Batch[  0] Beam[  0]: (-7.73): have you ever heard of sam harris ? he ' s an american singer , songwriter , and actor .  
    tokens: tensor([   1,   49,   15,  286,  474,   10, 1384, 5186,   20,   21, 8,   17,
      50,  241, 1789, 6, 6299, 6, 9, 2147, 5, 2])
    """
    self.assertListEqual(expected_tokens, generated_utterances[0])
