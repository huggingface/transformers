import unittest

from transformers import IsaacConfig, IsaacForConditionalGeneration, IsaacModel, is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ids_tensor


if is_torch_available():
    import torch


class IsaacModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=5,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        self.text_config = {
            "bos_token_id": 0,
            "eos_token_id": 1,
            "pad_token_id": 2,
            "hidden_act": "silu",
            "head_dim": hidden_size // num_attention_heads,
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "intermediate_size": hidden_size * 3,
            "max_position_embeddings": 128,
            "model_type": "qwen3",
            "num_attention_heads": num_attention_heads,
            "num_hidden_layers": num_hidden_layers,
            "num_key_value_heads": num_attention_heads,
            # Keep the same multi-RoPE setup as the reference checkpoints but shrink the
            # sections so they sum to the rotary half-dimension (4) of this tiny test model.
            "rope_parameters": {"rope_type": "default", "mrope_section": [2, 1, 1], "mrope_interleaved": True},
            # Qwen3 config expects `rope_theta` to be present on the text sub-config, so we
            # set it explicitly to mimic real checkpoints and keep attribute mirroring working.
            "rope_theta": 10000,
            "tie_word_embeddings": True,
        }

        self.vision_config = {
            "hidden_size": hidden_size,
            "intermediate_size": hidden_size * 2,
            "num_hidden_layers": 1,
            "num_attention_heads": num_attention_heads,
            "num_channels": 3,
            "num_patches": 64,
            "patch_size": 4,
            "pixel_shuffle_scale_factor": 1,
            "attention_dropout": 0.0,
            "layer_norm_eps": 1e-6,
        }

    def get_config(self):
        config = IsaacConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
        )
        # Rely on vanilla SDPA so the tests do not need flash attention.
        config._attn_implementation = "sdpa"
        config.text_config._attn_implementation = "sdpa"
        config.vision_attn_implementation = "sdpa"
        return config

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = torch.ones(
            (self.batch_size, self.seq_length),
            dtype=torch.long,
            device=torch_device,
        )
        labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        return config, input_ids, attention_mask, labels


@require_torch
class IsaacModelTest(unittest.TestCase):
    all_model_classes = (IsaacModel, IsaacForConditionalGeneration) if is_torch_available() else ()

    def setUp(self):
        self.model_tester = IsaacModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=IsaacConfig,
            has_text_modality=True,
            common_properties=["hidden_size"],
            text_config=self.model_tester.text_config,
            vision_config=self.model_tester.vision_config,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model_forward(self):
        config, input_ids, attention_mask, _ = self.model_tester.prepare_config_and_inputs()
        model = IsaacModel(config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_ids=input_ids, attention_mask=attention_mask)

        self.assertEqual(
            result.last_hidden_state.shape,
            (self.model_tester.batch_size, self.model_tester.seq_length, config.hidden_size),
        )

    def test_for_conditional_generation(self):
        config, input_ids, attention_mask, labels = self.model_tester.prepare_config_and_inputs()
        model = IsaacForConditionalGeneration(config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        self.assertEqual(
            result.logits.shape,
            (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.vocab_size),
        )
        self.assertIsNotNone(result.loss)

    def test_prepare_inputs_for_generation(self):
        config, input_ids, attention_mask, _ = self.model_tester.prepare_config_and_inputs()
        model = IsaacForConditionalGeneration(config)
        model.to(torch_device)

        prepared_inputs = model.prepare_inputs_for_generation(input_ids=input_ids, attention_mask=attention_mask)
        self.assertIn("input_ids", prepared_inputs)
        self.assertIn("position_ids", prepared_inputs)
        self.assertIsNone(prepared_inputs["position_ids"])
