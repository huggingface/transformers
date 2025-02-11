import tempfile
import unittest

from transformers import LlavaConfig


class LlavaConfigTest(unittest.TestCase):
    def test_llava_reload(self):
        """
        Simple test for reloading default llava configs
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = LlavaConfig()
            config.save_pretrained(tmp_dir)

            reloaded = LlavaConfig.from_pretrained(tmp_dir)
            assert config.to_dict() == reloaded.to_dict()

    def test_pixtral_reload(self):
        """
        Simple test for reloading pixtral configs
        """
        vision_config = {
            "model_type": "pixtral",
            "head_dim": 64,
            "hidden_act": "silu",
            "image_size": 1024,
            "is_composition": True,
            "patch_size": 16,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
        }

        text_config = {
            "model_type": "mistral",
            "hidden_size": 5120,
            "head_dim": 128,
            "num_attention_heads": 32,
            "intermediate_size": 14336,
            "is_composition": True,
            "max_position_embeddings": 1024000,
            "num_hidden_layers": 40,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-05,
            "rope_theta": 1000000000.0,
            "sliding_window": None,
            "vocab_size": 131072,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = LlavaConfig(vision_config=vision_config, text_config=text_config)
            config.save_pretrained(tmp_dir)

            reloaded = LlavaConfig.from_pretrained(tmp_dir)
            assert config.to_dict() == reloaded.to_dict()

    def test_arbitrary_reload(self):
        """
        Simple test for reloading arbirarily composed subconfigs
        """
        default_values = LlavaConfig().to_dict()
        default_values["vision_config"]["model_type"] = "qwen2_vl"
        default_values["text_config"]["model_type"] = "opt"

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = LlavaConfig(**default_values)
            config.save_pretrained(tmp_dir)

            reloaded = LlavaConfig.from_pretrained(tmp_dir)
            assert config.to_dict() == reloaded.to_dict()
