# Copyright 2026 The HuggingFace Team. All rights reserved.

import unittest

import torch

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    CogVLM2Config,
    CogVLM2ForConditionalGeneration,
)
from transformers.models.cogvlm2.modeling_cogvlm2 import build_position_ids
from transformers.testing_utils import require_torch


@require_torch
class CogVLM2ModelTest(unittest.TestCase):
    def get_config(self):
        return CogVLM2Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_multi_query_heads=2,
            max_position_embeddings=128,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            vision_config={
                "hidden_size": 32,
                "image_size": 28,
                "in_channels": 3,
                "intermediate_size": 64,
                "num_heads": 4,
                "num_hidden_layers": 2,
                "num_positions": 17,
                "patch_size": 7,
            },
        )

    def get_inputs(self):
        input_ids = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 5, 6, 7]])
        token_type_ids = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 0, 0, 0]])
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values_videos": torch.randn(1, 3, 28, 28),
        }

    def test_position_ids_compress_visual_span(self):
        token_type_ids = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 0, 0, 0]])
        position_ids = build_position_ids(token_type_ids)
        self.assertEqual(position_ids.tolist(), [[0, 1, 2, 2, 2, 2, 3, 4, 5, 6]])

    def test_forward_and_cache(self):
        model = CogVLM2ForConditionalGeneration(self.get_config()).eval()
        inputs = self.get_inputs()
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)
        self.assertEqual(outputs.logits.shape, (1, 10, 128))
        self.assertEqual(outputs.past_key_values.get_seq_length(), 10)
        self.assertTrue(torch.isfinite(outputs.logits).all())

    def test_visual_placeholder_mismatch_raises(self):
        model = CogVLM2ForConditionalGeneration(self.get_config()).eval()
        inputs = self.get_inputs()
        inputs["token_type_ids"][0, 6] = 0
        with self.assertRaisesRegex(ValueError, "visual placeholder count"):
            model(**inputs)

    def test_hidden_states_and_eager_attentions(self):
        config = self.get_config()
        config._attn_implementation = "eager"
        model = CogVLM2ForConditionalGeneration(config).eval()
        with torch.no_grad():
            outputs = model(
                **self.get_inputs(),
                output_hidden_states=True,
                output_attentions=True,
                use_cache=False,
            )
        self.assertEqual(len(outputs.hidden_states), config.num_hidden_layers + 1)
        self.assertEqual(len(outputs.attentions), config.num_hidden_layers)
        self.assertEqual(outputs.attentions[0].shape, (1, 4, 10, 10))

    def test_generate_with_video_and_cache(self):
        model = CogVLM2ForConditionalGeneration(self.get_config()).eval()
        inputs = self.get_inputs()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=2, do_sample=False)
        self.assertEqual(outputs.shape, (1, 12))

    def test_auto_classes(self):
        config = AutoConfig.for_model(
            "cogvlm2",
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_multi_query_heads=2,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            vision_config={
                "hidden_size": 32,
                "image_size": 28,
                "in_channels": 3,
                "intermediate_size": 64,
                "num_heads": 4,
                "num_hidden_layers": 1,
                "num_positions": 17,
                "patch_size": 7,
            },
        )
        self.assertEqual(type(AutoModel.from_config(config)).__name__, "CogVLM2Model")
        self.assertEqual(
            type(AutoModelForCausalLM.from_config(config)).__name__,
            "CogVLM2ForConditionalGeneration",
        )


if __name__ == "__main__":
    unittest.main()
