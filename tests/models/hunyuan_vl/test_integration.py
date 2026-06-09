# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import importlib.util
import tempfile
import unittest

from PIL import Image


class HunYuanVLIntegrationTest(unittest.TestCase):
    def setUp(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is required")
        if importlib.util.find_spec("tokenizers") is None:
            self.skipTest("tokenizers is required")

    def get_tokenizer(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace

        from transformers.tokenization_utils_tokenizers import PreTrainedTokenizerFast

        vocab = {
            "<unk>": 0,
            "<pad>": 1,
            "<bos>": 2,
            "<eos>": 3,
            "<image_start>": 4,
            "<image>": 5,
            "<image_end>": 6,
            "hello": 7,
            "<placeholder>": 8,
        }
        tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<bos>",
            eos_token="<eos>",
            additional_special_tokens=["<image_start>", "<image>", "<image_end>", "<placeholder>"],
            extra_special_tokens={
                "image_start_token": "<image_start>",
                "image_token": "<image>",
                "image_end_token": "<image_end>",
            },
        )
        fast_tokenizer.image_start_token = "<image_start>"
        fast_tokenizer.image_token = "<image>"
        fast_tokenizer.image_end_token = "<image_end>"
        return fast_tokenizer

    def get_image_processor(self):
        from transformers.models.hunyuan_vl.image_processing_hunyuan_vl import HunYuanVLImageProcessor

        return HunYuanVLImageProcessor(
            min_pixels=32 * 32,
            max_pixels=32 * 32,
            patch_size=16,
            temporal_patch_size=1,
            merge_size=1,
        )

    def get_config(self):
        from transformers.models.hunyuan_vl.configuration_hunyuan_vl import HunYuanVLConfig

        return HunYuanVLConfig(
            attn_implementation="eager",
            text_config={
                "vocab_size": 64,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "hidden_act": "silu",
                "max_position_embeddings": 64,
                "pad_token_id": 1,
                "bos_token_id": 2,
                "eos_token_id": 3,
                "head_dim": 8,
                "rope_theta": 10000.0,
                "tie_word_embeddings": False,
            },
            vision_config={
                "num_channels": 3,
                "patch_size": 16,
                "temporal_patch_size": 1,
                "spatial_merge_size": 1,
                "num_hidden_layers": 1,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "hidden_act": "silu",
                "out_hidden_size": 16,
                "text_hidden_size": 16,
                "max_image_size": 32,
                "min_image_size": 32,
                "anyres_vit_max_image_size": 32,
                "max_vit_seq_len": 4,
            },
            image_token_id=5,
            im_start_id=4,
            im_end_id=6,
        )

    def test_auto_interfaces_image_text_roundtrip(self):
        import torch

        from transformers.models.auto.configuration_auto import AutoConfig
        from transformers.models.auto.modeling_auto import AutoModelForImageTextToText
        from transformers.models.auto.processing_auto import AutoProcessor
        from transformers.models.hunyuan_vl.configuration_hunyuan_vl import HunYuanVLConfig
        from transformers.models.hunyuan_vl.processing_hunyuan_vl import HunYuanVLProcessor

        processor = HunYuanVLProcessor(image_processor=self.get_image_processor(), tokenizer=self.get_tokenizer())
        config = self.get_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            processor.save_pretrained(tmpdir)

            loaded_config = AutoConfig.from_pretrained(tmpdir)
            self.assertIsInstance(loaded_config, HunYuanVLConfig)
            loaded_config._attn_implementation = "eager"
            loaded_config.text_config._attn_implementation = "eager"

            loaded_processor = AutoProcessor.from_pretrained(tmpdir, backend="pil")
            self.assertIsInstance(loaded_processor, HunYuanVLProcessor)

            model = AutoModelForImageTextToText.from_config(loaded_config).to("cpu")
            model.eval()

            image = Image.new("RGB", (32, 32), color="white")
            inputs = loaded_processor(text=["<image> hello"], images=[image], padding=True, return_tensors="pt")
            inputs = {key: value.to("cpu") if hasattr(value, "to") else value for key, value in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            self.assertEqual(outputs.logits.shape[:2], inputs["input_ids"].shape)
            self.assertEqual(outputs.logits.shape[-1], loaded_config.text_config.vocab_size)

    def test_auto_interfaces_text_only_generate(self):
        import torch

        from transformers.models.auto.configuration_auto import AutoConfig
        from transformers.models.auto.modeling_auto import AutoModelForImageTextToText
        from transformers.models.auto.processing_auto import AutoProcessor
        from transformers.models.hunyuan_vl.configuration_hunyuan_vl import HunYuanVLConfig
        from transformers.models.hunyuan_vl.processing_hunyuan_vl import HunYuanVLProcessor

        processor = HunYuanVLProcessor(image_processor=self.get_image_processor(), tokenizer=self.get_tokenizer())
        config = self.get_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            processor.save_pretrained(tmpdir)

            loaded_config = AutoConfig.from_pretrained(tmpdir)
            self.assertIsInstance(loaded_config, HunYuanVLConfig)
            loaded_config._attn_implementation = "eager"
            loaded_config.text_config._attn_implementation = "eager"

            loaded_processor = AutoProcessor.from_pretrained(tmpdir, backend="pil")
            self.assertIsInstance(loaded_processor, HunYuanVLProcessor)

            model = AutoModelForImageTextToText.from_config(loaded_config).to("cpu")
            model.eval()

            inputs = loaded_processor(text=["hello"], padding=True, return_tensors="pt")
            inputs = {key: value.to("cpu") if hasattr(value, "to") else value for key, value in inputs.items()}

            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=2, do_sample=False)

            self.assertEqual(generated.shape[0], inputs["input_ids"].shape[0])
            self.assertGreaterEqual(generated.shape[1], inputs["input_ids"].shape[1] + 1)
