# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch ModernVBERT model."""

import tempfile
import unittest

import requests
from huggingface_hub import hf_hub_download
from PIL import Image
from typing import ClassVar

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    ModernVBertConfig,
    ModernVBertForMaskedLM,
    ModernVBertModel,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    require_torch,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
)


if is_torch_available():
    import torch


if is_vision_available():
    from PIL import Image


class ModernVBertModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        num_images=2,
        text_config={
            "vocab_size": 99,
            "pad_token_id": 0,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 64,
            "hidden_activation": "gelu",
            "mlp_dropout": 0.1,
            "attention_dropout": 0.1,
            "embedding_dropout": 0.1,
            "classifier_dropout": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "is_decoder": False,
            "initializer_range": 0.02,
            "tie_word_embeddings": False,
        },
        is_training=True,
        vision_config={
            "image_size": 16,
            "patch_size": 4,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 32,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "initializer_range": 0.02,
        },
        pixel_shuffle_factor=2,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.text_config = text_config
        self.vision_config = vision_config
        self.num_images = num_images
        self.image_token_id = self.text_config["vocab_size"] - 1
        self.image_size = vision_config["image_size"]
        self.pixel_shuffle_factor = pixel_shuffle_factor
        self.seq_length = (
            int(((vision_config["image_size"] // vision_config["patch_size"]) ** 2) / (pixel_shuffle_factor**2))
            * self.num_images
        )

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training

    def get_config(self):
        return ModernVBertConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            image_token_id=self.image_token_id,
            pixel_shuffle_factor=self.pixel_shuffle_factor,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([
            self.batch_size, 
            self.num_images, 
            3, 
            self.image_size, 
            self.image_size
        ])
        config = self.get_config()

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(torch_device)

        # For simplicity just set the last n tokens to the image token
        n_image_tokens_per_batch = self.seq_length
        input_ids[:, -n_image_tokens_per_batch:] = self.image_token_id
        attention_mask = input_ids.ne(1).to(torch_device)
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class ModernVBertModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Model tester for `ModernVBertForMaskedLM`.
    """

    all_model_classes = (
        (
            ModernVBertModel,
            ModernVBertForMaskedLM,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {"feature-extraction": ModernVBertModel, "fill-mask": ModernVBertForMaskedLM} if is_torch_available() else {}
    )

    _is_composite = True

    def setUp(self):
        self.model_tester = ModernVBertModelTester(self)
        self.config_tester = ConfigTester(
            self, 
            config_class=ModernVBertConfig, 
            has_text_modality=True, 
            has_vision_modality=True
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_sdpa_can_dispatch_composite_models(self):
        """
        Tests if composite models dispatch correctly on SDPA/eager when requested so when loading the model.
        This tests only by looking at layer names, as usually SDPA layers are called "SDPAAttention".
        In contrast to the above test, this one checks if the "config._attn_implementation" is a dict after the model
        is loaded, because we manually replicate requested attn implementation on each sub-config when loading.
        See https://github.com/huggingface/transformers/pull/32238 for more info

        The test tries to cover most general cases of composite models, VLMs with vision and text configs. Any model
        that has a different set of sub-configs has to overwrite this test.
        """
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if not self._is_composite:
            self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

        model_class = self.all_model_classes[0] # only ModernVBertModel is composite
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = model_class(config)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            model_sdpa = model_class.from_pretrained(tmpdirname)
            model_sdpa = model_sdpa.eval().to(torch_device)

            vision_model_names = {"visual", "image_tower", "vision_tower", "vision_model"}
            language_model_names = {"language_model", "model", "text_model"}
            vision_model_name = [name for name in vision_model_names if hasattr(model_sdpa, name)][0]
            language_model_name = [name for name in language_model_names if hasattr(model_sdpa, name)][0]

            vision_model_sdpa = getattr(model_sdpa, vision_model_name)
            language_model_sdpa = getattr(model_sdpa, language_model_name)
            text_attn = "sdpa" if language_model_sdpa._supports_sdpa else "eager"
            vision_attn = "sdpa" if vision_model_sdpa._supports_sdpa else "eager"

            # `None` as it is the requested one which will be assigned to each sub-config
            # Sub-model will dispatch to SDPA if it can (checked below that `SDPA` layers are present)
            self.assertTrue(language_model_sdpa.config._attn_implementation == text_attn)
            self.assertTrue(vision_model_sdpa.config._attn_implementation == vision_attn)

            model_eager = model_class.from_pretrained(tmpdirname, attn_implementation="eager")
            model_eager = model_eager.eval().to(torch_device)
            self.assertTrue(getattr(model_eager, language_model_name).config._attn_implementation == "eager")
            self.assertTrue(getattr(model_eager, vision_model_name).config._attn_implementation == "eager")

            for name, submodule in model_eager.named_modules():
                class_name = submodule.__class__.__name__
                if (
                    class_name.endswith("Attention")
                    and getattr(submodule, "config", None)
                    and submodule.config._attn_implementation == "sdpa"
                ):
                    raise ValueError("The eager model should not have SDPA attention layers")

    # skip test_training_gradient_checkpointing
    @unittest.skip(
        reason="ModernVBertModel does not implement gradient checkpointing."
    )
    def test_training_gradient_checkpointing(self):
        pass

    # skip test_training_gradient_checkpointing_use_reentrant
    @unittest.skip(
        reason="ModernVBertModel does not implement gradient checkpointing."
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="ModernVBertModel does not implement gradient checkpointing."
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass


@require_torch
class ModernVBertForMaskedLMIntegrationTest(unittest.TestCase):
    model_name: ClassVar[str] = "ModernVBERT/modernvbert"

    def setUp(self):
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.image = Image.open(hf_hub_download("HuggingFaceTB/SmolVLM", "example_images/rococo.jpg", repo_type="space"))
        self.text = "This [MASK] is on the wall."

    @slow
    def test_masked_lm_inference(self):
        model = ModernVBertForMaskedLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map=torch_device
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.text},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=False)
        inputs = self.processor(text=prompt, images=[self.image], return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        masked_index = inputs["input_ids"][0].tolist().index(self.tokenizer.mask_token_id)
        masked_token_logits = outputs.logits[0, masked_index, :]
        masked_token_probs = torch.softmax(masked_token_logits, dim=-1)
        top_5_probs, top_5_indices = torch.topk(masked_token_probs, k=5, dim=-1)

        EXPECTED_TOP_5_INDICES = torch.tensor([13497, 5406, 2460, 22946, 3665], device=torch_device)
        EXPECTED_TOP_5_VALUES = torch.tensor([0.4986, 0.3550, 0.0415, 0.0235, 0.0199], device=torch_device)

        self.assertTrue(torch.allclose(top_5_indices, EXPECTED_TOP_5_INDICES))
        self.assertTrue(torch.allclose(top_5_probs, EXPECTED_TOP_5_VALUES, atol=1e-4, rtol=1e-4))