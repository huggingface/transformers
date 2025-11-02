# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from transformers import is_torch_available
from transformers.testing_utils import require_torch, slow


if is_torch_available():
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image

    from transformers import AutoProcessor, AutoTokenizer, ModernVBertForMaskedLM


@require_torch
class ModernVBertIntegrationTest(unittest.TestCase):
    @slow
    def test_masked_lm_prediction(self):
        model_id = "ModernVBERT/modernvbert"

        processor = AutoProcessor.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = ModernVBertForMaskedLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # use torch_dtype=torch.bfloat16 for flash attention
            # _attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )

        image = Image.open(hf_hub_download("HuggingFaceTB/SmolVLM", "example_images/rococo.jpg", repo_type="space"))
        text = "This [MASK] is on the wall."

        # Create input messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text},
                ],
            },
        ]

        # Prepare inputs
        prompt = processor.apply_chat_template(messages, add_generation_prompt=False)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)

        # To get predictions for the mask:
        masked_index = inputs["input_ids"][0].tolist().index(tokenizer.mask_token_id)
        predicted_token_id = outputs.logits[0, masked_index].argmax(axis=-1)
        predicted_token = tokenizer.decode(predicted_token_id)

        self.assertEqual(predicted_token.strip(), "painting")


# from transformers import ModernVBertConfig
# from transformers.models.auto import MODEL_FOR_PRETRAINING_MAPPING, get_values
# from transformers.testing_utils import torch_device

# from ...test_configuration_common import ConfigTester
# from ...test_modeling_common import ModelTesterMixin, ids_tensor


# if is_torch_available():
#     from transformers import (
#         ModernVBertForMaskedLM,
#         ModernVBertModel,
#     )


# class ModernVBertModelTester:
#     def __init__(
#         self,
#         parent,
#         ignore_index=-100,
#         image_token_index=30522,  # TODO: check this
#         projector_hidden_act="gelu",
#         seq_length=7,
#         vision_feature_select_strategy="default",
#         vision_feature_layer=-2,
#         text_config={
#             "model_type": "modernbert",
#             "seq_length": 7,
#             "is_training": True,
#             "use_input_mask": True,
#             "use_labels": True,
#             "vocab_size": 99,
#             "hidden_size": 32,
#             "num_hidden_layers": 2,
#             "num_attention_heads": 4,
#             "intermediate_size": 37,
#             "hidden_act": "gelu",
#             "hidden_dropout_prob": 0.1,
#             "attention_probs_dropout_prob": 0.1,
#             "max_position_embeddings": 512,
#             "type_vocab_size": 16,
#             "type_sequence_label_size": 2,
#             "initializer_range": 0.02,
#             "num_labels": 3,
#             "num_choices": 4,
#             "pad_token_id": 0,
#         },
#         is_training=True,
#         vision_config={
#             "image_size": 30,
#             "patch_size": 2,
#             "num_channels": 3,
#             "is_training": True,
#             "hidden_size": 32,
#             "projection_dim": 32,
#             "num_hidden_layers": 2,
#             "num_attention_heads": 4,
#             "intermediate_size": 37,
#             "dropout": 0.1,
#             "attention_dropout": 0.1,
#             "initializer_range": 0.02,
#         },
#         batch_size=13,
#         num_labels=3,
#     ):
#         self.parent = parent
#         self.ignore_index = ignore_index
#         self.image_token_index = image_token_index
#         self.projector_hidden_act = projector_hidden_act
#         self.vision_feature_select_strategy = vision_feature_select_strategy
#         self.vision_feature_layer = vision_feature_layer
#         self.text_config = text_config
#         self.vision_config = vision_config
#         self.batch_size = batch_size
#         self.num_labels = num_labels
#         self.seq_length = seq_length

#         self.num_hidden_layers = text_config["num_hidden_layers"]
#         self.vocab_size = text_config["vocab_size"]
#         self.hidden_size = text_config["hidden_size"]
#         self.num_attention_heads = text_config["num_attention_heads"]
#         self.is_training = is_training

#         self.num_channels = vision_config["num_channels"]
#         self.image_size = vision_config["image_size"]

#     def get_config(self):
#         return ModernVBertConfig(
#             text_config=self.text_config,
#             vision_config=self.vision_config,
#             ignore_index=self.ignore_index,
#             image_token_index=self.image_token_index,
#             projector_hidden_act=self.projector_hidden_act,
#             vision_feature_select_strategy=self.vision_feature_select_strategy,
#             vision_feature_layer=self.vision_feature_layer,
#         )

#     def prepare_config_and_inputs(self):
#         pixel_values = torch.randn(
#             self.batch_size,
#             self.num_channels,
#             self.image_size,
#             self.image_size,
#         )
#         input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
#         config = self.get_config()
#         return config, pixel_values, input_ids

#     def create_and_check_for_masked_lm(self, config, pixel_values, input_ids):
#         model = ModernVBertForMaskedLM(config=config)
#         model.to(torch_device)
#         model.eval()
#         result = model(pixel_values=pixel_values, input_ids=input_ids)
#         self.parent.assertEqual(
#             result.logits.shape, (self.batch_size, self.seq_length, self.text_config["vocab_size"])
#         )

#     def prepare_config_and_inputs_for_common(self):
#         config_and_inputs = self.prepare_config_and_inputs()
#         config, pixel_values, input_ids = config_and_inputs
#         inputs_dict = {
#             "pixel_values": pixel_values,
#             "input_ids": input_ids,
#         }
#         return config, inputs_dict


# @require_torch
# class ModernVBertModelTest(ModelTesterMixin, unittest.TestCase):
#     all_model_classes = (ModernVBertModel, ModernVBertForMaskedLM) if is_torch_available() else ()
#     pipeline_model_mapping = {"feature-extraction": ModernVBertModel} if is_torch_available() else {}
#     test_pruning = False
#     test_resize_embeddings = False
#     test_head_masking = False

#     def setUp(self):
#         self.model_tester = ModernVBertModelTester(self)
#         self.config_tester = ConfigTester(self, config_class=ModernVBertConfig, has_text_modality=False)

#     def test_config(self):
#         self.config_tester.run_common_tests()

#     def test_for_masked_lm(self):
#         config_and_inputs = self.model_tester.prepare_config_and_inputs()
#         self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

#     def test_model(self):
#         config_and_inputs = self.model_tester.prepare_config_and_inputs()
#         config, pixel_values, input_ids = config_and_inputs
#         model = ModernVBertModel(config)
#         model.to(torch_device)
#         model.eval()
#         result = model(pixel_values=pixel_values, input_ids=input_ids)
#         self.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

#     @unittest.skip(reason="ModernVBert does not support training yet.")
#     def test_training(self):
#         pass

#     @unittest.skip(reason="ModernVBert does not support training yet.")
#     def test_training_gradient_checkpointing(self):
#         pass

#     @unittest.skip(reason="ModernVBert does not have separate vision and text models")
#     def test_load_vision_text_config(self):
#         pass

#     def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
#         inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

#         if model_class in get_values(MODEL_FOR_PRETRAINING_MAPPING):
#             inputs_dict["labels"] = torch.zeros(
#                 (self.model_tester.batch_size, self.model_tester.seq_length), dtype=torch.long, device=torch_device
#             )
#         return inputs_dict
