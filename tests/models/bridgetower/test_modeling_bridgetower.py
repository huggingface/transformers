# coding=utf-8
# Copyright 2022 The Intel Labs Team Authors, The Microsoft Research Team Authors and HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch BridgeTower model. """

import tempfile
import unittest

import numpy as np

from transformers import BridgeTowerConfig, is_torch_available, is_vision_available
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import BridgeTowerForImageAndTextRetrieval, BridgeTowerForMaskedLM, BridgeTowerModel
    from transformers.models.bridgetower.modeling_bridgetower import BRIDGETOWER_PRETRAINED_MODEL_ARCHIVE_LIST
    from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_10
else:
    is_torch_greater_or_equal_than_1_10 = False

if is_vision_available():
    from PIL import Image

    from transformers import BridgeTowerProcessor


class BridgeTowerModelTester:
    def __init__(
        self,
        parent,
        cross_modal_transform_shared=True,
        drop_rate=0.1,
        freeze_RoBERTa=False,
        freeze_ViT=False,
        freeze_layer_count_roberta=False,
        freeze_layer_count_vit=False,
        head_hidden_scale=2,
        hidden_size=768,
        image_size=288,
        input_image_embed_size=768,
        input_text_embed_size=768,
        is_encoder_decoder=False,
        layer_norm_eps=1e-05,
        link_tower_shared=False,
        link_tower_type="add",
        max_text_len=50,
        mlp_ratio=4,
        num_attention_heads=12,
        num_hidden_layers=6,
        resolution_before=224,
        stop_gradient=False,
        tie_word_embeddings=False,
        tokenizer="roberta-base",
        unfreeze_RoBERTa_attention=False,
        unfreeze_RoBERTa_embeddings=False,
        unfreeze_RoBERTa_encoder=False,
        unfreeze_RoBERTa_layernorm=False,
        unfreeze_ViT_attention=False,
        unfreeze_ViT_layernorm=False,
        vit_embed_dim=512,
        vit_layers=12,
        vit_layernorm_init_from_vit=False,
        vit_layernorm_shared=True,
        vit_patch_size=16,
        vit_transformer_width=512,
        vit_width=768,
        vit_remove_last=False,
        vocab_size=50265,
    ):
        self.parent = parent
        self.cross_modal_transform_shared = cross_modal_transform_shared
        self.drop_rate = drop_rate
        self.freeze_RoBERTa = freeze_RoBERTa
        self.freeze_ViT = freeze_ViT
        self.freeze_layer_count_roberta = freeze_layer_count_roberta
        self.freeze_layer_count_vit = freeze_layer_count_vit
        self.head_hidden_scale = head_hidden_scale
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.input_image_embed_size = input_image_embed_size
        self.input_text_embed_size = input_text_embed_size
        self.is_encoder_decoder = is_encoder_decoder
        self.layer_norm_eps = layer_norm_eps
        self.link_tower_shared = link_tower_shared
        self.link_tower_type = link_tower_type
        self.max_text_len = max_text_len
        self.mlp_ratio = mlp_ratio
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.resolution_before = resolution_before
        self.stop_gradient = stop_gradient
        self.tie_word_embeddings = tie_word_embeddings
        self.tokenizer = tokenizer
        self.unfreeze_RoBERTa_attention = unfreeze_RoBERTa_attention
        self.unfreeze_RoBERTa_embeddings = unfreeze_RoBERTa_embeddings
        self.unfreeze_RoBERTa_encoder = unfreeze_RoBERTa_encoder
        self.unfreeze_RoBERTa_layernorm = unfreeze_RoBERTa_layernorm
        self.unfreeze_ViT_attention = unfreeze_ViT_attention
        self.unfreeze_ViT_layernorm = unfreeze_ViT_layernorm
        self.vit_embed_dim = vit_embed_dim
        self.vit_layers = vit_layers
        self.vit_layernorm_init_from_vit = vit_layernorm_init_from_vit
        self.vit_layernorm_shared = vit_layernorm_shared
        self.vit_patch_size = vit_patch_size
        self.vit_remove_last = vit_remove_last
        self.vit_transformer_width = vit_transformer_width
        self.vit_width = vit_width
        self.vocab_size = vocab_size
        self.num_channels = 3
        self.seq_length = 4
        self.batch_size = 1
        self.is_training = False

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        pixel_mask = random_attention_mask([self.batch_size, self.image_size, self.image_size])

        config = self.get_config()

        return (config, input_ids, attention_mask, pixel_values, pixel_mask)

    def get_config(self):
        return BridgeTowerConfig(
            cross_modal_transform_shared=self.cross_modal_transform_shared,
            drop_rate=self.drop_rate,
            freeze_RoBERTa=self.freeze_RoBERTa,
            freeze_ViT=self.freeze_ViT,
            freeze_layer_count_roberta=self.freeze_layer_count_roberta,
            freeze_layer_count_vit=self.freeze_layer_count_vit,
            head_hidden_scale=self.head_hidden_scale,
            hidden_size=self.hidden_size,
            image_size=self.image_size,
            input_image_embed_size=self.input_image_embed_size,
            input_text_embed_size=self.input_text_embed_size,
            is_encoder_decoder=self.is_encoder_decoder,
            layer_norm_eps=self.layer_norm_eps,
            link_tower_shared=self.link_tower_shared,
            link_tower_type=self.link_tower_type,
            max_text_len=self.max_text_len,
            mlp_ratio=self.mlp_ratio,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            resolution_before=self.resolution_before,
            stop_gradient=self.stop_gradient,
            tie_word_embeddings=self.tie_word_embeddings,
            tokenizer=self.tokenizer,
            unfreeze_RoBERTa_attention=self.unfreeze_RoBERTa_attention,
            unfreeze_RoBERTa_embeddings=self.unfreeze_RoBERTa_embeddings,
            unfreeze_RoBERTa_encoder=self.unfreeze_RoBERTa_encoder,
            unfreeze_RoBERTa_layernorm=self.unfreeze_RoBERTa_layernorm,
            unfreeze_ViT_attention=self.unfreeze_ViT_attention,
            unfreeze_ViT_layernorm=self.unfreeze_ViT_layernorm,
            vit_embed_dim=self.vit_embed_dim,
            vit_layers=self.vit_layers,
            vit_layernorm_init_from_vit=self.vit_layernorm_init_from_vit,
            vit_layernorm_shared=self.vit_layernorm_shared,
            vit_patch_size=self.vit_patch_size,
            vit_remove_last=self.vit_remove_last,
            vit_transformer_width=self.vit_transformer_width,
            vit_width=self.vit_width,
            vocab_size=self.vocab_size,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        attention_mask,
        pixel_values,
        pixel_mask,
    ):
        model = BridgeTowerModel(config=config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values, pixel_mask=pixel_mask)
        result = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        self.parent.assertEqual(result["text_feats"].shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result["image_feats"].shape, (self.batch_size, 325, self.hidden_size))
        self.parent.assertEqual(result["pooler_output"].shape, (self.batch_size, 2 * self.hidden_size))

    def create_and_check_for_image_and_text_retrieval(
        self,
        config,
        input_ids,
        attention_mask,
        pixel_values,
        pixel_mask,
    ):
        bridgeTowerITMHead_output_last_dimension = 2

        model = BridgeTowerForImageAndTextRetrieval(config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values, pixel_mask=pixel_mask)
        result = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values)

        self.parent.assertEqual(result.logits.shape, (self.batch_size, bridgeTowerITMHead_output_last_dimension))

    def create_and_check_for_masked_language_modeling(
        self,
        config,
        input_ids,
        attention_mask,
        pixel_values,
        pixel_mask,
    ):
        model = BridgeTowerForMaskedLM(config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values, pixel_mask=pixel_mask)
        result = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values)

        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, attention_mask, pixel_values, pixel_mask) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
        }
        return config, inputs_dict


@require_torch
@unittest.skipIf(not is_torch_greater_or_equal_than_1_10, "BridgeTower is only available in torch v1.10+")
class BridgeTowerModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (
        (BridgeTowerModel, BridgeTowerForImageAndTextRetrieval, BridgeTowerForMaskedLM) if is_torch_available() else ()
    )

    is_training = False
    test_headmasking = False
    test_pruning = False
    test_torchscript = False
    test_resize_embeddings = False

    # function to extract meaningful tensor from output per different model_class
    def extract_output(self, outputs, model_class):
        return outputs["pooler_output"] if model_class == "BridgeTowerModel" else outputs["logits"]

    def setUp(self):
        self.model_tester = BridgeTowerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BridgeTowerConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_image_and_text_retrieval(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_and_text_retrieval(*config_and_inputs)

    def test_for_masked_language_modeling(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_language_modeling(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in BRIDGETOWER_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = BridgeTowerModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def test_save_load(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**input_dict)

            out_2 = self.extract_output(outputs, model_class.__name__)
            out_2 = out_2.cpu().numpy()
            out_2[np.isnan(out_2)] = 0

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname)
                model.to(torch_device)
                with torch.no_grad():
                    after_outputs = model(**input_dict)

                # Make sure we don't have nans
                out_1 = self.extract_output(after_outputs, model_class.__name__)
                out_1 = out_1.cpu().numpy()
                out_1[np.isnan(out_1)] = 0
                max_diff = np.amax(np.abs(out_1 - out_2))
                self.assertLessEqual(max_diff, 1e-5)

    def test_determinism(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                first = model(**input_dict)
                second = model(**input_dict)

            out_1 = self.extract_output(first, model_class.__name__).cpu().numpy()
            out_2 = self.extract_output(second, model_class.__name__).cpu().numpy()
            out_1 = out_1[~np.isnan(out_1)]
            out_2 = out_2[~np.isnan(out_2)]
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

    @unittest.skip(reason="""Bridge Tower model does not support this for now.""")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="""Bridge Tower model does not support this for now.""")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="""Bridge Tower model does not support this for now.""")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="""Bridge Tower model does not support this for now.""")
    def test_feed_forward_chunking(self):
        pass

    # Have not implemented model.get_input_embeddings() and model.get_output_embeddings()
    @unittest.skip(reason="""Bridge Tower model does not support this for now.""")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="""Bridge Tower model does not support this for now.""")
    def test_gradient_checkpointing_backward_compatibility(self):
        pass

    @unittest.skip(reason="""Bridge Tower model does not support this for now.""")
    def test_gradient_checkpointing_enable_disable(self):
        pass

    @unittest.skip(reason="""Bridge Tower model does not support this for now.""")
    def test_initialization(self):
        pass

    # Have not implemented model.get_input_embeddings()
    @unittest.skip(reason="""Bridge Tower model does not support this for now.""")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="""Bridge Tower model does not support this for now.""")
    def test_save_load_fast_init_to_base(self):
        pass


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
@unittest.skipIf(not is_torch_greater_or_equal_than_1_10, "BridgeTower is only available in torch v1.10+")
class BridgeTowerModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_processor(self):
        return (
            BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
            if is_vision_available()
            else None
        )

    @slow
    def test_image_and_text_retrieval(self):
        model = BridgeTowerForImageAndTextRetrieval.from_pretrained("BridgeTower/bridgetower-base-itm-mlm").to(
            torch_device
        )
        model.eval()
        processor = self.default_processor
        image = prepare_img()
        text = "a bunch of cats laying on a tower."
        inputs = processor(image, text, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size([1, 2])
        self.assertEqual(outputs.logits.shape, expected_shape)
        self.assertTrue(outputs.logits[0, 1].item() > outputs.logits[0, 0].item())

    @slow
    def test_masked_language_modeling(self):
        model = BridgeTowerForMaskedLM.from_pretrained("BridgeTower/bridgetower-base-itm-mlm").to(torch_device)
        model.eval()
        processor = self.default_processor
        image = prepare_img()
        text = "a bunch of <mask> laying on a tower."
        inputs = processor(image, text, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size([1, 11, 50265])
        self.assertEqual(outputs.logits.shape, expected_shape)

        # verify predicted word
        predicted_id = outputs.logits.argmax(dim=-1).squeeze(0).tolist()[4]
        self.assertTrue(processor.decode([predicted_id]) == " cats")
