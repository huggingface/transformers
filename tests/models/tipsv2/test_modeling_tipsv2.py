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
"""Testing suite for the PyTorch TIPSv2 model."""

import inspect
import tempfile
import unittest
from functools import cached_property

from parameterized import parameterized

from transformers import Tipsv2Config, Tipsv2TextConfig, Tipsv2VisionConfig
from transformers.testing_utils import (
    Expectations,
    is_flaky,
    require_sentencepiece,
    require_torch,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION,
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)
from ...test_pipeline_mixin import PipelineTesterMixin
from ...test_processing_common import url_to_local_path


if is_torch_available():
    import torch
    from torch import nn

    from transformers import (
        AutoModel,
        AutoModelForTextEncoding,
        AutoModelForZeroShotImageClassification,
        Tipsv2ImageProcessor,
        Tipsv2Model,
        Tipsv2Processor,
        Tipsv2TextModel,
        Tipsv2Tokenizer,
        Tipsv2VisionModel,
    )
    from transformers.image_utils import load_image_as_tensor


class Tipsv2ModelTesterMixin(ModelTesterMixin):
    def test_sdpa_can_dispatch_composite_models(self):
        for model_class in self.all_model_classes:
            config, _ = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_sdpa = model_class.from_pretrained(tmpdirname)
                model_eager = model_class.from_pretrained(tmpdirname, attn_implementation="eager")

            if hasattr(model_sdpa, "vision_model"):
                self.assertEqual(model_sdpa.vision_model.config._attn_implementation, "sdpa")
                self.assertEqual(model_eager.vision_model.config._attn_implementation, "eager")

            if hasattr(model_sdpa, "text_model"):
                self.assertEqual(model_sdpa.text_model.config._attn_implementation, "sdpa")
                self.assertEqual(model_eager.text_model.config._attn_implementation, "eager")

            self.assertEqual(model_sdpa.config._attn_implementation, "sdpa")
            self.assertEqual(model_eager.config._attn_implementation, "eager")

    @unittest.skip(reason="TIPSv2 does not use grouped expert implementations.")
    def test_eager_matches_batched_and_grouped_inference(self, name, dtype):
        pass

    @unittest.skip(reason="TIPSv2 SDPA flash dispatch is covered by generic accelerator tests when available.")
    def test_sdpa_can_dispatch_on_flash(self):
        pass


class Tipsv2VisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=4,
        image_size=28,
        patch_size=14,
        num_channels=3,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        mlp_ratio=2,
        num_register_tokens=1,
        initializer_range=0.02,
        is_training=True,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.num_register_tokens = num_register_tokens
        self.initializer_range = initializer_range
        self.is_training = is_training
        self.scope = scope

        self.num_patches = (image_size // patch_size) ** 2
        self.seq_length = self.num_patches + 1 + num_register_tokens
        self.mask_length = self.num_patches
        self.num_masks = max(1, self.num_patches // 2)

    def get_config(self):
        return Tipsv2VisionConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            mlp_ratio=self.mlp_ratio,
            num_register_tokens=self.num_register_tokens,
            initializer_range=self.initializer_range,
            use_swiglu_ffn=False,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()
        return config, pixel_values

    def create_and_check_model(self, config, pixel_values):
        model = Tipsv2VisionModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        return config, {"pixel_values": pixel_values}


@require_torch
class Tipsv2VisionModelTest(Tipsv2ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    TIPSv2VisionModel is a DINOv2-with-registers vision tower. It does not use token input ids, token embeddings, or
    text attention masks.
    """

    all_model_classes = (Tipsv2VisionModel,) if is_torch_available() else ()
    pipeline_model_mapping = {"image-feature-extraction": Tipsv2VisionModel} if is_torch_available() else {}

    test_resize_embeddings = False
    has_attentions = False

    def setUp(self):
        self.model_tester = Tipsv2VisionModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=Tipsv2VisionConfig,
            has_text_modality=False,
            hidden_size=16,
            num_attention_heads=4,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), nn.Module)
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            arg_names = [*signature.parameters.keys()]
            self.assertListEqual(arg_names[:1], ["pixel_values"])

    @unittest.skip(reason="TIPSv2VisionModel does not use text token inputs_embeds.")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="TIPSv2VisionModel does not use text token inputs_embeds.")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    @is_flaky()
    def test_eager_matches_sdpa_inference(self, *args):
        return getattr(ModelTesterMixin, self._testMethodName)(self)


class Tipsv2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=4,
        seq_length=7,
        vocab_size=99,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=16,
        initializer_range=0.02,
        is_training=True,
        use_input_mask=True,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.scope = scope

    def get_config(self):
        return Tipsv2TextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
        )

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size - 3) + 3
        attention_mask = None
        if self.use_input_mask:
            attention_mask = random_attention_mask([self.batch_size, self.seq_length])
            attention_mask[:, 0] = 1
            attention_mask[:, -1] = 0
            input_ids = input_ids.masked_fill(attention_mask == 0, 0)

        config = self.get_config()
        return config, input_ids, attention_mask

    def create_and_check_model(self, config, input_ids, attention_mask):
        model = Tipsv2TextModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_ids, attention_mask=attention_mask)
            result_without_mask = model(input_ids)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))
        self.parent.assertEqual(
            result_without_mask.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size)
        )

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask = self.prepare_config_and_inputs()
        return config, {"input_ids": input_ids, "attention_mask": attention_mask}


@require_torch
class Tipsv2TextModelTest(Tipsv2ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Tipsv2TextModel,) if is_torch_available() else ()

    test_resize_embeddings = False
    model_split_percents = [0.5, 0.8, 0.9]

    def setUp(self):
        self.model_tester = Tipsv2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Tipsv2TextConfig, hidden_size=16, num_attention_heads=4)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="TIPSv2TextModel is a bare encoder and does not return a training loss.")
    def test_training(self):
        pass

    @unittest.skip(reason="TIPSv2TextModel is a bare encoder and does not return a training loss.")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="TIPSv2TextModel is a bare encoder and does not return a training loss.")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="TIPSv2TextModel is a bare encoder and does not return a training loss.")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        pass

    def test_text_attention_mask_and_padding_mask_are_equivalent(self):
        config, input_ids, attention_mask = self.model_tester.prepare_config_and_inputs()
        model = Tipsv2TextModel(config).to(torch_device)
        model.eval()

        padding_mask = 1 - attention_mask

        with torch.no_grad():
            hf_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            official_outputs = model(input_ids=input_ids, padding_mask=padding_mask)

        torch.testing.assert_close(hf_outputs.last_hidden_state, official_outputs.last_hidden_state)
        torch.testing.assert_close(hf_outputs.pooler_output, official_outputs.pooler_output)

    def test_text_model_rejects_both_mask_conventions(self):
        config, input_ids, attention_mask = self.model_tester.prepare_config_and_inputs()
        model = Tipsv2TextModel(config)
        padding_mask = 1 - attention_mask

        with self.assertRaisesRegex(ValueError, "attention_mask and padding_mask"):
            model(input_ids=input_ids, attention_mask=attention_mask, padding_mask=padding_mask)


class Tipsv2ModelTester:
    def __init__(self, parent, text_kwargs=None, vision_kwargs=None, is_training=True):
        text_kwargs = {} if text_kwargs is None else text_kwargs
        vision_kwargs = {} if vision_kwargs is None else vision_kwargs

        self.parent = parent
        self.text_model_tester = Tipsv2TextModelTester(parent, **text_kwargs)
        self.vision_model_tester = Tipsv2VisionModelTester(parent, **vision_kwargs)
        self.batch_size = self.text_model_tester.batch_size
        self.hidden_size = self.text_model_tester.hidden_size
        self.num_hidden_layers = self.text_model_tester.num_hidden_layers
        self.num_attention_heads = self.text_model_tester.num_attention_heads
        self.seq_length = self.text_model_tester.seq_length
        self.is_training = is_training

    def get_config(self):
        return Tipsv2Config(
            text_config=self.text_model_tester.get_config().to_dict(),
            vision_config=self.vision_model_tester.get_config().to_dict(),
            temperature=0.01,
        )

    def prepare_config_and_inputs(self):
        _, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        _, pixel_values = self.vision_model_tester.prepare_config_and_inputs()
        config = self.get_config()
        return config, input_ids, attention_mask, pixel_values

    def create_and_check_model(self, config, input_ids, attention_mask, pixel_values):
        model = Tipsv2Model(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)

        self.parent.assertEqual(
            result.logits_per_image.shape,
            (self.vision_model_tester.batch_size, self.text_model_tester.batch_size),
        )
        self.parent.assertEqual(
            result.logits_per_text.shape,
            (self.text_model_tester.batch_size, self.vision_model_tester.batch_size),
        )
        self.parent.assertEqual(result.image_embeds.shape, (self.vision_model_tester.batch_size, self.hidden_size))
        self.parent.assertEqual(result.text_embeds.shape, (self.text_model_tester.batch_size, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask, pixel_values = self.prepare_config_and_inputs()
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "return_loss": False,
        }
        return config, inputs_dict


def get_b14_tipsv2_config(temperature=0.005065968260169029):
    hidden_size = 768
    return Tipsv2Config(
        text_config=Tipsv2TextConfig(
            vocab_size=32000,
            hidden_size=hidden_size,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            max_position_embeddings=64,
        ),
        vision_config=Tipsv2VisionConfig(
            image_size=448,
            patch_size=14,
            hidden_size=hidden_size,
            num_hidden_layers=12,
            num_attention_heads=12,
            mlp_ratio=4.0,
            use_swiglu_ffn=False,
        ),
        temperature=temperature,
    )


def max_abs_diff(left, right):
    return (left.detach().cpu() - right.detach().cpu()).abs().max().item()


@require_torch
class Tipsv2ModelTest(Tipsv2ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (Tipsv2Model,) if is_torch_available() else ()
    pipeline_model_mapping = {"feature-extraction": Tipsv2Model} if is_torch_available() else {}
    additional_model_inputs = ["pixel_values", "attention_mask", "return_loss"]

    test_resize_embeddings = False
    has_attentions = False
    _is_composite = True

    def setUp(self):
        self.model_tester = Tipsv2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Tipsv2Config, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="Hidden states are tested in the individual TIPSv2 text and vision tower tests.")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Retained gradients for hidden states are tested in individual tower tests.")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="TIPSv2Model does not expose a single input embedding layer for the composite model.")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(
        reason="Inputs_embeds behavior is covered by the text tower; the composite model has image inputs too."
    )
    def test_inputs_embeds(self):
        pass

    @unittest.skip(
        reason="Inputs_embeds behavior is covered by the text tower; the composite model has image inputs too."
    )
    def test_inputs_embeds_matches_input_ids(self):
        pass

    def test_load_vision_text_config(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = Tipsv2VisionConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.vision_config.to_dict(), vision_config.to_dict())

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            text_config = Tipsv2TextConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.text_config.to_dict(), text_config.to_dict())

    def test_model_forward_shapes_and_logits(self):
        config, input_ids, attention_mask, pixel_values = self.model_tester.prepare_config_and_inputs()
        model = Tipsv2Model(config).to(torch_device)
        model.eval()

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                return_loss=True,
            )

        self.assertEqual(outputs.logits_per_image.shape, (pixel_values.shape[0], input_ids.shape[0]))
        self.assertEqual(outputs.logits_per_text.shape, (input_ids.shape[0], pixel_values.shape[0]))
        self.assertEqual(outputs.image_embeds.shape, (pixel_values.shape[0], config.vision_config.hidden_size))
        self.assertEqual(outputs.text_embeds.shape, (input_ids.shape[0], config.text_config.hidden_size))
        self.assertEqual(outputs.loss.dim(), 0)

        torch.testing.assert_close(
            outputs.image_embeds.norm(dim=-1),
            torch.ones(pixel_values.shape[0], device=torch_device),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            outputs.text_embeds.norm(dim=-1),
            torch.ones(input_ids.shape[0], device=torch_device),
            atol=1e-5,
            rtol=1e-5,
        )
        expected_logits = torch.matmul(outputs.text_embeds, outputs.image_embeds.t()) / config.temperature
        torch.testing.assert_close(outputs.logits_per_text, expected_logits, atol=1e-5, rtol=1e-5)

    def test_auto_model_mappings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        self.assertIsInstance(AutoModel.from_config(config), Tipsv2Model)
        self.assertIsInstance(AutoModel.from_config(config.text_config), Tipsv2TextModel)
        self.assertIsInstance(AutoModel.from_config(config.vision_config), Tipsv2VisionModel)
        self.assertIsInstance(AutoModelForTextEncoding.from_config(config.text_config), Tipsv2TextModel)
        self.assertIsInstance(AutoModelForZeroShotImageClassification.from_config(config), Tipsv2Model)

    @unittest.skip(
        reason=(
            "TIPSv2 conversion uses packed qkv/in_proj split converters; the generic reverse-loading test does not "
            "cover this load-only alignment path reliably."
        )
    )
    def test_reverse_loading_mapping(self):
        pass


def prepare_img():
    image = load_image_as_tensor(
        url_to_local_path(
            "https://huggingface.co/datasets/hf-internal-testing/fixtures-coco/resolve/main/val2017/000000039769.jpg"
        )
    )
    return image


@require_torch
@require_vision
class Tipsv2ModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_processor(self):
        # Hub repo does not ship a processor config yet — instantiate both components explicitly.
        image_processor = Tipsv2ImageProcessor()
        tokenizer = Tipsv2Tokenizer.from_pretrained("google/tipsv2-b14")
        return Tipsv2Processor(image_processor=image_processor, tokenizer=tokenizer)

    @slow
    @require_sentencepiece
    def test_inference(self):
        model = Tipsv2Model.from_pretrained("google/tipsv2-b14", device_map=torch_device).eval()

        text_queries = ["two cats on a sofa", "a cat lying down", "a dog on a couch", "an empty room"]
        processor = self.default_processor
        inputs = processor(images=prepare_img(), text=text_queries, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        vision_cfg = model.config.vision_config
        _, _, height, width = inputs["pixel_values"].shape
        num_patches = (height // vision_cfg.patch_size) * (width // vision_cfg.patch_size)
        num_register_tokens = vision_cfg.num_register_tokens
        patch_tokens = outputs.last_hidden_state[:, 1 + num_register_tokens :]
        self.assertEqual(outputs.image_embeds.shape, torch.Size([1, vision_cfg.hidden_size]))
        self.assertEqual(patch_tokens.shape, torch.Size([1, num_patches, vision_cfg.hidden_size]))
        self.assertEqual(outputs.text_embeds.shape, torch.Size([4, model.config.text_config.hidden_size]))
        self.assertEqual(outputs.logits_per_image.shape, torch.Size([1, 4]))
        self.assertEqual(outputs.logits_per_text.shape, torch.Size([4, 1]))

        # Tolerance of 1e-3 for vision outputs because of difference in PIL vs. torch preprocessing.
        EXPECTED_IMAGE_EMBEDS = Expectations({("cuda", None): [0.03267, 0.02216, 0.00546, 0.01890, -0.05426]})
        expected_image_embeds = torch.tensor(EXPECTED_IMAGE_EMBEDS.get_expectation(), device=torch_device)
        torch.testing.assert_close(outputs.image_embeds[0, :5], expected_image_embeds, rtol=1e-3, atol=1e-3)

        EXPECTED_PATCH_TOKENS = Expectations({("cuda", None): [0.25287, -0.01092, -0.57542, 0.09660, -0.04010]})
        expected_patch_tokens = torch.tensor(EXPECTED_PATCH_TOKENS.get_expectation(), device=torch_device)
        torch.testing.assert_close(patch_tokens[0, 0, :5], expected_patch_tokens, rtol=1e-3, atol=1e-3)

        EXPECTED_TEXT_EMBEDS = Expectations({("cuda", None): [0.69319, 0.03710, 0.01194, 0.02136, -0.04281]})
        expected_text_embeds = torch.tensor(EXPECTED_TEXT_EMBEDS.get_expectation(), device=torch_device)
        torch.testing.assert_close(outputs.text_embeds[0, :5], expected_text_embeds, rtol=1e-4, atol=1e-4)

        EXPECTED_LOGITS_PER_IMAGE = Expectations({("cuda", None): [31.12190, 26.99341, 20.26748, 17.55544]})
        expected_logits_per_image = torch.tensor(EXPECTED_LOGITS_PER_IMAGE.get_expectation(), device=torch_device)
        torch.testing.assert_close(outputs.logits_per_image[0], expected_logits_per_image, rtol=1e-3, atol=1e-3)

        EXPECTED_LOGITS_PER_TEXT = Expectations({("cuda", None): [31.12190, 26.99341, 20.26748, 17.55544]})
        expected_logits_per_text = torch.tensor(EXPECTED_LOGITS_PER_TEXT.get_expectation(), device=torch_device)
        torch.testing.assert_close(outputs.logits_per_text[..., 0], expected_logits_per_text, rtol=1e-3, atol=1e-3)

        EXPECTED_VISION_POOLER_OUTPUT = Expectations({("cuda", None): [0.22055, 0.14960, 0.03689, 0.12756, -0.36632]})
        expected_vision_pooler_output = torch.tensor(
            EXPECTED_VISION_POOLER_OUTPUT.get_expectation(), device=torch_device
        )
        torch.testing.assert_close(
            outputs.vision_model_output.pooler_output[0, :5], expected_vision_pooler_output, rtol=1e-3, atol=1e-3
        )

        EXPECTED_TEXT_POOLER_OUTPUT = Expectations({("cuda", None): [12.07207, 0.64612, 0.20788, 0.37204, -0.74551]})
        expected_text_pooler_output = torch.tensor(EXPECTED_TEXT_POOLER_OUTPUT.get_expectation(), device=torch_device)
        torch.testing.assert_close(
            outputs.text_model_output.pooler_output[0, :5], expected_text_pooler_output, rtol=1e-4, atol=1e-4
        )
