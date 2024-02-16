# coding=utf-8
# Copyright 2024 Huggingface
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

import inspect
import tempfile
import unittest

from transformers import FLMRConfig, FLMRTextConfig, FLMRVisionConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        FLMRContextEncoderTokenizer,
        FLMRModelForRetrieval,
        FLMRQueryEncoderTokenizer,
        FLMRTextModel,
        FLMRVisionModel,
    )
    from transformers.models.flmr.modeling_flmr import (
        FLMR_PRETRAINED_MODEL_ARCHIVE_LIST,
    )


# Modified from tests.models.dpr.modeling_dpr with DPR to FLMR
class FLMRTextModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=False,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=20,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        scope=None,
        projection_dim=0,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.projection_dim = projection_dim

    def prepare_config_and_inputs(self, batch_size=None):
        self.batch_size = self.batch_size if batch_size is None else batch_size

        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return FLMRTextConfig(
            projection_dim=self.projection_dim,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
        )

    def create_and_check_text_encoder(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = FLMRTextModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.projection_dim or self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids}
        return config, inputs_dict


@require_torch
class FLMRTextModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (FLMRTextModel,) if is_torch_available() else ()
    pipeline_model_mapping = {"feature-extraction": FLMRTextModel} if is_torch_available() else {}

    test_resize_embeddings = False
    test_missing_keys = False  # why?
    test_pruning = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = FLMRTextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=FLMRTextConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_text_encoder_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_text_encoder(*config_and_inputs)

    def test_init_changed_config(self):
        config = self.model_tester.prepare_config_and_inputs()[0]

        model = FLMRTextModel(config=config)
        model.to(torch_device)
        model.eval()

        with tempfile.TemporaryDirectory() as tmp_dirname:
            model.save_pretrained(tmp_dirname)
            model = FLMRTextModel.from_pretrained(tmp_dirname, projection_dim=512)

        self.assertIsNotNone(model)


class FLMRVisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=30,
        patch_size=2,
        num_channels=3,
        is_training=True,
        hidden_size=32,
        projection_dim=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=20,
        dropout=0.1,
        attention_dropout=0.1,
        initializer_range=0.02,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.scope = scope

        # in ViT, the seq length equals the number of patches + 1 (we add 1 for the [CLS] token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1

    def prepare_config_and_inputs(self, batch_size=None, return_image_features=False):
        self.batch_size = self.batch_size if batch_size is None else batch_size

        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        if return_image_features:
            image_features = torch.randn(self.batch_size, 1, self.hidden_size)
            return config, pixel_values, image_features

        return config, pixel_values

    def get_config(self):
        return FLMRVisionConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            projection_dim=self.projection_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, pixel_values):
        model = FLMRVisionModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        image_size = (self.image_size, self.image_size)
        patch_size = (self.patch_size, self.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, num_patches + 1, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


# Modified from tests.models.clip.test_modeling_clip with CLIP to FLMR
@require_torch
class FLMRVisionModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as CLIP does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (FLMRVisionModel,) if is_torch_available() else ()
    fx_compatible = True
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = FLMRVisionModelTester(self)
        self.config_tester = ConfigTester(self, config_class=FLMRVisionConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="FLMR does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (torch.nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, torch.nn.Linear))

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_training(self):
        pass

    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="FLMRVisionModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="FLMRVisionModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @unittest.skip(reason="torch fx test is not needed for FLMRVisionModel")
    def test_torch_fx(self):
        pass

    @unittest.skip(reason="torch fx test is not needed for FLMRVisionModel")
    def test_torch_fx_output_loss(self):
        pass

    # @slow
    # def test_model_from_pretrained(self):
    #     for model_name in CLIP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
    #         model = CLIPVisionModel.from_pretrained(model_name)
    #         self.assertIsNotNone(model)

    # @slow
    # def test_model_with_projection_from_pretrained(self):
    #     for model_name in CLIP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
    #         model = CLIPVisionModelWithProjection.from_pretrained(model_name)
    #         self.assertIsNotNone(model)
    #         self.assertTrue(hasattr(model, "visual_projection"))


class FLMRModelTester:
    def __init__(
        self,
        parent,
        text_kwargs=None,
        vision_kwargs=None,
        batch_size=13,
        mask_punctuation=True,
        mapping_network_prefix_length=4,
        dim=16,
        use_vision_encoder=True,
        initializer_range=0.02,
        vision_model_version="openai/clip-vit-base-patch32",
        separate_query_and_context_text_encoder=False,
        separate_query_and_context_vision_encoder=False,
        query_concat_output_from_vision_encoder=True,
        query_concat_output_from_text_encoder=True,
        context_concat_output_from_vision_encoder=False,
        context_concat_output_from_text_encoder=True,
        use_transformer_mapping_network=False,
        transformer_mapping_config_base=None,
        transformer_mapping_num_hidden_layers=None,
        load_cpu_extension=False,
        mask_instruction_token=None,
        transformer_mapping_cross_attention_length=32,
        num_negative_examples=1,
        is_training=True,
    ):
        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}

        self.parent = parent
        self.text_model_tester = FLMRTextModelTester(parent, **text_kwargs)
        self.vision_model_tester = FLMRVisionModelTester(parent, **vision_kwargs)

        self.batch_size = batch_size
        self.dim = dim
        self.initializer_range = initializer_range
        self.mask_punctuation = mask_punctuation
        self.mapping_network_prefix_length = mapping_network_prefix_length
        self.vision_model_version = vision_model_version
        self.use_vision_encoder = use_vision_encoder
        self.separate_query_and_context_text_encoder = separate_query_and_context_text_encoder
        self.separate_query_and_context_vision_encoder = separate_query_and_context_vision_encoder
        self.query_concat_output_from_vision_encoder = query_concat_output_from_vision_encoder
        self.query_concat_output_from_text_encoder = query_concat_output_from_text_encoder
        self.context_concat_output_from_vision_encoder = context_concat_output_from_vision_encoder
        self.context_concat_output_from_text_encoder = context_concat_output_from_text_encoder
        self.use_transformer_mapping_network = use_transformer_mapping_network
        self.transformer_mapping_config_base = transformer_mapping_config_base
        self.transformer_mapping_num_hidden_layers = transformer_mapping_num_hidden_layers
        self.load_cpu_extension = load_cpu_extension
        self.mask_instruction_token = mask_instruction_token
        self.transformer_mapping_cross_attention_length = transformer_mapping_cross_attention_length
        self.num_negative_examples = num_negative_examples
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        # from a DPR-like model
        (
            text_config,
            query_input_ids,
            _,
            query_attention_mask,
            _,
            _,
            _,
        ) = self.text_model_tester.prepare_config_and_inputs(batch_size=self.batch_size)
        (
            text_config,
            context_input_ids,
            _,
            context_attention_mask,
            _,
            _,
            _,
        ) = self.text_model_tester.prepare_config_and_inputs(
            batch_size=self.batch_size * (1 + self.num_negative_examples)
        )
        # from a CLIP-ViT-like model
        vision_config, query_pixel_values, query_image_features = self.vision_model_tester.prepare_config_and_inputs(
            batch_size=self.batch_size, return_image_features=True
        )
        (
            vision_config,
            context_pixel_values,
            context_image_features,
        ) = self.vision_model_tester.prepare_config_and_inputs(
            batch_size=self.batch_size * (1 + self.num_negative_examples), return_image_features=True
        )

        config = self.get_config()

        return (
            config,
            query_input_ids,
            query_attention_mask,
            context_input_ids,
            context_attention_mask,
            query_pixel_values,
            context_pixel_values,
            query_image_features,
            context_image_features,
        )

    def get_config(self):
        return FLMRConfig.from_text_vision_configs(
            self.text_model_tester.get_config(),
            self.vision_model_tester.get_config(),
            dim=self.dim,
            initializer_range=self.initializer_range,
            mask_punctuation=self.mask_punctuation,
            mapping_network_prefix_length=self.mapping_network_prefix_length,
            vision_model_version=self.vision_model_version,
            use_vision_encoder=self.use_vision_encoder,
            separate_query_and_context_text_encoder=self.separate_query_and_context_text_encoder,
            separate_query_and_context_vision_encoder=self.separate_query_and_context_vision_encoder,
            query_concat_output_from_vision_encoder=self.query_concat_output_from_vision_encoder,
            query_concat_output_from_text_encoder=self.query_concat_output_from_text_encoder,
            context_concat_output_from_vision_encoder=self.context_concat_output_from_vision_encoder,
            context_concat_output_from_text_encoder=self.context_concat_output_from_text_encoder,
            use_transformer_mapping_network=self.use_transformer_mapping_network,
            transformer_mapping_config_base=self.transformer_mapping_config_base,
            transformer_mapping_num_hidden_layers=self.transformer_mapping_num_hidden_layers,
            load_cpu_extension=self.load_cpu_extension,
            mask_instruction_token=self.mask_instruction_token,
            transformer_mapping_cross_attention_length=self.transformer_mapping_cross_attention_length,
        )

    def create_and_check_model(
        self,
        config,
        query_input_ids,
        query_attention_mask,
        context_input_ids,
        context_attention_mask,
        query_pixel_values,
        context_pixel_values,
        query_image_features,
        context_image_features,
    ):
        model = FLMRModelForRetrieval(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(
                query_input_ids=query_input_ids,
                query_attention_mask=query_attention_mask,
                context_input_ids=context_input_ids,
                context_attention_mask=context_attention_mask,
                query_pixel_values=query_pixel_values,
                context_pixel_values=context_pixel_values,
            )

        self.parent.assertEqual(result.scores.shape, (self.batch_size, 1 + self.num_negative_examples))
        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.in_batch_negative_loss.shape, ())

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            query_input_ids,
            query_attention_mask,
            context_input_ids,
            context_attention_mask,
            query_pixel_values,
            context_pixel_values,
            query_image_features,
            context_image_features,
        ) = config_and_inputs
        inputs_dict = {
            "query_input_ids": query_input_ids,
            "query_attention_mask": query_attention_mask,
            "context_input_ids": context_input_ids,
            "context_attention_mask": context_attention_mask,
            "query_pixel_values": query_pixel_values,
            "context_pixel_values": context_pixel_values,
        }
        return config, inputs_dict

    def prepare_config_and_inputs_with_image_features(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            query_input_ids,
            query_attention_mask,
            context_input_ids,
            context_attention_mask,
            query_pixel_values,
            context_pixel_values,
            query_image_features,
            context_image_features,
        ) = config_and_inputs
        inputs_dict = {
            "query_input_ids": query_input_ids,
            "query_attention_mask": query_attention_mask,
            "context_input_ids": context_input_ids,
            "context_attention_mask": context_attention_mask,
            "query_image_features": query_image_features,
            "context_image_features": context_image_features,
        }
        return config, inputs_dict


@require_torch
class FLMRModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (FLMRModelForRetrieval,) if is_torch_available() else ()

    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    pipeline_model_mapping = {"feature-extraction": FLMRModelForRetrieval} if is_torch_available() else {}

    def setUp(self):
        self.model_tester = FLMRModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Inputs_embeds is tested in individual model tests")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="FLMR does not have input/output embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="This test does not apply to FLMR")
    def _create_and_check_torchscript(self, config, inputs_dict):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]
            if model.config.is_encoder_decoder:
                expected_arg_names = [
                    "query_input_ids",
                    "query_attention_mask",
                    "query_pixel_values",
                    "context_input_ids",
                    "context_attention_mask",
                    "context_pixel_values",
                ]
                expected_arg_names.extend([])
                self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    def test_from_pretrained_no_checkpoint(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            state_dict = model.state_dict()

            new_model = model_class.from_pretrained(name_or_path=None, config=config, state_dict=state_dict)
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))

    def test_replacing_pixel_values_with_image_features(self):
        """
        In some cases, image features are pre-extracted to speed up training
        """
        config_and_inputs = self.model_tester.prepare_config_and_inputs_with_image_features()
        config, inputs_dict = config_and_inputs
        model = FLMRModelForRetrieval(config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(**inputs_dict)

        self.assertEqual(
            result.scores.shape, (self.model_tester.batch_size, 1 + self.model_tester.num_negative_examples)
        )
        self.assertEqual(result.loss.shape, ())
        self.assertEqual(result.in_batch_negative_loss.shape, ())

    def test_different_concatenation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        config, inputs_dict = config_and_inputs
        model = FLMRModelForRetrieval(config)
        model.to(torch_device)
        model.eval()

        # enable all
        for to_enable in [
            "query_concat_output_from_vision_encoder",
            "context_concat_output_from_vision_encoder",
            "query_concat_output_from_text_encoder",
            "context_concat_output_from_text_encoder",
        ]:
            inputs_dict[to_enable] = True

        text_len = self.model_tester.text_model_tester.seq_length
        vision_len = self.model_tester.mapping_network_prefix_length
        total_len = text_len + vision_len
        context_batch_size = self.model_tester.batch_size * (1 + self.model_tester.num_negative_examples)

        for to_disable in [
            "query_concat_output_from_vision_encoder",
            "context_concat_output_from_vision_encoder",
            "query_concat_output_from_text_encoder",
            "context_concat_output_from_text_encoder",
        ]:
            dup_inputs_dict = inputs_dict.copy()
            dup_inputs_dict[to_disable] = False
            with torch.no_grad():
                result = model(**dup_inputs_dict)

            if to_disable == "query_concat_output_from_vision_encoder":
                self.assertEqual(
                    result.query_late_interaction_output.shape,
                    (self.model_tester.batch_size, text_len, self.model_tester.dim),
                )
                self.assertEqual(
                    result.context_late_interaction_output.shape,
                    (context_batch_size, total_len, self.model_tester.dim),
                )
            elif to_disable == "context_concat_output_from_vision_encoder":
                self.assertEqual(
                    result.query_late_interaction_output.shape,
                    (self.model_tester.batch_size, total_len, self.model_tester.dim),
                )
                self.assertEqual(
                    result.context_late_interaction_output.shape, (context_batch_size, text_len, self.model_tester.dim)
                )
            elif to_disable == "query_concat_output_from_text_encoder":
                self.assertEqual(
                    result.query_late_interaction_output.shape,
                    (self.model_tester.batch_size, vision_len, self.model_tester.dim),
                )
                self.assertEqual(
                    result.context_late_interaction_output.shape,
                    (context_batch_size, total_len, self.model_tester.dim),
                )
            elif to_disable == "context_concat_output_from_text_encoder":
                self.assertEqual(
                    result.query_late_interaction_output.shape,
                    (self.model_tester.batch_size, total_len, self.model_tester.dim),
                )
                self.assertEqual(
                    result.context_late_interaction_output.shape,
                    (context_batch_size, vision_len, self.model_tester.dim),
                )

    def test_different_num_of_negative_examples(self):
        for num_negative_examples in [1, 2, 3, 4]:
            model_tester = FLMRModelTester(self, num_negative_examples=num_negative_examples)
            # new tester
            config_and_inputs = model_tester.prepare_config_and_inputs_for_common()
            config, inputs_dict = config_and_inputs
            model = FLMRModelForRetrieval(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                result = model(
                    **inputs_dict,
                    num_negative_examples=num_negative_examples,
                )

            self.assertEqual(result.scores.shape, (model_tester.batch_size, 1 + num_negative_examples))
            self.assertEqual(result.loss.shape, ())
            self.assertEqual(result.in_batch_negative_loss.shape, ())

    @slow
    def test_model_from_pretrained(self):
        for model_name in FLMR_PRETRAINED_MODEL_ARCHIVE_LIST:
            model = FLMRModelForRetrieval.from_pretrained(model_name)
            self.assertIsNotNone(model)


@require_torch
class FLMRModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference(self):
        checkpoint_path = "LinWeizheDragon/PreFLMR_ViT-G"
        query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(checkpoint_path, subfolder="query_tokenizer")
        context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(checkpoint_path, subfolder="context_tokenizer")

        model = FLMRModelForRetrieval.from_pretrained(
            checkpoint_path,
            query_tokenizer=query_tokenizer,
            context_tokenizer=context_tokenizer,
        ).to("cuda")
        # Some toy inputs
        Q_encoding = query_tokenizer(
            [
                "Using the provided image, obtain documents that address the subsequent question: What is the capital of France?",
                "Extract documents linked to the question provided in conjunction with the image: What is the capital of France?",
            ]
        )
        D_encoding = context_tokenizer(
            [
                "Paris is the capital of France.",
                "Beijing is the capital of China.",
                "Paris is the capital of France.",
                "Beijing is the capital of China.",
            ]
        )

        Q_pixel_values = torch.zeros(2, 3, 224, 224)

        inputs = {
            "query_input_ids": Q_encoding["input_ids"],
            "query_attention_mask": Q_encoding["attention_mask"],
            "query_pixel_values": Q_pixel_values,
            "context_input_ids": D_encoding["input_ids"],
            "context_attention_mask": D_encoding["attention_mask"],
            "use_in_batch_negatives": True,
        }

        # set to inference mode
        model.eval()
        with torch.no_grad():
            forward_results = model.forward(**inputs)

        expected_loss = torch.tensor(
            7.1585e-05,
            dtype=torch.half,
            device=torch_device,
        )
        expected_scores = torch.tensor(
            [[42.1562, 31.9688], [41.4375, 32.2812]],
            dtype=torch.half,
            device=torch_device,
        )
        expected_in_batch_negative_loss = torch.tensor(
            0.6932,
            dtype=torch.float,
            device=torch_device,
        )
        expected_query_late_interaction_output = torch.tensor(
            [[[0.0976, 0.0195], [-0.0216, -0.1124]], [[0.1090, 0.0369], [-0.0194, -0.1110]]],
            dtype=torch.float,
            device=torch_device,
        )
        expected_context_late_interaction_output = torch.tensor(
            [
                [[0.0410, -0.0591], [0.0278, -0.1041]],
                [[0.0384, -0.0287], [0.0200, -0.0552]],
                [[0.0410, -0.0591], [0.0278, -0.1041]],
                [[0.0384, -0.0287], [0.0200, -0.0552]],
            ],
            dtype=torch.half,
            device=torch_device,
        )

        self.assertTrue(torch.allclose(forward_results.loss, expected_loss, atol=1e-3))
        self.assertTrue(torch.allclose(forward_results.scores, expected_scores, atol=5))
        self.assertTrue(
            torch.allclose(forward_results.in_batch_negative_loss, expected_in_batch_negative_loss, atol=1e-2)
        )
        self.assertTrue(
            torch.allclose(
                forward_results.query_late_interaction_output[:, :2, :2],
                expected_query_late_interaction_output,
                atol=1e-2,
            )
        )
        self.assertTrue(
            torch.allclose(
                forward_results.context_late_interaction_output[:, :2, :2],
                expected_context_late_interaction_output,
                atol=1e-2,
            )
        )
