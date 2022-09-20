# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch LayoutLMv2 model. """


import copy
import os
import random
import tempfile
import unittest

from transformers.models.auto import get_values
from transformers.testing_utils import require_detectron2, require_torch, require_torch_multi_gpu, slow, torch_device
from transformers.utils import is_detectron2_available, is_torch_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import (
        MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING,
        MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        MODEL_MAPPING,
        LayoutLMv2Config,
        LayoutLMv2ForQuestionAnswering,
        LayoutLMv2ForRelationExtraction,
        LayoutLMv2ForSequenceClassification,
        LayoutLMv2ForTokenClassification,
        LayoutLMv2Model,
    )
    from transformers.models.layoutlmv2.modeling_layoutlmv2 import LAYOUTLMV2_PRETRAINED_MODEL_ARCHIVE_LIST

if is_detectron2_available():
    from detectron2.structures.image_list import ImageList


class LayoutLMv2ModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        num_channels=3,
        image_size=4,
        text_seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=36,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        image_feature_pool_shape=[7, 7, 256],
        coordinate_size=6,
        shape_size=6,
        num_labels=3,
        num_choices=4,
        scope=None,
        range_bbox=1000,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.text_seq_length = text_seq_length
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
        self.image_feature_pool_shape = image_feature_pool_shape
        self.coordinate_size = coordinate_size
        self.shape_size = shape_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.range_bbox = range_bbox

        # in LayoutLMv2, the seq length equals the number of text tokens + number of image tokens
        self.seq_length = self.text_seq_length + self.image_feature_pool_shape[0] * self.image_feature_pool_shape[1]

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.text_seq_length], self.vocab_size)

        bbox = ids_tensor([self.batch_size, self.text_seq_length, 4], self.range_bbox)
        # Ensure that bbox is legal
        for i in range(bbox.shape[0]):
            for j in range(bbox.shape[1]):
                if bbox[i, j, 3] < bbox[i, j, 1]:
                    t = bbox[i, j, 3]
                    bbox[i, j, 3] = bbox[i, j, 1]
                    bbox[i, j, 1] = t
                if bbox[i, j, 2] < bbox[i, j, 0]:
                    t = bbox[i, j, 2]
                    bbox[i, j, 2] = bbox[i, j, 0]
                    bbox[i, j, 0] = t

        image = ImageList(
            torch.zeros(self.batch_size, self.num_channels, self.image_size, self.image_size, device=torch_device),
            self.image_size,
        )

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.text_seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.text_seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        entities = None
        relations = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.text_seq_length], self.num_labels)
            # we choose some random entities and relations
            entities = [{"start": [0, 4], "end": [3, 6], "label": [2, 1]} for _ in range(self.batch_size)]
            relations = [
                {"start_index": [0], "end_index": [5], "head": [0], "tail": [1]} for _ in range(self.batch_size)
            ]

        config = LayoutLMv2Config(
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
            is_decoder=False,
            initializer_range=self.initializer_range,
            image_feature_pool_shape=self.image_feature_pool_shape,
            coordinate_size=self.coordinate_size,
            shape_size=self.shape_size,
        )

        # use smaller resnet backbone to make tests faster
        config.detectron2_config_args["MODEL.RESNETS.DEPTH"] = 18
        config.detectron2_config_args["MODEL.RESNETS.RES2_OUT_CHANNELS"] = 64
        config.detectron2_config_args["MODEL.RESNETS.NUM_GROUPS"] = 1

        return (
            config,
            input_ids,
            bbox,
            image,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            entities,
            relations,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        bbox,
        image,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        entities,
        relations,
    ):
        model = LayoutLMv2Model(config=config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids, bbox=bbox, image=image, attention_mask=input_mask, token_type_ids=token_type_ids)
        result = model(input_ids, bbox=bbox, image=image, token_type_ids=token_type_ids)
        result = model(input_ids, bbox=bbox, image=image)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def create_and_check_for_sequence_classification(
        self,
        config,
        input_ids,
        bbox,
        image,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        entities,
        relations,
    ):
        config.num_labels = self.num_labels
        model = LayoutLMv2ForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            bbox=bbox,
            image=image,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            labels=sequence_labels,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_token_classification(
        self,
        config,
        input_ids,
        bbox,
        image,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        entities,
        relations,
    ):
        config.num_labels = self.num_labels
        model = LayoutLMv2ForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            bbox=bbox,
            image=image,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            labels=token_labels,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.text_seq_length, self.num_labels))

    def create_and_check_for_question_answering(
        self,
        config,
        input_ids,
        bbox,
        image,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        entities,
        relations,
    ):
        model = LayoutLMv2ForQuestionAnswering(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            bbox=bbox,
            image=image,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
        )
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.text_seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.text_seq_length))

    def create_and_check_for_relation_extraction(
        self,
        config,
        input_ids,
        bbox,
        image,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        entities,
        relations,
    ):
        model = LayoutLMv2ForRelationExtraction(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            bbox=bbox,
            image=image,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            entities=entities,
            relations=relations,
        )
        self.parent.assertTrue(result.pred_relations)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            bbox,
            image,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            entities,
            relations,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "bbox": bbox,
            "image": image,
            "token_type_ids": token_type_ids,
            "attention_mask": input_mask,
        }
        return config, inputs_dict


@require_torch
@require_detectron2
class LayoutLMv2ModelTest(ModelTesterMixin, unittest.TestCase):

    test_pruning = False
    test_torchscript = True
    test_mismatched_shapes = False

    all_model_classes = (
        (
            LayoutLMv2Model,
            LayoutLMv2ForSequenceClassification,
            LayoutLMv2ForTokenClassification,
            LayoutLMv2ForQuestionAnswering,
            LayoutLMv2ForRelationExtraction,
        )
        if is_torch_available()
        else ()
    )

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = copy.deepcopy(inputs_dict)

        if model_class.__name__ == "LayoutLMv2ForRelationExtraction":
            # we choose some random entities and relations
            entities = [{"start": [0, 4], "end": [3, 6], "label": [2, 1]} for _ in range(self.model_tester.batch_size)]
            relations = [
                {"start_index": [0], "end_index": [5], "head": [0], "tail": [1]}
                for _ in range(self.model_tester.batch_size)
            ]
            inputs_dict["entities"] = entities
            inputs_dict["relations"] = relations

        if return_labels:
            if model_class in [
                *get_values(MODEL_FOR_QUESTION_ANSWERING_MAPPING),
                *get_values(MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING),
            ]:

                inputs_dict["start_positions"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )
                inputs_dict["end_positions"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )
            elif model_class in get_values(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING):
                inputs_dict["labels"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )
            elif model_class in get_values(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING):
                inputs_dict["labels"] = torch.zeros(
                    (self.model_tester.batch_size, self.model_tester.text_seq_length), dtype=torch.long, device=torch_device
                )

        return inputs_dict

    def setUp(self):
        self.model_tester = LayoutLMv2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LayoutLMv2Config, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @require_torch_multi_gpu
    @unittest.skip(
        reason=(
            "LayoutLMV2 and its dependency `detectron2` have some layers using `add_module` which doesn't work well"
            " with `nn.DataParallel`"
        )
    )
    def test_multi_gpu_data_parallel_forward(self):
        pass

    def test_model_various_embeddings(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        for type in ["absolute", "relative_key", "relative_key_query"]:
            config_and_inputs[0].position_embedding_type = type
            self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    def test_for_relation_extraction(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_relation_extraction(*config_and_inputs)

    def test_save_load_fast_init_from_base(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        base_class = MODEL_MAPPING[config.__class__]

        if isinstance(base_class, tuple):
            base_class = base_class[0]

        for model_class in self.all_model_classes:
            if model_class == base_class:
                continue

            # make a copy of model class to not break future tests
            # from https://stackoverflow.com/questions/9541025/how-to-copy-a-python-class
            class CopyClass(model_class):
                pass

            model_class_copy = CopyClass

            # make sure that all keys are expected for test
            model_class_copy._keys_to_ignore_on_load_missing = []

            # make init deterministic, but make sure that
            # non-initialized weights throw errors nevertheless
            model_class_copy._init_weights = self._mock_init_weights

            model = base_class(config)
            state_dict = model.state_dict()

            # this will often delete a single weight of a multi-weight module
            # to test an edge case
            random_key_to_del = random.choice(list(state_dict.keys()))
            del state_dict[random_key_to_del]

            # check that certain keys didn't get saved with the model
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                torch.save(state_dict, os.path.join(tmpdirname, "pytorch_model.bin"))

                model_fast_init = model_class_copy.from_pretrained(tmpdirname)
                model_slow_init = model_class_copy.from_pretrained(tmpdirname, _fast_init=False)

                for key in model_fast_init.state_dict().keys():
                    if key == "layoutlmv2.visual_segment_embedding":
                        # we skip the visual segment embedding as it has a custom initialization scheme
                        continue
                    max_diff = (model_slow_init.state_dict()[key] - model_fast_init.state_dict()[key]).sum().item()
                    self.assertLessEqual(max_diff, 1e-3, msg=f"{key} not identical")

    def test_save_load_fast_init_to_base(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        base_class = MODEL_MAPPING[config.__class__]

        if isinstance(base_class, tuple):
            base_class = base_class[0]

        for model_class in self.all_model_classes:

            if model_class == base_class:
                continue

            # make a copy of model class to not break future tests
            # from https://stackoverflow.com/questions/9541025/how-to-copy-a-python-class
            class CopyClass(base_class):
                pass

            base_class_copy = CopyClass

            # make sure that all keys are expected for test
            base_class_copy._keys_to_ignore_on_load_missing = []

            # make init deterministic, but make sure that
            # non-initialized weights throw errors nevertheless
            base_class_copy._init_weights = self._mock_init_weights

            model = model_class(config)
            state_dict = model.state_dict()

            # this will often delete a single weight of a multi-weight module
            # to test an edge case
            random_key_to_del = random.choice(list(state_dict.keys()))
            del state_dict[random_key_to_del]

            # check that certain keys didn't get saved with the model
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.config.save_pretrained(tmpdirname)
                torch.save(state_dict, os.path.join(tmpdirname, "pytorch_model.bin"))

                model_fast_init = base_class_copy.from_pretrained(tmpdirname)
                model_slow_init = base_class_copy.from_pretrained(tmpdirname, _fast_init=False)

                for key in model_fast_init.state_dict().keys():
                    if key == "layoutlmv2.visual_segment_embedding":
                        # we skip the visual segment embedding as it has a custom initialization scheme
                        continue
                    max_diff = (model_slow_init.state_dict()[key] - model_fast_init.state_dict()[key]).sum().item()
                    self.assertLessEqual(max_diff, 1e-3, msg=f"{key} not identical")
    
    @slow
    def test_model_from_pretrained(self):
        for model_name in LAYOUTLMV2_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = LayoutLMv2Model.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def test_training(self):
        if not self.model_tester.is_training:
            return
    
        for model_class in self.all_model_classes:
            print("Model class:", model_class)
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.return_dict = True
    
            if model_class in get_values(MODEL_MAPPING):
                continue
    
            model = model_class(config)
            model.to(torch_device)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
    
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if "backbone" in name or "visual_segment_embedding" in name:
                    continue

                if param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

    # we overwrite this as LayoutLMv2ForRelationExtraction is not supported
    def test_headmasking(self):
        if not self.test_head_masking:
            return

        global_rng = random.Random()

        global_rng.seed(42)
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        global_rng.seed()

        inputs_dict["output_attentions"] = True
        config.output_hidden_states = True
        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        for model_class in self.all_model_classes:
            if model_class.__name__ == "LayoutLMv2ForRelationExtraction":
                continue

            model = model_class(config=configs_no_init)
            model.to(torch_device)
            model.eval()

            # Prepare head_mask
            # Set require_grad after having prepared the tensor to avoid error (leaf variable has been moved into the graph interior)
            head_mask = torch.ones(
                self.model_tester.num_hidden_layers,
                self.model_tester.num_attention_heads,
                device=torch_device,
            )
            head_mask[0, 0] = 0
            head_mask[-1, :-1] = 0
            head_mask.requires_grad_(requires_grad=True)
            inputs = self._prepare_for_class(inputs_dict, model_class).copy()
            inputs["head_mask"] = head_mask
            outputs = model(**inputs, return_dict=True)

            # Test that we can get a gradient back for importance score computation
            output = sum(t.sum() for t in outputs[0])
            output = output.sum()
            output.backward()
            multihead_outputs = head_mask.grad

            self.assertIsNotNone(multihead_outputs)
            self.assertEqual(len(multihead_outputs), self.model_tester.num_hidden_layers)

            def check_attentions_validity(attentions):
                # Remove Nan
                for t in attentions:
                    self.assertLess(
                        torch.sum(torch.isnan(t)), t.numel() / 4
                    )  # Check we don't have more than 25% nans (arbitrary)
                attentions = [
                    t.masked_fill(torch.isnan(t), 0.0) for t in attentions
                ]  # remove them (the test is less complete)

                self.assertAlmostEqual(attentions[0][..., 0, :, :].flatten().sum().item(), 0.0)
                self.assertNotEqual(attentions[0][..., -1, :, :].flatten().sum().item(), 0.0)
                if len(attentions) > 2:  # encoder-decoder models have only 2 layers in each module
                    self.assertNotEqual(attentions[1][..., 0, :, :].flatten().sum().item(), 0.0)
                self.assertAlmostEqual(attentions[-1][..., -2, :, :].flatten().sum().item(), 0.0)
                self.assertNotEqual(attentions[-1][..., -1, :, :].flatten().sum().item(), 0.0)

            check_attentions_validity(outputs.attentions)
    
    # we overwrite this as LayoutLMv2 requires special inputs + LayoutLMv2ForRelationExtraction is not supported
    def _create_and_check_torchscript(self, config, inputs_dict):
        if not self.test_torchscript:
            return

        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        configs_no_init.torchscript = True
        for model_class in self.all_model_classes:
            if model_class.__name__ == "LayoutLMv2ForRelationExtraction":
                continue

            model = model_class(config=configs_no_init)
            model.to(torch_device)
            model.eval()
            inputs = self._prepare_for_class(inputs_dict, model_class)

            try:
                input_ids = inputs["input_ids"]
                bbox = inputs["bbox"]
                image = inputs["image"].tensor
                traced_model = torch.jit.trace(
                    model, (input_ids, bbox, image), check_trace=False
                )  # when traced model is checked, an error is produced due to name mangling
            except RuntimeError:
                self.fail("Couldn't trace module.")

            with tempfile.TemporaryDirectory() as tmp_dir_name:
                pt_file_name = os.path.join(tmp_dir_name, "traced_model.pt")

                try:
                    torch.jit.save(traced_model, pt_file_name)
                except Exception:
                    self.fail("Couldn't save module.")

                try:
                    loaded_model = torch.jit.load(pt_file_name)
                except Exception:
                    self.fail("Couldn't load module.")

            model.to(torch_device)
            model.eval()

            loaded_model.to(torch_device)
            loaded_model.eval()

            model_state_dict = model.state_dict()
            loaded_model_state_dict = loaded_model.state_dict()

            non_persistent_buffers = {}
            for key in loaded_model_state_dict.keys():
                if key not in model_state_dict.keys():
                    non_persistent_buffers[key] = loaded_model_state_dict[key]

            loaded_model_state_dict = {
                key: value for key, value in loaded_model_state_dict.items() if key not in non_persistent_buffers
            }

            self.assertEqual(set(model_state_dict.keys()), set(loaded_model_state_dict.keys()))

            model_buffers = list(model.buffers())
            for non_persistent_buffer in non_persistent_buffers.values():
                found_buffer = False
                for i, model_buffer in enumerate(model_buffers):
                    if torch.equal(non_persistent_buffer, model_buffer):
                        found_buffer = True
                        break

                self.assertTrue(found_buffer)
                model_buffers.pop(i)

            models_equal = True
            for layer_name, p1 in model_state_dict.items():
                if layer_name in loaded_model_state_dict:
                    p2 = loaded_model_state_dict[layer_name]
                    if p1.data.ne(p2.data).sum() > 0:
                        models_equal = False

            self.assertTrue(models_equal)

            # Avoid memory leak. Without this, each call increase RAM usage by ~20MB.
            # (Even with this call, there are still memory leak by ~0.04MB)
            self.clear_torch_jit_class_registry()


def prepare_layoutlmv2_batch_inputs():
    # Here we prepare a batch of 2 sequences to test a LayoutLMv2 forward pass on:
    # fmt: off
    input_ids = torch.tensor([[101,1019,1014,1016,1037,12849,4747,1004,14246,2278,5439,4524,5002,2930,2193,2930,4341,3208,1005,1055,2171,2848,11300,3531,102],[101,4070,4034,7020,1024,3058,1015,1013,2861,1013,6070,19274,2772,6205,27814,16147,16147,4343,2047,10283,10969,14389,1012,2338,102]])  # noqa: E231
    bbox = torch.tensor([[[0,0,0,0],[423,237,440,251],[427,272,441,287],[419,115,437,129],[961,885,992,912],[256,38,330,58],[256,38,330,58],[336,42,353,57],[360,39,401,56],[360,39,401,56],[411,39,471,59],[479,41,528,59],[533,39,630,60],[67,113,134,131],[141,115,209,132],[68,149,133,166],[141,149,187,164],[195,148,287,165],[195,148,287,165],[195,148,287,165],[295,148,349,165],[441,149,492,166],[497,149,546,164],[64,201,125,218],[1000,1000,1000,1000]],[[0,0,0,0],[662,150,754,166],[665,199,742,211],[519,213,554,228],[519,213,554,228],[134,433,187,454],[130,467,204,480],[130,467,204,480],[130,467,204,480],[130,467,204,480],[130,467,204,480],[314,469,376,482],[504,684,582,706],[941,825,973,900],[941,825,973,900],[941,825,973,900],[941,825,973,900],[610,749,652,765],[130,659,168,672],[176,657,237,672],[238,657,312,672],[443,653,628,672],[443,653,628,672],[716,301,825,317],[1000,1000,1000,1000]]])  # noqa: E231
    image = ImageList(torch.randn((2,3,224,224)), image_sizes=[(224,224), (224,224)])  # noqa: E231
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],])  # noqa: E231
    token_type_ids = torch.tensor([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])  # noqa: E231
    # fmt: on

    return input_ids, bbox, image, attention_mask, token_type_ids


@require_torch
@require_detectron2
class LayoutLMv2ModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_no_head(self):
        model = LayoutLMv2Model.from_pretrained("microsoft/layoutlmv2-base-uncased").to(torch_device)

        (
            input_ids,
            bbox,
            image,
            attention_mask,
            token_type_ids,
        ) = prepare_layoutlmv2_batch_inputs()

        # forward pass
        outputs = model(
            input_ids=input_ids.to(torch_device),
            bbox=bbox.to(torch_device),
            image=image.to(torch_device),
            attention_mask=attention_mask.to(torch_device),
            token_type_ids=token_type_ids.to(torch_device),
        )

        # verify the sequence output
        expected_shape = torch.Size(
            (
                2,
                input_ids.shape[1]
                + model.config.image_feature_pool_shape[0] * model.config.image_feature_pool_shape[1],
                model.config.hidden_size,
            )
        )
        self.assertEqual(outputs.last_hidden_state.shape, expected_shape)

        expected_slice = torch.tensor(
            [[-0.1087, 0.0727, -0.3075], [0.0799, -0.0427, -0.0751], [-0.0367, 0.0480, -0.1358]], device=torch_device
        )
        self.assertTrue(torch.allclose(outputs.last_hidden_state[0, :3, :3], expected_slice, atol=1e-3))

        # verify the pooled output
        expected_shape = torch.Size((2, model.config.hidden_size))
        self.assertEqual(outputs.pooler_output.shape, expected_shape)
