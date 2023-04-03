# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the TensorFlow Blip model. """
import unittest

import numpy as np

from transformers import BlipTextConfig
from transformers.testing_utils import require_tf, slow
from transformers.utils import is_tf_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, ids_tensor, random_attention_mask


if is_tf_available():
    import tensorflow as tf

    from transformers import TFBlipTextModel
    from transformers.models.blip.modeling_tf_blip import TF_BLIP_PRETRAINED_MODEL_ARCHIVE_LIST


class BlipTextModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        projection_dim=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        bos_token_id=0,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.scope = scope
        self.bos_token_id = bos_token_id

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        if input_mask is not None:
            input_mask = input_mask.numpy()
            batch_size, seq_length = input_mask.shape
            rnd_start_indices = np.random.randint(1, seq_length - 1, size=(batch_size,))
            for batch_idx, start_index in enumerate(rnd_start_indices):
                input_mask[batch_idx, :start_index] = 1
                input_mask[batch_idx, start_index:] = 0

        config = self.get_config()

        return config, input_ids, tf.convert_to_tensor(input_mask)

    def get_config(self):
        return BlipTextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            projection_dim=self.projection_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            bos_token_id=self.bos_token_id,
        )

    def create_and_check_model(self, config, input_ids, input_mask):
        model = TFBlipTextModel(config=config)
        result = model(input_ids, attention_mask=input_mask, training=False)
        result = model(input_ids, training=False)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_tf
class BlipTextModelTest(TFModelTesterMixin, unittest.TestCase):
    all_model_classes = (TFBlipTextModel,) if is_tf_available() else ()
    test_onnx = False
    test_pruning = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = BlipTextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BlipTextConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_training(self):
        pass

    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="Blip does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="BlipTextModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="BlipTextModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        for model_name in TF_BLIP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            try:
                model = TFBlipTextModel.from_pretrained(model_name)
            except OSError:
                model = TFBlipTextModel.from_pretrained(model_name, from_pt=True)
            self.assertIsNotNone(model)

    # @unittest.skip(reason="This test class covers encoder-decoder models that the base test does not work with.")
    def test_pt_tf_model_equivalence(self):
        import transformers
        import inspect
        import tempfile
        import os
        import torch

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            # Output all for aggressive testing
            config.output_hidden_states = True
            config.output_attentions = self.has_attentions

            # Make sure no sequence has all zeros as attention mask, otherwise some tests fail due to the inconsistency
            # of the usage `1e-4`, `1e-9`, `1e-30`, `-inf`.
            self._make_attention_mask_non_null(inputs_dict)

            pt_model_class_name = model_class.__name__[2:]  # Skip the "TF" at the beginning
            pt_model_class = getattr(transformers, pt_model_class_name)

            tf_model = model_class(config)
            pt_model = pt_model_class(config)

            tf_inputs_dict = self._prepare_for_class(inputs_dict, model_class)
            tf_inputs_dict_with_labels = self._prepare_for_class(
                inputs_dict,
                model_class,
                # Not all models accept "labels" in the forward pass (yet :) )
                return_labels=True if "labels" in inspect.signature(model_class.call).parameters.keys() else False,
            )

            # For some models (e.g. base models), there is no label returned.
            # Set the input dict to `None` to avoid check outputs twice for the same input dicts.
            if not set(tf_inputs_dict_with_labels.keys()).symmetric_difference(tf_inputs_dict.keys()):
                tf_inputs_dict_with_labels = None
            # Check we can load pt model in tf and vice-versa with model => model functions
            breakpoint()
            tf_model = transformers.load_pytorch_model_in_tf2_model(tf_model, pt_model, tf_inputs=tf_inputs_dict)
            pt_model = transformers.load_tf2_model_in_pytorch_model(pt_model, tf_model, allow_missing_keys=True)

            # Original test: check without `labels`
            self.check_pt_tf_models(tf_model, pt_model, tf_inputs_dict)
            # check with `labels`
            if tf_inputs_dict_with_labels:
                self.check_pt_tf_models(tf_model, pt_model, tf_inputs_dict_with_labels)

            # Check we can load pt model in tf and vice-versa with checkpoint => model functions
            with tempfile.TemporaryDirectory() as tmpdirname:
                pt_checkpoint_path = os.path.join(tmpdirname, "pt_model.bin")
                torch.save(pt_model.state_dict(), pt_checkpoint_path)
                tf_model = transformers.load_pytorch_checkpoint_in_tf2_model(tf_model, pt_checkpoint_path)

                tf_checkpoint_path = os.path.join(tmpdirname, "tf_model.h5")
                tf_model.save_weights(tf_checkpoint_path)
                pt_model = transformers.load_tf2_checkpoint_in_pytorch_model(pt_model, tf_checkpoint_path)

            # Original test: check without `labels`
            self.check_pt_tf_models(tf_model, pt_model, tf_inputs_dict)
            # check with `labels`
            if tf_inputs_dict_with_labels:
                self.check_pt_tf_models(tf_model, pt_model, tf_inputs_dict_with_labels)
