# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch EGT model. """


import copy
import inspect
import os
import tempfile
import unittest

from transformers import EGTConfig, is_torch_available
from transformers.testing_utils import require_dgl, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import tensor

    from transformers import EGTForGraphClassification, EGTModel
    from transformers.models.egt.modeling_egt import EGT_PRETRAINED_MODEL_ARCHIVE_LIST

NODE_FEATURES_OFFSET = 128
EDGE_FEATURES_OFFSET = 8


class EGTModelTester:
    def __init__(
        self,
        parent,
        num_atoms=3,
        num_edges=1,
        feat_size=128,
        edge_feat_size=32,
        num_heads=8,
        num_layers=4,
        dropout=0.0,
        attn_dropout=0.0,
        activation="ELU",
        egt_simple=False,
        upto_hop=16,
        mlp_ratios=[1.0, 1.0],
        num_virtual_nodes=4,
        svd_pe_size=8,
        num_classes=1,
        batch_size=10,
        graph_size=20,
        is_training=True,
    ):
        self.num_atoms = num_atoms
        self.num_edges = num_edges
        self.parent = parent
        self.feat_size = feat_size
        self.edge_feat_size = edge_feat_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.activation = activation
        self.egt_simple = egt_simple
        self.upto_hop = upto_hop
        self.mlp_ratios = mlp_ratios
        self.num_virtual_nodes = num_virtual_nodes
        self.svd_pe_size = svd_pe_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.graph_size = graph_size
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        featm = ids_tensor(
            [self.batch_size, self.graph_size, self.graph_size, 1], self.num_edges * EDGE_FEATURES_OFFSET + 1
        )
        dm = ids_tensor([self.batch_size, self.graph_size, self.graph_size], self.upto_hop + 2)
        node_feat = ids_tensor([self.batch_size, self.graph_size, 1], self.num_atoms * NODE_FEATURES_OFFSET + 1)
        svd_pe = floats_tensor([self.batch_size, self.graph_size, self.svd_pe_size * 2])
        attn_mask = random_attention_mask([self.batch_size, self.graph_size])
        labels = ids_tensor([self.batch_size], self.num_classes)

        config = self.get_config()

        return config, featm, dm, node_feat, svd_pe, attn_mask, labels

    def get_config(self):
        return EGTConfig(
            feat_size=self.feat_size,
            edge_feat_size=self.edge_feat_size,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
            attn_dropout=self.attn_dropout,
            activation=self.activation,
            egt_simple=self.egt_simple,
            upto_hop=self.upto_hop,
            mlp_ratios=self.mlp_ratios,
            num_virtual_nodes=self.num_virtual_nodes,
            svd_pe_size=self.svd_pe_size,
            num_classes=self.num_classes,
        )

    def create_and_check_model(self, config, featm, dm, node_feat, svd_pe, attn_mask, labels):
        model = EGTModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            node_feat=node_feat,
            dm=dm,
            featm=featm,
            attn_mask=attn_mask,
            svd_pe=svd_pe,
            labels=labels,
        )
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.num_classes))

    def create_and_check_for_graph_classification(self, config, featm, dm, node_feat, svd_pe, attn_mask, labels):
        model = EGTForGraphClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(
            node_feat=node_feat,
            dm=dm,
            featm=featm,
            svd_pe=svd_pe,
            attn_mask=attn_mask,
            labels=labels,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_classes))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            featm,
            dm,
            node_feat,
            svd_pe,
            attn_mask,
            labels,
        ) = config_and_inputs
        inputs_dict = {
            "featm": featm,
            "dm": dm,
            "node_feat": node_feat,
            "svd_pe": svd_pe,
            "attn_mask": attn_mask,
            "labels": labels,
        }
        return config, inputs_dict


@require_dgl
@require_torch
class EGTModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (EGTForGraphClassification, EGTModel) if is_torch_available() else ()
    all_generative_model_classes = ()
    pipeline_model_mapping = {"feature-extraction": EGTModel} if is_torch_available() else {}
    test_pruning = False
    test_head_masking = False
    test_resize_embeddings = False
    main_input_name_nodes = "input_nodes"
    main_input_name_edges = "attn_edge_type"
    has_attentions = False  # does not output attention

    def setUp(self):
        self.model_tester = EGTModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=EGTConfig,
            common_properties=["feat_size", "edge_feat_size", "num_heads", "num_layers"],
            has_text_modality=False,
        )

    # overwrite from common as `EGT` requires more input arguments
    def _create_and_check_torchscript(self, config, inputs_dict):
        if not self.test_torchscript:
            return

        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        configs_no_init.torchscript = True
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            model.to(torch_device)
            model.eval()
            inputs = self._prepare_for_class(inputs_dict, model_class)

            try:
                required_keys = (
                    "node_feat",
                    "dm",
                    "featm",
                    "svd_pe",
                    "attn_mask",
                )
                required_inputs = tuple(inputs[k] for k in required_keys)
                model(*required_inputs)
                traced_model = torch.jit.trace(model, required_inputs)
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

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="EGT does not use one single inputs_embedding but three")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="EGTModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="EGTModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @unittest.skip(reason="EGT does not implement feed forward chunking")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="EGT does not share input and output embeddings")
    def test_model_common_attributes(self):
        pass

    def test_initialization(self):
        def _config_zero_init(config):
            configs_no_init = copy.deepcopy(config)
            for key in configs_no_init.__dict__.keys():
                if "_range" in key or "_std" in key or "initializer_factor" in key or "layer_scale" in key:
                    setattr(configs_no_init, key, 1e-10)
            return configs_no_init

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.assertTrue(
                        -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

    @unittest.skip(reason="EGT does not record hidden states")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="EGT does not record hidden states")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    # Inputs are 'input_nodes' and 'attn_edge_type' not 'input_ids'
    def test_model_main_input_name(self):
        for model_class in self.all_model_classes:
            model_signature = inspect.signature(getattr(model_class, "forward"))
            # The main input is the name of the argument after `self`
            observed_main_input_name_nodes = list(model_signature.parameters.keys())[1]
            observed_main_input_name_edges = list(model_signature.parameters.keys())[2]
            self.assertEqual(model_class.main_input_name_nodes, observed_main_input_name_nodes)
            self.assertEqual(model_class.main_input_name_edges, observed_main_input_name_edges)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["node_feat", "featm"]
            self.assertListEqual(arg_names[:2], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_graph_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_graph_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in EGT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = EGTForGraphClassification.from_pretrained(model_name)
            self.assertIsNotNone(model)


@require_torch
class EGTModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_graph_classification(self):
        model = EGTForGraphClassification.from_pretrained("Zhiteng/dgl-egt")

        # Actual real graph data from the MUTAG dataset
        # fmt: off
        model_input = {
            "featm": tensor(
                [
                    [
                        [[0], [2], [0], [0], [0], [2], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[2], [0], [2], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [2], [0], [2], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [2], [0], [2], [0], [0], [0], [0], [2], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [2], [0], [2], [2], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[2], [0], [0], [0], [2], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [2], [0], [0], [2], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [2], [0], [2], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [2], [0], [2], [0], [0], [0], [2], [0], [0], [0]],
                        [[0], [0], [0], [2], [0], [0], [0], [0], [2], [0], [2], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0], [0], [2], [0], [2], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [2], [0], [2], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [2], [0], [2], [2], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0], [2], [0], [0], [0], [2], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [2], [0], [0], [2], [2]],
                        [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [2], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [2], [0], [0]],
                    ],
                    [
                        [[0], [2], [0], [0], [0], [0], [0], [0], [0], [2], [0], [0], [0], [0], [0], [0], [0]],
                        [[2], [0], [2], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [2], [0], [2], [0], [0], [0], [2], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [2], [0], [2], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [2], [0], [2], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [2], [0], [2], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [2], [0], [2], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [2], [0], [0], [0], [2], [0], [2], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [2], [0], [2], [2], [0], [0], [0], [0], [0], [0]],
                        [[2], [0], [0], [0], [0], [0], [0], [0], [2], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0], [2], [0], [0], [2], [2], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [2], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [2], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                    ],
                ]
            ),
            # fmt: on
            "dm": tensor(
                [
                    [
                        [0, 1, 2, 3, 2, 1, 3, 4, 5, 4, 5, 6, 7, 6, 8, 9, 9],
                        [1, 0, 1, 2, 3, 2, 4, 5, 4, 3, 4, 5, 6, 5, 7, 8, 8],
                        [2, 1, 0, 1, 2, 3, 3, 4, 3, 2, 3, 4, 5, 4, 6, 7, 7],
                        [3, 2, 1, 0, 1, 2, 2, 3, 2, 1, 2, 3, 4, 3, 5, 6, 6],
                        [2, 3, 2, 1, 0, 1, 1, 2, 3, 2, 3, 4, 5, 4, 6, 7, 7],
                        [1, 2, 3, 2, 1, 0, 2, 3, 4, 3, 4, 5, 6, 5, 7, 8, 8],
                        [3, 4, 3, 2, 1, 2, 0, 1, 2, 3, 4, 5, 4, 3, 5, 6, 6],
                        [4, 5, 4, 3, 2, 3, 1, 0, 1, 2, 3, 4, 3, 2, 4, 5, 5],
                        [5, 4, 3, 2, 3, 4, 2, 1, 0, 1, 2, 3, 2, 1, 3, 4, 4],
                        [4, 3, 2, 1, 2, 3, 3, 2, 1, 0, 1, 2, 3, 2, 4, 5, 5],
                        [5, 4, 3, 2, 3, 4, 4, 3, 2, 1, 0, 1, 2, 3, 3, 4, 4],
                        [6, 5, 4, 3, 4, 5, 5, 4, 3, 2, 1, 0, 1, 2, 2, 3, 3],
                        [7, 6, 5, 4, 5, 6, 4, 3, 2, 3, 2, 1, 0, 1, 1, 2, 2],
                        [6, 5, 4, 3, 4, 5, 3, 2, 1, 2, 3, 2, 1, 0, 2, 3, 3],
                        [8, 7, 6, 5, 6, 7, 5, 4, 3, 4, 3, 2, 1, 2, 0, 1, 1],
                        [9, 8, 7, 6, 7, 8, 6, 5, 4, 5, 4, 3, 2, 3, 1, 0, 2],
                        [9, 8, 7, 6, 7, 8, 6, 5, 4, 5, 4, 3, 2, 3, 1, 2, 0]
                    ],
                    [
                        [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 3, 4, 4, 0, 0, 0, 0],
                        [1, 0, 1, 2, 3, 4, 3, 2, 3, 2, 4, 5, 5, 0, 0, 0, 0],
                        [2, 1, 0, 1, 2, 3, 2, 1, 2, 3, 3, 4, 4, 0, 0, 0, 0],
                        [3, 2, 1, 0, 1, 2, 3, 2, 3, 4, 4, 5, 5, 0, 0, 0, 0],
                        [4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 5, 6, 6, 0, 0, 0, 0],
                        [5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 4, 5, 5, 0, 0, 0, 0],
                        [4, 3, 2, 3, 2, 1, 0, 1, 2, 3, 3, 4, 4, 0, 0, 0, 0],
                        [3, 2, 1, 2, 3, 2, 1, 0, 1, 2, 2, 3, 3, 0, 0, 0, 0],
                        [2, 3, 2, 3, 4, 3, 2, 1, 0, 1, 1, 2, 2, 0, 0, 0, 0],
                        [1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 2, 3, 3, 0, 0, 0, 0],
                        [3, 4, 3, 4, 5, 4, 3, 2, 1, 2, 0, 1, 1, 0, 0, 0, 0],
                        [4, 5, 4, 5, 6, 5, 4, 3, 2, 3, 1, 0, 2, 0, 0, 0, 0],
                        [4, 5, 4, 5, 6, 5, 4, 3, 2, 3, 1, 2, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    ],
                ]
            ),
            "node_feat": tensor(
                [
                    [[2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2]],
                    [[2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [0], [0], [0], [0]],
                ]
            ),
            "svd_pe": tensor(
                [
                    [
                        [-7.8307e-16, -2.9407e-01, -4.5033e-01,  6.7591e-03,  6.3101e-01,
                         -2.1299e-02,  1.6331e-03,  4.7326e-02,  2.9407e-01, -3.9521e-16,
                          6.7591e-03,  4.5033e-01, -2.1299e-02, -6.3101e-01, -4.7326e-02,
                          1.6331e-03],
                        [-3.1084e-01,  4.4883e-16, -6.4028e-03, -4.2660e-01,  1.9575e-02,
                          5.7994e-01,  3.6495e-01, -1.2594e-02,  1.2181e-15,  3.1084e-01,
                          4.2660e-01, -6.4028e-03, -5.7994e-01,  1.9575e-02, -1.2594e-02,
                         -3.6495e-01],
                        [-8.7008e-16, -4.6958e-01, -4.4607e-01,  6.6952e-03,  3.6095e-01,
                         -1.2183e-02,  1.6979e-02,  4.9203e-01,  4.6958e-01, -5.3888e-16,
                          6.6952e-03,  4.4607e-01, -1.2183e-02, -3.6095e-01, -4.9203e-01,
                          1.6979e-02],
                        [ 8.4280e-01, -1.0622e-15,  7.6656e-03,  5.1073e-01, -1.2644e-03,
                         -3.7459e-02, -3.6221e-01,  1.2499e-02, -1.1801e-15, -8.4280e-01,
                         -5.1073e-01,  7.6656e-03,  3.7459e-02, -1.2644e-03,  1.2499e-02,
                          3.6221e-01],
                        [ 1.0819e-15,  7.1716e-01,  6.4166e-01, -9.6308e-03, -2.2316e-01,
                          7.5325e-03,  1.6678e-02,  4.8331e-01, -7.1716e-01,  9.3042e-16,
                         -9.6308e-03, -6.4166e-01,  7.5325e-03,  2.2316e-01, -4.8331e-01,
                          1.6678e-02],
                        [ 4.1162e-01, -4.4883e-16,  7.7999e-03,  5.1968e-01, -1.6856e-02,
                         -4.9938e-01,  2.9501e-01, -1.0180e-02, -7.4297e-16, -4.1162e-01,
                         -5.1968e-01,  7.7999e-03,  4.9938e-01, -1.6856e-02, -1.0180e-02,
                         -2.9501e-01],
                        [ 5.0744e-01, -6.2900e-16,  4.7715e-03,  3.1791e-01,  5.2362e-03,
                          1.5513e-01,  7.8148e-01, -2.6967e-02, -6.7363e-16, -5.0744e-01,
                         -3.1791e-01,  4.7715e-03, -1.5513e-01,  5.2362e-03, -2.6967e-02,
                         -7.8148e-01],
                        [-6.5714e-16, -5.2949e-01, -2.6351e-02,  3.9551e-04, -4.8850e-01,
                          1.6489e-02, -2.3176e-02, -6.7163e-01,  5.2949e-01, -5.8186e-16,
                          3.9551e-04,  2.6351e-02,  1.6489e-02,  4.8850e-01,  6.7163e-01,
                         -2.3176e-02],
                        [-7.9338e-01,  1.1280e-15,  3.9404e-03,  2.6253e-01, -2.2967e-02,
                         -6.8044e-01, -2.1110e-01,  7.2847e-03,  9.4079e-16,  7.9338e-01,
                         -2.6253e-01,  3.9404e-03,  6.8044e-01, -2.2967e-02,  7.2847e-03,
                          2.1110e-01],
                        [-1.1318e-15, -8.8379e-01,  1.4538e-02, -2.1821e-04, -5.2004e-01,
                          1.7553e-02,  1.8172e-02,  5.2659e-01,  8.8379e-01, -1.2072e-15,
                         -2.1821e-04, -1.4538e-02,  1.7553e-02,  5.2004e-01, -5.2659e-01,
                          1.8172e-02],
                        [ 5.3505e-01, -8.5638e-16, -4.1837e-03, -2.7875e-01,  8.3213e-03,
                          2.4653e-01, -6.2713e-01,  2.1641e-02, -6.3409e-16, -5.3505e-01,
                          2.7875e-01, -4.1837e-03, -2.4653e-01,  8.3213e-03,  2.1641e-02,
                          6.2713e-01],
                        [ 4.8499e-16,  4.3068e-01, -5.7119e-01,  8.5730e-03, -9.8360e-02,
                          3.3200e-03, -1.3811e-02, -4.0024e-01, -4.3068e-01,  8.5755e-16,
                          8.5730e-03,  5.7119e-01,  3.3200e-03,  9.8360e-02,  4.0024e-01,
                         -1.3811e-02],
                        [ 5.2302e-01, -8.5509e-16, -1.3831e-02, -9.2149e-01, -1.4000e-02,
                         -4.1477e-01,  3.5627e-02, -1.2294e-03, -5.5741e-16, -5.2302e-01,
                          9.2149e-01, -1.3831e-02,  4.1477e-01, -1.4000e-02, -1.2294e-03,
                         -3.5627e-02],
                        [-6.0984e-16, -5.3583e-01,  5.6347e-01, -8.4572e-03, -1.5532e-01,
                          5.2426e-03, -5.7611e-03, -1.6695e-01,  5.3583e-01, -7.1094e-16,
                         -8.4572e-03, -5.6347e-01,  5.2426e-03,  1.5532e-01,  1.6695e-01,
                         -5.7611e-03],
                        [-3.3225e-16, -3.1840e-01,  8.0165e-01, -1.2032e-02,  7.6641e-01,
                         -2.5869e-02, -9.8673e-03, -2.8594e-01,  3.1840e-01, -2.6527e-16,
                         -1.2032e-02, -8.0165e-01, -2.5869e-02, -7.6641e-01,  2.8594e-01,
                         -9.8673e-03],
                        [-1.2960e-01, -4.9727e-17,  5.7260e-03,  3.8150e-01,  1.5124e-02,
                          4.4807e-01, -1.9348e-01,  6.6766e-03,  1.2942e-16,  1.2960e-01,
                         -3.8150e-01,  5.7260e-03, -4.4807e-01,  1.5124e-02,  6.6766e-03,
                          1.9348e-01],
                        [-1.2960e-01, -4.9727e-17,  5.7260e-03,  3.8150e-01,  1.5124e-02,
                          4.4807e-01, -1.9348e-01,  6.6766e-03,  1.2942e-16,  1.2960e-01,
                         -3.8150e-01,  5.7260e-03, -4.4807e-01,  1.5124e-02,  6.6766e-03,
                          1.9348e-01]
                    ],
                    [
                        [-8.5768e-16, -4.7616e-01,  6.4296e-03,  1.1986e-01,  3.1722e-04,
                         -8.3400e-01,  6.8728e-06, -5.8059e-02,  4.7616e-01, -5.0020e-16,
                          1.1986e-01, -6.4296e-03, -8.3400e-01, -3.1722e-04, -5.8059e-02,
                         -6.8728e-06],
                        [ 5.6321e-01, -6.5394e-16, -2.2227e-01,  1.1924e-02, -7.9471e-01,
                         -3.0227e-04,  2.5130e-01,  2.9748e-05, -8.5768e-16, -5.6321e-01,
                         -1.1924e-02, -2.2227e-01,  3.0227e-04, -7.9471e-01, -2.9748e-05,
                          2.5130e-01],
                        [-9.4344e-16, -8.6834e-01, -2.8296e-02, -5.2748e-01,  1.4446e-04,
                         -3.7979e-01, -4.0716e-05,  3.4395e-01,  8.6834e-01, -1.0735e-15,
                         -5.2748e-01,  2.8296e-02, -3.7979e-01, -1.4446e-04,  3.4395e-01,
                          4.0716e-05],
                        [-5.2803e-01,  6.2587e-16,  6.4061e-01, -3.4365e-02, -4.8723e-02,
                         -1.8532e-05, -6.5571e-01, -7.7620e-05,  5.0315e-16,  5.2803e-01,
                          3.4365e-02,  6.4061e-01,  1.8532e-05, -4.8723e-02,  7.7620e-05,
                         -6.5571e-01],
                        [-3.5003e-16, -3.9218e-01, -3.4726e-02, -6.4734e-01, -1.7276e-04,
                          4.5421e-01, -4.7589e-05,  4.0201e-01,  3.9218e-01, -3.1458e-16,
                         -6.4734e-01,  3.4726e-02,  4.5421e-01,  1.7276e-04,  4.0201e-01,
                          4.7589e-05],
                        [-4.0817e-01,  2.0027e-16,  5.4656e-01, -2.9320e-02, -6.4501e-01,
                         -2.4533e-04,  1.9836e-01,  2.3481e-05,  3.4089e-16,  4.0817e-01,
                          2.9320e-02,  5.4656e-01,  2.4533e-04, -6.4501e-01, -2.3481e-05,
                          1.9836e-01],
                        [ 5.1245e-16,  5.8219e-01,  1.9044e-02,  3.5500e-01,  2.0195e-04,
                         -5.3094e-01, -7.4302e-05,  6.2768e-01, -5.8219e-01,  5.0050e-16,
                          3.5500e-01, -1.9044e-02, -5.3094e-01, -2.0195e-04,  6.2768e-01,
                          7.4302e-05],
                        [-9.8164e-01,  1.0802e-15,  1.0448e-01, -5.6047e-03, -1.6592e-01,
                         -6.3107e-05,  5.1571e-01,  6.1048e-05,  8.8239e-16,  9.8164e-01,
                          5.6047e-03,  1.0448e-01,  6.3107e-05, -1.6592e-01, -6.1048e-05,
                          5.1571e-01],
                        [ 8.4209e-16,  8.9284e-01, -3.7062e-02, -6.9088e-01,  3.8897e-05,
                         -1.0226e-01, -3.5865e-05,  3.0298e-01, -8.9284e-01,  4.7313e-16,
                         -6.9088e-01,  3.7062e-02, -1.0226e-01, -3.8897e-05,  3.0298e-01,
                          3.5865e-05],
                        [ 5.7348e-01, -4.3952e-16,  4.4208e-01, -2.3715e-02, -4.7909e-01,
                         -1.8223e-04, -3.1735e-01, -3.7567e-05, -6.2792e-16, -5.7348e-01,
                          2.3715e-02,  4.4208e-01,  1.8223e-04, -4.7909e-01,  3.7567e-05,
                         -3.1735e-01],
                        [-5.7625e-01,  2.1847e-16, -9.2941e-01,  4.9858e-02, -4.6937e-01,
                         -1.7853e-04, -4.8839e-01, -5.7813e-05,  4.9991e-16,  5.7625e-01,
                         -4.9858e-02, -9.2941e-01,  1.7853e-04, -4.6937e-01,  5.7813e-05,
                         -4.8839e-01],
                        [ 2.0941e-16,  2.4139e-01, -2.7186e-02, -5.0679e-01,  1.1689e-04,
                         -3.0731e-01,  5.0818e-05, -4.2929e-01, -2.4139e-01,  5.7273e-17,
                         -5.0679e-01,  2.7186e-02, -3.0731e-01, -1.1689e-04, -4.2929e-01,
                         -5.0818e-05],
                        [-2.0941e-16, -2.4139e-01,  2.7186e-02,  5.0679e-01, -1.1689e-04,
                          3.0731e-01, -5.0818e-05,  4.2929e-01,  2.4139e-01,  7.0525e-18,
                          5.0679e-01, -2.7186e-02,  3.0731e-01,  1.1689e-04,  4.2929e-01,
                          5.0818e-05],
                        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                          0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                          0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                          0.0000e+00],
                        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                          0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                          0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                          0.0000e+00],
                        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                          0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                          0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                          0.0000e+00],
                        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                          0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                          0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                          0.0000e+00]
                    ]
                ]
            ),
            "attn_mask": tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                ]
            ),
            "labels": tensor([1, 0]),
        }

        output = model(**model_input)["logits"]

        expected_shape = torch.Size((2, 1))
        self.assertEqual(output.shape, expected_shape)

        expected_logs = torch.tensor(
            [[7.7899], [9.0817]]
        )

        self.assertTrue(torch.allclose(output, expected_logs, atol=1e-4))
