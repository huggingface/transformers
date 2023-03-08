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
""" Testing suite for the PyTorch Graphormer model. """


import unittest
from typing import Callable

from transformers import TokenGTConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_modeling_common import ids_tensor


if is_torch_available():
    import torch
    from torch import tensor

    from transformers import TokenGTForGraphClassification  # , GraphormerModel


class TokenGTModelTester:
    def __init__(
        self,
        parent,
        num_classes: int = 1,
        num_atoms: int = 512 * 9,
        num_in_degree: int = 512,
        num_out_degree: int = 512,
        num_edges: int = 512 * 3,
        num_spatial: int = 512,
        num_edge_dis: int = 128,
        edge_type: str = "multi_hop",
        multi_hop_max_dist: int = 5,
        max_nodes: int = 128,
        spatial_pos_max: int = 1024,
        # for tokenization
        rand_node_id: bool = False,
        rand_node_id_dim: int = 64,
        orf_node_id: bool = False,
        orf_node_id_dim: int = 64,
        lap_node_id: bool = True,
        lap_node_id_k: int = 16,
        lap_node_id_sign_flip: bool = True,
        lap_node_id_eig_dropout: float = 0.0,
        type_id: bool = True,
        share_encoder_input_output_embed: bool = False,
        prenorm: bool = True,
        postnorm: bool = False,
        stochastic_depth: bool = False,
        performer: bool = False,
        performer_finetune: bool = False,
        performer_nb_features: int = None,
        performer_feature_redraw_interval: int = 1000,
        performer_generalized_attention: bool = False,
        performer_auto_check_redraw: bool = True,
        num_layers: int = 12,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 768,  # 3072 in TokenGTGraphEncoderLayer
        num_attention_heads: int = 32,
        dropout: float = 0.0,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0.0,
        encoder_normalize_before: bool = True,
        layernorm_style: str = "prenorm",
        apply_graphormer_init: bool = True,
        activation_fn: str = "gelu",
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        n_trans_layers_to_freeze: int = 0,
        traceable: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        init_fn: Callable = None,
        kdim: int = None,
        vdim: int = None,
        bias: bool = True,
        self_attention: bool = True,
        uses_fixed_gaussian_features: bool = False,
        return_attention: bool = False,
        batch_size=10,
        graph_size=20,
        is_training=True,
    ):
        self.parent = parent
        self.num_classes = num_classes
        self.num_atoms = num_atoms
        self.num_in_degree = num_in_degree
        self.num_out_degree = num_out_degree
        self.num_edges = num_edges
        self.num_spatial = num_spatial
        self.num_edge_dis = num_edge_dis
        self.edge_type = edge_type
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max
        self.max_nodes = max_nodes
        self.rand_node_id = rand_node_id
        self.rand_node_id_dim = rand_node_id_dim
        self.orf_node_id = orf_node_id
        self.orf_node_id_dim = orf_node_id_dim
        self.lap_node_id = lap_node_id
        self.lap_node_id_k = lap_node_id_k
        self.lap_node_id_sign_flip = lap_node_id_sign_flip
        self.lap_node_id_eig_dropout = lap_node_id_eig_dropout
        self.type_id = type_id
        self.share_encoder_input_output_embed = share_encoder_input_output_embed
        self.prenorm = prenorm
        self.postnorm = postnorm
        self.stochastic_depth = stochastic_depth
        self.performer = performer
        self.performer_finetune = performer_finetune
        self.performer_nb_features = performer_nb_features
        self.performer_feature_redraw_interval = performer_feature_redraw_interval
        self.performer_generalized_attention = performer_generalized_attention
        self.performer_auto_check_redraw = performer_auto_check_redraw
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        # self.hidden_size = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.layerdrop = layerdrop
        self.encoder_normalize_before = encoder_normalize_before
        self.layernorm_style = layernorm_style
        self.apply_graphormer_init = apply_graphormer_init
        self.activation_fn = activation_fn
        self.embed_scale = embed_scale
        self.freeze_embeddings = freeze_embeddings
        self.n_trans_layers_to_freeze = n_trans_layers_to_freeze
        self.traceable = traceable
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size
        self.init_fn = init_fn
        self.kdim = kdim
        self.vdim = vdim
        self.self_attention = self_attention
        self.uses_fixed_gaussian_features = uses_fixed_gaussian_features
        self.return_attention = return_attention
        self.bias = bias
        self.batch_size = batch_size
        self.graph_size = graph_size
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        # TODO  I don't know what to do with ids_tensor , right now hardcoded
        node_data = ids_tensor([34, 9], self.num_out_degree)
        num_nodes = ids_tensor([2], self.num_out_degree)
        edge_index = ids_tensor([2, 68], self.num_out_degree)
        edge_data = ids_tensor([68, 3], self.num_out_degree)
        edge_num = ids_tensor([2], self.num_out_degree)
        in_degree = ids_tensor([34], self.num_out_degree)
        out_degree = ids_tensor([34], self.num_out_degree)
        lap_eigvec = ids_tensor([34], self.num_out_degree)
        lap_eigval = ids_tensor([34, 17], self.num_out_degree)
        labels = ids_tensor([34, 17], self.num_classes)
        config = self.get_config()
        return (
            config,
            node_data,
            num_nodes,
            edge_index,
            edge_data,
            edge_num,
            in_degree,
            out_degree,
            lap_eigvec,
            lap_eigval,
            labels,
        )

    def get_config(self):
        return TokenGTConfig(
            tasks_weights=None,
            num_classes=self.num_classes,
            num_atoms=self.num_atoms,
            num_in_degree=self.num_in_degree,
            num_out_degree=self.num_out_degree,
            num_edges=self.num_edges,
            num_spatial=self.num_spatial,
            num_edge_dis=self.num_edge_dis,
            edge_type=self.edge_type,
            multi_hop_max_dist=self.multi_hop_max_dist,
            spatial_pos_max=self.spatial_pos_max,
            max_nodes=self.max_nodes,
            rand_node_id=self.rand_node_id,
            rand_node_id_dim=self.rand_node_id_dim,
            orf_node_id=self.orf_node_id,
            orf_node_id_dim=self.orf_node_id_dim,
            lap_node_id=self.lap_node_id,
            lap_node_id_k=self.lap_node_id_k,
            lap_node_id_sign_flip=self.lap_node_id_sign_flip,
            lap_node_id_eig_dropout=self.lap_node_id_eig_dropout,
            type_id=self.type_id,
            share_encoder_input_output_embed=self.share_encoder_input_output_embed,
            prenorm=self.prenorm,
            postnorm=self.postnorm,
            stochastic_depth=self.stochastic_depth,
            performer=self.performer,
            performer_finetune=self.performer_finetune,
            performer_nb_features=self.performer_nb_features,
            performer_feature_redraw_interval=self.performer_feature_redraw_interval,
            performer_generalized_attention=self.performer_generalized_attention,
            performer_auto_check_redraw=self.performer_auto_check_redraw,
            num_layers=self.num_layers,
            embedding_dim=self.embedding_dim,
            # hidden_size = self.embedding_dim,
            ffn_embedding_dim=self.ffn_embedding_dim,
            num_attention_heads=self.num_attention_heads,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            activation_dropout=self.activation_dropout,
            layerdrop=self.layerdrop,
            encoder_normalize_before=self.encoder_normalize_before,
            layernorm_style=self.layernorm_style,
            apply_graphormer_init=self.apply_graphormer_init,
            activation_fn=self.activation_fn,
            embed_scale=self.embed_scale,
            freeze_embeddings=self.freeze_embeddings,
            n_trans_layers_to_freeze=self.n_trans_layers_to_freeze,
            traceable=self.traceable,
            q_noise=self.q_noise,
            qn_block_size=self.qn_block_size,
            init_fn=self.init_fn,
            kdim=self.kdim,
            vdim=self.vdim,
            self_attention=self.self_attention,
            uses_fixed_gaussian_features=self.uses_fixed_gaussian_features,
            return_attention=self.return_attention,
            bias=self.bias,
        )

    def create_and_check_for_graph_classification(
        self,
        config,
        node_data,
        num_nodes,
        edge_index,
        edge_data,
        edge_num,
        in_degree,
        out_degree,
        lap_eigvec,
        lap_eigval,
        labels,
    ):
        model = TokenGTForGraphClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(
            node_data=node_data,
            num_nodes=num_nodes,
            edge_index=edge_index,
            edge_data=edge_data,
            edge_num=edge_num,
            in_degree=in_degree,
            out_degree=out_degree,
            lap_eigvec=lap_eigvec,
            lap_eigval=lap_eigval,
            labels=labels,
        )
        # self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))
        self.parent.assertEqual(result.logits.shape, (2, 1))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            node_data,
            num_nodes,
            edge_index,
            edge_data,
            edge_num,
            in_degree,
            out_degree,
            lap_eigvec,
            lap_eigval,
            labels,
        ) = config_and_inputs
        inputs_dict = {
            "node_data": node_data,
            "num_nodes": num_nodes,
            "edge_index": edge_index,
            "edge_data": edge_data,
            "edge_num": edge_num,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "lap_eigvec": lap_eigvec,
            "lap_eigval": lap_eigval,
            "labels": labels,
        }
        return config, inputs_dict


# @require_torch
# class TokenGTModelTest(ModelTesterMixin, unittest.TestCase):
#     all_model_classes = (TokenGTForGraphClassification, TokenGTForGraphClassification) if is_torch_available() else ()
#     all_generative_model_classes = ()
#     test_pruning = False
#     test_head_masking = False
#     test_resize_embeddings = False
#     main_input_name_nodes = "input_nodes"
#     main_input_name_edges = "input_edges"
#     has_attentions = False  # does not output attention

#     def setUp(self):
#         self.model_tester = TokenGTModelTester(self)
#         self.config_tester = ConfigTester(self, config_class=TokenGTConfig, has_text_modality=False)


#     # overwrite from common as `TokenGT` requires more input arguments
#     def _create_and_check_torchscript(self, config, inputs_dict):
#         if not self.test_torchscript:
#             return

#         configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
#         configs_no_init.torchscript = True
#         for model_class in self.all_model_classes:
#             model = model_class(config=configs_no_init)
#             model.to(torch_device)
#             model.eval()
#             inputs = self._prepare_for_class(inputs_dict, model_class)

#             try:
#                 required_keys = (
#                     "node_data",
#                     "num_nodes",
#                     "edge_index",
#                     "edge_data",
#                     "edge_num",
#                     "in_degree",
#                     "out_degree",
#                     "lap_eigvec",
#                     "lap_eigval",
#                 )
#                 required_inputs = tuple(inputs[k] for k in required_keys)
#                 model(*required_inputs)
#                 traced_model = torch.jit.trace(model, required_inputs)
#             except RuntimeError:
#                 self.fail("Couldn't trace module.")

#             with tempfile.TemporaryDirectory() as tmp_dir_name:
#                 pt_file_name = os.path.join(tmp_dir_name, "traced_model.pt")

#                 try:
#                     torch.jit.save(traced_model, pt_file_name)
#                 except Exception:
#                     self.fail("Couldn't save module.")

#                 try:
#                     loaded_model = torch.jit.load(pt_file_name)
#                 except Exception:
#                     self.fail("Couldn't load module.")

#             model.to(torch_device)
#             model.eval()

#             loaded_model.to(torch_device)
#             loaded_model.eval()

#             model_state_dict = model.state_dict()
#             loaded_model_state_dict = loaded_model.state_dict()

#             non_persistent_buffers = {}
#             for key in loaded_model_state_dict.keys():
#                 if key not in model_state_dict.keys():
#                     non_persistent_buffers[key] = loaded_model_state_dict[key]

#             loaded_model_state_dict = {
#                 key: value for key, value in loaded_model_state_dict.items() if key not in non_persistent_buffers
#             }

#             self.assertEqual(set(model_state_dict.keys()), set(loaded_model_state_dict.keys()))

#             model_buffers = list(model.buffers())
#             for non_persistent_buffer in non_persistent_buffers.values():
#                 found_buffer = False
#                 for i, model_buffer in enumerate(model_buffers):
#                     if torch.equal(non_persistent_buffer, model_buffer):
#                         found_buffer = True
#                         break

#                 self.assertTrue(found_buffer)
#                 model_buffers.pop(i)

#             models_equal = True
#             for layer_name, p1 in model_state_dict.items():
#                 if layer_name in loaded_model_state_dict:
#                     p2 = loaded_model_state_dict[layer_name]
#                     if p1.data.ne(p2.data).sum() > 0:
#                         models_equal = False

#             self.assertTrue(models_equal)

#             # Avoid memory leak. Without this, each call increase RAM usage by ~20MB.
#             # (Even with this call, there are still memory leak by ~0.04MB)
#             self.clear_torch_jit_class_registry()

#     def test_config(self):
#         self.config_tester.run_common_tests()


#     @unittest.skip(reason="TokenGT does not use one single inputs_embedding but three")
#     def test_inputs_embeds(self):
#         pass

#     @unittest.skip(reason="TokenGT does not implement feed forward chunking")
#     def test_feed_forward_chunking(self):
#         pass

#     @unittest.skip(reason="TokenGT does not share input and output embeddings")
#     def test_model_common_attributes(self):
#         pass

#     def test_initialization(self):
#         def _config_zero_init(config):
#             configs_no_init = copy.deepcopy(config)
#             for key in configs_no_init.__dict__.keys():
#                 if "_range" in key or "_std" in key or "initializer_factor" in key or "layer_scale" in key:
#                     setattr(configs_no_init, key, 1e-10)
#             return configs_no_init

#         config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

#         configs_no_init = _config_zero_init(config)
#         for model_class in self.all_model_classes:
#             model = model_class(config=configs_no_init)
#             for name, param in model.named_parameters():
#                 if param.requires_grad:
#                     self.assertTrue(
#                         -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
#                         msg=f"Parameter {name} of model {model_class} seems not properly initialized",
#                     )

#     def test_hidden_states_output(self):
#         def check_hidden_states_output(inputs_dict, config, model_class):
#             model = model_class(config)
#             model.to(torch_device)
#             model.eval()

#             with torch.no_grad():
#                 outputs = model(**self._prepare_for_class(inputs_dict, model_class))

#             hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

#             expected_num_layers = getattr(
#                 self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
#             )
#             self.assertEqual(len(hidden_states), expected_num_layers)

#             batch_size = self.model_tester.batch_size

#             self.assertListEqual(
#                 list(hidden_states[0].shape[-2:]),
#                 [batch_size, self.model_tester.hidden_size],
#             )

#         config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

#         for model_class in self.all_model_classes:
#             # Always returns hidden_states
#             check_hidden_states_output(inputs_dict, config, model_class)

#     def test_retain_grad_hidden_states_attentions(self):
#         config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
#         config.output_hidden_states = True
#         config.output_attentions = False

#         # no need to test all models as different heads yield the same functionality
#         model_class = self.all_model_classes[0]
#         model = model_class(config)
#         model.to(torch_device)

#         outputs = model(**inputs_dict)
#         output = outputs[0]

#         hidden_states = outputs.hidden_states[0]
#         hidden_states.retain_grad()

#         output.flatten()[0].backward(retain_graph=True)

#         self.assertIsNotNone(hidden_states.grad)

#     # Inputs are 'input_nodes' and 'input_edges' not 'input_ids'
#     def test_model_main_input_name(self):
#         for model_class in self.all_model_classes:
#             model_signature = inspect.signature(getattr(model_class, "forward"))
#             # The main input is the name of the argument after `self`
#             observed_main_input_name_nodes = list(model_signature.parameters.keys())[1]
#             observed_main_input_name_edges = list(model_signature.parameters.keys())[2]
#             self.assertEqual(model_class.main_input_name_nodes, observed_main_input_name_nodes)
#             self.assertEqual(model_class.main_input_name_edges, observed_main_input_name_edges)

#     def test_forward_signature(self):
#         config, _ = self.model_tester.prepare_config_and_inputs_for_common()

#         for model_class in self.all_model_classes:
#             model = model_class(config)
#             signature = inspect.signature(model.forward)
#             # signature.parameters is an OrderedDict => so arg_names order is deterministic
#             arg_names = [*signature.parameters.keys()]

#             expected_arg_names = ["input_nodes", "input_edges"]
#             self.assertListEqual(arg_names[:2], expected_arg_names)

#     def test_model(self):
#         config_and_inputs = self.model_tester.prepare_config_and_inputs()
#         self.model_tester.create_and_check_model(*config_and_inputs)

#     def test_for_graph_classification(self):
#         config_and_inputs = self.model_tester.prepare_config_and_inputs()
#         self.model_tester.create_and_check_for_graph_classification(*config_and_inputs)

#     @slow
#     def test_model_from_pretrained(self):
#         for model_name in TOKENGT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
#             model = TokenGTForGraphClassification.from_pretrained(model_name)
#             self.assertIsNotNone(model)


@require_torch
class TokenGTModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_graph_classification(self):
        model = TokenGTForGraphClassification.from_pretrained("raman-ai/tokengt-base-lap-pcqm4mv2")

        # Actual real graph data from the pcqm4mv2 dataset
        # fmt: off
        model_input = {
            "node_data": tensor(
                [
                    [   7,  514, 1030, 1543, 2053, 2562, 3076, 3586, 4098],
                    [   9,  514, 1028, 1543, 2050, 2562, 3075, 3586, 4098],
                    [   7,  514, 1029, 1543, 2050, 2562, 3075, 3587, 4099],
                    [   7,  514, 1029, 1543, 2051, 2562, 3075, 3587, 4099],
                    [   7,  514, 1029, 1543, 2051, 2562, 3075, 3587, 4099],
                    [   7,  514, 1029, 1543, 2051, 2562, 3075, 3587, 4099],
                    [   7,  514, 1029, 1543, 2051, 2562, 3075, 3587, 4099],
                    [   7,  514, 1029, 1543, 2050, 2562, 3075, 3587, 4099],
                    [   8,  514, 1029, 1543, 2051, 2562, 3075, 3586, 4098],
                    [   7,  516, 1030, 1543, 2051, 2562, 3076, 3586, 4098],
                    [   7,  514, 1029, 1543, 2050, 2562, 3075, 3586, 4098],
                    [   8,  514, 1028, 1543, 2050, 2562, 3075, 3586, 4098],
                    [   7,  514, 1029, 1543, 2050, 2562, 3075, 3586, 4098],
                    [   8,  514, 1028, 1543, 2051, 2562, 3075, 3586, 4098],
                    [   9,  514, 1028, 1543, 2051, 2562, 3075, 3586, 4098],
                    [   9,  514, 1028, 1543, 2051, 2562, 3075, 3586, 4098],
                    [   7,  514, 1030, 1543, 2053, 2562, 3076, 3586, 4098],
                    [   7,  514, 1030, 1543, 2053, 2562, 3076, 3586, 4098],
                    [   9,  514, 1028, 1543, 2050, 2562, 3075, 3586, 4098],
                    [   7,  514, 1029, 1543, 2050, 2562, 3075, 3587, 4099],
                    [   7,  514, 1029, 1543, 2051, 2562, 3075, 3587, 4099],
                    [   7,  514, 1029, 1543, 2051, 2562, 3075, 3587, 4099],
                    [   7,  514, 1029, 1543, 2051, 2562, 3075, 3587, 4099],
                    [   7,  514, 1029, 1543, 2051, 2562, 3075, 3587, 4099],
                    [   7,  514, 1029, 1543, 2050, 2562, 3075, 3587, 4099],
                    [   8,  514, 1029, 1543, 2051, 2562, 3075, 3586, 4098],
                    [   7,  516, 1030, 1543, 2051, 2562, 3076, 3586, 4098],
                    [   7,  514, 1029, 1543, 2050, 2562, 3075, 3586, 4098],
                    [   8,  514, 1028, 1543, 2050, 2562, 3075, 3586, 4098],
                    [   7,  514, 1029, 1543, 2050, 2562, 3075, 3586, 4098],
                    [   8,  514, 1028, 1543, 2051, 2562, 3075, 3586, 4098],
                    [   9,  514, 1028, 1543, 2051, 2562, 3075, 3586, 4098],
                    [   9,  514, 1028, 1543, 2051, 2562, 3075, 3586, 4098],
                    [   7,  514, 1030, 1543, 2053, 2562, 3076, 3586, 4098]
                ]
            ),
            "num_nodes": tensor([17, 17]),
            "edge_index": tensor(
                [
                    [
                        0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  7,  7,  8,  8,  9,
                        9, 10, 10, 11, 11, 12, 12, 13, 12, 14, 10, 15,  9, 16,  7,  2,  0,  1,
                        1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  7,  7,  8,  8,  9,  9, 10,
                        10, 11, 11, 12, 12, 13, 12, 14, 10, 15,  9, 16,  7,  2
                    ],
                    [
                        1,  0,  2,  1,  3,  2,  4,  3,  5,  4,  6,  5,  7,  6,  8,  7,  9,  8,
                        10,  9, 11, 10, 12, 11, 13, 12, 14, 12, 15, 10, 16,  9,  2,  7,  1,  0,
                        2,  1,  3,  2,  4,  3,  5,  4,  6,  5,  7,  6,  8,  7,  9,  8, 10,  9,
                        11, 10, 12, 11, 13, 12, 14, 12, 15, 10, 16,  9,  2,  7
                    ]
                ]
            ),
            # fmt: on
            "edge_data": tensor(
                    [
                        [   2,  514, 1026],
                        [   2,  514, 1026],
                        [   2,  514, 1027],
                        [   2,  514, 1027],
                        [   5,  514, 1027],
                        [   5,  514, 1027],
                        [   5,  514, 1027],
                        [   5,  514, 1027],
                        [   5,  514, 1027],
                        [   5,  514, 1027],
                        [   5,  514, 1027],
                        [   5,  514, 1027],
                        [   5,  514, 1027],
                        [   5,  514, 1027],
                        [   2,  514, 1027],
                        [   2,  514, 1027],
                        [   2,  514, 1026],
                        [   2,  514, 1026],
                        [   2,  514, 1026],
                        [   2,  514, 1026],
                        [   3,  516, 1027],
                        [   3,  516, 1027],
                        [   2,  514, 1027],
                        [   2,  514, 1027],
                        [   3,  514, 1027],
                        [   3,  514, 1027],
                        [   2,  514, 1027],
                        [   2,  514, 1027],
                        [   2,  514, 1027],
                        [   2,  514, 1027],
                        [   2,  514, 1026],
                        [   2,  514, 1026],
                        [   5,  514, 1027],
                        [   5,  514, 1027],
                        [   2,  514, 1026],
                        [   2,  514, 1026],
                        [   2,  514, 1027],
                        [   2,  514, 1027],
                        [   5,  514, 1027],
                        [   5,  514, 1027],
                        [   5,  514, 1027],
                        [   5,  514, 1027],
                        [   5,  514, 1027],
                        [   5,  514, 1027],
                        [   5,  514, 1027],
                        [   5,  514, 1027],
                        [   5,  514, 1027],
                        [   5,  514, 1027],
                        [   2,  514, 1027],
                        [   2,  514, 1027],
                        [   2,  514, 1026],
                        [   2,  514, 1026],
                        [   2,  514, 1026],
                        [   2,  514, 1026],
                        [   3,  516, 1027],
                        [   3,  516, 1027],
                        [   2,  514, 1027],
                        [   2,  514, 1027],
                        [   3,  514, 1027],
                        [   3,  514, 1027],
                        [   2,  514, 1027],
                        [   2,  514, 1027],
                        [   2,  514, 1027],
                        [   2,  514, 1027],
                        [   2,  514, 1026],
                        [   2,  514, 1026],
                        [   5,  514, 1027],
                        [   5,  514, 1027]
                    ]
            ),
            "edge_num": tensor( [34, 34]),
            "in_degree": tensor(
               [2, 3, 4, 3, 3, 3, 3, 4, 3, 4, 4, 3, 4, 2, 2, 2, 2, 2, 3, 4, 3, 3, 3, 3, 4, 3, 4, 4, 3, 4, 2, 2, 2, 2]
            ),
            "out_degree": tensor(
              [2, 3, 4, 3, 3, 3, 3, 4, 3, 4, 4, 3, 4, 2, 2, 2, 2, 2, 3, 4, 3, 3, 3, 3, 4, 3, 4, 4, 3, 4, 2, 2, 2, 2]
            ),
            "lap_eigvec": tensor(
                [
                    [ 1.7150e-01,  1.8449e-01,  1.6448e-01,  4.9311e-01, -4.4299e-02,
                    -6.9469e-02, -3.6682e-01, -2.2258e-01,  1.9602e-02,  8.6046e-03,
                    -3.6682e-01, -6.9469e-02, -4.4299e-02, -4.9311e-01, -1.6448e-01,
                    1.8449e-01, -1.7150e-01],
                    [ 2.4254e-01,  2.5192e-01,  1.9613e-01,  5.4505e-01, -3.7527e-02,
                    -4.5612e-02, -1.9667e-01,  2.9990e-16, -2.0834e-17, -1.5943e-17,
                    1.9667e-01,  4.5612e-02,  3.7527e-02,  5.4505e-01,  1.9613e-01,
                    -2.5192e-01,  2.4254e-01],
                    [ 2.9704e-01,  2.7628e-01,  1.2018e-01,  1.8942e-01,  2.1666e-02,
                    6.8451e-02,  4.5272e-01,  3.8552e-01, -3.3952e-02, -1.4904e-02,
                    4.5272e-01,  6.8451e-02,  2.1666e-02, -1.8942e-01, -1.2018e-01,
                    2.7628e-01, -2.9704e-01],
                    [ 2.4254e-01,  2.4469e-01,  1.2324e-01, -1.1740e-01, -2.3330e-01,
                    4.1086e-01,  3.5930e-01, -2.5566e-16,  1.4756e-16,  7.8031e-17,
                    -3.5930e-01, -4.1086e-01,  2.3330e-01, -1.1740e-01,  1.2324e-01,
                    -2.4469e-01,  2.4254e-01],
                    [ 2.4254e-01,  2.4695e-01,  1.0969e-01, -3.3817e-01, -2.9719e-01,
                    3.2562e-01, -9.7213e-02, -3.1478e-01,  2.7722e-02,  1.2169e-02,
                    -9.7213e-02,  3.2562e-01, -2.9719e-01,  3.3817e-01, -1.0969e-01,
                    2.4695e-01, -2.4254e-01],
                    [ 2.4254e-01,  2.3220e-01,  6.1736e-02, -4.1123e-01, -1.2274e-01,
                    -1.0851e-01, -4.3301e-01,  8.0317e-17,  9.0280e-18, -2.0209e-17,
                    4.3301e-01,  1.0851e-01,  1.2274e-01, -4.1123e-01,  6.1736e-02,
                    -2.3220e-01,  2.4254e-01],
                    [ 2.4254e-01,  2.0145e-01, -5.5806e-03, -3.0466e-01,  1.5015e-01,
                    -4.2637e-01, -2.3111e-01,  3.1478e-01, -2.7722e-02, -1.2169e-02,
                    -2.3111e-01, -4.2637e-01,  1.5015e-01,  3.0466e-01,  5.5806e-03,
                    2.0145e-01, -2.4254e-01],
                    [ 2.9704e-01,  1.9207e-01, -8.7137e-02, -7.9619e-02,  3.7063e-01,
                    -3.5199e-01,  3.1571e-01, -3.9424e-16, -1.4577e-16, -4.6898e-17,
                    -3.1571e-01,  3.5199e-01, -3.7063e-01, -7.9619e-02, -8.7137e-02,
                    -1.9207e-01,  2.9704e-01],
                    [ 2.4254e-01,  2.7243e-02, -2.7251e-01, -2.4298e-03,  3.7598e-01,
                    -2.9822e-02,  1.5464e-01, -6.2955e-01,  5.5444e-02,  2.4338e-02,
                    1.5464e-01, -2.9822e-02,  3.7598e-01,  2.4298e-03,  2.7251e-01,
                    2.7243e-02, -2.4254e-01],
                    [ 2.9704e-01, -1.2764e-01, -4.7569e-01,  7.4967e-02,  1.8103e-01,
                    3.1808e-01, -1.7211e-01,  1.9162e-16,  9.9184e-17,  1.7459e-17,
                    1.7211e-01, -3.1808e-01, -1.8103e-01,  7.4967e-02, -4.7569e-01,
                    1.2764e-01,  2.9704e-01],
                    [ 2.9704e-01, -2.7090e-01, -3.0533e-01,  8.2842e-02, -4.3737e-01,
                    -2.0555e-01,  6.8828e-02,  1.8394e-16,  4.1227e-17, -2.0661e-16,
                    6.8828e-02, -2.0555e-01, -4.3737e-01, -8.2842e-02,  3.0533e-01,
                    -2.7090e-01, -2.9704e-01],
                    [ 2.4254e-01, -3.0742e-01,  5.3463e-02,  1.0850e-02, -1.9339e-01,
                    -1.3198e-01,  5.6205e-02,  1.9422e-02,  4.7760e-01, -5.8562e-01,
                    -5.6205e-02,  1.3198e-01,  1.9339e-01,  1.0850e-02,  5.3463e-02,
                    3.0742e-01,  2.4254e-01],
                    [ 2.9704e-01, -4.5619e-01,  4.1575e-01, -6.2069e-02,  1.5362e-01,
                    5.5454e-02, -1.6635e-02,  1.2413e-16,  1.2385e-16, -2.6299e-16,
                    -1.6635e-02,  5.5454e-02,  1.5362e-01,  6.2069e-02, -4.1575e-01,
                    -4.5619e-01, -2.9704e-01],
                    [ 1.7150e-01, -2.7278e-01,  2.8468e-01, -4.5850e-02,  1.4807e-01,
                    6.8960e-02, -2.5333e-02,  5.8347e-02,  3.7573e-01,  6.5334e-01,
                    2.5333e-02, -6.8960e-02, -1.4807e-01, -4.5850e-02,  2.8468e-01,
                    2.7278e-01,  1.7150e-01],
                    [ 1.7150e-01, -2.7278e-01,  2.8468e-01, -4.5850e-02,  1.4807e-01,
                    6.8960e-02, -2.5333e-02, -7.2081e-02, -7.1344e-01, -2.3924e-01,
                    2.5333e-02, -6.8960e-02, -1.4807e-01, -4.5850e-02,  2.8468e-01,
                    2.7278e-01,  1.7150e-01],
                    [ 1.7150e-01, -1.6198e-01, -2.0907e-01,  6.1194e-02, -4.2156e-01,
                    -2.5561e-01,  1.0482e-01, -1.3734e-02, -3.3771e-01,  4.1410e-01,
                    -1.0482e-01,  2.5561e-01,  4.2156e-01,  6.1194e-02, -2.0907e-01,
                    1.6198e-01,  1.7150e-01],
                    [ 1.7150e-01, -7.6320e-02, -3.2573e-01,  5.5377e-02,  1.7449e-01,
                    3.9555e-01, -2.6210e-01,  4.4516e-01, -3.9205e-02, -1.7209e-02,
                    -2.6210e-01,  3.9555e-01,  1.7449e-01, -5.5377e-02,  3.2573e-01,
                    -7.6320e-02, -1.7150e-01],
                    [ 1.7150e-01,  1.8449e-01,  1.6448e-01,  4.9311e-01, -4.4299e-02,
                    -6.9469e-02, -3.6682e-01, -2.2258e-01,  1.9602e-02,  8.6046e-03,
                    -3.6682e-01, -6.9469e-02, -4.4299e-02, -4.9311e-01, -1.6448e-01,
                    1.8449e-01, -1.7150e-01],
                    [ 2.4254e-01,  2.5192e-01,  1.9613e-01,  5.4505e-01, -3.7527e-02,
                    -4.5612e-02, -1.9667e-01,  2.9990e-16, -2.0834e-17, -1.5943e-17,
                    1.9667e-01,  4.5612e-02,  3.7527e-02,  5.4505e-01,  1.9613e-01,
                    -2.5192e-01,  2.4254e-01],
                    [ 2.9704e-01,  2.7628e-01,  1.2018e-01,  1.8942e-01,  2.1666e-02,
                    6.8451e-02,  4.5272e-01,  3.8552e-01, -3.3952e-02, -1.4904e-02,
                    4.5272e-01,  6.8451e-02,  2.1666e-02, -1.8942e-01, -1.2018e-01,
                    2.7628e-01, -2.9704e-01],
                    [ 2.4254e-01,  2.4469e-01,  1.2324e-01, -1.1740e-01, -2.3330e-01,
                    4.1086e-01,  3.5930e-01, -2.5566e-16,  1.4756e-16,  7.8031e-17,
                    -3.5930e-01, -4.1086e-01,  2.3330e-01, -1.1740e-01,  1.2324e-01,
                    -2.4469e-01,  2.4254e-01],
                    [ 2.4254e-01,  2.4695e-01,  1.0969e-01, -3.3817e-01, -2.9719e-01,
                    3.2562e-01, -9.7213e-02, -3.1478e-01,  2.7722e-02,  1.2169e-02,
                    -9.7213e-02,  3.2562e-01, -2.9719e-01,  3.3817e-01, -1.0969e-01,
                    2.4695e-01, -2.4254e-01],
                    [ 2.4254e-01,  2.3220e-01,  6.1736e-02, -4.1123e-01, -1.2274e-01,
                    -1.0851e-01, -4.3301e-01,  8.0317e-17,  9.0280e-18, -2.0209e-17,
                    4.3301e-01,  1.0851e-01,  1.2274e-01, -4.1123e-01,  6.1736e-02,
                    -2.3220e-01,  2.4254e-01],
                    [ 2.4254e-01,  2.0145e-01, -5.5806e-03, -3.0466e-01,  1.5015e-01,
                    -4.2637e-01, -2.3111e-01,  3.1478e-01, -2.7722e-02, -1.2169e-02,
                    -2.3111e-01, -4.2637e-01,  1.5015e-01,  3.0466e-01,  5.5806e-03,
                    2.0145e-01, -2.4254e-01],
                    [ 2.9704e-01,  1.9207e-01, -8.7137e-02, -7.9619e-02,  3.7063e-01,
                    -3.5199e-01,  3.1571e-01, -3.9424e-16, -1.4577e-16, -4.6898e-17,
                    -3.1571e-01,  3.5199e-01, -3.7063e-01, -7.9619e-02, -8.7137e-02,
                    -1.9207e-01,  2.9704e-01],
                    [ 2.4254e-01,  2.7243e-02, -2.7251e-01, -2.4298e-03,  3.7598e-01,
                    -2.9822e-02,  1.5464e-01, -6.2955e-01,  5.5444e-02,  2.4338e-02,
                    1.5464e-01, -2.9822e-02,  3.7598e-01,  2.4298e-03,  2.7251e-01,
                    2.7243e-02, -2.4254e-01],
                    [ 2.9704e-01, -1.2764e-01, -4.7569e-01,  7.4967e-02,  1.8103e-01,
                    3.1808e-01, -1.7211e-01,  1.9162e-16,  9.9184e-17,  1.7459e-17,
                    1.7211e-01, -3.1808e-01, -1.8103e-01,  7.4967e-02, -4.7569e-01,
                    1.2764e-01,  2.9704e-01],
                    [ 2.9704e-01, -2.7090e-01, -3.0533e-01,  8.2842e-02, -4.3737e-01,
                    -2.0555e-01,  6.8828e-02,  1.8394e-16,  4.1227e-17, -2.0661e-16,
                    6.8828e-02, -2.0555e-01, -4.3737e-01, -8.2842e-02,  3.0533e-01,
                    -2.7090e-01, -2.9704e-01],
                    [ 2.4254e-01, -3.0742e-01,  5.3463e-02,  1.0850e-02, -1.9339e-01,
                    -1.3198e-01,  5.6205e-02,  1.9422e-02,  4.7760e-01, -5.8562e-01,
                    -5.6205e-02,  1.3198e-01,  1.9339e-01,  1.0850e-02,  5.3463e-02,
                    3.0742e-01,  2.4254e-01],
                    [ 2.9704e-01, -4.5619e-01,  4.1575e-01, -6.2069e-02,  1.5362e-01,
                    5.5454e-02, -1.6635e-02,  1.2413e-16,  1.2385e-16, -2.6299e-16,
                    -1.6635e-02,  5.5454e-02,  1.5362e-01,  6.2069e-02, -4.1575e-01,
                    -4.5619e-01, -2.9704e-01],
                    [ 1.7150e-01, -2.7278e-01,  2.8468e-01, -4.5850e-02,  1.4807e-01,
                    6.8960e-02, -2.5333e-02,  5.8347e-02,  3.7573e-01,  6.5334e-01,
                    2.5333e-02, -6.8960e-02, -1.4807e-01, -4.5850e-02,  2.8468e-01,
                    2.7278e-01,  1.7150e-01],
                    [ 1.7150e-01, -2.7278e-01,  2.8468e-01, -4.5850e-02,  1.4807e-01,
                    6.8960e-02, -2.5333e-02, -7.2081e-02, -7.1344e-01, -2.3924e-01,
                    2.5333e-02, -6.8960e-02, -1.4807e-01, -4.5850e-02,  2.8468e-01,
                    2.7278e-01,  1.7150e-01],
                    [ 1.7150e-01, -1.6198e-01, -2.0907e-01,  6.1194e-02, -4.2156e-01,
                    -2.5561e-01,  1.0482e-01, -1.3734e-02, -3.3771e-01,  4.1410e-01,
                    -1.0482e-01,  2.5561e-01,  4.2156e-01,  6.1194e-02, -2.0907e-01,
                    1.6198e-01,  1.7150e-01],
                    [ 1.7150e-01, -7.6320e-02, -3.2573e-01,  5.5377e-02,  1.7449e-01,
                    3.9555e-01, -2.6210e-01,  4.4516e-01, -3.9205e-02, -1.7209e-02,
                    -2.6210e-01,  3.9555e-01,  1.7449e-01, -5.5377e-02,  3.2573e-01,
                    -7.6320e-02, -1.7150e-01]
                ]
            ),
            "lap_eigval": tensor(
                [
                    [3.7835e-08, 3.4440e-02, 1.5683e-01, 2.1841e-01, 4.0099e-01, 5.3572e-01,
                    6.2089e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.3791e+00, 1.4643e+00,
                    1.5990e+00, 1.7816e+00, 1.8432e+00, 1.9656e+00, 2.0000e+00],
                    [3.7835e-08, 3.4440e-02, 1.5683e-01, 2.1841e-01, 4.0099e-01, 5.3572e-01,
                    6.2089e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.3791e+00, 1.4643e+00,
                    1.5990e+00, 1.7816e+00, 1.8432e+00, 1.9656e+00, 2.0000e+00],
                    [3.7835e-08, 3.4440e-02, 1.5683e-01, 2.1841e-01, 4.0099e-01, 5.3572e-01,
                    6.2089e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.3791e+00, 1.4643e+00,
                    1.5990e+00, 1.7816e+00, 1.8432e+00, 1.9656e+00, 2.0000e+00],
                    [3.7835e-08, 3.4440e-02, 1.5683e-01, 2.1841e-01, 4.0099e-01, 5.3572e-01,
                    6.2089e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.3791e+00, 1.4643e+00,
                    1.5990e+00, 1.7816e+00, 1.8432e+00, 1.9656e+00, 2.0000e+00],
                    [3.7835e-08, 3.4440e-02, 1.5683e-01, 2.1841e-01, 4.0099e-01, 5.3572e-01,
                    6.2089e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.3791e+00, 1.4643e+00,
                    1.5990e+00, 1.7816e+00, 1.8432e+00, 1.9656e+00, 2.0000e+00],
                    [3.7835e-08, 3.4440e-02, 1.5683e-01, 2.1841e-01, 4.0099e-01, 5.3572e-01,
                    6.2089e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.3791e+00, 1.4643e+00,
                    1.5990e+00, 1.7816e+00, 1.8432e+00, 1.9656e+00, 2.0000e+00],
                    [3.7835e-08, 3.4440e-02, 1.5683e-01, 2.1841e-01, 4.0099e-01, 5.3572e-01,
                    6.2089e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.3791e+00, 1.4643e+00,
                    1.5990e+00, 1.7816e+00, 1.8432e+00, 1.9656e+00, 2.0000e+00],
                    [3.7835e-08, 3.4440e-02, 1.5683e-01, 2.1841e-01, 4.0099e-01, 5.3572e-01,
                    6.2089e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.3791e+00, 1.4643e+00,
                    1.5990e+00, 1.7816e+00, 1.8432e+00, 1.9656e+00, 2.0000e+00],
                    [3.7835e-08, 3.4440e-02, 1.5683e-01, 2.1841e-01, 4.0099e-01, 5.3572e-01,
                    6.2089e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.3791e+00, 1.4643e+00,
                    1.5990e+00, 1.7816e+00, 1.8432e+00, 1.9656e+00, 2.0000e+00],
                    [3.7835e-08, 3.4440e-02, 1.5683e-01, 2.1841e-01, 4.0099e-01, 5.3572e-01,
                    6.2089e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.3791e+00, 1.4643e+00,
                    1.5990e+00, 1.7816e+00, 1.8432e+00, 1.9656e+00, 2.0000e+00],
                    [3.7835e-08, 3.4440e-02, 1.5683e-01, 2.1841e-01, 4.0099e-01, 5.3572e-01,
                    6.2089e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.3791e+00, 1.4643e+00,
                    1.5990e+00, 1.7816e+00, 1.8432e+00, 1.9656e+00, 2.0000e+00],
                    [3.7835e-08, 3.4440e-02, 1.5683e-01, 2.1841e-01, 4.0099e-01, 5.3572e-01,
                    6.2089e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.3791e+00, 1.4643e+00,
                    1.5990e+00, 1.7816e+00, 1.8432e+00, 1.9656e+00, 2.0000e+00],
                    [3.7835e-08, 3.4440e-02, 1.5683e-01, 2.1841e-01, 4.0099e-01, 5.3572e-01,
                    6.2089e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.3791e+00, 1.4643e+00,
                    1.5990e+00, 1.7816e+00, 1.8432e+00, 1.9656e+00, 2.0000e+00],
                    [3.7835e-08, 3.4440e-02, 1.5683e-01, 2.1841e-01, 4.0099e-01, 5.3572e-01,
                    6.2089e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.3791e+00, 1.4643e+00,
                    1.5990e+00, 1.7816e+00, 1.8432e+00, 1.9656e+00, 2.0000e+00],
                    [3.7835e-08, 3.4440e-02, 1.5683e-01, 2.1841e-01, 4.0099e-01, 5.3572e-01,
                    6.2089e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.3791e+00, 1.4643e+00,
                    1.5990e+00, 1.7816e+00, 1.8432e+00, 1.9656e+00, 2.0000e+00],
                    [3.7835e-08, 3.4440e-02, 1.5683e-01, 2.1841e-01, 4.0099e-01, 5.3572e-01,
                    6.2089e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.3791e+00, 1.4643e+00,
                    1.5990e+00, 1.7816e+00, 1.8432e+00, 1.9656e+00, 2.0000e+00],
                    [3.7835e-08, 3.4440e-02, 1.5683e-01, 2.1841e-01, 4.0099e-01, 5.3572e-01,
                    6.2089e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.3791e+00, 1.4643e+00,
                    1.5990e+00, 1.7816e+00, 1.8432e+00, 1.9656e+00, 2.0000e+00],
                    [3.7835e-08, 3.4440e-02, 1.5683e-01, 2.1841e-01, 4.0099e-01, 5.3572e-01,
                    6.2089e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.3791e+00, 1.4643e+00,
                    1.5990e+00, 1.7816e+00, 1.8432e+00, 1.9656e+00, 2.0000e+00],
                    [3.7835e-08, 3.4440e-02, 1.5683e-01, 2.1841e-01, 4.0099e-01, 5.3572e-01,
                    6.2089e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.3791e+00, 1.4643e+00,
                    1.5990e+00, 1.7816e+00, 1.8432e+00, 1.9656e+00, 2.0000e+00],
                    [3.7835e-08, 3.4440e-02, 1.5683e-01, 2.1841e-01, 4.0099e-01, 5.3572e-01,
                    6.2089e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.3791e+00, 1.4643e+00,
                    1.5990e+00, 1.7816e+00, 1.8432e+00, 1.9656e+00, 2.0000e+00],
                    [3.7835e-08, 3.4440e-02, 1.5683e-01, 2.1841e-01, 4.0099e-01, 5.3572e-01,
                    6.2089e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.3791e+00, 1.4643e+00,
                    1.5990e+00, 1.7816e+00, 1.8432e+00, 1.9656e+00, 2.0000e+00],
                    [3.7835e-08, 3.4440e-02, 1.5683e-01, 2.1841e-01, 4.0099e-01, 5.3572e-01,
                    6.2089e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.3791e+00, 1.4643e+00,
                    1.5990e+00, 1.7816e+00, 1.8432e+00, 1.9656e+00, 2.0000e+00],
                    [3.7835e-08, 3.4440e-02, 1.5683e-01, 2.1841e-01, 4.0099e-01, 5.3572e-01,
                    6.2089e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.3791e+00, 1.4643e+00,
                    1.5990e+00, 1.7816e+00, 1.8432e+00, 1.9656e+00, 2.0000e+00],
                    [3.7835e-08, 3.4440e-02, 1.5683e-01, 2.1841e-01, 4.0099e-01, 5.3572e-01,
                    6.2089e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.3791e+00, 1.4643e+00,
                    1.5990e+00, 1.7816e+00, 1.8432e+00, 1.9656e+00, 2.0000e+00],
                    [3.7835e-08, 3.4440e-02, 1.5683e-01, 2.1841e-01, 4.0099e-01, 5.3572e-01,
                    6.2089e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.3791e+00, 1.4643e+00,
                    1.5990e+00, 1.7816e+00, 1.8432e+00, 1.9656e+00, 2.0000e+00],
                    [3.7835e-08, 3.4440e-02, 1.5683e-01, 2.1841e-01, 4.0099e-01, 5.3572e-01,
                    6.2089e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.3791e+00, 1.4643e+00,
                    1.5990e+00, 1.7816e+00, 1.8432e+00, 1.9656e+00, 2.0000e+00],
                    [3.7835e-08, 3.4440e-02, 1.5683e-01, 2.1841e-01, 4.0099e-01, 5.3572e-01,
                    6.2089e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.3791e+00, 1.4643e+00,
                    1.5990e+00, 1.7816e+00, 1.8432e+00, 1.9656e+00, 2.0000e+00],
                    [3.7835e-08, 3.4440e-02, 1.5683e-01, 2.1841e-01, 4.0099e-01, 5.3572e-01,
                    6.2089e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.3791e+00, 1.4643e+00,
                    1.5990e+00, 1.7816e+00, 1.8432e+00, 1.9656e+00, 2.0000e+00],
                    [3.7835e-08, 3.4440e-02, 1.5683e-01, 2.1841e-01, 4.0099e-01, 5.3572e-01,
                    6.2089e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.3791e+00, 1.4643e+00,
                    1.5990e+00, 1.7816e+00, 1.8432e+00, 1.9656e+00, 2.0000e+00],
                    [3.7835e-08, 3.4440e-02, 1.5683e-01, 2.1841e-01, 4.0099e-01, 5.3572e-01,
                    6.2089e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.3791e+00, 1.4643e+00,
                    1.5990e+00, 1.7816e+00, 1.8432e+00, 1.9656e+00, 2.0000e+00],
                    [3.7835e-08, 3.4440e-02, 1.5683e-01, 2.1841e-01, 4.0099e-01, 5.3572e-01,
                    6.2089e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.3791e+00, 1.4643e+00,
                    1.5990e+00, 1.7816e+00, 1.8432e+00, 1.9656e+00, 2.0000e+00],
                    [3.7835e-08, 3.4440e-02, 1.5683e-01, 2.1841e-01, 4.0099e-01, 5.3572e-01,
                    6.2089e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.3791e+00, 1.4643e+00,
                    1.5990e+00, 1.7816e+00, 1.8432e+00, 1.9656e+00, 2.0000e+00],
                    [3.7835e-08, 3.4440e-02, 1.5683e-01, 2.1841e-01, 4.0099e-01, 5.3572e-01,
                    6.2089e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.3791e+00, 1.4643e+00,
                    1.5990e+00, 1.7816e+00, 1.8432e+00, 1.9656e+00, 2.0000e+00],
                    [3.7835e-08, 3.4440e-02, 1.5683e-01, 2.1841e-01, 4.0099e-01, 5.3572e-01,
                    6.2089e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.3791e+00, 1.4643e+00,
                    1.5990e+00, 1.7816e+00, 1.8432e+00, 1.9656e+00, 2.0000e+00]
                ]
            ),
            "labels": tensor([4.5878, 4.9715]),
        }

        output = model(**model_input)["logits"]

        expected_shape = torch.Size((2, 1))
        self.assertEqual(output.shape, expected_shape)

        expected_logs = torch.tensor(
            [[4.6461], [4.6461]]
        )

        self.assertTrue(torch.allclose(output, expected_logs, atol=1e-4))