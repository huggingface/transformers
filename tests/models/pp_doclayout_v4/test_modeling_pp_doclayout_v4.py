# coding = utf-8
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
"""Testing suite for the PP-DocLayoutV4 model."""

import inspect
import math
import unittest

from parameterized import parameterized

from transformers import (
    PPDocLayoutV4Config,
    PPDocLayoutV4ForObjectDetection,
    PPDocLayoutV4ImageProcessor,
    is_torch_available,
    is_vision_available,
)
from transformers.image_utils import load_image
from transformers.testing_utils import (
    require_torch,
    require_torch_accelerator,
    require_vision,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor
from ...test_pipeline_mixin import PipelineTesterMixin
from ...test_processing_common import url_to_local_path


if is_torch_available():
    import torch

    from transformers.models.pp_doclayout_v4.modeling_pp_doclayout_v4 import (
        PPDocLayoutV4GlobalPointer,
        PPDocLayoutV4S2RFusion,
    )


class PPDocLayoutV4ModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        is_training=False,
        num_labels=25,
        initializer_range=0.01,
        layer_norm_eps=1e-5,
        batch_norm_eps=1e-5,
        # encoder HybridEncoder
        encoder_hidden_dim=32,
        encoder_in_channels=[32, 32, 32],
        feat_strides=[8, 16, 32],
        encoder_layers=1,
        encoder_ffn_dim=64,
        encoder_attention_heads=2,
        dropout=0.0,
        activation_dropout=0.0,
        encode_proj_layers=[2],
        positional_encoding_temperature=10000,
        encoder_activation_function="gelu",
        activation_function="silu",
        eval_size=None,
        normalize_before=False,
        # decoder
        d_model=32,
        num_queries=30,
        decoder_in_channels=[32, 32, 32],
        decoder_ffn_dim=8,
        num_feature_levels=3,
        decoder_n_points=4,
        decoder_layers=2,
        decoder_attention_heads=2,
        decoder_activation_function="relu",
        attention_dropout=0.0,
        num_denoising=0,
        learn_initial_query=False,
        anchor_image_size=None,
        image_size=128,
        disable_custom_kernels=True,
        # quad boxes and reading order
        num_coords=10,
        global_pointer_head_size=8,
        gp_dropout_value=0.0,
        use_s2r=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = 3
        self.is_training = is_training
        self.num_labels = num_labels
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.batch_norm_eps = batch_norm_eps
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_in_channels = encoder_in_channels
        self.feat_strides = feat_strides
        self.encoder_layers = encoder_layers
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.encode_proj_layers = encode_proj_layers
        self.positional_encoding_temperature = positional_encoding_temperature
        self.encoder_activation_function = encoder_activation_function
        self.activation_function = activation_function
        self.eval_size = eval_size
        self.normalize_before = normalize_before
        self.d_model = d_model
        self.num_queries = num_queries
        self.decoder_in_channels = decoder_in_channels
        self.decoder_ffn_dim = decoder_ffn_dim
        self.num_feature_levels = num_feature_levels
        self.decoder_n_points = decoder_n_points
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_activation_function = decoder_activation_function
        self.attention_dropout = attention_dropout
        self.num_denoising = num_denoising
        self.learn_initial_query = learn_initial_query
        self.anchor_image_size = anchor_image_size
        self.image_size = image_size
        self.disable_custom_kernels = disable_custom_kernels
        self.num_coords = num_coords
        self.global_pointer_head_size = global_pointer_head_size
        self.gp_dropout_value = gp_dropout_value
        self.use_s2r = use_s2r

        self.encoder_seq_length = math.ceil(self.image_size / 32) * math.ceil(self.image_size / 32)

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        return self.get_config(), pixel_values

    def get_config(self):
        hidden_sizes = [10, 20, 30, 40]
        backbone_config = {
            "model_type": "hgnet_v2",
            "arch": "L",
            "return_idx": [1, 2, 3],
            "hidden_sizes": [32, 32, 32, 32],
            "stem_channels": [3, 32, 32],
            "stage_in_channels": [32, 32, 32, 32],
            "stage_mid_channels": [32, 32, 32, 32],
            "stage_out_channels": [32, 32, 32, 32],
            "freeze_stem_only": True,
            "freeze_at": 0,
            "freeze_norm": True,
            "lr_mult_list": [0, 0.05, 0.05, 0.05, 0.05],
            "out_features": ["stage2", "stage3", "stage4"],
        }
        return PPDocLayoutV4Config(
            backbone_config=backbone_config,
            num_labels=self.num_labels,
            encoder_hidden_dim=self.encoder_hidden_dim,
            encoder_in_channels=hidden_sizes[1:],
            feat_strides=self.feat_strides,
            encoder_layers=self.encoder_layers,
            encoder_ffn_dim=self.encoder_ffn_dim,
            encoder_attention_heads=self.encoder_attention_heads,
            dropout=self.dropout,
            activation_dropout=self.activation_dropout,
            encode_proj_layers=self.encode_proj_layers,
            positional_encoding_temperature=self.positional_encoding_temperature,
            encoder_activation_function=self.encoder_activation_function,
            activation_function=self.activation_function,
            eval_size=self.eval_size,
            normalize_before=self.normalize_before,
            d_model=self.d_model,
            num_queries=self.num_queries,
            decoder_in_channels=self.decoder_in_channels,
            decoder_ffn_dim=self.decoder_ffn_dim,
            num_feature_levels=self.num_feature_levels,
            decoder_n_points=self.decoder_n_points,
            decoder_layers=self.decoder_layers,
            decoder_attention_heads=self.decoder_attention_heads,
            decoder_activation_function=self.decoder_activation_function,
            attention_dropout=self.attention_dropout,
            num_denoising=self.num_denoising,
            learn_initial_query=self.learn_initial_query,
            anchor_image_size=self.anchor_image_size,
            image_size=self.image_size,
            disable_custom_kernels=self.disable_custom_kernels,
            num_coords=self.num_coords,
            global_pointer_head_size=self.global_pointer_head_size,
            gp_dropout_value=self.gp_dropout_value,
            use_s2r=self.use_s2r,
        )

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        return config, {"pixel_values": pixel_values}


@require_torch
class PPDocLayoutV4ModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (PPDocLayoutV4ForObjectDetection,) if is_torch_available() else ()
    pipeline_model_mapping = {"object-detection": PPDocLayoutV4ForObjectDetection} if is_torch_available() else {}
    is_encoder_decoder = True

    test_missing_keys = False
    test_torch_exportable = True

    def setUp(self):
        self.model_tester = PPDocLayoutV4ModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=PPDocLayoutV4Config,
            has_text_modality=False,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="PPDocLayoutV4 does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="PPDocLayoutV4 does not use test_inputs_embeds_matches_input_ids")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(reason="PPDocLayoutV4 does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="PPDocLayoutV4 does not support input and output embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="PPDocLayoutV4 does not use token embeddings")
    def test_resize_tokens_embeddings(self):
        pass

    @unittest.skip(reason="Feed forward chunking is not implemented")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="PPDocLayoutV4 does not support training")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            arg_names = [*signature.parameters.keys()]
            self.assertListEqual(arg_names[:1], ["pixel_values"])

    def test_object_detection_head_shapes(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = PPDocLayoutV4ForObjectDetection(config).to(torch_device).eval()

        with torch.no_grad():
            outputs = model(**self._prepare_for_class(inputs_dict, PPDocLayoutV4ForObjectDetection))

        batch_size = self.model_tester.batch_size
        num_queries = self.model_tester.num_queries
        self.assertEqual(outputs.logits.shape, torch.Size((batch_size, num_queries, config.num_labels)))
        # PP-DocLayoutV4 regresses a four point quad instead of a `(cx, cy, w, h)` box.
        self.assertEqual(outputs.pred_boxes.shape, torch.Size((batch_size, num_queries, config.num_coords)))
        order_shape = torch.Size((batch_size, num_queries, num_queries))
        self.assertEqual(outputs.relative_order_logits.shape, order_shape)
        self.assertEqual(outputs.successor_order_logits.shape, order_shape)

    def test_anchor_image_size_and_eval_size_accept_sizes(self):
        """
        Both are documented as `tuple[int, int]` and `__post_init__` normalizes them to lists, so a scalar-only
        annotation would make every documented value unconstructible under `@strict` and leave the cached-anchor
        and pos-embed-disable branches unreachable.
        """
        for size in ((800, 800), [800, 800]):
            config = PPDocLayoutV4Config(anchor_image_size=size, eval_size=size)
            self.assertEqual(config.anchor_image_size, [800, 800])
            self.assertEqual(config.eval_size, [800, 800])

        # The cached-anchor branch only runs when `anchor_image_size` is set, so exercise a real forward pass.
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.anchor_image_size = list(inputs_dict["pixel_values"].shape[-2:])
        model = PPDocLayoutV4ForObjectDetection(config).to(torch_device).eval()
        with torch.no_grad():
            outputs = model(**self._prepare_for_class(inputs_dict, PPDocLayoutV4ForObjectDetection))
        self.assertEqual(
            outputs.pred_boxes.shape,
            torch.Size((self.model_tester.batch_size, self.model_tester.num_queries, config.num_coords)),
        )

    def test_global_pointer_symmetries(self):
        """
        The two order heads differ only in how they symmetrize.

        These are properties of `PPDocLayoutV4GlobalPointer` itself, so they are tested on the module with random
        inputs: a randomly initialized tiny model produces decoder hidden states small enough that its order logits
        vanish to ~1e-11, where every symmetry assertion holds trivially.
        """
        config = self.model_tester.get_config()
        torch.manual_seed(0)
        hidden_states = torch.randn(2, 6, config.d_model, device=torch_device)

        relative_head = PPDocLayoutV4GlobalPointer(config, antisymmetric=True).to(torch_device).eval()
        successor_head = PPDocLayoutV4GlobalPointer(config, antisymmetric=False).to(torch_device).eval()
        with torch.no_grad():
            relative = relative_head(hidden_states)
            successor = successor_head(hidden_states)

        self.assertGreater(relative.abs().max().item(), 1e-3)
        torch.testing.assert_close(relative, -relative.transpose(-2, -1), rtol=1e-4, atol=1e-4)

        # The successor head only masks self loops and is deliberately *not* antisymmetric: "j follows i" and
        # "i follows j" are scored independently.
        self.assertTrue(bool((successor.diagonal(dim1=-2, dim2=-1) < -1e3).all()))
        off_diagonal = ~torch.eye(successor.shape[-1], dtype=torch.bool, device=successor.device)
        symmetric_part = (successor + successor.transpose(-2, -1))[:, off_diagonal]
        self.assertGreater(symmetric_part.abs().max().item(), 1e-3)

    def test_s2r_fusion_mixes_successor_into_relative_order(self):
        config = self.model_tester.get_config()
        fusion = PPDocLayoutV4S2RFusion(config).to(torch_device).eval()

        torch.manual_seed(0)
        relative = torch.randn(2, 6, 6, device=torch_device)
        relative = relative - relative.transpose(-2, -1)
        successor = torch.randn(2, 6, 6, device=torch_device)

        gate = 0.7
        with torch.no_grad():
            fusion.a.zero_()
            unfused = fusion(relative, successor)
            fusion.a.fill_(gate)
            fused = fusion(relative, successor)

        # `a` gates the closure term, so `a = 0` leaves the relative order logits alone (`b` defaults to 1.0) ...
        torch.testing.assert_close(unfused, relative, rtol=1e-4, atol=1e-4)
        # ... and a non-zero gate has to actually move them, otherwise the successor head is not contributing.
        self.assertGreater((fused - relative).abs().max().item(), 1e-3)
        # Antisymmetry survives *any* gate value, because the closure term is antisymmetrized before it is added.
        torch.testing.assert_close(fused, -fused.transpose(-2, -1), rtol=1e-4, atol=1e-4)

        num_queries = successor.shape[-1]
        eye = torch.eye(num_queries, device=successor.device, dtype=successor.dtype)
        adjacency = successor.sigmoid() * (1.0 - eye)
        adjacency = adjacency / adjacency.sum(-1, keepdim=True).clamp(min=1.0)
        closure, power = adjacency, adjacency
        for _ in range(config.s2r_steps - 1):
            power = config.s2r_damping * torch.bmm(adjacency, power)
            closure = closure + power
        expected = relative + gate * (closure - closure.transpose(-2, -1))
        torch.testing.assert_close(fused, expected, rtol=1e-4, atol=1e-4)

    def test_s2r_b_is_read_from_the_config(self):
        """`b` only lives in the checkpoint when `s2r_learnable_b=True`, otherwise it is a plain config float."""
        config = self.model_tester.get_config()
        fusion = PPDocLayoutV4S2RFusion(config).to(torch_device).eval()
        self.assertNotIn("b", fusion.state_dict())
        self.assertIsInstance(fusion.b, float)
        self.assertEqual(fusion.b, config.s2r_b_init)

        torch.manual_seed(0)
        relative = torch.randn(2, 6, 6, device=torch_device)
        relative = relative - relative.transpose(-2, -1)
        successor = torch.randn(2, 6, 6, device=torch_device)
        with torch.no_grad():
            baseline = fusion(relative, successor)
            fusion.b = 2.0
            scaled = fusion(relative, successor)
        # Only the relative term is scaled by `b`, the gated closure term is untouched.
        closure_term = baseline - config.s2r_b_init * relative
        torch.testing.assert_close(scaled, closure_term + 2.0 * relative, rtol=1e-4, atol=1e-4)

        config.s2r_learnable_b = True
        learnable = PPDocLayoutV4S2RFusion(config)
        self.assertIn("b", learnable.state_dict())
        self.assertEqual(learnable.b.item(), config.s2r_b_init)

    def test_order_heads_are_wired_into_the_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        inputs = self._prepare_for_class(inputs_dict, PPDocLayoutV4ForObjectDetection)
        order_shape = torch.Size((self.model_tester.batch_size,) + (self.model_tester.num_queries,) * 2)

        for use_s2r in (True, False):
            config.use_s2r = use_s2r
            model = PPDocLayoutV4ForObjectDetection(config).to(torch_device).eval()
            self.assertEqual(model.model.s2r_fusion is not None, use_s2r)

            with torch.no_grad():
                outputs = model(**inputs)

            self.assertEqual(outputs.relative_order_logits.shape, order_shape)
            self.assertEqual(outputs.successor_order_logits.shape, order_shape)
            # Only self loops are masked out, and only on the successor head.
            self.assertTrue(bool((outputs.successor_order_logits.diagonal(dim1=-2, dim2=-1) < -1e3).all()))
            self.assertTrue(bool((outputs.relative_order_logits.diagonal(dim1=-2, dim2=-1).abs() < 1e-3).all()))

    def test_training_raises(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = PPDocLayoutV4ForObjectDetection(config).to(torch_device)
        labels = [{"class_labels": torch.zeros(1, dtype=torch.long), "boxes": torch.zeros(1, 4)}]
        with self.assertRaises(ValueError):
            model(pixel_values=inputs_dict["pixel_values"][:1].to(torch_device), labels=labels)

    @parameterized.expand(["float32", "float16", "bfloat16"])
    @require_torch_accelerator
    @slow
    def test_inference_with_different_dtypes(self, dtype_str):
        dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_str]

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device).to(dtype)
            model.eval()
            for key, tensor in inputs_dict.items():
                inputs_dict[key] = tensor.to(dtype)
            with torch.no_grad():
                _ = model(**self._prepare_for_class(inputs_dict, model_class))

    # PP-DocLayoutV4 has no `num_hidden_layers`, the encoder depth follows `encoder_in_channels`.
    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states
            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", len(self.model_tester.encoder_in_channels) - 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)
            self.assertListEqual(
                list(hidden_states[1].shape[-2:]),
                [
                    self.model_tester.image_size // self.model_tester.feat_strides[-1],
                    self.model_tester.image_size // self.model_tester.feat_strides[-1],
                ],
            )

            hidden_states = outputs.decoder_hidden_states
            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.decoder_layers + 1
            )
            self.assertIsInstance(hidden_states, (list, tuple))
            self.assertEqual(len(hidden_states), expected_num_layers)
            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [self.model_tester.num_queries, self.model_tester.d_model],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True
            check_hidden_states_output(inputs_dict, config, model_class)

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            model = model_class._from_config(config, attn_implementation="eager")
            config = model.config
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            attentions = outputs.encoder_attentions
            self.assertEqual(len(attentions), self.model_tester.encoder_layers)
            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [
                    self.model_tester.encoder_attention_heads,
                    self.model_tester.encoder_seq_length,
                    self.model_tester.encoder_seq_length,
                ],
            )

            self.assertEqual(len(outputs.decoder_attentions), self.model_tester.decoder_layers)
            self.assertListEqual(
                list(outputs.decoder_attentions[0].shape[-3:]),
                [
                    self.model_tester.decoder_attention_heads,
                    self.model_tester.num_queries,
                    self.model_tester.num_queries,
                ],
            )


@require_torch
@require_vision
@slow
class PPDocLayoutV4ModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        model_path = "PaddlePaddle/PP-DocLayoutV4_safetensors"
        self.model = PPDocLayoutV4ForObjectDetection.from_pretrained(model_path).to(torch_device)
        self.image_processor = (
            PPDocLayoutV4ImageProcessor.from_pretrained(model_path) if is_vision_available() else None
        )
        img_url = url_to_local_path(
            "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout_demo.jpg"
        )
        self.image = load_image(img_url)

    def test_inference_object_detection_head(self):
        inputs = self.image_processor(images=self.image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        expected_shape_logits = torch.Size((1, 300, self.model.config.num_labels))
        expected_logits = torch.tensor(
            [[-3.5623, -4.5347, -5.0415], [-3.7752, -3.6966, -4.4212], [-4.4829, -4.3740, -4.5478]]
        ).to(torch_device)
        self.assertEqual(outputs.logits.shape, expected_shape_logits)
        torch.testing.assert_close(outputs.logits[0, :3, :3], expected_logits, rtol=2e-4, atol=2e-2)

        expected_shape_boxes = torch.Size((1, 300, self.model.config.num_coords))
        expected_boxes = torch.tensor(
            [[0.3719, 0.1785, 0.3313], [0.7257, 0.4412, 0.3301], [0.7253, 0.2664, 0.3306]]
        ).to(torch_device)
        self.assertEqual(outputs.pred_boxes.shape, expected_shape_boxes)
        torch.testing.assert_close(outputs.pred_boxes[0, :3, :3], expected_boxes, rtol=2e-4, atol=2e-2)

        expected_shape_order_logits = torch.Size((1, 300, 300))
        self.assertEqual(outputs.relative_order_logits.shape, expected_shape_order_logits)
        self.assertEqual(outputs.successor_order_logits.shape, expected_shape_order_logits)
        expected_relative_order_logits = torch.tensor(
            [[0.0000, 42.6447, 46.4521], [-42.6447, 0.0000, -28.6615], [-46.4521, 28.6615, 0.0000]]
        ).to(torch_device)
        torch.testing.assert_close(
            outputs.relative_order_logits[0, :3, :3], expected_relative_order_logits, rtol=2e-2, atol=2e-2
        )

        # verify postprocessing
        results = self.image_processor.post_process_object_detection(
            outputs, threshold=0.5, target_sizes=[self.image.size[::-1]]
        )[0]

        expected_scores = torch.tensor(
            [0.9885, 0.9781, 0.9938, 0.9900, 0.9871, 0.9833, 0.9771, 0.9010, 0.9529, 0.6550, 0.7850, 0.9787, 0.9286]
        ).to(torch_device)
        torch.testing.assert_close(results["scores"], expected_scores, rtol=2e-2, atol=2e-2)

        expected_labels = [22, 17, 22, 22, 22, 22, 22, 22, 22, 22, 10, 16, 8]
        self.assertSequenceEqual(results["labels"].tolist(), expected_labels)

        # Results come back sorted by reading order, which the model resolves into a single chain here.
        self.assertSequenceEqual(results["order_seq"].tolist(), list(range(13)))

        expected_slice_boxes = torch.tensor(
            [
                [336.0739, 182.0364, 894.1705, 652.6191],
                [336.4460, 681.8829, 868.7751, 796.9087],
                [334.0145, 840.8432, 889.1123, 1452.2927],
                [920.6475, 183.6178, 1476.7504, 462.7547],
            ]
        ).to(torch_device)
        torch.testing.assert_close(results["boxes"][:4], expected_slice_boxes, rtol=2e-2, atol=2e-2)

        # Unlike PP-DocLayoutV3 the polygon is always the four regressed corners, in TL, TR, BR, BL order.
        expected_polygon_points = torch.tensor(
            [[336.0739, 182.0364], [893.9496, 182.2223], [894.1705, 652.5367], [336.4081, 652.6191]]
        ).to(torch_device)
        self.assertEqual(results["polygon_points"].shape, torch.Size((13, 4, 2)))
        torch.testing.assert_close(results["polygon_points"][0], expected_polygon_points, rtol=2e-2, atol=2e-2)
