# Copyright 2026 H Company and the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch NeoMME model."""

import unittest
from typing import ClassVar
from unittest.mock import patch

import pytest
from datasets import load_dataset
from huggingface_hub.errors import StrictDataclassClassValidationError, StrictDataclassFieldValidationError

from transformers import NeoMMEConfig, is_torch_available
from transformers.modeling_outputs import BaseModelOutput
from transformers.testing_utils import cleanup, require_torch, require_vision, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import (
        AutoModelForMaskedLM,
        AutoProcessor,
        NeoMMEForMaskedLM,
        NeoMMEForRetrieval,
        NeoMMEModel,
        NeoMMEProcessor,
    )
    from transformers import initialization as init
    from transformers.models.neomme.modeling_neomme import (
        NeoMMEAttention,
        NeoMMEEncoderLayer,
        NeoMMEMLP,
        NeoMMEPreTrainedModel,
        apply_rotary_pos_emb,
    )


def _patch_residual_init(test_case: unittest.TestCase) -> None:
    """Activate NeoMME's zero-initialized paths for mixin comparisons.

    Attention and MLP outputs, XSA scaling, value embeddings, and initial-state mixing otherwise begin as no-ops.
    """
    initialize = NeoMMEPreTrainedModel._init_weights

    @torch.no_grad()
    def initialize_with_live_residual_branches(self: NeoMMEPreTrainedModel, module: torch.nn.Module) -> None:
        initialize(self, module)
        if isinstance(module, NeoMMEAttention):
            init.normal_(module.o_proj.weight, mean=0.0, std=self.config.initializer_range)
            if module.alpha is not None:
                # Use O(1) values so `tanh(alpha)` exercises the XSA branch.
                init.normal_(module.alpha, mean=0.0, std=1.0)
        elif isinstance(module, NeoMMEMLP):
            init.normal_(module.down_proj.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, NeoMMEEncoderLayer):
            # The default is `[1.0, 0.0]`, so replacing only all-zero parameters would miss it.
            init.copy_(module.lambdas, torch.tensor([1.0, 0.5]))
        elif isinstance(module, NeoMMEModel):
            init.normal_(module.value_embeddings.weight, mean=0.0, std=self.config.initializer_range)

    patcher = patch.object(NeoMMEPreTrainedModel, "_init_weights", initialize_with_live_residual_branches)
    patcher.start()
    test_case.addCleanup(patcher.stop)


def _layer_types(num_hidden_layers: int, full_attention_every_n_layers: int) -> list[str]:
    return [
        "full_attention"
        if (index + 1) % full_attention_every_n_layers == 0 or index == num_hidden_layers - 1
        else "sliding_attention"
        for index in range(num_hidden_layers)
    ]


class NeoMMEModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=13,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        embedding_rank=16,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        # 16 ensures the default 0.25 rotary factor yields four dimensions, enough for both M-RoPE axes.
        head_dim=16,
        layer_types=None,
        sliding_window_short=3,
        sliding_window_long=6,
        patch_size=4,
        embedding_dim=8,
        max_position_embeddings=128,
        initializer_range=0.02,
        pad_token_id=0,
        document_token_id=5,
        image_token_id=6,
        row_token_id=8,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.embedding_rank = embedding_rank
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.layer_types = layer_types or _layer_types(num_hidden_layers, 2)
        self.sliding_window_short = sliding_window_short
        self.sliding_window_long = sliding_window_long
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.document_token_id = document_token_id
        self.image_token_id = image_token_id
        self.row_token_id = row_token_id

    def get_config(self, **kwargs):
        config_kwargs = {
            "vocab_size": self.vocab_size,
            "embedding_rank": self.embedding_rank,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "layer_types": self.layer_types,
            "sliding_window_short": self.sliding_window_short,
            "sliding_window_long": self.sliding_window_long,
            "patch_size": self.patch_size,
            "embedding_dim": self.embedding_dim,
            "max_position_embeddings": self.max_position_embeddings,
            "initializer_range": self.initializer_range,
            "pad_token_id": self.pad_token_id,
            "document_token_id": self.document_token_id,
            "image_token_id": self.image_token_id,
        }
        config_kwargs.update(kwargs)
        return NeoMMEConfig(**config_kwargs)

    def prepare_config_and_inputs(self):
        # Keep random text IDs above the reserved special-token range.
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size - 64) + 64
        input_mask = random_attention_mask([self.batch_size, self.seq_length]) if self.use_input_mask else None
        token_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size) if self.use_labels else None
        return self.get_config(), input_ids, input_mask, token_labels

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, input_mask, _ = self.prepare_config_and_inputs()
        return config, {"input_ids": input_ids, "attention_mask": input_mask}

    def prepare_image_config_and_inputs(self, grid_height=2, grid_width=3):
        """Build one processor-style image sequence per batch item."""
        config = self.get_config()
        sequence = [config.document_token_id, config.image_token_id]
        for _ in range(grid_height):
            sequence += [config.image_token_id] * grid_width + [self.row_token_id]
        input_ids = torch.tensor([sequence] * self.batch_size)
        pixel_values = floats_tensor([self.batch_size * grid_height * grid_width, config.patch_dim])
        return config, input_ids, pixel_values

    def create_and_check_model(self, config, input_ids, input_mask, token_labels):
        model = NeoMMEModel(config=config).to(torch_device).eval()
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertTrue(torch.isfinite(result.last_hidden_state).all())

    def create_and_check_for_masked_lm(self, config, input_ids, input_mask, token_labels):
        model = NeoMMEForMaskedLM(config=config).to(torch_device).eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))
        self.parent.assertTrue(torch.isfinite(result.loss))

    def create_and_check_for_retrieval(self, config, input_ids, input_mask, token_labels):
        model = NeoMMEForRetrieval(config=config).to(torch_device).eval()
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result.embeddings.shape, (self.batch_size, self.seq_length, self.embedding_dim))
        self.parent.assertEqual(result.dense_embeddings.shape, (self.batch_size, self.hidden_size))


@require_torch
class NeoMMEModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (NeoMMEModel, NeoMMEForMaskedLM) if is_torch_available() else ()
    # The common batch is text-only, so the vision stem legitimately receives no gradient. The dedicated
    # `test_patch_stem_gradients` covers it instead.
    test_all_params_have_gradient = False

    def setUp(self):
        self.model_tester = NeoMMEModelTester(self)
        self.config_tester = ConfigTester(self, config_class=NeoMMEConfig)
        _patch_residual_init(self)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        self.model_tester.create_and_check_model(*self.model_tester.prepare_config_and_inputs())

    def test_for_masked_lm(self):
        self.model_tester.create_and_check_for_masked_lm(*self.model_tester.prepare_config_and_inputs())

    @unittest.skip(reason="value embeddings require token ids and are omitted by inputs_embeds-only forwards")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    def test_requires_exactly_one_model_input(self):
        config, input_ids, _, _ = self.model_tester.prepare_config_and_inputs()
        model = NeoMMEModel(config).to(torch_device).eval()
        inputs_embeds = model.get_input_embeddings()(input_ids)

        with self.assertRaisesRegex(ValueError, "exactly one"):
            model()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            model(input_ids=input_ids, inputs_embeds=inputs_embeds)

    @unittest.skip(reason="the generic check compares an unused layer spectrum that differs by one floating-point ULP")
    def test_model_rope_scaling_frequencies(self):
        pass

    @unittest.skip(
        reason="every NeoMME layer passes a 4-D mask; SDPA's flash kernel rejects masks. The real flash path "
        "for a windowed bidirectional model is the flash-attention package, covered by "
        "test_flash_attn_2_inference_equivalence."
    )
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    def test_grouped_query_heads_validated(self):
        with self.assertRaises(StrictDataclassFieldValidationError):
            NeoMMEConfig(num_attention_heads=4, num_key_value_heads=0)
        with self.assertRaisesRegex(StrictDataclassClassValidationError, "must divide"):
            NeoMMEConfig(num_attention_heads=4, num_key_value_heads=3)

    def test_layer_types_validated(self):
        base = {"num_hidden_layers": 3}
        with self.assertRaises(ValueError):  # one entry short
            NeoMMEConfig(**base, layer_types=["sliding_attention"] + ["full_attention"])
        with self.assertRaises(ValueError):  # not a known layer type
            NeoMMEConfig(**base, layer_types=["sliding_attention", "gdn", "full_attention"])
        with self.assertRaises(ValueError):  # value embeddings require a full-attention layer
            NeoMMEConfig(**base, layer_types=["sliding_attention"] * 3)

        pattern = ["sliding_attention", "sliding_attention", "full_attention"]
        config = NeoMMEConfig(num_hidden_layers=3, layer_types=pattern)
        self.assertEqual(config.layer_types, pattern)

    def test_window_widths_validated(self):
        """Equal widths select one band; runtime configs reject the research zero encoding."""
        base = {"num_hidden_layers": 3, "layer_types": _layer_types(3, 3)}
        uniform = NeoMMEConfig(**base, sliding_window_short=256, sliding_window_long=256)
        self.assertEqual([w for w in uniform.layer_window_sizes if w is not None], [256, 256])

        for short, long in ((256, 0), (256, 128), (0, 256)):
            with self.subTest(short=short, long=long), self.assertRaises(StrictDataclassClassValidationError):
                NeoMMEConfig(**base, sliding_window_short=short, sliding_window_long=long)

    def test_rope_parameters_follow_layer_types(self):
        self.assertEqual(list(NeoMMEConfig(num_hidden_layers=1).rope_parameters), ["full_attention"])

    def test_flat_rope_theta(self):
        config = NeoMMEConfig(rope_theta=123456.0)
        self.assertEqual(
            {layer_type: params["rope_theta"] for layer_type, params in config.rope_parameters.items()},
            {"full_attention": 123456.0, "sliding_attention": 123456.0},
        )
        self.assertNotIn("rope_theta", config.to_dict())

        explicit = NeoMMEConfig(rope_theta=123456.0, rope_parameters={"sliding_attention": {"rope_theta": 7.0}})
        self.assertEqual(explicit.rope_parameters["sliding_attention"]["rope_theta"], 7.0)
        self.assertEqual(explicit.rope_parameters["full_attention"]["rope_theta"], 123456.0)

    def test_rope_theta_must_be_positive(self):
        for theta in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(theta=theta), self.assertRaises(StrictDataclassClassValidationError):
                NeoMMEConfig(rope_theta=theta)
            with self.subTest(theta=theta, nested=True), self.assertRaises(StrictDataclassClassValidationError):
                NeoMMEConfig(rope_parameters={"sliding_attention": {"rope_theta": theta}})

    def test_architecture_dimensions_must_be_positive(self):
        for name in (
            "num_hidden_layers",
            "num_attention_heads",
            "head_dim",
            "max_position_embeddings",
            "patch_size",
            "embedding_dim",
        ):
            with self.subTest(name=name), self.assertRaises(StrictDataclassFieldValidationError):
                NeoMMEConfig(**{name: 0})

    def test_legacy_rope_scaling_type_alias(self):
        config = NeoMMEConfig(rope_scaling={"type": "linear", "factor": 2.0})
        self.assertEqual(
            {params["rope_type"] for params in config.rope_parameters.values()},
            {"linear"},
        )

    def test_partial_rotary_factor_multiple_of_four(self):
        """Rotating dims must be a multiple of 4 (two M-RoPE axes × pairs); used to silently round down."""
        with self.assertRaisesRegex(StrictDataclassClassValidationError, "needs at least 4"):
            NeoMMEConfig(head_dim=8)
        with self.assertRaisesRegex(StrictDataclassClassValidationError, "not a multiple of 4"):
            NeoMMEConfig(head_dim=64, rope_parameters={"full_attention": {"partial_rotary_factor": 0.3}})
        with self.assertRaisesRegex(StrictDataclassClassValidationError, "needs at least 4"):
            NeoMMEConfig(head_dim=2, layer_types=["full_attention"] * 17)

        # `0.75 * 64 = 48`, a valid rotary width.
        config = NeoMMEConfig(head_dim=64, rope_parameters={"full_attention": {"partial_rotary_factor": 0.75}})
        self.assertEqual(config.rope_parameters["full_attention"]["partial_rotary_factor"], 0.75)

    def test_partial_rotary_factor_unit_interval(self):
        for factor in (2.0, 0.0, -0.25):
            with self.assertRaisesRegex(StrictDataclassClassValidationError, r"outside \(0.0, 1.0\]"):
                NeoMMEConfig(rope_parameters={"sliding_attention": {"partial_rotary_factor": factor}})

    def test_config_dict_roundtrip(self):
        config = self.model_tester.get_config()
        reloaded = NeoMMEConfig.from_dict(config.to_dict())

        self.assertEqual(reloaded.layer_types, config.layer_types)
        self.assertEqual(reloaded.layer_window_sizes, config.layer_window_sizes)
        self.assertEqual(reloaded.rope_parameters, config.rope_parameters)

    def test_sliding_windows_alternate(self):
        # Three layers [sliding, sliding, global]: both short/long widths plus the always-global last layer.
        config = self.model_tester.get_config(num_hidden_layers=3, layer_types=_layer_types(3, 3))
        windows = [window for window in config.layer_window_sizes if window is not None]
        expected = [
            config.sliding_window_long if index % 2 else config.sliding_window_short for index in range(len(windows))
        ]
        self.assertEqual(windows, expected)
        self.assertEqual(
            [window is None for window in config.layer_window_sizes],
            [layer_type == "full_attention" for layer_type in config.layer_types],
        )

    def test_bidirectional_attention_windows(self):
        """Each layer is bidirectional; sliding layers are zero outside `abs(i - j) <= window`."""
        config = self.model_tester.get_config(num_hidden_layers=3, layer_types=_layer_types(3, 3))
        config._attn_implementation = "eager"  # only the eager path returns attention probabilities
        model = NeoMMEModel(config).to(torch_device).eval()

        seq_length = self.model_tester.seq_length
        input_ids = ids_tensor([1, seq_length], config.vocab_size - 64) + 64
        with torch.no_grad():
            attentions = model(
                input_ids=input_ids, attention_mask=torch.ones_like(input_ids), output_attentions=True
            ).attentions

        self.assertEqual(len(attentions), config.num_hidden_layers)
        positions = torch.arange(seq_length, device=torch_device)
        distance = (positions[:, None] - positions[None, :]).abs()

        for layer_idx, (attention, window) in enumerate(zip(attentions, config.layer_window_sizes)):
            inside = distance <= window if window is not None else torch.ones_like(distance, dtype=torch.bool)
            with self.subTest(layer=layer_idx, window=window):
                self.assertTrue((attention[0, :, inside] > 0).all(), "a reachable pair got zero weight")
                if window is not None and (~inside).any():
                    self.assertTrue((attention[0, :, ~inside] == 0).all(), "attention leaked outside the band")

                # The upper triangle carries the bidirectionality: a causal mask would zero it.
                upper = torch.triu(inside, diagonal=1)
                if upper.any():
                    self.assertTrue((attention[0, :, upper] > 0).all(), "layer is causal")

    def test_patch_embedding_scatter(self):
        """The `<img>` marker after `<doc>` must not consume a patch; multi-image rows scatter in order."""
        config = self.model_tester.get_config()
        model = NeoMMEModel(config).to(torch_device).eval()

        with self.subTest(case="single_image_forward"):
            _, input_ids, pixel_values = self.model_tester.prepare_image_config_and_inputs()
            input_ids, pixel_values = input_ids.to(torch_device), pixel_values.to(torch_device)
            output = model(input_ids=input_ids, pixel_values=pixel_values)
            self.assertEqual(output.last_hidden_state.shape[1], input_ids.shape[1])
            self.assertTrue(torch.isfinite(output.last_hidden_state).all())

            with self.assertRaises(ValueError):
                model(input_ids=input_ids, pixel_values=pixel_values[:-1])
            with self.assertRaises(ValueError):
                model(input_ids=input_ids, pixel_values=pixel_values[:, :-1])

            inputs_embeds = model.get_input_embeddings()(input_ids)
            with self.assertRaisesRegex(ValueError, "requires `input_ids`"):
                model(inputs_embeds=inputs_embeds, pixel_values=pixel_values)

        with self.subTest(case="multi_image_order"):
            grids = [(2, 3), (1, 2)]
            sequence: list[int] = []
            patch_positions: list[int] = []
            for grid_height, grid_width in grids:
                sequence += [config.document_token_id, config.image_token_id]
                for _ in range(grid_height):
                    patch_positions += [len(sequence) + offset for offset in range(grid_width)]
                    sequence += [config.image_token_id] * grid_width + [self.model_tester.row_token_id]
            input_ids = torch.tensor([sequence], device=torch_device)
            pixel_values = floats_tensor([len(patch_positions), config.patch_dim]).to(torch_device)

            inputs_embeds = model.embeddings(input_ids=input_ids)
            scattered = model._scatter_patch_embeddings(input_ids, inputs_embeds, pixel_values)
            expected = model.patch_embeddings(pixel_values)

            self.assertEqual(len(patch_positions), sum(h * w for h, w in grids))
            torch.testing.assert_close(scattered[0, patch_positions], expected)
            untouched = [i for i in range(len(sequence)) if i not in patch_positions]
            torch.testing.assert_close(scattered[0, untouched], inputs_embeds[0, untouched])

    @pytest.mark.torch_compile_test
    def test_image_path_torch_compile(self):
        """Image path must compile under `fullgraph=True` (patch-count used to be data-dependent)."""
        config = self.model_tester.get_config()
        config._attn_implementation = "sdpa"
        model = NeoMMEModel(config).to(torch_device).eval()

        grid_width = 3
        sequence = [config.document_token_id, config.image_token_id]
        sequence += [config.image_token_id] * grid_width + [self.model_tester.row_token_id]
        input_ids = torch.tensor([sequence], device=torch_device)
        pixel_values = floats_tensor([grid_width, config.patch_dim]).to(torch_device)

        with torch.no_grad():
            compiled = torch.compile(model, fullgraph=True)(input_ids=input_ids, pixel_values=pixel_values)
            eager = model(input_ids=input_ids, pixel_values=pixel_values)

        torch.testing.assert_close(compiled.last_hidden_state, eager.last_hidden_state)

        with self.assertRaises((ValueError, RuntimeError)):
            torch.compile(model, fullgraph=True)(input_ids=input_ids, pixel_values=pixel_values[:-1])
        with self.assertRaises((ValueError, RuntimeError)):
            torch.compile(model, fullgraph=True)(
                input_ids=input_ids, pixel_values=torch.cat([pixel_values, pixel_values[:1]])
            )

    def test_masked_lm_adds_no_parameters(self):
        config = self.model_tester.get_config()
        self.assertEqual(NeoMMEForMaskedLM(config).num_parameters(), NeoMMEModel(config).num_parameters())
        self.assertIsNone(NeoMMEForMaskedLM(config).get_output_embeddings())

    def test_partial_rotary_standard_layout(self):
        """Rotation acts on the standard half layout and leaves the NoPE tail untouched."""
        head_dim, rotary_dim = 8, 4
        states = torch.arange(head_dim, dtype=torch.float32).view(1, 1, 1, head_dim)
        cos = torch.zeros(1, 1, rotary_dim)
        sin = torch.ones(1, 1, rotary_dim)
        rotated, _ = apply_rotary_pos_emb(states, states, cos, sin, unsqueeze_dim=2)
        torch.testing.assert_close(rotated.flatten(), torch.tensor([-2.0, -3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 7.0]))

    def test_patch_stem_gradients(self):
        config, input_ids, pixel_values = self.model_tester.prepare_image_config_and_inputs()
        model = NeoMMEForMaskedLM(config).to(torch_device).train()
        input_ids, pixel_values = input_ids.to(torch_device), pixel_values.to(torch_device)

        model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids).loss.backward()

        for name, parameter in model.model.patch_embeddings.named_parameters():
            self.assertIsNotNone(parameter.grad, f"patch_embeddings.{name} received no gradient")
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0)

    def test_text_parameters_receive_gradients(self):
        config, input_ids, attention_mask, labels = self.model_tester.prepare_config_and_inputs()
        model = NeoMMEForMaskedLM(config).to(torch_device).train()
        input_ids, attention_mask, labels = (
            input_ids.to(torch_device),
            attention_mask.to(torch_device),
            labels.to(torch_device),
        )

        model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss.backward()
        missing = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and "patch_embeddings" not in name and parameter.grad is None
        ]
        self.assertEqual(missing, [])

    def test_padded_row_stays_finite(self):
        """A padded query can have no keys after intersecting padding with a sliding window."""
        config = self.model_tester.get_config()
        model = NeoMMEModel(config).to(torch_device).eval()
        seq_length = 4 * config.sliding_window_long
        input_ids = ids_tensor([2, seq_length], config.vocab_size - 64) + 64
        attention_mask = torch.ones_like(input_ids)
        attention_mask[1, 2:] = 0

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        self.assertTrue(torch.isfinite(output.last_hidden_state).all())

    def test_two_axis_position_ids(self):
        config, input_ids, input_mask, _ = self.model_tester.prepare_config_and_inputs()
        model = NeoMMEModel(config).to(torch_device).eval()
        one_axis = torch.arange(input_ids.shape[1], device=torch_device).expand(input_ids.shape[0], -1)

        with torch.no_grad():
            default = model(input_ids=input_ids, attention_mask=input_mask).last_hidden_state
            explicit = model(input_ids=input_ids, attention_mask=input_mask, position_ids=one_axis).last_hidden_state
            stacked = model(
                input_ids=input_ids, attention_mask=input_mask, position_ids=torch.stack([one_axis, one_axis])
            ).last_hidden_state

        torch.testing.assert_close(default, explicit)
        torch.testing.assert_close(default, stacked)

    def test_position_ids_shape_validated(self):
        config, input_ids, input_mask, _ = self.model_tester.prepare_config_and_inputs()
        model = NeoMMEModel(config).to(torch_device).eval()
        input_ids = input_ids[:2].to(torch_device)
        input_mask = input_mask[:2].to(torch_device)
        batch_size, seq_len = input_ids.shape

        invalid_shapes = [(3, batch_size, seq_len), (batch_size, seq_len, 2), (batch_size, seq_len + 1)]
        for shape in invalid_shapes:
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "position_ids must have shape"):
                model(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    position_ids=torch.zeros(shape, dtype=torch.long, device=torch_device),
                )


@require_torch
class NeoMMEForRetrievalModelTest(ModelTesterMixin, unittest.TestCase):
    """`NeoMMEForRetrieval` produces embeddings rather than a loss, so it is tested on its own."""

    all_model_classes = (NeoMMEForRetrieval,) if is_torch_available() else ()

    def setUp(self):
        self.model_tester = NeoMMEModelTester(self, is_training=False)
        _patch_residual_init(self)

    @unittest.skip(
        reason="every NeoMME layer passes a 4-D mask; SDPA's flash kernel rejects masks. The real flash path "
        "for a windowed bidirectional model is the flash-attention package, covered by "
        "test_flash_attn_2_inference_equivalence."
    )
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    def test_for_retrieval(self):
        self.model_tester.create_and_check_for_retrieval(*self.model_tester.prepare_config_and_inputs())

    def test_multivector_padding_and_norm(self):
        config, input_ids, input_mask, _ = self.model_tester.prepare_config_and_inputs()
        input_mask[0, 3:] = 0
        model = NeoMMEForRetrieval(config).to(torch_device).eval()
        embeddings = model(input_ids=input_ids, attention_mask=input_mask).embeddings

        self.assertTrue((embeddings[0, 3:] == 0).all())
        real = input_mask.bool()
        torch.testing.assert_close(
            embeddings[real].norm(dim=-1), torch.ones_like(embeddings[real][:, 0]), rtol=1e-4, atol=1e-4
        )

    def test_dense_dim_out_of_range(self):
        """`dense_dim` must be in 1..hidden_size; invalid widths used to be sliced silently."""
        config, input_ids, input_mask, _ = self.model_tester.prepare_config_and_inputs()
        model = NeoMMEForRetrieval(config).to(torch_device).eval()
        for dense_dim in (0, -1, config.hidden_size + 1):
            with self.subTest(dense_dim=dense_dim), self.assertRaises(ValueError):
                model(input_ids=input_ids, attention_mask=input_mask, dense_dim=dense_dim)

    def test_dense_head_truncation(self):
        config, input_ids, input_mask, _ = self.model_tester.prepare_config_and_inputs()
        model = NeoMMEForRetrieval(config).to(torch_device).eval()
        full = model(input_ids=input_ids, attention_mask=input_mask).dense_embeddings
        truncated = model(input_ids=input_ids, attention_mask=input_mask, dense_dim=8).dense_embeddings

        self.assertEqual(truncated.shape[-1], 8)
        torch.testing.assert_close(truncated.norm(dim=-1), torch.ones_like(truncated[:, 0]), rtol=1e-4, atol=1e-4)
        # Renormalizing a truncated prefix differs from slicing the normalized full vector.
        self.assertFalse(torch.allclose(truncated, full[:, :8], atol=1e-3))

    def test_retrieval_head_selection(self):
        config, input_ids, input_mask, _ = self.model_tester.prepare_config_and_inputs()
        model = NeoMMEForRetrieval(config).to(torch_device).eval()

        output = model(input_ids=input_ids, output_dense=False)
        self.assertIsInstance(output, BaseModelOutput)
        self.assertIsNone(output.dense_embeddings)
        self.assertIsNone(model(input_ids=input_ids, output_multivector=False).embeddings)

        with self.assertRaises(ValueError):
            model(input_ids=input_ids, output_dense=False, output_multivector=False)

    def test_retrieval_class_is_not_auto_mapped(self):
        from transformers.models.auto.modeling_auto import MODEL_FOR_RETRIEVAL_MAPPING_NAMES

        # Keep the combined two-head class directly importable, but do not let AutoModelForRetrieval or
        # Sentence Transformers select it instead of the AutoModel backbone.
        self.assertNotIn("neomme", MODEL_FOR_RETRIEVAL_MAPPING_NAMES)

    def test_fully_padded_row_pooling(self):
        config, input_ids, input_mask, _ = self.model_tester.prepare_config_and_inputs()
        input_mask[0] = 0
        model = NeoMMEForRetrieval(config).to(torch_device).eval()
        output = model(input_ids=input_ids, attention_mask=input_mask)

        self.assertTrue(torch.isfinite(output.dense_embeddings).all())
        self.assertTrue(torch.isfinite(output.embeddings).all())
        self.assertTrue((output.embeddings[0] == 0).all())
        self.assertTrue((output.dense_embeddings[0] == 0).all())

    def test_dense_head_uses_mean_pooling(self):
        config, input_ids, input_mask, _ = self.model_tester.prepare_config_and_inputs()
        input_mask[0, 3:] = 0
        model = NeoMMEForRetrieval(config).to(torch_device).eval()

        with torch.no_grad():
            hidden_states = model.model(input_ids=input_ids, attention_mask=input_mask).last_hidden_state
            actual = model(input_ids=input_ids, attention_mask=input_mask).dense_embeddings

        expanded_mask = input_mask.unsqueeze(-1).expand(hidden_states.shape).to(hidden_states.dtype)
        expected = (hidden_states * expanded_mask).sum(1) / expanded_mask.sum(1).clamp_min(1e-9)
        torch.testing.assert_close(actual, torch.nn.functional.normalize(expected, dim=-1))


@unittest.skip("NeoMME checkpoints are not public yet.")
@slow
@require_torch
@require_vision
class NeoMMEModelIntegrationTest(unittest.TestCase):
    model_name: ClassVar[str] = "Hcompany/NeoMME-260M-Retriever"
    model_dtype: ClassVar["torch.dtype"] = torch.float32 if is_torch_available() else None

    TEXT_QUERIES: ClassVar[list[str]] = [
        "How many people live in the capital of France?",
        "When was the United States Declaration of Independence proclaimed?",
    ]
    TEXT_DOCUMENTS: ClassVar[list[str]] = [
        "Paris is the capital of France and has a population of about 2.1 million people.",
        "The United States Declaration of Independence was proclaimed in 1776.",
    ]

    @classmethod
    def setUpClass(cls):
        cls.processor = NeoMMEProcessor.from_pretrained(cls.model_name)
        cls.model, cls.loading_info = NeoMMEForRetrieval.from_pretrained(
            cls.model_name, dtype=cls.model_dtype, output_loading_info=True
        )
        cls.model = cls.model.to(torch_device).eval()
        cls.dataset = load_dataset("hf-internal-testing/document-visual-retrieval-test", split="test")

    @classmethod
    def tearDownClass(cls):
        del cls.model
        del cls.processor
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_checkpoint_loads_cleanly(self):
        self.assertFalse(self.loading_info["missing_keys"])
        self.assertFalse(self.loading_info["unexpected_keys"])
        self.assertFalse(self.loading_info["mismatched_keys"])
        self.assertFalse(self.loading_info["error_msgs"])

    def test_multivector_image_retrieval(self):
        """Each query ranks its paired image first with MaxSim."""
        queries, images = self._embed_image_pair(head="multivector")
        self._assert_diagonal_retrieval(self.processor.score_retrieval(queries, images))

    def test_dense_image_retrieval(self):
        """Each query ranks its paired image first with dense cosine similarity."""
        queries, images = self._embed_image_pair(head="dense")

        self.assertEqual(images.shape, (len(self.dataset), self.model.config.hidden_size))
        torch.testing.assert_close(images.norm(dim=-1), torch.ones_like(images[:, 0]), rtol=1e-3, atol=1e-3)
        self._assert_diagonal_retrieval(self.processor.score_retrieval(queries, images))

    def test_multivector_text_retrieval(self):
        """Each query ranks its paired text first with MaxSim."""
        queries, documents = self._embed_text_pair(head="multivector")
        self._assert_diagonal_retrieval(self.processor.score_retrieval(queries, documents))

    def test_dense_text_retrieval(self):
        """Each query ranks its paired text first with dense cosine similarity."""
        queries, documents = self._embed_text_pair(head="dense")

        torch.testing.assert_close(documents.norm(dim=-1), torch.ones_like(documents[:, 0]), rtol=1e-3, atol=1e-3)
        self._assert_diagonal_retrieval(self.processor.score_retrieval(queries, documents))

    def _embed(self, batch, head: str) -> "torch.Tensor":
        """Compute only the requested retrieval head."""
        only_this_head = {"output_multivector": False} if head == "dense" else {"output_dense": False}
        with torch.inference_mode():
            outputs = self.model(**batch.to(torch_device), **only_this_head)
        return outputs.dense_embeddings if head == "dense" else outputs.embeddings

    def _embed_image_pair(self, head: str) -> tuple["torch.Tensor", "torch.Tensor"]:
        queries = self.processor(text=self.dataset["query"][:], task="query")
        images = self.processor(images=self.dataset["image"][:])
        return self._embed(queries, head), self._embed(images, head)

    def _embed_text_pair(self, head: str) -> tuple["torch.Tensor", "torch.Tensor"]:
        queries = self.processor(text=self.TEXT_QUERIES, task="query")
        documents = self.processor(text=self.TEXT_DOCUMENTS, task="document")
        return self._embed(queries, head), self._embed(documents, head)

    def _assert_diagonal_retrieval(self, scores: "torch.Tensor") -> None:
        self.assertEqual(scores.shape[0], scores.shape[1])
        self.assertTrue((scores.argmax(dim=1) == torch.arange(len(scores), device=scores.device)).all())
        self.assertTrue(((scores >= -1.0) & (scores <= 1.0)).all())


@unittest.skip("NeoMME checkpoints are not public yet.")
@slow
@require_torch
@require_vision
class NeoMMEBaseModelIntegrationTest(unittest.TestCase):
    model_name: ClassVar[str] = "Hcompany/NeoMME-260M"
    model_dtype: ClassVar["torch.dtype"] = torch.float32 if is_torch_available() else None

    @classmethod
    def setUpClass(cls):
        cls.processor = NeoMMEProcessor.from_pretrained(cls.model_name)
        cls.model, cls.loading_info = NeoMMEModel.from_pretrained(
            cls.model_name, dtype=cls.model_dtype, output_loading_info=True
        )
        cls.model = cls.model.to(torch_device).eval()
        cls.dataset = load_dataset("hf-internal-testing/document-visual-retrieval-test", split="test")

    @classmethod
    def tearDownClass(cls):
        del cls.model
        del cls.processor
        cleanup(torch_device, gc_collect=True)

    def test_checkpoint_loads_cleanly(self):
        self.assertFalse(self.loading_info["missing_keys"])
        self.assertFalse(self.loading_info["unexpected_keys"])
        self.assertFalse(self.loading_info["mismatched_keys"])
        self.assertFalse(self.loading_info["error_msgs"])

    def test_text_and_image_forward(self):
        inputs = [
            self.processor(text=["When was the Declaration of Independence proclaimed?"], task="query"),
            self.processor(images=[self.dataset[0]["image"]]),
        ]

        for batch in inputs:
            with self.subTest(batch=batch.keys()), torch.inference_mode():
                outputs = self.model(**batch.to(torch_device))

            self.assertEqual(outputs.last_hidden_state.shape[:2], batch.input_ids.shape)
            self.assertTrue(torch.isfinite(outputs.last_hidden_state).all())


@unittest.skip("NeoMME checkpoints are not public yet.")
@slow
@require_torch
class NeoMMEMaskedLMIntegrationTest(unittest.TestCase):
    model_name: ClassVar[str] = "Hcompany/NeoMME-260M"
    model_dtype: ClassVar["torch.dtype"] = torch.float32 if is_torch_available() else None

    @classmethod
    def setUpClass(cls):
        cls.processor = AutoProcessor.from_pretrained(cls.model_name)
        cls.model = AutoModelForMaskedLM.from_pretrained(cls.model_name, dtype=cls.model_dtype)
        cls.model = cls.model.to(torch_device).eval()

    @classmethod
    def tearDownClass(cls):
        del cls.model
        del cls.processor
        cleanup(torch_device, gc_collect=True)

    def test_fill_mask(self):
        text = f"The capital of {self.processor.tokenizer.mask_token} is London."
        inputs = self.processor(text=[text], task="document").to(torch_device)

        with torch.inference_mode():
            outputs = self.model(**inputs)

        masked_index = (inputs.input_ids[0] == self.processor.tokenizer.mask_token_id).nonzero().item()
        predicted_token_id = outputs.logits[0, masked_index].argmax(dim=-1)
        self.assertEqual(self.processor.tokenizer.decode(predicted_token_id).strip(), "UK")
