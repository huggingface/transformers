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
from unittest.mock import patch

import pytest
from huggingface_hub.errors import StrictDataclassClassValidationError, StrictDataclassFieldValidationError

from transformers import NeoMMEConfig, is_torch_available
from transformers.modeling_outputs import BaseModelOutput
from transformers.testing_utils import require_torch, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import (
        NeoMMEForMaskedLM,
        NeoMMEForRetrieval,
        NeoMMEModel,
    )
    from transformers import initialization as init
    from transformers.models.neomme.modeling_neomme import (
        NeoMMEDenseHead,
        NeoMMEEncoderLayer,
        NeoMMEExclusiveSelfAttention,
        NeoMMEMLP,
        NeoMMEMultiVectorHead,
        NeoMMEPreTrainedModel,
        NeoMMESigmoidGatedProjection,
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
        if isinstance(module, NeoMMESigmoidGatedProjection):
            init.normal_(module.o_proj.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, NeoMMEExclusiveSelfAttention):
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


def _per_layer_window_config(layer_types: list[str], alternate_sliding_window: int) -> dict[int, dict]:
    per_layer_config = {}
    sliding_idx = 0
    for layer_idx, layer_type in enumerate(layer_types):
        if layer_type == "full_attention":
            per_layer_config[layer_idx] = {"sliding_window": None}
            continue
        if sliding_idx % 2:
            per_layer_config[layer_idx] = {"sliding_window": alternate_sliding_window}
        sliding_idx += 1
    return per_layer_config


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
        sliding_window=3,
        alternate_sliding_window=6,
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
        self.sliding_window = sliding_window
        self.alternate_sliding_window = alternate_sliding_window
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
            "sliding_window": self.sliding_window,
            "patch_size": self.patch_size,
            "embedding_dim": self.embedding_dim,
            "max_position_embeddings": self.max_position_embeddings,
            "initializer_range": self.initializer_range,
            "pad_token_id": self.pad_token_id,
            "document_token_id": self.document_token_id,
            "image_token_id": self.image_token_id,
        }
        config_kwargs.update(kwargs)
        config_kwargs.setdefault(
            "per_layer_config",
            _per_layer_window_config(config_kwargs["layer_types"], self.alternate_sliding_window),
        )
        config = NeoMMEConfig(**config_kwargs)
        # Generic model tests inspect the global window even though NeoMME resolves windows per layer.
        config.allow_global_per_layer_attribute_access = True
        return config

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

    def _image_features_prepare_config_and_inputs(self):
        config = self.model_tester.get_config()
        pixel_values = floats_tensor([self.model_tester.batch_size, config.patch_dim])
        return config, {"pixel_values": pixel_values}

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        self.model_tester.create_and_check_model(*self.model_tester.prepare_config_and_inputs())

    def test_for_masked_lm(self):
        self.model_tester.create_and_check_for_masked_lm(*self.model_tester.prepare_config_and_inputs())

    @unittest.skip(reason="NeoMME value embeddings require token IDs")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="NeoMME value embeddings require token IDs")
    def test_inputs_embeds_matches_input_ids(self):
        pass

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

    @unittest.skip(reason="the generic test cannot read a heterogeneous global window; custom coverage is below")
    def test_sliding_window_mask(self):
        pass

    @unittest.skip(reason="NeoMME's image feature extractor is a single patch MLP without intermediate states")
    def test_get_image_features_hidden_states(self):
        pass

    @unittest.skip(reason="NeoMME's image feature extractor is a patch MLP without attention layers")
    def test_get_image_features_attentions(self):
        pass

    def test_grouped_query_heads_validated(self):
        with self.assertRaises(StrictDataclassFieldValidationError):
            NeoMMEConfig(num_attention_heads=4, num_key_value_heads=0)
        with self.assertRaisesRegex(StrictDataclassClassValidationError, "must divide"):
            NeoMMEConfig(num_attention_heads=4, num_key_value_heads=3)

    def test_layer_types_validated(self):
        base = {"num_hidden_layers": 3}
        invalid_cases = (
            ("too short", ["sliding_attention", "full_attention"], "must be equal"),
            ("too long", ["sliding_attention"] * 3 + ["full_attention"], "must be equal"),
            ("unknown", ["sliding_attention", "gdn", "full_attention"], "must be one of"),
            (
                "unsupported",
                ["sliding_attention", "chunked_attention", "full_attention"],
                "must be one of",
            ),
            ("no full attention", ["sliding_attention"] * 3, "must contain"),
        )
        for name, layer_types, error_pattern in invalid_cases:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, error_pattern):
                NeoMMEConfig(**base, layer_types=layer_types)

        pattern = ["sliding_attention", "sliding_attention", "full_attention"]
        config = NeoMMEConfig(num_hidden_layers=3, layer_types=pattern)
        self.assertEqual(config.layer_types, pattern)
        self.assertEqual(set(config.rope_parameters), {"full_attention", "sliding_attention"})

    def test_default_per_layer_windows(self):
        config = NeoMMEConfig()
        self.assertEqual(
            [layer.sliding_window for layer in config.per_layer_config],
            [256, 1024, 256, 1024, 256, None, 1024, 256, 1024, 256, 1024, None, 256, 1024, 256, 1024, None],
        )

    def test_attention_helper_modules(self):
        config = self.model_tester.get_config()
        model = NeoMMEModel(config)
        attention = model.layers[0].self_attn
        self.assertIsInstance(attention.exclusive_self_attention, NeoMMEExclusiveSelfAttention)
        self.assertIsInstance(attention.output_projection, NeoMMESigmoidGatedProjection)

    def test_window_widths_validated(self):
        """Global and per-layer windows must be positive; `None` selects full attention."""
        base = {"num_hidden_layers": 3, "layer_types": _layer_types(3, 3)}
        default = NeoMMEConfig(**base)
        self.assertEqual([layer.sliding_window for layer in default.per_layer_config], [256, 1024, None])

        configured = NeoMMEConfig(**base, per_layer_config={1: {"sliding_window": 1024}, 2: {"sliding_window": None}})
        self.assertEqual([layer.sliding_window for layer in configured.per_layer_config], [256, 1024, None])

        overridden = NeoMMEConfig(**base, per_layer_config={0: {"sliding_window": 128}})
        self.assertEqual([layer.sliding_window for layer in overridden.per_layer_config], [128, 256, 256])

        for value in (0, -1, 1.5, True):
            with (
                self.subTest(value=value, global_value=True),
                self.assertRaises(
                    (ValueError, StrictDataclassClassValidationError, StrictDataclassFieldValidationError)
                ),
            ):
                NeoMMEConfig(**base, sliding_window=value)
            with (
                self.subTest(value=value, global_value=False),
                self.assertRaises(
                    (ValueError, StrictDataclassClassValidationError, StrictDataclassFieldValidationError)
                ),
            ):
                NeoMMEConfig(**base, per_layer_config={0: {"sliding_window": value}})

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
            with self.subTest(theta=theta), self.assertRaises(ValueError):
                NeoMMEConfig(rope_theta=theta)
            with self.subTest(theta=theta, nested=True), self.assertRaises(ValueError):
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

    def test_residual_multiplier_defaults_from_depth(self):
        config = NeoMMEConfig(num_hidden_layers=8)
        self.assertEqual(config.residual_multiplier, (2 * 8) ** -0.5)
        self.assertNotIn("residual_scale", config.to_dict())

    def test_partial_rotary_factor_multiple_of_four(self):
        """Rotating dims must be a multiple of 4 (two M-RoPE axes × pairs); used to silently round down."""
        with self.assertRaisesRegex(ValueError, "which is not a multiple of 4"):
            NeoMMEConfig(head_dim=8)
        with self.assertRaisesRegex(ValueError, "which is not a multiple of 4"):
            NeoMMEConfig(head_dim=64, rope_parameters={"full_attention": {"partial_rotary_factor": 0.3}})
        with self.assertRaisesRegex(ValueError, "which is not a multiple of 4"):
            NeoMMEConfig(head_dim=2, layer_types=["full_attention"] * 17)

        # `0.75 * 64 = 48`, a valid rotary width.
        config = NeoMMEConfig(head_dim=64, rope_parameters={"full_attention": {"partial_rotary_factor": 0.75}})
        self.assertEqual(config.rope_parameters["full_attention"]["partial_rotary_factor"], 0.75)

    def test_partial_rotary_factor_unit_interval(self):
        for factor in (2.0, 0.0, -0.25):
            with self.assertRaisesRegex(ValueError, r"must be in \(0.0, 1.0\]"):
                NeoMMEConfig(rope_parameters={"sliding_attention": {"partial_rotary_factor": factor}})

    def test_config_dict_roundtrip(self):
        config = self.model_tester.get_config()
        reloaded = NeoMMEConfig.from_dict(config.to_dict())

        self.assertEqual(reloaded.layer_types, config.layer_types)
        self.assertEqual(
            [layer.sliding_window for layer in reloaded.per_layer_config],
            [layer.sliding_window for layer in config.per_layer_config],
        )
        self.assertEqual(reloaded.rope_parameters, config.rope_parameters)

    def test_sliding_windows_alternate(self):
        # Three layers [sliding, sliding, global]: both short/long widths plus the always-global last layer.
        config = self.model_tester.get_config(num_hidden_layers=3, layer_types=_layer_types(3, 3))
        windows = [layer.sliding_window for layer in config.per_layer_config]
        self.assertEqual(windows, [self.model_tester.sliding_window, self.model_tester.alternate_sliding_window, None])
        self.assertEqual(
            [window is None for window in windows],
            [layer_type == "full_attention" for layer_type in config.layer_types],
        )

        moved_full = self.model_tester.get_config(
            num_hidden_layers=3,
            layer_types=["full_attention", "sliding_attention", "sliding_attention"],
        )
        self.assertEqual(
            [layer.sliding_window for layer in moved_full.per_layer_config],
            [None, self.model_tester.sliding_window, self.model_tester.alternate_sliding_window],
        )

        homogeneous = self.model_tester.get_config(
            num_hidden_layers=3,
            layer_types=_layer_types(3, 3),
            per_layer_config=None,
        )
        model = NeoMMEModel(homogeneous)
        self.assertEqual(
            [layer.self_attn.sliding_window for layer in model.layers],
            [self.model_tester.sliding_window + 1, self.model_tester.sliding_window + 1, None],
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

        windows = [layer.sliding_window for layer in config.per_layer_config]
        for layer_idx, (attention, window) in enumerate(zip(attentions, windows)):
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

            inputs_embeds = model.embed_tokens(input_ids)
            image_features = model.get_image_features(pixel_values).pooler_output
            image_mask = model.get_placeholder_mask(input_ids, image_features)
            scattered = inputs_embeds.masked_scatter(image_mask, image_features)

            self.assertEqual(len(patch_positions), sum(h * w for h, w in grids))
            self.assertFalse(image_mask[0, 1].any(), "the <img> marker immediately after <doc> is not a patch")
            torch.testing.assert_close(scattered[0, patch_positions], image_features)
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

    def test_masked_lm_ties_word_embeddings(self):
        config, input_ids, attention_mask, _ = self.model_tester.prepare_config_and_inputs()
        model = NeoMMEForMaskedLM(config).to(torch_device).eval()
        input_ids, attention_mask = input_ids.to(torch_device), attention_mask.to(torch_device)

        self.assertEqual(model.num_parameters(), NeoMMEModel(config).num_parameters())
        self.assertIs(model.get_output_embeddings().weight, model.model.embed_tokens.word_embeddings.weight)
        self.assertIs(model.unembedding_projection.weight, model.model.embed_tokens.embedding_projection.weight)

        with torch.no_grad():
            hidden_states = model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            expected = (hidden_states @ model.model.embed_tokens.embedding_projection.weight) @ (
                model.model.embed_tokens.word_embeddings.weight.t()
            )
            actual = model(input_ids=input_ids, attention_mask=attention_mask).logits
        torch.testing.assert_close(actual, expected)

    def test_masked_lm_can_untie_word_embeddings(self):
        config, input_ids, attention_mask, _ = self.model_tester.prepare_config_and_inputs()
        config.tie_word_embeddings = False
        model = NeoMMEForMaskedLM(config).to(torch_device).eval()
        input_ids, attention_mask = input_ids.to(torch_device), attention_mask.to(torch_device)

        self.assertIsNot(model.lm_head.weight, model.model.embed_tokens.word_embeddings.weight)
        self.assertIsNot(model.unembedding_projection.weight, model.model.embed_tokens.embedding_projection.weight)
        self.assertEqual(
            model.num_parameters() - NeoMMEModel(config).num_parameters(),
            config.embedding_rank * (config.vocab_size + config.hidden_size),
        )

        with torch.no_grad():
            hidden_states = model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            expected = model.lm_head(hidden_states @ model.unembedding_projection.weight)
            actual = model(input_ids=input_ids, attention_mask=attention_mask).logits
        torch.testing.assert_close(actual, expected)

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
        seq_length = 4 * max(
            layer.sliding_window for layer in config.per_layer_config if layer.sliding_window is not None
        )
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


@require_torch
class NeoMMEForRetrievalModelTest(ModelTesterMixin, unittest.TestCase):
    """`NeoMMEForRetrieval` produces embeddings rather than a loss, so it is tested on its own."""

    all_model_classes = (NeoMMEForRetrieval,) if is_torch_available() else ()

    def setUp(self):
        self.model_tester = NeoMMEModelTester(self, is_training=False)
        _patch_residual_init(self)

    @unittest.skip(reason="NeoMME value embeddings require token IDs")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="the generic test cannot read a heterogeneous global window; covered on NeoMMEModel")
    def test_sliding_window_mask(self):
        pass

    @unittest.skip(
        reason="every NeoMME layer passes a 4-D mask; SDPA's flash kernel rejects masks. The real flash path "
        "for a windowed bidirectional model is the flash-attention package, covered by "
        "test_flash_attn_2_inference_equivalence."
    )
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    def test_for_retrieval(self):
        self.model_tester.create_and_check_for_retrieval(*self.model_tester.prepare_config_and_inputs())

    def test_retrieval_head_modules(self):
        config = self.model_tester.get_config()
        model = NeoMMEForRetrieval(config)
        self.assertIsInstance(model.multi_vector_head, NeoMMEMultiVectorHead)
        self.assertIsInstance(model.dense_head, NeoMMEDenseHead)

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
