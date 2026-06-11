import importlib
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import NamedTuple
from unittest.mock import patch

from parameterized import parameterized

from transformers.testing_utils import (
    cleanup,
    is_torch_available,
    require_torch,
    torch_device,
)


if is_torch_available():
    import torch

    from tests.heterogeneity.model_fixtures import MODEL_FIXTURES
    from transformers import (
        DynamicCache,
        LlamaConfig,
        LlamaForCausalLM,
        PreTrainedModel,
        StaticCache,
    )
    from transformers.cache_utils import DynamicSlidingWindowLayer, StaticSlidingWindowLayer
    from transformers.heterogeneity import (
        HeterogeneousModelingSpec,
        ReturnEntry,
        get_heterogeneous_modeling_spec,
        get_skip_replacement,
    )
    from transformers.masking_utils import create_chunked_causal_mask, create_sliding_window_causal_mask
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM
    from transformers.models.llama4.configuration_llama4 import Llama4TextConfig
    from transformers.models.llama4.modeling_llama4 import Llama4ForCausalLM
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM


# ──────────────────────────────────────────────────────────────────────
# Tiny config factories
# ──────────────────────────────────────────────────────────────────────


def _tiny_llama_config(per_layer_config=None, **overrides):
    return LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        vocab_size=32,
        max_position_embeddings=64,
        per_layer_config=per_layer_config,
        **overrides,
    )


def _tiny_gpt_oss_config(per_layer_config=None, **overrides):
    return GptOssConfig(
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        vocab_size=32,
        max_position_embeddings=64,
        num_local_experts=4,
        num_experts_per_tok=2,
        sliding_window=32,
        per_layer_config=per_layer_config,
        **overrides,
    )


def _tiny_llama4_config(per_layer_config=None, **overrides):
    return Llama4TextConfig(
        hidden_size=64,
        intermediate_size=32,
        intermediate_size_mlp=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        vocab_size=32,
        max_position_embeddings=64,
        num_local_experts=4,
        num_experts_per_tok=1,
        moe_layers=[1, 3],
        attention_chunk_size=32,
        use_qk_norm=False,
        attn_temperature_tuning=False,
        per_layer_config=per_layer_config,
        **overrides,
    )


def _tiny_nemotron_h_config(per_layer_config=None, **overrides):
    return NemotronHConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        vocab_size=32,
        max_position_embeddings=64,
        layers_block_type=["attention", "mamba", "moe", "attention"],
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        moe_shared_expert_intermediate_size=32,
        ssm_state_size=16,
        mamba_num_heads=4,
        mamba_head_dim=16,
        n_groups=2,
        per_layer_config=per_layer_config,
        **overrides,
    )


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


@contextmanager
def _hetero_context(model_key):
    """Temporarily set the production heterogeneous modeling spec on a model class."""
    fixture = MODEL_FIXTURES[model_key]
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(fixture.pretrained_cls, "_heterogeneous_modeling_spec", fixture.spec_factory(), create=True)
        )
        yield


def _build_model(config, model_cls, seed=42):
    """Build a model deterministically on CPU."""
    torch.manual_seed(seed)
    return model_cls(config).eval()


def _forward_logits(model, input_ids):
    """Run a forward pass and return logits."""
    with torch.no_grad():
        return model(input_ids).logits


def _dummy_input_ids(batch=1, seq_len=8):
    return torch.randint(0, 32, (batch, seq_len))


def _build_reference(hetero_model, base_config, model_cls, model_key):
    """Build a reference model with skip-aware layers that match the heterogeneous one.

    The reference layers read per-layer config natively: they create modules at the right
    dimensions and simply don't create skipped modules. This is what the model would look
    like if skip support was built in natively rather than through the heterogeneity mechanism.
    """
    hetero_config = hetero_model.config
    ref = _build_model(base_config, model_cls)
    ref_layer_cls = MODEL_FIXTURES[model_key].ref_layer_cls
    for i in range(len(ref.model.layers)):
        ref.model.layers[i] = ref_layer_cls(hetero_config, layer_idx=i)
    ref.load_state_dict(hetero_model.state_dict())
    return ref.eval()


if is_torch_available():
    # Toy composite models for testing the nested model construction paths.
    class _DefaultContextMarker(torch.nn.Module):
        pass

    class _SameLayerContextMarker(torch.nn.Module):
        pass

    class _OuterContextMarker(torch.nn.Module):
        pass

    class _InnerContextMarker(torch.nn.Module):
        pass

    def _init_toy_layer(layer, config, layer_idx):
        torch.nn.Module.__init__(layer)
        layer.layer_idx = layer_idx
        layer.intermediate_size = config.intermediate_size
        layer.context_marker = _DefaultContextMarker()

    class _SameLayerToyLayer(torch.nn.Module):
        def __init__(self, config, layer_idx):
            _init_toy_layer(self, config, layer_idx)

    class _OuterToyLayer(torch.nn.Module):
        def __init__(self, config, layer_idx):
            _init_toy_layer(self, config, layer_idx)

    class _InnerToyLayer(torch.nn.Module):
        def __init__(self, config, layer_idx):
            _init_toy_layer(self, config, layer_idx)

    def _toy_modeling_spec(layer_cls, marker_replacement_cls):
        return HeterogeneousModelingSpec(
            layer_cls=layer_cls,
            layer_idx_variable_name="layer_idx",
            skip_descriptors={"context_marker": {"context_marker": marker_replacement_cls}},
        )

    def _toy_config(intermediate_size, replace_context_marker=False):
        layer_config = {"intermediate_size": intermediate_size}
        if replace_context_marker:
            layer_config["skip"] = ["context_marker"]
        return _tiny_llama_config(per_layer_config={0: layer_config})

    class _NestedSameLayerToyModel(PreTrainedModel):
        config_class = LlamaConfig
        _heterogeneous_modeling_spec = _toy_modeling_spec(_SameLayerToyLayer, _SameLayerContextMarker)

        def __init__(self, config, inner_config=None):
            super().__init__(config)
            if inner_config is not None:
                self.inner_model = _NestedSameLayerToyModel(inner_config)
            self.layer = _SameLayerToyLayer(config, layer_idx=0)

    class _ToyPreTrainedModel(PreTrainedModel):
        config_class = LlamaConfig

    class _InterleavedOuterToyModel(_ToyPreTrainedModel):
        _heterogeneous_modeling_spec = _toy_modeling_spec(_OuterToyLayer, _OuterContextMarker)

        def __init__(self, config, inner_config):
            super().__init__(config)
            self.inner_model = _InterleavedInnerToyModel(
                inner_config,
                outer_layer_factory=lambda: _OuterToyLayer(config, layer_idx=0),
            )

    class _InterleavedInnerToyModel(_ToyPreTrainedModel):
        _heterogeneous_modeling_spec = _toy_modeling_spec(_InnerToyLayer, _InnerContextMarker)

        def __init__(self, config, outer_layer_factory):
            super().__init__(config)
            # Build an outer-owned layer while the inner model context is active.
            self.outer_layer_from_inner_init = outer_layer_factory()
            self.layer = _InnerToyLayer(config, layer_idx=0)


# ──────────────────────────────────────────────────────────────────────
# Test case definitions
# ──────────────────────────────────────────────────────────────────────


class WeightCheck(NamedTuple):
    """An expected weight shape in the built model."""

    layer_idx: int
    attr_path: str  # dot-separated path to the weight tensor
    shape_dim: int
    expected: int


@dataclass
class HeteroCase:
    """A single heterogeneous modeling test scenario.

    Fields describe the model under test AND how to build a manually-constructed
    reference model (ground truth) that does not use any heterogeneity code.
    """

    name: str
    model_key: str  # key into MODEL_FIXTURES
    config_factory: callable
    model_cls: type
    per_layer_config: dict
    # --- Structure verification fields ---
    structure_weight_checks: list[WeightCheck] = field(default_factory=list)


HETERO_CASES = [
    # ── Llama ──
    HeteroCase(
        name="llama_dim",
        model_key="llama",
        config_factory=_tiny_llama_config,
        model_cls=LlamaForCausalLM,
        per_layer_config={0: {"intermediate_size": 64}, 2: {"intermediate_size": 96}},
        structure_weight_checks=[
            WeightCheck(0, "mlp.gate_proj.weight", 0, 64),
            WeightCheck(1, "mlp.gate_proj.weight", 0, 128),
            WeightCheck(2, "mlp.gate_proj.weight", 0, 96),
            WeightCheck(3, "mlp.gate_proj.weight", 0, 128),
        ],
    ),
    HeteroCase(
        name="llama_multi_attr",
        model_key="llama",
        config_factory=_tiny_llama_config,
        model_cls=LlamaForCausalLM,
        per_layer_config={
            0: {"intermediate_size": 64, "num_key_value_heads": 2},
            2: {"intermediate_size": 96, "num_key_value_heads": 1},
        },
        structure_weight_checks=[
            WeightCheck(0, "mlp.gate_proj.weight", 0, 64),
            WeightCheck(0, "self_attn.k_proj.weight", 0, 32),  # 2 kv_heads × 16
            WeightCheck(1, "mlp.gate_proj.weight", 0, 128),
            WeightCheck(1, "self_attn.k_proj.weight", 0, 64),  # 4 kv_heads × 16 (default)
            WeightCheck(2, "mlp.gate_proj.weight", 0, 96),
            WeightCheck(2, "self_attn.k_proj.weight", 0, 16),  # 1 kv_head × 16
            WeightCheck(3, "mlp.gate_proj.weight", 0, 128),
            WeightCheck(3, "self_attn.k_proj.weight", 0, 64),  # 4 kv_heads × 16 (default)
        ],
    ),
    HeteroCase(
        name="llama_skip_attn",
        model_key="llama",
        config_factory=_tiny_llama_config,
        model_cls=LlamaForCausalLM,
        per_layer_config={1: {"skip": ["attention"]}},
    ),
    HeteroCase(
        name="llama_skip_mlp",
        model_key="llama",
        config_factory=_tiny_llama_config,
        model_cls=LlamaForCausalLM,
        per_layer_config={2: {"skip": ["mlp"]}},
    ),
    HeteroCase(
        name="llama_skip_both",
        model_key="llama",
        config_factory=_tiny_llama_config,
        model_cls=LlamaForCausalLM,
        per_layer_config={1: {"skip": ["attention", "mlp"]}},
    ),
    # ── GPT-OSS ──
    HeteroCase(
        name="gpt_oss_dim",
        model_key="gpt_oss",
        config_factory=_tiny_gpt_oss_config,
        model_cls=GptOssForCausalLM,
        per_layer_config={0: {"intermediate_size": 16}, 2: {"intermediate_size": 48}},
        structure_weight_checks=[
            WeightCheck(0, "mlp.experts.down_proj", 1, 16),
            WeightCheck(1, "mlp.experts.down_proj", 1, 32),
            WeightCheck(2, "mlp.experts.down_proj", 1, 48),
            WeightCheck(3, "mlp.experts.down_proj", 1, 32),
        ],
    ),
    HeteroCase(
        name="gpt_oss_skip_attn",
        model_key="gpt_oss",
        config_factory=_tiny_gpt_oss_config,
        model_cls=GptOssForCausalLM,
        per_layer_config={1: {"skip": ["attention"]}},
    ),
    HeteroCase(
        name="gpt_oss_skip_mlp",
        model_key="gpt_oss",
        config_factory=_tiny_gpt_oss_config,
        model_cls=GptOssForCausalLM,
        per_layer_config={2: {"skip": ["mlp"]}},
    ),
    HeteroCase(
        name="gpt_oss_skip_both",
        model_key="gpt_oss",
        config_factory=_tiny_gpt_oss_config,
        model_cls=GptOssForCausalLM,
        per_layer_config={1: {"skip": ["attention", "mlp"]}},
    ),
    # ── Llama4 ──
    HeteroCase(
        name="llama4_dim",
        model_key="llama4",
        config_factory=_tiny_llama4_config,
        model_cls=Llama4ForCausalLM,
        per_layer_config={0: {"intermediate_size_mlp": 64}},
        structure_weight_checks=[
            WeightCheck(0, "feed_forward.up_proj.weight", 0, 64),
            WeightCheck(2, "feed_forward.up_proj.weight", 0, 128),  # default
        ],
    ),
    HeteroCase(
        name="llama4_skip_attn",
        model_key="llama4",
        config_factory=_tiny_llama4_config,
        model_cls=Llama4ForCausalLM,
        per_layer_config={0: {"skip": ["attention"]}},
    ),
    HeteroCase(
        name="llama4_skip_mlp",
        model_key="llama4",
        config_factory=_tiny_llama4_config,
        model_cls=Llama4ForCausalLM,
        per_layer_config={0: {"skip": ["mlp"]}},
    ),
    HeteroCase(
        name="llama4_skip_both",
        model_key="llama4",
        config_factory=_tiny_llama4_config,
        model_cls=Llama4ForCausalLM,
        per_layer_config={0: {"skip": ["attention", "mlp"]}},
    ),
    # ── NemotronH (layers: attention, mamba, moe, attention) ──
    HeteroCase(
        name="nemotron_h_dim",
        model_key="nemotron_h",
        config_factory=_tiny_nemotron_h_config,
        model_cls=NemotronHForCausalLM,
        per_layer_config={0: {"num_key_value_heads": 2}},
        structure_weight_checks=[
            WeightCheck(0, "mixer.k_proj.weight", 0, 32),  # 2 kv_heads × head_dim 16
            WeightCheck(3, "mixer.k_proj.weight", 0, 64),  # 4 kv_heads × 16 (default)
        ],
    ),
    HeteroCase(
        name="nemotron_h_skip_attn",
        model_key="nemotron_h",
        config_factory=_tiny_nemotron_h_config,
        model_cls=NemotronHForCausalLM,
        per_layer_config={0: {"skip": ["mixer"]}},
    ),
    HeteroCase(
        name="nemotron_h_skip_mamba",
        model_key="nemotron_h",
        config_factory=_tiny_nemotron_h_config,
        model_cls=NemotronHForCausalLM,
        per_layer_config={1: {"skip": ["mixer"]}},
    ),
    HeteroCase(
        name="nemotron_h_skip_moe",
        model_key="nemotron_h",
        config_factory=_tiny_nemotron_h_config,
        model_cls=NemotronHForCausalLM,
        per_layer_config={2: {"skip": ["mixer"]}},
    ),
]


def _case_name(f, n, p):
    return f"{f.__name__}_{p.args[0].name}"


def _build_hetero_and_ref(case):
    """Build a heterogeneous model (code under test) and a matching reference model (ground truth).

    The reference uses skip-aware layer subclasses that implement skip/dim natively,
    representing what the model would look like without the generic heterogeneity mechanism.
    """
    with _hetero_context(case.model_key):
        hetero = _build_model(case.config_factory(per_layer_config=case.per_layer_config), case.model_cls)

    ref = _build_reference(hetero, case.config_factory(), case.model_cls, case.model_key)
    return hetero, ref


# ──────────────────────────────────────────────────────────────────────
# Tests: Modeling
# ──────────────────────────────────────────────────────────────────────


@require_torch
class TestHeterogeneousModeling(unittest.TestCase):
    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_get_heterogeneous_modeling_spec_uses_custom_model_spec(self):
        spec = HeterogeneousModelingSpec(layer_cls=torch.nn.Linear, layer_idx_variable_name="layer_idx")

        class CustomModel:
            pass

        CustomModel.__module__ = "remote_code.modeling_custom"
        CustomModel._heterogeneous_modeling_spec = spec

        self.assertIs(get_heterogeneous_modeling_spec(CustomModel()), spec)

    def test_get_heterogeneous_modeling_spec_uses_supported_model_registry(self):
        spec = HeterogeneousModelingSpec(layer_cls=torch.nn.Linear, layer_idx_variable_name="layer_idx")

        class BuiltInModel:
            pass

        BuiltInModel.__module__ = "transformers.models.test_model.modeling_test_model"

        supported_models = importlib.import_module("transformers.heterogeneity.supported_models")
        with patch.dict(supported_models.MODEL_TO_SPEC_FACTORY, {"test_model": lambda: spec}):
            self.assertIs(get_heterogeneous_modeling_spec(BuiltInModel()), spec)

    def test_get_heterogeneous_modeling_spec_raises_for_unsupported_builtin_model(self):
        class UnsupportedModel:
            pass

        UnsupportedModel.__module__ = "transformers.models.fake.modeling_fake"

        with self.assertRaisesRegex(ValueError, "No heterogeneous modeling spec is defined for `fake`"):
            get_heterogeneous_modeling_spec(UnsupportedModel())

    @parameterized.expand(HETERO_CASES, name_func=_case_name)
    def test_structure(self, case):
        """Verify the entire model structure: skip replacements and weight shapes."""
        config = case.config_factory(per_layer_config=case.per_layer_config)
        with _hetero_context(case.model_key):
            model = _build_model(config, case.model_cls)

        # Check skip structure: compare hetero model against reference layer expectations.
        # Modules that are None in the reference (skipped) should be NoOpReplacement in the hetero model.
        ref_layer_cls = MODEL_FIXTURES[case.model_key].ref_layer_cls
        for i in range(config.num_hidden_layers):
            ref_layer = ref_layer_cls(config, layer_idx=i)
            hetero_layer = model.model.layers[i]
            for name, _ in hetero_layer.named_children():
                is_noop = type(getattr(hetero_layer, name)).__name__ == "NoOpReplacement"
                if getattr(ref_layer, name) is None:
                    self.assertTrue(is_noop, f"Layer {i}.{name} should be NoOpReplacement")
                else:
                    self.assertFalse(is_noop, f"Layer {i}.{name} should NOT be NoOpReplacement")

        # Check weight shapes on all specified layers
        for layer_idx, attr_path, shape_dim, expected in case.structure_weight_checks:
            obj = model.model.layers[layer_idx]
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            self.assertEqual(obj.shape[shape_dim], expected, f"Layer {layer_idx} {attr_path} shape mismatch")

    @parameterized.expand(HETERO_CASES, name_func=_case_name)
    def test_forward(self, case):
        """Heterogeneous model forward should match a manually-constructed reference."""
        hetero, ref = _build_hetero_and_ref(case)
        input_ids = _dummy_input_ids()
        torch.testing.assert_close(_forward_logits(hetero, input_ids), _forward_logits(ref, input_ids))

    @parameterized.expand(HETERO_CASES, name_func=_case_name)
    def test_generate(self, case):
        """Heterogeneous model generate should match a manually-constructed reference."""
        hetero, ref = _build_hetero_and_ref(case)
        input_ids = _dummy_input_ids()
        gen_kwargs = {"max_new_tokens": 4, "do_sample": False}
        self.assertTrue(
            torch.equal(
                hetero.generate(input_ids, **gen_kwargs),
                ref.generate(input_ids, **gen_kwargs),
            )
        )

    def test_layer_configs_reflect_model_init_attention_implementation(self):
        config = _tiny_llama_config(per_layer_config={0: {"intermediate_size": 64}})
        self.assertIsNone(config._attn_implementation)

        with _hetero_context("llama"):
            model = _build_model(config, LlamaForCausalLM)

        expected_attn_implementation = model.config._attn_implementation
        self.assertIsNotNone(expected_attn_implementation)
        for layer in model.model.layers:
            self.assertEqual(layer.self_attn.config._attn_implementation, expected_attn_implementation)

    def test_no_leaked_state_after_heterogeneous_init(self):
        """After heterogeneous model init, no context token should remain and non-heterogeneous models should work."""
        with _hetero_context("llama"):
            config = _tiny_llama_config(per_layer_config={1: {"skip": ["attention"]}})
            model = _build_model(config, LlamaForCausalLM)

        self.assertFalse(hasattr(model, "_layer_init_context_token"))

        # A new non-heterogeneous model should work fine
        plain_model = _build_model(_tiny_llama_config(), LlamaForCausalLM)
        _forward_logits(plain_model, _dummy_input_ids())  # should not raise

    def test_get_skip_replacement_forward(self):
        """Unit test for get_skip_replacement / ReturnEntry."""
        replacement_cls = get_skip_replacement(
            torch.nn.Linear, ReturnEntry(arg_name="input", transform=lambda x: x * 2)
        )
        module = replacement_cls()
        dummy = torch.randn(2, 4, 64)
        result = module(dummy)
        torch.testing.assert_close(result, dummy * 2)

    def test_save_pretrained_model_round_trip(self):
        """Full model save/load: config, weight shapes, and forward output should survive."""
        per_layer = {0: {"intermediate_size": 64}, 2: {"intermediate_size": 96}}
        hetero_config = _tiny_llama_config(per_layer_config=per_layer)
        with _hetero_context("llama"):
            hetero_model = _build_model(hetero_config, LlamaForCausalLM)

        input_ids = _dummy_input_ids()
        expected_logits = _forward_logits(hetero_model, input_ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            hetero_model.save_pretrained(tmpdir)
            with _hetero_context("llama"):
                loaded_model = LlamaForCausalLM.from_pretrained(tmpdir)

        loaded_model.eval()
        for i in range(4):
            orig_shape = hetero_model.model.layers[i].mlp.gate_proj.weight.shape
            loaded_shape = loaded_model.model.layers[i].mlp.gate_proj.weight.shape
            self.assertEqual(orig_shape, loaded_shape, f"Layer {i} weight shape mismatch")

        torch.testing.assert_close(_forward_logits(loaded_model, input_ids), expected_logits)

    def test_per_layer_kv_cache_shapes(self):
        """KV cache tensors should reflect per-layer num_key_value_heads after a cached forward pass."""
        config = _tiny_llama_config(per_layer_config={0: {"num_key_value_heads": 2}, 2: {"num_key_value_heads": 1}})
        with _hetero_context("llama"):
            model = _build_model(config, LlamaForCausalLM)
        input_ids = _dummy_input_ids()
        with torch.no_grad():
            cache = model(input_ids, use_cache=True).past_key_values
        # keys shape: [batch, num_heads, seq_len, head_dim]
        self.assertEqual(cache.layers[0].keys.shape[1], 2)
        self.assertEqual(cache.layers[1].keys.shape[1], 4)  # default
        self.assertEqual(cache.layers[2].keys.shape[1], 1)
        self.assertEqual(cache.layers[3].keys.shape[1], 4)  # default

    def test_error_missing_skip_descriptor(self):
        """Requesting a skip type without a matching descriptor should raise ValueError."""
        config = _tiny_llama_config(per_layer_config={1: {"skip": ["attention"]}})
        fixture = MODEL_FIXTURES["llama"]
        modeling_spec = fixture.spec_factory()
        modeling_spec = HeterogeneousModelingSpec(
            layer_cls=modeling_spec.layer_cls,
            layer_idx_variable_name=modeling_spec.layer_idx_variable_name,
            skip_descriptors={},
        )
        with patch.object(fixture.pretrained_cls, "_heterogeneous_modeling_spec", modeling_spec, create=True):
            with self.assertRaisesRegex(ValueError, "No-op descriptors are missing"):
                _build_model(config, LlamaForCausalLM)

    def test_model_with_same_layer_class_submodel_initializes_each_model_layers(self):
        """A model containing a same-layer-class submodel should initialize each model's layers correctly."""
        model = _NestedSameLayerToyModel(
            _toy_config(intermediate_size=32),
            inner_config=_toy_config(intermediate_size=64, replace_context_marker=True),
        )

        self.assertEqual(model.layer.intermediate_size, 32)
        self.assertIsInstance(model.layer.context_marker, _DefaultContextMarker)
        self.assertEqual(model.inner_model.layer.intermediate_size, 64)
        self.assertIsInstance(model.inner_model.layer.context_marker, _SameLayerContextMarker)

    def test_submodel_can_construct_outer_model_layer_during_initialization(self):
        """A submodel should be able to construct an outer-model layer during its own initialization."""
        model = _InterleavedOuterToyModel(
            _toy_config(intermediate_size=32, replace_context_marker=True),
            inner_config=_toy_config(intermediate_size=64, replace_context_marker=True),
        )

        outer_layer = model.inner_model.outer_layer_from_inner_init
        inner_layer = model.inner_model.layer

        self.assertEqual(outer_layer.intermediate_size, 32)
        self.assertIsInstance(outer_layer.context_marker, _OuterContextMarker)
        self.assertEqual(inner_layer.intermediate_size, 64)
        self.assertIsInstance(inner_layer.context_marker, _InnerContextMarker)

    def test_sequential_heterogeneous_models_no_interference(self):
        """Two heterogeneous models built sequentially should each have correct per-layer weights."""
        per_layer_a = {0: {"intermediate_size": 64}}
        per_layer_b = {0: {"intermediate_size": 96}}

        with _hetero_context("llama"):
            model_a = _build_model(_tiny_llama_config(per_layer_config=per_layer_a), LlamaForCausalLM)
            model_b = _build_model(_tiny_llama_config(per_layer_config=per_layer_b), LlamaForCausalLM, seed=123)

        self.assertEqual(model_a.model.layers[0].mlp.gate_proj.weight.shape[0], 64)
        self.assertEqual(model_b.model.layers[0].mlp.gate_proj.weight.shape[0], 96)

        input_ids = _dummy_input_ids()
        _forward_logits(model_a, input_ids)
        _forward_logits(model_b, input_ids)


# ──────────────────────────────────────────────────────────────────────
# Tests: Cache and Mask
# ──────────────────────────────────────────────────────────────────────


@require_torch
class TestHeterogeneousCacheAndMask(unittest.TestCase):
    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_dynamic_cache_heterogeneous_sliding_window(self):
        """DynamicCache should create sliding layers matching per-layer sliding_window."""
        config = _tiny_llama_config(
            sliding_window=None, per_layer_config={0: {"sliding_window": 32}, 2: {"sliding_window": 16}}
        )
        cache = DynamicCache(config=config)
        layers = cache.layers

        self.assertEqual(len(layers), 4)
        self.assertIsInstance(layers[0], DynamicSlidingWindowLayer)
        self.assertEqual(layers[0].sliding_window, 32)
        self.assertFalse(layers[1].is_sliding)
        self.assertIsInstance(layers[2], DynamicSlidingWindowLayer)
        self.assertEqual(layers[2].sliding_window, 16)
        self.assertFalse(layers[3].is_sliding)

    def test_static_cache_heterogeneous_sliding_window(self):
        """StaticCache should create sliding layers for the right layers."""
        config = _tiny_llama_config(
            sliding_window=None, per_layer_config={1: {"sliding_window": 24}, 3: {"sliding_window": 48}}
        )
        cache = StaticCache(config=config, batch_size=1, max_cache_len=64)
        layers = cache.layers

        self.assertEqual(len(layers), 4)
        self.assertFalse(layers[0].is_sliding)
        self.assertIsInstance(layers[1], StaticSlidingWindowLayer)
        # StaticSlidingWindowLayer caps max_cache_len = min(sliding_window, max_cache_len)
        self.assertEqual(layers[1].max_cache_len, 24)
        self.assertFalse(layers[2].is_sliding)
        self.assertIsInstance(layers[3], StaticSlidingWindowLayer)
        self.assertEqual(layers[3].max_cache_len, 48)

    def test_sliding_window_mask_returns_dict(self):
        """For heterogeneous sliding_window, create_sliding_window_causal_mask should return a dict."""
        config = _tiny_llama_config(
            sliding_window=None,
            per_layer_config={0: {"sliding_window": 32}, 1: {"sliding_window": 32}, 2: {"sliding_window": 16}},
        )
        config._attn_implementation = "sdpa"

        inputs_embeds = torch.randn(1, 8, 64)
        cache = DynamicCache(config=config)

        mask = create_sliding_window_causal_mask(config, inputs_embeds, attention_mask=None, past_key_values=cache)
        self.assertIsInstance(mask, dict)
        # Keyed by distinct sliding_window values (deduplication), not layer indices
        self.assertEqual(set(mask.keys()), {32, 16})

    def test_chunked_attention_mask_returns_dict(self):
        """For heterogeneous attention_chunk_size, create_chunked_causal_mask should return a dict."""
        config = _tiny_llama4_config(
            per_layer_config={
                0: {"attention_chunk_size": 16},
                2: {"attention_chunk_size": 16},
                3: {"attention_chunk_size": 8},
            },
        )
        config._attn_implementation = "sdpa"

        inputs_embeds = torch.randn(1, 8, 64)
        cache = DynamicCache(config=config)

        mask = create_chunked_causal_mask(config, inputs_embeds, attention_mask=None, past_key_values=cache)
        self.assertIsInstance(mask, dict)
        self.assertEqual(set(mask.keys()), {32, 16, 8})
