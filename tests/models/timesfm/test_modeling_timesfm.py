# Copyright 2025 Google LLC and HuggingFace Inc. team.
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
import unittest

import numpy as np
import torch
from parameterized import parameterized

from transformers import TimesFmConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION, ModelTesterMixin


if is_torch_available():
    from transformers import TimesFmModelForPrediction

TOLERANCE = 1e-4


class TimesFmModelTester:
    def __init__(
        self,
        parent,
        patch_length: int = 32,
        context_length: int = 512,
        horizon_length: int = 128,
        freq_size: int = 3,
        num_hidden_layers: int = 1,
        hidden_size: int = 16,
        intermediate_size: int = 32,
        head_dim: int = 8,
        num_heads: int = 2,
        tolerance: float = 1e-6,
        rms_norm_eps: float = 1e-6,
        quantiles: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        pad_val: float = 1123581321.0,
        use_positional_embedding: bool = True,
        initializer_factor: float = 0.0,
        is_training: bool = False,
        batch_size: int = 3,
    ):
        self.parent = parent
        self.patch_length = patch_length
        self.context_length = context_length
        self.horizon_length = horizon_length
        self.quantiles = quantiles
        self.pad_val = pad_val
        self.freq_size = freq_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_heads
        self.tolerance = tolerance
        self.rms_norm_eps = rms_norm_eps
        self.use_positional_embedding = use_positional_embedding
        self.initializer_factor = initializer_factor
        self.is_training = is_training
        self.batch_size = batch_size

        # The size of test input
        self.seq_length = context_length // patch_length
        self.hidden_size = hidden_size

    def get_config(self):
        return TimesFmConfig(
            patch_length=self.patch_length,
            context_length=self.context_length,
            horizon_length=self.horizon_length,
            quantiles=self.quantiles,
            pad_val=self.pad_val,
            freq_size=self.freq_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            head_dim=self.head_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            tolerance=self.tolerance,
            rms_norm_eps=self.rms_norm_eps,
            use_positional_embedding=self.use_positional_embedding,
            initializer_factor=self.initializer_factor,
        )

    def get_pipeline_config(self):
        return self.get_config()

    def prepare_config_and_inputs(self):
        forecast_input = [
            torch.tensor(np.sin(np.linspace(0, 20, 100)), dtype=torch.float32, device=torch_device),
            torch.tensor(np.cos(np.linspace(0, 20, 100)), dtype=torch.float32, device=torch_device),
            torch.tensor(np.tan(np.linspace(0, 20, 100)), dtype=torch.float32, device=torch_device),
        ]
        frequency_input = torch.tensor([0, 1, 2], dtype=torch.long, device=torch_device)

        return (self.get_config(), torch.stack(forecast_input, dim=0), frequency_input)

    def prepare_config_and_inputs_for_common(self):
        (config, forecast_input, frequency_input) = self.prepare_config_and_inputs()

        inputs_dict = {
            "past_values": forecast_input,
            "freq": frequency_input,
        }
        return config, inputs_dict


@require_torch
class TimesFmModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (TimesFmModelForPrediction,) if is_torch_available() else ()
    all_generative_model_classes = ()

    test_resize_embeddings = False
    is_encoder_decoder = False
    test_inputs_embeds = False
    test_torch_exportable = False

    def setUp(self):
        self.model_tester = TimesFmModelTester(self)
        self.config_tester = ConfigTester(self, config_class=TimesFmConfig)

    def test_create_and_run_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = TimesFmModelForPrediction(config)
        model.to(torch_device)
        model.eval()
        results = model(**inputs_dict)
        assert results.mean_predictions is not None

    @unittest.skip(reason="Compile not yet supported because of masks")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    def test_eager_matches_sdpa_inference(
        self, name, dtype, padding_side, use_attention_mask, output_attentions, enable_kernels
    ):
        """TimesFM computes its own attention mask internally, so the generic test
        (which injects external masks) is not compatible. This override directly
        verifies eager vs SDPA equivalence on model outputs."""
        if not self.all_model_classes[0]._supports_sdpa:
            self.skipTest("Model does not support SDPA")

        if dtype == "fp16":
            dtype = torch.float16
        elif dtype == "bf16":
            dtype = torch.bfloat16
        elif dtype == "fp32":
            dtype = torch.float32

        tolerance = {torch.float32: 1e-5, torch.bfloat16: 1e-3, torch.float16: 1e-3}[dtype]

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True

        model_eager = TimesFmModelForPrediction._from_config(config, attn_implementation="eager")
        model_eager.to(dtype=dtype, device=torch_device)
        model_eager.eval()

        model_sdpa = TimesFmModelForPrediction._from_config(config, attn_implementation="sdpa")
        model_sdpa.load_state_dict(model_eager.state_dict())
        model_sdpa.to(dtype=dtype, device=torch_device)
        model_sdpa.eval()

        past_values = inputs_dict["past_values"].to(dtype=dtype, device=torch_device)
        freq = inputs_dict["freq"].to(device=torch_device)

        with torch.no_grad():
            out_eager = model_eager(past_values=past_values, freq=freq)
            out_sdpa = model_sdpa(past_values=past_values, freq=freq)

        self.assertTrue(
            torch.allclose(out_eager.mean_predictions, out_sdpa.mean_predictions, atol=tolerance),
            f"mean_predictions max diff: {(out_eager.mean_predictions - out_sdpa.mean_predictions).abs().max().item():.2e}",
        )
        self.assertTrue(
            torch.allclose(out_eager.full_predictions, out_sdpa.full_predictions, atol=tolerance),
            f"full_predictions max diff: {(out_eager.full_predictions - out_sdpa.full_predictions).abs().max().item():.2e}",
        )
        hs_eager = out_eager.hidden_states[-1]
        hs_sdpa = out_sdpa.hidden_states[-1]
        self.assertTrue(
            torch.allclose(hs_eager, hs_sdpa, atol=tolerance),
            f"hidden_states max diff: {(hs_eager - hs_sdpa).abs().max().item():.2e}",
        )

    @unittest.skip(reason="Model does not have input embeddings")
    def test_model_get_set_embeddings(self):
        pass

    # the main input name is `inputs`
    def test_model_main_input_name(self):
        model_signature = inspect.signature(getattr(TimesFmModelForPrediction, "forward"))
        # The main input is the name of the argument after `self`
        observed_main_input_name = list(model_signature.parameters.keys())[1]
        self.assertEqual(TimesFmModelForPrediction.main_input_name, observed_main_input_name)


@require_torch
class TimesFmForwardInputVariantsTest(unittest.TestCase):
    def setUp(self):
        config = TimesFmConfig(
            patch_length=32,
            context_length=64,
            horizon_length=32,
            hidden_size=16,
            intermediate_size=32,
            head_dim=8,
            num_hidden_layers=1,
            num_attention_heads=2,
        )
        self.model = TimesFmModelForPrediction(config).to(torch_device).eval()
        self.horizon_len = config.horizon_length

    def test_different_length_series(self):
        """forward() handles a list of series with different lengths."""
        inputs = [
            torch.randn(20, device=torch_device),
            torch.randn(50, device=torch_device),
            torch.randn(100, device=torch_device),
        ]
        with torch.no_grad():
            out = self.model(past_values=inputs, freq=[0, 0, 0])
        self.assertEqual(out.mean_predictions.shape, (3, self.horizon_len))

    def test_very_short_and_very_long_series(self):
        """forward() works when one series is tiny and another exceeds context_len."""
        inputs = [
            torch.randn(5, device=torch_device),
            torch.randn(500, device=torch_device),
        ]
        with torch.no_grad():
            out = self.model(past_values=inputs, freq=[0, 0])
        self.assertEqual(out.mean_predictions.shape, (2, self.horizon_len))

    def test_2d_tensor_input(self):
        """forward() accepts a 2D tensor and produces correct output shape."""
        inputs = torch.randn(4, 80, device=torch_device)
        with torch.no_grad():
            out = self.model(past_values=inputs, freq=[0, 0, 0, 0])
        self.assertEqual(out.mean_predictions.shape, (4, self.horizon_len))

    def test_list_vs_tensor_parity(self):
        """forward() with list and 2D tensor of equal-length series gives identical output."""
        raw = [torch.randn(50, device=torch_device) for _ in range(2)]
        stacked = torch.stack(raw)
        freq = [0, 0]
        with torch.no_grad():
            out_list = self.model(past_values=raw, freq=freq)
            out_tensor = self.model(past_values=stacked, freq=freq)
        self.assertTrue(torch.allclose(out_list.mean_predictions, out_tensor.mean_predictions, atol=1e-5))
        self.assertTrue(torch.allclose(out_list.full_predictions, out_tensor.full_predictions, atol=1e-5))

    def test_long_series_truncated(self):
        """forward() with a long series produces the same output as passing only the tail."""
        long_series = torch.randn(500, device=torch_device)
        tail = long_series[-64:]
        with torch.no_grad():
            out_long = self.model(past_values=[long_series], freq=[0])
            out_tail = self.model(past_values=[tail], freq=[0])
        self.assertTrue(torch.allclose(out_long.mean_predictions, out_tail.mean_predictions, atol=1e-5))

    def test_truncate_negative_with_positive_input(self):
        """truncate_negative clamps outputs to zero when all inputs are non-negative."""
        inputs = torch.rand(2, 80, device=torch_device).abs() + 1.0
        with torch.no_grad():
            out = self.model(past_values=inputs, freq=[0, 0], truncate_negative=True)
        self.assertTrue((out.mean_predictions >= 0).all())
        self.assertTrue((out.full_predictions >= 0).all())

    def test_truncate_negative_with_negative_input(self):
        """truncate_negative leaves outputs untouched when inputs contain negatives."""
        inputs = torch.randn(2, 80, device=torch_device) - 5.0
        with torch.no_grad():
            out_trunc = self.model(past_values=inputs, freq=[0, 0], truncate_negative=True)
            out_plain = self.model(past_values=inputs, freq=[0, 0], truncate_negative=False)
        self.assertTrue(torch.allclose(out_trunc.mean_predictions, out_plain.mean_predictions))
        self.assertTrue(torch.allclose(out_trunc.full_predictions, out_plain.full_predictions))

    def test_onnx_export_and_inference(self):
        """Export to ONNX, verify dynamic batch and truncate_negative both work."""
        try:
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnxruntime not installed")

        import tempfile

        from torch.export import Dim

        class Wrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, past_values, freq):
                o = self.m(past_values, freq=freq, truncate_negative=True)
                return o.mean_predictions, o.full_predictions

        wrapped = Wrapper(self.model).cpu().eval()
        export_input = torch.randn(2, 80)
        export_freq = torch.zeros(2, 1, dtype=torch.int32)
        batch = Dim("batch", min=1, max=64)
        seq = Dim("seq", min=1, max=512)
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/model.onnx"
            torch.onnx.export(
                wrapped,
                (export_input, export_freq),
                path,
                input_names=["past_values", "freq"],
                output_names=["mean_predictions", "full_predictions"],
                dynamo=True,
                dynamic_shapes={
                    "past_values": {0: batch, 1: seq},
                    "freq": {0: batch},
                },
            )
            import onnx

            onnx_model = onnx.load(path, load_external_data=False)
            op_types = {n.op_type for n in onnx_model.graph.node}

            # 1. Dynamic dims: input batch & seq must be symbolic strings, not fixed ints
            inp = onnx_model.graph.input[0]
            dims = [d.dim_param or d.dim_value for d in inp.type.tensor_type.shape.dim]
            self.assertIsInstance(dims[0], str, f"batch dim should be symbolic, got {dims[0]}")
            self.assertIsInstance(dims[1], str, f"seq dim should be symbolic, got {dims[1]}")
            for out in onnx_model.graph.output:
                out_batch = out.type.tensor_type.shape.dim[0]
                self.assertTrue(out_batch.dim_param, f"output '{out.name}' batch dim not dynamic")

            # 2. No If nodes: all Python branches (forecast_context_len, window_size,
            #    freq is None, return_forecast_on_context, future_values) must be
            #    frozen at export time, not traced as conditional ops
            if_nodes = [n.name for n in onnx_model.graph.node if n.op_type == "If"]
            self.assertEqual(len(if_nodes), 0, f"Graph has If nodes (unfrozen branches): {if_nodes}")

            # 3. truncate_negative: the inp_min >= 0 check must be branchless (torch.where),
            #    so we expect a Where op in the graph instead of an If
            self.assertIn("Where", op_types, "Missing Where op — truncate_negative not branchless")

            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

            # (a) different batch size AND seq length to verify both dims are dynamic
            diff_input = torch.randn(3, 50)
            diff_freq = torch.zeros(3, 1, dtype=torch.int32)
            with torch.no_grad():
                pt_out = self.model(past_values=diff_input, freq=diff_freq, truncate_negative=True)
            onnx_mean, onnx_full = sess.run(
                None, {"past_values": diff_input.numpy(), "freq": diff_freq.numpy()}
            )
            np.testing.assert_allclose(onnx_mean, pt_out.mean_predictions.numpy(), rtol=1e-3, atol=1e-3)
            np.testing.assert_allclose(onnx_full, pt_out.full_predictions.numpy(), rtol=1e-3, atol=1e-3)

            # (b) all-positive input triggers the truncate_negative clamp path
            pos_input = torch.rand(2, 80).abs() + 1.0
            pos_freq = torch.zeros(2, 1, dtype=torch.int32)
            with torch.no_grad():
                pt_pos = self.model(past_values=pos_input, freq=pos_freq, truncate_negative=True)
            onnx_mean_pos, onnx_full_pos = sess.run(
                None, {"past_values": pos_input.numpy(), "freq": pos_freq.numpy()}
            )
            np.testing.assert_allclose(onnx_mean_pos, pt_pos.mean_predictions.numpy(), rtol=1e-3, atol=1e-3)
            self.assertTrue((onnx_mean_pos >= 0).all())

            # (c) freq=None path: wrapper does not pass freq, so `if freq is None` runs. Using
            # `[0] * past_values.shape[0]` there bakes batch into the graph; this block fails
            # if that regression returns. The (a)/(b) paths above never hit freq=None.
            class WrapperNoFreq(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m

                def forward(self, past_values):
                    o = self.m(past_values, truncate_negative=True)
                    return o.mean_predictions, o.full_predictions

            path_nf = f"{tmp}/model_no_freq.onnx"
            torch.onnx.export(
                WrapperNoFreq(self.model).cpu().eval(),
                (export_input,),
                path_nf,
                input_names=["past_values"],
                output_names=["mean_predictions", "full_predictions"],
                dynamo=True,
                dynamic_shapes={"past_values": {0: batch, 1: seq}},
            )
            onnx_nf = onnx.load(path_nf, load_external_data=False)
            inp_nf = onnx_nf.graph.input[0]
            dims_nf = [d.dim_param or d.dim_value for d in inp_nf.type.tensor_type.shape.dim]
            self.assertIsInstance(dims_nf[0], str, f"no-freq export: batch dim should be symbolic, got {dims_nf[0]}")
            sess_nf = ort.InferenceSession(path_nf, providers=["CPUExecutionProvider"])
            nf_input = torch.randn(3, 50)
            with torch.no_grad():
                pt_nf = self.model(past_values=nf_input, truncate_negative=True)
            onnx_m_nf, onnx_f_nf = sess_nf.run(None, {"past_values": nf_input.numpy()})
            np.testing.assert_allclose(onnx_m_nf, pt_nf.mean_predictions.numpy(), rtol=1e-3, atol=1e-3)
            np.testing.assert_allclose(onnx_f_nf, pt_nf.full_predictions.numpy(), rtol=1e-3, atol=1e-3)

    def test_onnx_export_with_forecast_context_len(self):
        """Export with forecast_context_len baked in; verify ONNX uses truncated context."""
        try:
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnxruntime not installed")

        import tempfile

        from torch.export import Dim

        short_ctx = 32

        class Wrapper(torch.nn.Module):
            def __init__(self, m, ctx):
                super().__init__()
                self.m = m
                self.ctx = ctx

            def forward(self, past_values):
                o = self.m(past_values, forecast_context_len=self.ctx)
                return o.mean_predictions, o.full_predictions

        wrapped = Wrapper(self.model, short_ctx).cpu().eval()
        export_input = torch.randn(2, 80)
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/model.onnx"
            batch = Dim("batch", min=1, max=64)
            seq = Dim("seq", min=1, max=512)
            torch.onnx.export(
                wrapped,
                (export_input,),
                path,
                input_names=["past_values"],
                output_names=["mean_predictions", "full_predictions"],
                dynamo=True,
                dynamic_shapes={"past_values": {0: batch, 1: seq}},
            )
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

            # ONNX graph has forecast_context_len=32 baked in, so passing 80 values
            # should give the same result as PyTorch with the same override
            test_input = torch.randn(2, 80)
            with torch.no_grad():
                pt_out = self.model(past_values=test_input, forecast_context_len=short_ctx)
            onnx_mean, onnx_full = sess.run(None, {"past_values": test_input.numpy()})
            np.testing.assert_allclose(onnx_mean, pt_out.mean_predictions.numpy(), rtol=1e-3, atol=1e-3)
            np.testing.assert_allclose(onnx_full, pt_out.full_predictions.numpy(), rtol=1e-3, atol=1e-3)


@require_torch
@slow
class TimesFmModelIntegrationTests(unittest.TestCase):
    def test_inference(self):
        model = TimesFmModelForPrediction.from_pretrained("google/timesfm-2.0-500m-pytorch").to(torch_device)
        forecast_input = [
            np.sin(np.linspace(0, 20, 100)),
            np.sin(np.linspace(0, 20, 200)),
            np.sin(np.linspace(0, 20, 400)),
        ]
        forecast_input_tensor = [torch.tensor(ts, dtype=torch.float32, device=torch_device) for ts in forecast_input]
        frequency_input = [0, 1, 2]

        with torch.no_grad():
            output = model(past_values=forecast_input_tensor, freq=frequency_input)

        mean_predictions = output.mean_predictions
        self.assertEqual(mean_predictions.shape, torch.Size([3, model.config.horizon_length]))
        # fmt: off
        expected_slice = torch.tensor(
            [ 0.9813,  1.0086,  0.9985,  0.9432,  0.8505,  0.7203,  0.5596,  0.3788,
              0.1796, -0.0264, -0.2307, -0.4255, -0.5978, -0.7642, -0.8772, -0.9670,
             -1.0110, -1.0162, -0.9848, -0.9151, -0.8016, -0.6511, -0.4707, -0.2842,
             -0.0787,  0.1260,  0.3293,  0.5104,  0.6818,  0.8155,  0.9172,  0.9843,
              1.0101,  1.0025,  0.9529,  0.8588,  0.7384,  0.5885,  0.4022,  0.2099,
             -0.0035, -0.2104, -0.4146, -0.6033, -0.7661, -0.8818, -0.9725, -1.0191,
             -1.0190, -0.9874, -0.9137, -0.8069, -0.6683, -0.4939, -0.3086, -0.1106,
              0.0846,  0.2927,  0.4832,  0.6612,  0.8031,  0.9051,  0.9772,  1.0064
            ],
            device=torch_device)
        # fmt: on
        self.assertTrue(torch.allclose(mean_predictions[0, :64], expected_slice, atol=TOLERANCE))
