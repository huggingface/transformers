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

from transformers import TimesFmConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin


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


@require_torch
class TimesFmCovariatesTest(unittest.TestCase):
    """Test TimesFM covariates functionality."""

    def setUp(self):
        self.model_tester = TimesFmModelTester(
            self,
            patch_length=32,
            context_length=128,
            horizon_length=32,
            num_hidden_layers=1,
            hidden_size=16,
            intermediate_size=32,
            batch_size=2,
        )
        self.config = self.model_tester.get_config()
        self.model = TimesFmModelForPrediction(self.config).to(torch_device)
        self.model.eval()

        # Create test data with consistent lengths
        self.context_len = 60  # Use a fixed context length
        self.horizon_len = 16
        self.past_values = [
            torch.tensor(np.sin(np.linspace(0, 10, self.context_len)), dtype=torch.float32, device=torch_device),
            torch.tensor(np.cos(np.linspace(0, 10, self.context_len)), dtype=torch.float32, device=torch_device),
        ]
        self.total_len = self.context_len + self.horizon_len

    def _create_test_covariates(self):
        """Create comprehensive test covariates."""
        # Dynamic numerical covariates
        dynamic_numerical = {
            "temperature": [
                (20 + 5 * np.sin(2 * np.pi * np.arange(self.total_len) / 10)).tolist(),
                (25 + 3 * np.cos(2 * np.pi * np.arange(self.total_len) / 8)).tolist(),
            ],
            "humidity": [
                (60 + np.random.RandomState(42).randn(self.total_len) * 2).tolist(),
                (55 + np.random.RandomState(43).randn(self.total_len) * 3).tolist(),
            ],
        }

        # Dynamic categorical covariates
        dynamic_categorical = {
            "weekday": [
                [i % 7 for i in range(self.total_len)],
                [(i + 1) % 7 for i in range(self.total_len)],
            ],
            "season": [
                [["spring", "summer", "fall", "winter"][i % 4] for i in range(self.total_len)],
                [["spring", "summer", "fall", "winter"][i % 4] for i in range(self.total_len)],
            ],
        }

        # Static covariates
        static_numerical = {
            "store_size": [100.0, 150.0],
            "avg_income": [50000.0, 60000.0],
        }

        static_categorical = {
            "store_type": ["supermarket", "convenience"],
            "region": ["north", "south"],
        }

        return {
            "dynamic_numerical_covariates": dynamic_numerical,
            "dynamic_categorical_covariates": dynamic_categorical,
            "static_numerical_covariates": static_numerical,
            "static_categorical_covariates": static_categorical,
        }

    def test_forecast_with_covariates_basic_functionality(self):
        """Test basic covariates functionality."""
        covariates = self._create_test_covariates()

        with torch.no_grad():
            output = self.model.forecast_with_covariates(
                past_values=self.past_values,
                ridge=0.5,  # Use higher ridge for test stability
                **covariates,
            )

        # Check output structure
        self.assertTrue(hasattr(output, "combined_predictions"))
        self.assertTrue(hasattr(output, "xreg_predictions"))
        self.assertTrue(hasattr(output, "mean_predictions"))

        # Check tensor shapes
        batch_size = len(self.past_values)
        expected_shape = torch.Size([batch_size, self.horizon_len])

        self.assertEqual(output.combined_predictions.shape, expected_shape)
        self.assertEqual(output.xreg_predictions.shape, expected_shape)
        self.assertTrue(output.mean_predictions.shape[0] == batch_size)

        # Check that predictions are finite
        self.assertTrue(torch.isfinite(output.combined_predictions).all())
        self.assertTrue(torch.isfinite(output.xreg_predictions).all())
        self.assertTrue(torch.isfinite(output.mean_predictions).all())

    def test_forecast_with_covariates_both_modes(self):
        """Test both XReg modes."""
        covariates = self._create_test_covariates()

        for mode in ["xreg + timesfm", "timesfm + xreg"]:
            with self.subTest(mode=mode):
                with torch.no_grad():
                    output = self.model.forecast_with_covariates(
                        past_values=self.past_values, xreg_mode=mode, ridge=0.5, **covariates
                    )

                # Both modes should produce valid outputs
                self.assertTrue(torch.isfinite(output.combined_predictions).all())
                self.assertTrue(torch.isfinite(output.xreg_predictions).all())

                # Check shapes are consistent
                batch_size = len(self.past_values)
                expected_shape = torch.Size([batch_size, self.horizon_len])
                self.assertEqual(output.combined_predictions.shape, expected_shape)

    def test_forecast_with_covariates_individual_types(self):
        """Test individual covariate types."""
        test_cases = [
            {
                "name": "dynamic_numerical_only",
                "covariates": {
                    "dynamic_numerical_covariates": self._create_test_covariates()["dynamic_numerical_covariates"]
                },
            },
            {
                "name": "dynamic_categorical_only",
                "covariates": {
                    "dynamic_categorical_covariates": self._create_test_covariates()["dynamic_categorical_covariates"]
                },
            },
            {
                "name": "static_numerical_only",
                "covariates": {
                    "static_numerical_covariates": self._create_test_covariates()["static_numerical_covariates"]
                },
            },
            {
                "name": "static_categorical_only",
                "covariates": {
                    "static_categorical_covariates": self._create_test_covariates()["static_categorical_covariates"]
                },
            },
        ]

        for test_case in test_cases:
            with self.subTest(covariate_type=test_case["name"]):
                with torch.no_grad():
                    output = self.model.forecast_with_covariates(
                        past_values=self.past_values,
                        ridge=1.0,  # Higher ridge for stability with fewer covariates
                        **test_case["covariates"],
                    )

                # All individual types should work
                self.assertTrue(torch.isfinite(output.combined_predictions).all())
                self.assertTrue(torch.isfinite(output.xreg_predictions).all())

    def test_forecast_with_covariates_error_handling(self):
        """Test error handling for invalid inputs."""

        # Test no covariates provided
        with self.assertRaises(ValueError) as context:
            self.model.forecast_with_covariates(past_values=self.past_values)
        self.assertIn("At least one of", str(context.exception))

        # Test invalid xreg_mode
        with self.assertRaises(ValueError) as context:
            self.model.forecast_with_covariates(
                past_values=self.past_values,
                static_numerical_covariates={"test": [1.0, 2.0]},
                xreg_mode="invalid_mode",
            )
        self.assertIn("xreg_mode must be", str(context.exception))

        # Test horizon too long
        long_covariates = {
            "dynamic_numerical_covariates": {
                "test": [
                    list(range(len(self.past_values[0]) + 1000)),  # Much longer than model horizon
                    list(range(len(self.past_values[1]) + 1000)),
                ]
            }
        }
        with self.assertRaises(ValueError) as context:
            self.model.forecast_with_covariates(past_values=self.past_values, **long_covariates)
        self.assertIn("exceeds model horizon", str(context.exception))

    def test_forecast_with_covariates_ridge_regularization(self):
        """Test different ridge regularization values."""
        covariates = self._create_test_covariates()
        ridge_values = [0.0, 0.1, 1.0, 10.0]

        for ridge in ridge_values:
            with self.subTest(ridge=ridge):
                with torch.no_grad():
                    output = self.model.forecast_with_covariates(
                        past_values=self.past_values, ridge=ridge, **covariates
                    )

                # All ridge values should produce finite outputs
                self.assertTrue(torch.isfinite(output.combined_predictions).all())
                self.assertTrue(torch.isfinite(output.xreg_predictions).all())

    def test_forecast_with_covariates_normalization(self):
        """Test normalization option."""
        covariates = self._create_test_covariates()

        for normalize in [True, False]:
            with self.subTest(normalize=normalize):
                with torch.no_grad():
                    output = self.model.forecast_with_covariates(
                        past_values=self.past_values,
                        normalize_xreg_target_per_input=normalize,
                        ridge=0.5,
                        **covariates,
                    )

                # Both options should work
                self.assertTrue(torch.isfinite(output.combined_predictions).all())
                self.assertTrue(torch.isfinite(output.xreg_predictions).all())

    def test_forecast_with_covariates_truncate_negative(self):
        """Test negative value truncation."""
        # Create positive-only past values
        positive_past_values = [torch.abs(ts) + 1.0 for ts in self.past_values]
        covariates = self._create_test_covariates()

        with torch.no_grad():
            output = self.model.forecast_with_covariates(
                past_values=positive_past_values, truncate_negative=True, ridge=0.5, **covariates
            )

        # Check that outputs are non-negative when truncate_negative=True
        self.assertTrue((output.combined_predictions >= 0).all())
        self.assertTrue((output.xreg_predictions >= 0).all())

    def test_forecast_with_covariates_variable_lengths(self):
        """Test with variable sequence lengths."""
        # Create sequences of different lengths
        var_past_values = [
            torch.tensor(np.sin(np.linspace(0, 5, 30)), dtype=torch.float32, device=torch_device),
            torch.tensor(np.cos(np.linspace(0, 8, 45)), dtype=torch.float32, device=torch_device),
        ]

        # Adjust covariates for variable lengths
        max_context = max(len(ts) for ts in var_past_values)
        total_len = max_context + self.horizon_len

        covariates = {
            "dynamic_numerical_covariates": {
                "feature1": [
                    np.random.RandomState(42).randn(total_len).tolist(),
                    np.random.RandomState(43).randn(total_len).tolist(),
                ]
            },
            "static_categorical_covariates": {"category": ["A", "B"]},
        }

        with torch.no_grad():
            output = self.model.forecast_with_covariates(past_values=var_past_values, ridge=1.0, **covariates)

        # Should handle variable lengths correctly
        self.assertTrue(torch.isfinite(output.combined_predictions).all())
        self.assertTrue(torch.isfinite(output.xreg_predictions).all())

    def test_forecast_with_covariates_return_dict(self):
        """Test return_dict parameter."""
        covariates = self._create_test_covariates()

        # Test return_dict=True (default)
        with torch.no_grad():
            output_dict = self.model.forecast_with_covariates(
                past_values=self.past_values, return_dict=True, ridge=0.5, **covariates
            )

        self.assertTrue(hasattr(output_dict, "combined_predictions"))
        self.assertTrue(hasattr(output_dict, "xreg_predictions"))

        # Test return_dict=False
        with torch.no_grad():
            output_tuple = self.model.forecast_with_covariates(
                past_values=self.past_values, return_dict=False, ridge=0.5, **covariates
            )

        self.assertIsInstance(output_tuple, tuple)
        self.assertTrue(len(output_tuple) > 0)

    def test_forecast_with_covariates_device_consistency(self):
        """Test that outputs are on the correct device."""
        covariates = self._create_test_covariates()

        with torch.no_grad():
            output = self.model.forecast_with_covariates(past_values=self.past_values, ridge=0.5, **covariates)

        # All outputs should be on the same device as the model
        expected_device = next(self.model.parameters()).device
        self.assertEqual(output.combined_predictions.device, expected_device)
        self.assertEqual(output.xreg_predictions.device, expected_device)
        self.assertEqual(output.mean_predictions.device, expected_device)

    def test_forecast_with_covariates_realistic_example(self):
        """Test with realistic ice cream/sunscreen sales data similar to covariates.ipynb."""
        # Based on the ice cream and sunscreen sales example from covariates.ipynb
        batch_size = 2
        context_len = 50
        horizon_len = 10

        # Create realistic time series (ice cream and sunscreen sales)
        np.random.seed(42)
        time_points = np.arange(context_len)

        # Ice cream sales: higher in summer, affected by temperature
        seasonal_pattern = 50 + 30 * np.sin(2 * np.pi * time_points / 12 - np.pi / 2)
        ice_cream_sales = seasonal_pattern + np.random.randn(context_len) * 5

        # Sunscreen sales: also seasonal but different pattern
        seasonal_pattern2 = 40 + 25 * np.sin(2 * np.pi * time_points / 12)
        sunscreen_sales = seasonal_pattern2 + np.random.randn(context_len) * 4

        past_values = [
            torch.tensor(ice_cream_sales, dtype=torch.float32, device=torch_device),
            torch.tensor(sunscreen_sales, dtype=torch.float32, device=torch_device),
        ]

        # Create realistic covariates
        total_len = context_len + horizon_len

        # Temperature covariate - main driver
        temperature = 20 + 15 * np.sin(2 * np.pi * np.arange(total_len) / 12) + np.random.randn(total_len) * 2

        # Day of week effect
        weekday_pattern = np.tile([0, 1, 2, 3, 4, 5, 6], (total_len // 7) + 1)[:total_len]

        # Promotion effect (binary)
        promotion = np.random.choice([0, 1], size=total_len, p=[0.8, 0.2])

        dynamic_numerical = {
            "temperature": [temperature.tolist(), temperature.tolist()],
            "promotion": [promotion.tolist(), promotion.tolist()],
        }

        dynamic_categorical = {"weekday": [weekday_pattern.tolist(), weekday_pattern.tolist()]}

        static_numerical = {
            "store_size": [1000.0, 800.0]  # sq ft
        }

        static_categorical = {"store_type": ["mall", "street"], "region": ["north", "south"]}

        # Test both modes
        for xreg_mode in ["xreg + timesfm", "timesfm + xreg"]:
            with torch.no_grad():
                output = self.model.forecast_with_covariates(
                    past_values=past_values,
                    dynamic_numerical_covariates=dynamic_numerical,
                    dynamic_categorical_covariates=dynamic_categorical,
                    static_numerical_covariates=static_numerical,
                    static_categorical_covariates=static_categorical,
                    xreg_mode=xreg_mode,
                    ridge=0.1,
                )

            # Validate realistic predictions
            self.assertEqual(output.combined_predictions.shape, (batch_size, horizon_len))
            self.assertEqual(output.xreg_predictions.shape, (batch_size, horizon_len))
            self.assertEqual(output.mean_predictions.shape, (batch_size, horizon_len))

            # Ensure finite predictions (main technical requirement)
            self.assertTrue(torch.isfinite(output.combined_predictions).all())
            self.assertTrue(torch.isfinite(output.xreg_predictions).all())

            # Predictions should not be extreme values (reasonable sanity check)
            self.assertTrue(torch.abs(output.combined_predictions).max() < 1e6)  # Avoid extreme values

    def test_forecast_with_covariates_epf_style_data(self):
        """Test with EPF (Electricity Price Forecasting) style data like in covariates.ipynb."""
        # Based on EPF example from covariates.ipynb
        batch_size = 3  # 3 different market regions
        context_len = 48  # 48 hours of historical data
        horizon_len = 24  # 24 hour forecast

        # Create realistic electricity price data with daily patterns
        np.random.seed(123)

        past_values = []
        for region in range(batch_size):
            time_points = np.arange(context_len)

            # Daily pattern: higher during day, lower at night
            daily_pattern = 50 + 20 * np.sin(2 * np.pi * time_points / 24)
            # Weekly pattern: higher on weekdays
            weekly_pattern = 5 * np.sin(2 * np.pi * time_points / (24 * 7))
            # Regional base price
            regional_base = 40 + region * 10
            # Random noise
            noise = np.random.randn(context_len) * 5

            prices = regional_base + daily_pattern + weekly_pattern + noise
            past_values.append(torch.tensor(prices, dtype=torch.float32, device=torch_device))

        # EPF-style covariates
        total_len = context_len + horizon_len

        # Load covariates (MW) - main driver for electricity prices
        base_load = 1000 + 300 * np.sin(2 * np.pi * np.arange(total_len) / 24)
        load_variation = np.random.randn(total_len) * 50

        dynamic_numerical = {
            "load_mw": [(base_load + load_variation + i * 100).tolist() for i in range(batch_size)],
            "temperature": [
                (
                    20 + 10 * np.sin(2 * np.pi * np.arange(total_len) / (24 * 30)) + np.random.randn(total_len) * 3
                ).tolist()
                for _ in range(batch_size)
            ],
            "renewable_share": [
                np.clip(0.3 + 0.2 * np.random.randn(total_len), 0.1, 0.8).tolist() for _ in range(batch_size)
            ],
        }

        dynamic_categorical = {
            "hour": [[i % 24 for i in range(total_len)] for _ in range(batch_size)],
            "day_type": [
                ["weekday" if (i // 24) % 7 < 5 else "weekend" for i in range(total_len)] for _ in range(batch_size)
            ],
        }

        static_numerical = {
            "market_capacity_mw": [5000.0, 4500.0, 6000.0],
            "transmission_capacity": [800.0, 700.0, 900.0],
        }

        static_categorical = {
            "market_type": ["competitive", "regulated", "competitive"],
            "primary_fuel": ["gas", "coal", "nuclear"],
        }

        # Test with higher ridge for stability with many covariates
        with torch.no_grad():
            output = self.model.forecast_with_covariates(
                past_values=past_values,
                dynamic_numerical_covariates=dynamic_numerical,
                dynamic_categorical_covariates=dynamic_categorical,
                static_numerical_covariates=static_numerical,
                static_categorical_covariates=static_categorical,
                xreg_mode="xreg + timesfm",
                ridge=0.5,  # Higher ridge for stability
            )

        # Validate EPF-style predictions
        self.assertEqual(output.combined_predictions.shape, (batch_size, horizon_len))

        # Electricity prices should be positive
        self.assertTrue((output.combined_predictions > 0).all())
        self.assertTrue((output.xreg_predictions > 0).all())

        # Should be in reasonable range for electricity prices (0-500 $/MWh)
        self.assertTrue((output.combined_predictions < 500).all())

        # Predictions should be finite
        self.assertTrue(torch.isfinite(output.combined_predictions).all())
        self.assertTrue(torch.isfinite(output.xreg_predictions).all())

        # Test that covariates model provides useful signal
        # XReg predictions should capture some of the load-price relationship
        mean_price = output.combined_predictions.mean()
        self.assertTrue(20 < mean_price < 200)  # Reasonable electricity price range

    def test_covariates_training_backward(self):
        """Ensure loss computes and gradients flow for covariate training."""
        covariates = self._create_test_covariates()

        # Fresh small model for training step
        model = TimesFmModelForPrediction(self.config).to(torch_device)
        model.train()

        # Future values matching the covariate-driven horizon per series
        future_values = torch.zeros(len(self.past_values), self.horizon_len, dtype=torch.float32, device=torch_device)

        # Use residual training path (xreg + timesfm) by default
        output = model.forecast_with_covariates(
            past_values=self.past_values,
            future_values=future_values,
            ridge=0.1,
            **covariates,
        )

        self.assertIsNotNone(output.loss)
        # Backward pass should produce non-zero gradients on some parameters
        output.loss.backward()

        total_grad = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_grad += float(p.grad.detach().abs().sum().item())

        self.assertGreater(total_grad, 0.0)
