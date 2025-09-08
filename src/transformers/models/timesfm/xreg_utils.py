# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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
"""Helper utilities for TimesFM covariates and in-context regression."""

import itertools
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Optional, Union

import numpy as np
import torch
from sklearn import preprocessing


Category = Union[int, str]
XRegMode = Literal["timesfm + xreg", "xreg + timesfm"]

_TOL = 1e-6


def _unnest(nested: Sequence[Sequence[Any]]) -> np.ndarray:
    """Flatten a nested sequence into a 1D numpy array."""
    return np.array(list(itertools.chain.from_iterable(nested)))


def _repeat(elements: Sequence[Any], counts: Sequence[int]) -> np.ndarray:
    """Repeat elements according to counts."""
    return np.array(list(itertools.chain.from_iterable(map(itertools.repeat, elements, counts))))


def _normalize(targets: list[np.ndarray], eps: float = _TOL) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    """Normalize each target series independently.

    Args:
        targets: List of target arrays to normalize
        eps: Small value for numerical stability

    Returns:
        Normalized targets and their statistics (mean, std) for denormalization
    """
    normalized = []
    stats = []

    for target in targets:
        target = np.array(target)
        mean = np.mean(target)
        std = np.std(target)
        if std < eps:
            std = 1.0
        normalized.append((target - mean) / std)
        stats.append((mean, std))

    return normalized, stats


def _renormalize(predictions: list[np.ndarray], stats: list[tuple[float, float]]) -> list[np.ndarray]:
    """Denormalize predictions using saved statistics.

    Args:
        predictions: List of normalized predictions
        stats: List of (mean, std) tuples from normalization

    Returns:
        Denormalized predictions
    """
    denormalized = []
    for pred, (mean, std) in zip(predictions, stats):
        denormalized.append(pred * std + mean)
    return denormalized


class BatchedInContextXRegBase:
    """Base class for in-context regression with covariates.

    This class handles the formatting and validation of covariates for
    batched in-context regression used with TimesFM.

    Attributes:
        targets: List of target values for regression
        train_lens: List of context lengths for each series
        test_lens: List of horizon lengths for each series
        train_dynamic_numerical_covariates: Dict of dynamic numerical covariates for context
        train_dynamic_categorical_covariates: Dict of dynamic categorical covariates for context
        test_dynamic_numerical_covariates: Dict of dynamic numerical covariates for horizon
        test_dynamic_categorical_covariates: Dict of dynamic categorical covariates for horizon
        static_numerical_covariates: Dict of static numerical covariates per series
        static_categorical_covariates: Dict of static categorical covariates per series
    """

    def __init__(
        self,
        targets: Sequence[Sequence[float]],
        train_lens: Sequence[int],
        test_lens: Sequence[int],
        train_dynamic_numerical_covariates: Optional[Mapping[str, Sequence[Sequence[float]]]] = None,
        train_dynamic_categorical_covariates: Optional[Mapping[str, Sequence[Sequence[Category]]]] = None,
        test_dynamic_numerical_covariates: Optional[Mapping[str, Sequence[Sequence[float]]]] = None,
        test_dynamic_categorical_covariates: Optional[Mapping[str, Sequence[Sequence[Category]]]] = None,
        static_numerical_covariates: Optional[Mapping[str, Sequence[float]]] = None,
        static_categorical_covariates: Optional[Mapping[str, Sequence[Category]]] = None,
    ) -> None:
        """Initialize with exogenous covariate inputs.

        Args:
            targets: Target values for each series in the batch
            train_lens: Length of context for each series
            test_lens: Length of horizon for each series
            train_dynamic_numerical_covariates: Dynamic numerical features for context
            train_dynamic_categorical_covariates: Dynamic categorical features for context
            test_dynamic_numerical_covariates: Dynamic numerical features for horizon
            test_dynamic_categorical_covariates: Dynamic categorical features for horizon
            static_numerical_covariates: Static numerical features per series
            static_categorical_covariates: Static categorical features per series
        """
        self.targets = targets
        self.train_lens = train_lens
        self.test_lens = test_lens

        # Initialize covariate dictionaries
        self.train_dynamic_numerical_covariates = train_dynamic_numerical_covariates or {}
        self.train_dynamic_categorical_covariates = train_dynamic_categorical_covariates or {}
        self.test_dynamic_numerical_covariates = test_dynamic_numerical_covariates or {}
        self.test_dynamic_categorical_covariates = test_dynamic_categorical_covariates or {}
        self.static_numerical_covariates = static_numerical_covariates or {}
        self.static_categorical_covariates = static_categorical_covariates or {}

    def _assert_covariates(self, assert_covariate_shapes: bool = False) -> None:
        """Validate covariate consistency and shapes.

        Args:
            assert_covariate_shapes: Whether to validate detailed shapes

        Raises:
            ValueError: If covariates are inconsistent or have wrong shapes
        """
        # Check that train and test dynamic covariates are paired
        if (self.train_dynamic_numerical_covariates and not self.test_dynamic_numerical_covariates) or (
            not self.train_dynamic_numerical_covariates and self.test_dynamic_numerical_covariates
        ):
            raise ValueError(
                "train_dynamic_numerical_covariates and test_dynamic_numerical_covariates "
                "must be both present or both absent."
            )

        if (self.train_dynamic_categorical_covariates and not self.test_dynamic_categorical_covariates) or (
            not self.train_dynamic_categorical_covariates and self.test_dynamic_categorical_covariates
        ):
            raise ValueError(
                "train_dynamic_categorical_covariates and test_dynamic_categorical_covariates "
                "must be both present or both absent."
            )

        # Check that keys match between train and test
        for dict_a, dict_b, dict_a_name, dict_b_name in [
            (
                self.train_dynamic_numerical_covariates,
                self.test_dynamic_numerical_covariates,
                "train_dynamic_numerical_covariates",
                "test_dynamic_numerical_covariates",
            ),
            (
                self.train_dynamic_categorical_covariates,
                self.test_dynamic_categorical_covariates,
                "train_dynamic_categorical_covariates",
                "test_dynamic_categorical_covariates",
            ),
        ]:
            if w := set(dict_a.keys()) - set(dict_b.keys()):
                raise ValueError(f"{dict_a_name} has keys not present in {dict_b_name}: {w}")
            if w := set(dict_b.keys()) - set(dict_a.keys()):
                raise ValueError(f"{dict_b_name} has keys not present in {dict_a_name}: {w}")

        # Detailed shape checking
        if assert_covariate_shapes:
            if len(self.targets) != len(self.train_lens):
                raise ValueError("targets and train_lens must have the same number of elements.")

            if len(self.train_lens) != len(self.test_lens):
                raise ValueError("train_lens and test_lens must have the same number of elements.")

            # Check target lengths match train_lens
            for i, (target, train_len) in enumerate(zip(self.targets, self.train_lens)):
                if len(target) != train_len:
                    raise ValueError(f"targets[{i}] has length {len(target)} != expected {train_len}.")

            # Check static covariates have correct batch size
            for key, values in self.static_numerical_covariates.items():
                if len(values) != len(self.train_lens):
                    raise ValueError(
                        f"static_numerical_covariates['{key}'] has {len(values)} examples "
                        f"!= expected {len(self.train_lens)}."
                    )

            for key, values in self.static_categorical_covariates.items():
                if len(values) != len(self.train_lens):
                    raise ValueError(
                        f"static_categorical_covariates['{key}'] has {len(values)} examples "
                        f"!= expected {len(self.train_lens)}."
                    )

            # Check dynamic covariates have correct lengths
            for lens, dict_cov, dict_cov_name in [
                (self.train_lens, self.train_dynamic_numerical_covariates, "train_dynamic_numerical_covariates"),
                (self.train_lens, self.train_dynamic_categorical_covariates, "train_dynamic_categorical_covariates"),
                (self.test_lens, self.test_dynamic_numerical_covariates, "test_dynamic_numerical_covariates"),
                (self.test_lens, self.test_dynamic_categorical_covariates, "test_dynamic_categorical_covariates"),
            ]:
                for key, cov_values in dict_cov.items():
                    if len(cov_values) != len(lens):
                        raise ValueError(
                            f"{dict_cov_name}['{key}'] has {len(cov_values)} examples != expected {len(lens)}."
                        )
                    for i, cov_value in enumerate(cov_values):
                        if len(cov_value) != lens[i]:
                            raise ValueError(
                                f"{dict_cov_name}['{key}'][{i}] has length {len(cov_value)} != expected {lens[i]}."
                            )

    def create_covariate_matrix(
        self,
        one_hot_encoder_drop: Optional[str] = "first",
        use_intercept: bool = True,
        assert_covariates: bool = False,
        assert_covariate_shapes: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create target vector and covariate matrices for regression.

        Args:
            one_hot_encoder_drop: Strategy for dropping columns in one-hot encoding
            use_intercept: Whether to add an intercept column
            assert_covariates: Whether to validate covariates
            assert_covariate_shapes: Whether to validate covariate shapes

        Returns:
            Tuple of (target_vector, train_covariate_matrix, test_covariate_matrix)
        """
        if assert_covariates:
            self._assert_covariates(assert_covariate_shapes)

        x_train, x_test = [], []

        # Process numerical features
        for name in sorted(self.train_dynamic_numerical_covariates):
            x_train.append(_unnest(self.train_dynamic_numerical_covariates[name])[:, np.newaxis])
            x_test.append(_unnest(self.test_dynamic_numerical_covariates[name])[:, np.newaxis])

        for name in sorted(self.static_numerical_covariates):
            covs = self.static_numerical_covariates[name]
            x_train.append(_repeat(covs, self.train_lens)[:, np.newaxis])
            x_test.append(_repeat(covs, self.test_lens)[:, np.newaxis])

        # Normalize numerical features if present
        if x_train:
            x_train = np.concatenate(x_train, axis=1)
            x_test = np.concatenate(x_test, axis=1)

            # Normalize for numerical stability
            x_mean = np.mean(x_train, axis=0, keepdims=True)
            x_std = np.where((w := np.std(x_train, axis=0, keepdims=True)) > _TOL, w, 1.0)
            x_train = [(x_train - x_mean) / x_std]
            x_test = [(x_test - x_mean) / x_std]

        # Process categorical features
        one_hot_encoder = preprocessing.OneHotEncoder(
            drop=one_hot_encoder_drop,
            sparse_output=False,
            handle_unknown="ignore",
        )

        for name in sorted(self.train_dynamic_categorical_covariates.keys()):
            ohe_train = _unnest(self.train_dynamic_categorical_covariates[name])[:, np.newaxis]
            ohe_test = _unnest(self.test_dynamic_categorical_covariates[name])[:, np.newaxis]
            x_train.append(np.array(one_hot_encoder.fit_transform(ohe_train)))
            x_test.append(np.array(one_hot_encoder.transform(ohe_test)))

        for name in sorted(self.static_categorical_covariates.keys()):
            covs = self.static_categorical_covariates[name]
            ohe = one_hot_encoder.fit_transform(np.array(covs)[:, np.newaxis])
            x_train.append(_repeat(ohe, self.train_lens))
            x_test.append(_repeat(ohe, self.test_lens))

        # Concatenate all features
        x_train = np.concatenate(x_train, axis=1) if x_train else np.zeros((sum(self.train_lens), 0))
        x_test = np.concatenate(x_test, axis=1) if x_test else np.zeros((sum(self.test_lens), 0))

        # Add intercept if requested
        if use_intercept:
            x_train = np.pad(x_train, ((0, 0), (1, 0)), constant_values=1.0)
            x_test = np.pad(x_test, ((0, 0), (1, 0)), constant_values=1.0)

        return _unnest(self.targets), x_train, x_test

    def fit(self) -> Any:
        """Fit the model. To be implemented by subclasses."""
        raise NotImplementedError("fit() must be implemented by subclasses.")


class BatchedInContextXRegLinear(BatchedInContextXRegBase):
    """Linear regression model for in-context covariates.

    This class implements a batched linear regression model that can be used
    with TimesFM for incorporating covariates into forecasts.
    """

    def fit(
        self,
        ridge: float = 0.0,
        one_hot_encoder_drop: Optional[str] = "first",
        use_intercept: bool = True,
        force_on_cpu: bool = False,
        max_rows_per_col: int = 0,
        max_rows_per_col_sample_seed: int = 42,
        debug_info: bool = False,
        assert_covariates: bool = False,
        assert_covariate_shapes: bool = False,
        device: Optional[torch.device] = None,
    ) -> Union[list[np.ndarray], tuple[list[np.ndarray], list[np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Fit a linear regression model with optional ridge regularization.

        Args:
            ridge: Ridge regularization parameter (L2 penalty)
            one_hot_encoder_drop: Strategy for dropping columns in one-hot encoding
            use_intercept: Whether to add an intercept term
            force_on_cpu: Whether to force computation on CPU
            max_rows_per_col: Maximum ratio of rows to columns for stability (0 for no limit)
            max_rows_per_col_sample_seed: Random seed for sampling rows
            debug_info: Whether to return debug information
            assert_covariates: Whether to validate covariates
            assert_covariate_shapes: Whether to validate covariate shapes
            device: PyTorch device to use for computation

        Returns:
            If debug_info is False: List of predictions for each series
            If debug_info is True: Tuple of (predictions, predictions_on_context,
                                            coeff_matrix, train_matrix, test_matrix)
        """
        # Create covariate matrices
        y, x_train, x_test = self.create_covariate_matrix(
            one_hot_encoder_drop=one_hot_encoder_drop,
            use_intercept=use_intercept,
            assert_covariates=assert_covariates,
            assert_covariate_shapes=assert_covariate_shapes,
        )

        # Determine device
        if device is None:
            if force_on_cpu:
                device = torch.device("cpu")
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert to PyTorch tensors
        y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32, device=device)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32, device=device)

        # Handle max_rows_per_col constraint
        if max_rows_per_col > 0 and x_train.shape[0] > max_rows_per_col * x_train.shape[1]:
            # Sample rows to maintain stability
            np.random.seed(max_rows_per_col_sample_seed)
            n_samples = max_rows_per_col * x_train.shape[1]
            indices = np.random.choice(x_train.shape[0], n_samples, replace=False)
            indices_tensor = torch.tensor(indices, device=device)
            x_train_tensor = x_train_tensor[indices_tensor]
            y_tensor = y_tensor[indices_tensor]

        # Solve linear regression with ridge regularization
        if x_train_tensor.shape[1] == 0:
            # No covariates, predict zeros
            predictions_flat = torch.zeros(x_test_tensor.shape[0], device=device)
            predictions_on_context_flat = torch.zeros(len(y), device=device)
            coeffs = torch.zeros(0, device=device)
        else:
            # Compute (X^T X + ridge * I)
            xtx = x_train_tensor.T @ x_train_tensor
            if ridge > 0:
                xtx = xtx + ridge * torch.eye(xtx.shape[0], device=device)

            # Compute X^T y
            xty = x_train_tensor.T @ y_tensor

            # Solve for coefficients
            try:
                coeffs = torch.linalg.solve(xtx, xty)
            except torch.linalg.LinAlgError:
                # Fallback to least squares if solve fails
                result = torch.linalg.lstsq(x_train_tensor, y_tensor, rcond=None)
                coeffs = result.solution[: x_train_tensor.shape[1]]  # Trim to correct size

            # Make predictions
            predictions_flat = x_test_tensor @ coeffs

            # Reconstruct predictions on training data for debug
            x_train_full = torch.tensor(x_train, dtype=torch.float32, device=device)
            predictions_on_context_flat = x_train_full @ coeffs

        # Convert back to numpy and reshape to original batch structure
        predictions_flat = predictions_flat.cpu().numpy()
        predictions_on_context_flat = predictions_on_context_flat.cpu().numpy()

        # Reshape predictions to match original batch structure
        predictions = []
        predictions_on_context = []

        test_start = 0
        train_start = 0
        for train_len, test_len in zip(self.train_lens, self.test_lens):
            predictions.append(predictions_flat[test_start : test_start + test_len])
            predictions_on_context.append(predictions_on_context_flat[train_start : train_start + train_len])
            test_start += test_len
            train_start += train_len

        if debug_info:
            return (
                predictions,
                predictions_on_context,
                coeffs.cpu() if x_train_tensor.shape[1] > 0 else coeffs,
                x_train_tensor.cpu(),
                x_test_tensor.cpu(),
            )
        else:
            return predictions
