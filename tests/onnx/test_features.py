from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import MagicMock, patch

from transformers import AutoModel, TFAutoModel
from transformers.onnx import FeaturesManager
from transformers.testing_utils import SMALL_MODEL_IDENTIFIER, require_tf, require_torch


@require_torch
@require_tf
class DetermineFrameworkTest(TestCase):
    """
    Test `FeaturesManager.determine_framework`
    """

    def setUp(self):
        self.test_model = SMALL_MODEL_IDENTIFIER
        self.framework_pt = "pt"
        self.framework_tf = "tf"

    def _setup_pt_ckpt(self, save_dir):
        model_pt = AutoModel.from_pretrained(self.test_model)
        model_pt.save_pretrained(save_dir)

    def _setup_tf_ckpt(self, save_dir):
        model_tf = TFAutoModel.from_pretrained(self.test_model, from_pt=True)
        model_tf.save_pretrained(save_dir)

    def test_framework_provided(self):
        """
        Ensure the that the provided framework is returned.
        """
        mock_framework = "mock_framework"

        # Framework provided - return whatever the user provides
        result = FeaturesManager.determine_framework(self.test_model, mock_framework)
        self.assertEqual(result, mock_framework)

        # Local checkpoint and framework provided - return provided framework
        # PyTorch checkpoint
        with TemporaryDirectory() as local_pt_ckpt:
            self._setup_pt_ckpt(local_pt_ckpt)
            result = FeaturesManager.determine_framework(local_pt_ckpt, mock_framework)
            self.assertEqual(result, mock_framework)

        # TensorFlow checkpoint
        with TemporaryDirectory() as local_tf_ckpt:
            self._setup_tf_ckpt(local_tf_ckpt)
            result = FeaturesManager.determine_framework(local_tf_ckpt, mock_framework)
            self.assertEqual(result, mock_framework)

    def test_checkpoint_provided(self):
        """
        Ensure that the determined framework is the one used for the local checkpoint.

        For the functionality to execute, local checkpoints are provided but framework is not.
        """
        # PyTorch checkpoint
        with TemporaryDirectory() as local_pt_ckpt:
            self._setup_pt_ckpt(local_pt_ckpt)
            result = FeaturesManager.determine_framework(local_pt_ckpt)
            self.assertEqual(result, self.framework_pt)

        # TensorFlow checkpoint
        with TemporaryDirectory() as local_tf_ckpt:
            self._setup_tf_ckpt(local_tf_ckpt)
            result = FeaturesManager.determine_framework(local_tf_ckpt)
            self.assertEqual(result, self.framework_tf)

        # Invalid local checkpoint
        with TemporaryDirectory() as local_invalid_ckpt:
            with self.assertRaises(FileNotFoundError):
                result = FeaturesManager.determine_framework(local_invalid_ckpt)

    def test_from_environment(self):
        """
        Ensure that the determined framework is the one available in the environment.

        For the functionality to execute, framework and local checkpoints are not provided.
        """
        # Framework not provided, hub model is used (no local checkpoint directory)
        # TensorFlow not in environment -> use PyTorch
        mock_tf_available = MagicMock(return_value=False)
        with patch("transformers.onnx.features.is_tf_available", mock_tf_available):
            result = FeaturesManager.determine_framework(self.test_model)
            self.assertEqual(result, self.framework_pt)

        # PyTorch not in environment -> use TensorFlow
        mock_torch_available = MagicMock(return_value=False)
        with patch("transformers.onnx.features.is_torch_available", mock_torch_available):
            result = FeaturesManager.determine_framework(self.test_model)
            self.assertEqual(result, self.framework_tf)

        # Both in environment -> use PyTorch
        mock_tf_available = MagicMock(return_value=True)
        mock_torch_available = MagicMock(return_value=True)
        with patch("transformers.onnx.features.is_tf_available", mock_tf_available), patch(
            "transformers.onnx.features.is_torch_available", mock_torch_available
        ):
            result = FeaturesManager.determine_framework(self.test_model)
            self.assertEqual(result, self.framework_pt)

        # Both not in environment -> raise error
        mock_tf_available = MagicMock(return_value=False)
        mock_torch_available = MagicMock(return_value=False)
        with patch("transformers.onnx.features.is_tf_available", mock_tf_available), patch(
            "transformers.onnx.features.is_torch_available", mock_torch_available
        ):
            with self.assertRaises(EnvironmentError):
                result = FeaturesManager.determine_framework(self.test_model)
