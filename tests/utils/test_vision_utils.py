import unittest

from parameterized import parameterized

from transformers.testing_utils import require_vision, require_torchvision

from transformers import AutoConfig, AutoModelForImageClassification, is_torch_available
from transformers.utils.vision_utils import image_encoder_size, encode_images

MODELS_CONFIGS = {
    "WinKawaks/vit-tiny-patch16-224": 192,
    "microsoft/swinv2-tiny-patch4-window16-256": 768,
    "google/vit-base-patch16-224": 768,
    "microsoft/resnet-18": 512,
    "apple/mobilevit-xx-small": 80,
}


def image_encoder(model_name):
    # Create an image encoder model from config, so it does not load the weights
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_config(config)
    model.eval()
    return model


MODELS = [[name, size, image_encoder(name)] for name, size in MODELS_CONFIGS.items()]

if is_torch_available():
    import torch


@require_vision
@require_torchvision
class ImageEncoderSizeTest(unittest.TestCase):
    @parameterized.expand(MODELS)
    def test_image_encoder_size(self, model_name, expected_size, model):
        """Test that image_encoder_size returns the expected hidden size for each model."""
        actual_size = image_encoder_size(model)
        assert actual_size == expected_size, f"Expected {expected_size}, got {actual_size} for {model_name}"

    @parameterized.expand(MODELS)
    def test_image_encoder_size_when_called(self, model_name, expected_size, model):
        random_images = torch.randn(1, 3, 64, 64)
        embedding = encode_images(model, random_images)
        assert embedding.shape == (1, expected_size)
