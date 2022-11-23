"""Feature extractor class for ViViT."""

from ...utils import logging
from .image_processing_vivit import ViViTImageProcessor


logger = logging.get_logger(__name__)


ViViTFeatureExtractor = ViViTImageProcessor
