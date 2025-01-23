import unittest
import torch
import requests
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
