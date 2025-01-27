import unittest
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.testing_utils import require_torch, torch_device

