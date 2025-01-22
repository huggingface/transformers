import unittest
import torch
from transformers import SwitchTransformersConfig
from transformers.models.switch_transformers.modeling_switch_transformers import (
    SwitchTransformersTop1Router,
    SwitchTransformersSparseMLP
)
