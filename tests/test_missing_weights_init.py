import unittest
import torch
import torch.nn as nn
import tempfile
import os
from transformers import PreTrainedModel, PretrainedConfig
from transformers.testing_utils import require_torch

@require_torch
class TestMissingWeightsInit(unittest.TestCase):
    """Test class for proper initialization of missing weights during from_pretrained()"""
    
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        
        class TestConfig(PretrainedConfig):
            def __init__(self, use_new=False, **kwargs):
                super().__init__(**kwargs)
                self.use_new = use_new

        class TestModel(PreTrainedModel):
            config_class = TestConfig
            
            def __init__(self, config):
                super().__init__(config)
                # Initialize base_layer with ones to make testing easier
                self.base_layer = nn.Linear(10, 10)
                with torch.no_grad():
                    self.base_layer.weight.fill_(1.0)
                    self.base_layer.bias.fill_(0.0)
                    
                if config.use_new:
                    self.new_layer = nn.Linear(10, 10)
                    
            def _init_weights(self, module):
                if isinstance(module, nn.Linear):
                    module.weight.data.fill_(1.0)
                    if module.bias is not None:
                        module.bias.data.fill_(0.0)

            def forward(self, x):
                x = self.base_layer(x)
                if hasattr(self, 'new_layer'):
                    x = self.new_layer(x)
                return x

        self.TestConfig = TestConfig
        self.TestModel = TestModel