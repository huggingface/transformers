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
        
    def test_missing_weights_initialization(self):
        # 1. Create and save base model
        base_config = self.TestConfig(use_new=False)
        base_model = self.TestModel(base_config)
        
        # Verify initial state
        self.assertTrue(torch.all(base_model.base_layer.weight.data == 1.0))
        base_model.save_pretrained(self.tmp_dir)
        
        # 2. Load with new layer
        new_config = self.TestConfig(use_new=True)
        loaded_model = self.TestModel.from_pretrained(self.tmp_dir, config=new_config)
        
        # 3. Print debug information
        print("\nBase layer weight:", loaded_model.base_layer.weight.data)
        print("New layer weight:", loaded_model.new_layer.weight.data)
        
        # 4. Verify initialization
        # Base layer should keep pretrained weights
        self.assertTrue(torch.all(loaded_model.base_layer.weight.data == 1.0), 
                       f"Base layer weights not preserved: {loaded_model.base_layer.weight.data}")
        
        # New layer should be properly initialized
        self.assertTrue(torch.all(loaded_model.new_layer.weight.data == 1.0),
                       f"New layer not properly initialized: {loaded_model.new_layer.weight.data}")

    def test_backward_compatibility(self):
        """Test that existing behavior is preserved for matched weights"""
        # 1. Create and save base model
        base_config = self.TestConfig(use_new=False)
        base_model = self.TestModel(base_config)
        original_weights = base_model.base_layer.weight.data.clone()
        base_model.save_pretrained(self.tmp_dir)
        
        # 2. Load model without new layers
        loaded_model = self.TestModel.from_pretrained(self.tmp_dir)
        
        # 3. Verify weights are exactly preserved
        torch.testing.assert_close(loaded_model.base_layer.weight.data, original_weights)

    def test_initialization_without_fast_init(self):
        """Test that initialization works the same with and without _fast_init"""
        # 1. Create and save base model
        base_config = self.TestConfig(use_new=False)
        base_model = self.TestModel(base_config)
        base_model.save_pretrained(self.tmp_dir)
        
        # 2. Load with new layer using both methods
        new_config = self.TestConfig(use_new=True)
        model_with_fast_init = self.TestModel.from_pretrained(self.tmp_dir, config=new_config)
        model_without_fast_init = self.TestModel.from_pretrained(self.tmp_dir, config=new_config, _fast_init=False)
        
        # 3. Verify both methods give same initialization
        torch.testing.assert_close(
            model_with_fast_init.new_layer.weight.data,
            model_without_fast_init.new_layer.weight.data,
            msg="Initialization differs between _fast_init=True and _fast_init=False"
        )
        
        # 4. Verify both are properly initialized
        self.assertTrue(torch.all(model_with_fast_init.new_layer.weight.data == 1.0),
                       "New layer not properly initialized with _fast_init=True")

    def test_original_issue_reproduction(self):
        """Test the specific case from issue #35437"""
        # 1. Create and save base model
        base_config = self.TestConfig(use_new=False)
        base_model = self.TestModel(base_config)
        base_model.save_pretrained(self.tmp_dir)
        
        # 2. Load with new layer - this should now work correctly without _fast_init=False
        new_config = self.TestConfig(use_new=True)
        new_model = self.TestModel.from_pretrained(self.tmp_dir, use_new=True)
        
        # 3. Verify new weights are properly initialized
        self.assertFalse(torch.isnan(new_model.new_layer.weight.data).any(),
                         "New weights contain NaN values")
        self.assertTrue(torch.all(new_model.new_layer.weight.data == 1.0),
                       "New weights not properly initialized")

    def tearDown(self):
        # Clean up
        for file in os.listdir(self.tmp_dir):
            os.remove(os.path.join(self.tmp_dir, file))
        os.rmdir(self.tmp_dir)

if __name__ == '__main__':
    unittest.main() 
