import unittest
from transformers import MatchboxNetConfig

class MatchboxNetConfigTest(unittest.TestCase):
    def test_config_to_json_and_back(self):
        
        
        config = MatchboxNetConfig(input_channels=32, num_classes=10, B=2, R=1, C=16, kernel_sizes=[13,15])
        
        json_str = config.to_json_string()
      
        new_config = MatchboxNetConfig.from_json_string(json_str)
        self.assertEqual(new_config.input_channels, 32)
        self.assertEqual(new_config.num_classes, 10)
        self.assertEqual(new_config.B, 2)
        self.assertEqual(new_config.R, 1)
        self.assertEqual(new_config.C, 16)
        self.assertEqual(new_config.kernel_sizes, [13,15])
