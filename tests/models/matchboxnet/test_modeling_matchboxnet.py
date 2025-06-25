import unittest
import torch
from transformers import MatchboxNetConfig, MatchboxNetForAudioClassification

class ModelTest(unittest.TestCase):
    def test_forward_loss_and_logits(self):
      
        config = MatchboxNetConfig(input_channels=8, num_classes=4, B=2, R=1, C=16, kernel_sizes=[3,3], target_sr=8000, n_mfcc=8, fixed_length=32)
        model = MatchboxNetForAudioClassification(config)
        
     
        batch_size = 2
        inputs = torch.randn(batch_size, config.input_channels, config.fixed_length)
        
        labels = torch.tensor([0, 1], dtype=torch.long)
        outputs = model(input_ids=inputs, labels=labels)
        
        logits = outputs["logits"]
        self.assertEqual(logits.shape, (batch_size, config.num_classes))
        loss = outputs["loss"]
        
        self.assertIsNotNone(loss)
        self.assertTrue(loss.item() >= 0.0)
        
        loss.backward()
