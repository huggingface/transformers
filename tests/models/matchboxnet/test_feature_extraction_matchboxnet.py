import unittest
import numpy as np
import torch
from transformers import MatchboxNetFeatureExtractor

class FeatureExtractorTest(unittest.TestCase):
    
    def test_extract_shape_and_save_load(self):
        
      
        fe = MatchboxNetFeatureExtractor(target_sr=16000, n_mfcc=32, fixed_length=64)
        
        sr = 16000
        dummy = np.random.randn(sr).astype(np.float32)
        out = fe(dummy, sampling_rate=sr)
        
        self.assertIn("input_ids", out)
        arr = out["input_ids"]
        
        self.assertEqual(arr.shape, (32, 64))

        
        save_dir = "tmp_matchbox_fe"  
        fe.save_pretrained(save_dir)
        fe2 = MatchboxNetFeatureExtractor.from_pretrained(save_dir)
        
        out2 = fe2(dummy, sampling_rate=sr)
        self.assertEqual(out2["input_ids"].shape, (32, 64))
