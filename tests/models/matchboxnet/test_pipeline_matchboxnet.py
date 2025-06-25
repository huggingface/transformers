import unittest
import numpy as np
import torch
from transformers import MatchboxNetConfig, MatchboxNetForAudioClassification, MatchboxNetFeatureExtractor, pipeline

class PipelineTest(unittest.TestCase):
    def test_pipeline_audio_classification(self):
        
        config = MatchboxNetConfig(input_channels=8, num_classes=3, B=2, R=1, C=16, kernel_sizes=[3,3], target_sr=8000, n_mfcc=8, fixed_length=32)
        model = MatchboxNetForAudioClassification(config)
        fe = MatchboxNetFeatureExtractor(target_sr=8000, n_mfcc=8, fixed_length=32)
        
        clf = pipeline("audio-classification", model=model, feature_extractor=fe, torch_dtype=None,)
       
        sr = 8000
        dummy = np.random.randn(int(0.5*sr)).astype(np.float32)
        #dummy = torch.randn(int(0.5 * sr), dtype=torch.float32).unsqueeze(0)
        
        #dummy = torch.
        res = clf(dummy, sampling_rate=sr)
        
       
        self.assertIsInstance(res, list)
     
        self.assertTrue(all("score" in item for item in res))
