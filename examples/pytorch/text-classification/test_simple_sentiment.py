"""
Unit tests for simple sentiment analysis example.
"""

import unittest
import sys
import os

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add the example directory to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from run_simple_sentiment import preprocess_function, compute_metrics
except ImportError:
    # If running from different directory
    pass


class TestSimpleSentiment(unittest.TestCase):
    """Test cases for simple sentiment analysis."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across tests."""
        cls.model_name = "distilbert-base-uncased"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.model = AutoModelForSequenceClassification.from_pretrained(
            cls.model_name,
            num_labels=2
        )
    
    def test_tokenizer_loading(self):
        """Test that tokenizer loads correctly."""
        self.assertIsNotNone(self.tokenizer)
        self.assertTrue(hasattr(self.tokenizer, 'encode'))
    
    def test_model_loading(self):
        """Test that model loads correctly."""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.config.num_labels, 2)
    
    def test_preprocess_function(self):
        """Test text preprocessing and tokenization."""
        examples = {
            'text': [
                'This is a positive review.',
                'This is a negative review.'
            ]
        }
        
        result = preprocess_function(examples, self.tokenizer, max_length=128)
        
        self.assertIn('input_ids', result)
        self.assertIn('attention_mask', result)
        self.assertEqual(len(result['input_ids']), 2)
        self.assertEqual(len(result['input_ids'][0]), 128)
    
    def test_model_inference(self):
        """Test model can perform inference."""
        text = "This movie was great!"
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        self.assertEqual(outputs.logits.shape, (1, 2))
        
        # Test softmax probabilities sum to 1
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        self.assertAlmostEqual(probs.sum().item(), 1.0, places=5)
    
    def test_compute_metrics(self):
        """Test metrics computation."""
        import numpy as np
        from collections import namedtuple
        
        # Create a proper EvalPrediction-like object
        EvalPrediction = namedtuple('EvalPrediction', ['predictions', 'label_ids'])
        
        # Create mock predictions
        # Predictions (logits for 2 classes, 4 samples)
        predictions = np.array([
            [0.9, 0.1],  # Predicts class 0
            [0.2, 0.8],  # Predicts class 1
            [0.7, 0.3],  # Predicts class 0
            [0.1, 0.9],  # Predicts class 1
        ])
        # True labels
        labels = np.array([0, 1, 0, 1])
        
        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
        metrics = compute_metrics(eval_pred)
        
        self.assertIn('accuracy', metrics)
        self.assertIn('f1', metrics)
        self.assertEqual(metrics['accuracy'], 1.0)  # All correct
        self.assertEqual(metrics['f1'], 1.0)


if __name__ == '__main__':
    unittest.main()