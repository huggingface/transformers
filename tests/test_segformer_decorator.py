"""Test that @capture_outputs and @can_return_tuple decorators work on SegformerForImageClassification."""
import torch
import torch.nn as nn

# Create a minimal model to test
class DummySegformerConfig:
    num_labels = 1
    hidden_sizes = [64]
    decoder_hidden_size = 64
    reshape_last_stage = False

class DummyEncoder:
    def __init__(self):
        pass
    
    @property
    def _can_record_outputs(self):
        return {"hidden_states": "DummyEncoder", "attentions": "DummyEncoder"}

class DummySegformerPreTrainedModel:
    def __init__(self):
        pass

class TestSegformerForImageClassification:
    def __init__(self):
        self.num_labels = 1
        self.segformer = DummySegformer()
        self.classifier = nn.Linear(64, 1)
    
    @capture_outputs
    @can_return_tuple
    def forward(self, pixel_values):
        # Call encoder
        outputs = self.segformer(pixel_values)
        sequence_output = outputs[0]
        # Test tuple handling - this should work with @can_return_tuple
        if not isinstance(outputs, tuple):
            return sequence_output
        else:
            return (sequence_output,) + outputs[1:]
    
    def test_capture_outputs_decorator(self):
        """Test that @capture_outputs decorator is present and working."""
        import inspect
        # Get the forward method
        forward_method = getattr(self, 'forward')
        
        # Check for decorator
        has_capture_outputs = hasattr(forward_method, '__wrapped__')
        
        print(f'Has @capture_outputs decorator: {has_capture_outputs}')
        
        # Check that forward can handle return_dict=False
        from ..utils.generic import modeling_utils
        result = self.forward(torch.randn(2, 3, 224, 3), return_dict=False)
        
        # Should return a tuple since @can_return_tuple is present
        assert isinstance(result, tuple), f'Expected tuple, got {type(result)}'
        print(f'✓ Tuple handling works correctly')
        print(f'Result type: {type(result)}')
        print(f'Result keys: {result.keys() if hasattr(result, \"keys\") else \"N/A\"}')

if __name__ == '__main__':
    test = TestSegformerForImageClassification()
    test.test_capture_outputs_decorator()
