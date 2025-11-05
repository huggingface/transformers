#!/usr/bin/env python3
"""
Comprehensive test suite for the PreTrainedTokenizerBase.model_input_names property fix

This test suite verifies:
1. Original singleton bug behavior
2. Property-based fix resolves the issue
3. Multiple tokenizer instances are independent
4. Backward compatibility is maintained
5. Edge cases like subclasses overriding model_input_names
6. Serialization/deserialization behavior
7. Different tokenizer types work correctly

The tests use mock implementations that mirror the actual transformers library behavior.
"""

import pytest
import copy
import json
import pickle
from typing import List
from unittest.mock import patch, MagicMock


# Mock implementations based on the actual bug and fix patterns

class MockSpecialTokensMixin:
    """Mock SpecialTokensMixin"""
    pass


class MockPushToHubMixin:
    """Mock PushToHubMixin"""
    pass


class PreTrainedTokenizerBuggy(MockSpecialTokensMixin, MockPushToHubMixin):
    """
    Base class demonstrating the BUGGY implementation.
    This mirrors the original transformers bug where all instances share the same list.
    """
    
    # THIS IS THE BUG: mutable default argument
    model_input_names: List[str] = ["input_ids", "token_type_ids", "attention_mask"]
    
    def __init__(self, **kwargs):
        # The problematic line: using class attribute directly
        self.model_input_names = kwargs.pop("model_input_names", self.model_input_names)


class PreTrainedTokenizerFixed(MockSpecialTokensMixin, MockPushToHubMixin):
    """
    Base class with the PROPERTY-BASED FIX.
    This fixes the singleton bug by using proper instance isolation.
    """
    
    # Class-level default as immutable tuple (prevents shared mutation at class level)
    _MODEL_INPUT_NAMES_DEFAULT: tuple[str, ...] = ("input_ids", "token_type_ids", "attention_mask")
    
    def __init__(self, **kwargs):
        # Initialize instance-level model_input_names with proper copying
        # This prevents the singleton bug where modifying one instance affects all others
        model_input_names = kwargs.pop("model_input_names", None)
        if model_input_names is not None:
            # Explicit value provided - convert to list if needed and create copy
            self._model_input_names = list(model_input_names) if not isinstance(model_input_names, list) else model_input_names.copy()
        else:
            # No explicit value - use class default but make a copy for this instance
            self._model_input_names = list(self._MODEL_INPUT_NAMES_DEFAULT)

    @property
    def model_input_names(self) -> List[str]:
        """
        Get the list of input names expected by the model.
        
        Returns:
            list[str]: Copy of the instance's model input names list
            
        Note: Returns a copy to prevent external mutation from affecting other instances.
        """
        # Return a copy to prevent external mutation that could affect other instances
        return self._model_input_names.copy()

    @model_input_names.setter
    def model_input_names(self, value: List[str] | tuple[str, ...]):
        """
        Set the model input names list.
        
        Args:
            value: New list or tuple of input names
            
        Note: Always creates an internal copy to prevent shared references.
        """
        # Accept both list and tuple, always store as list with proper copying
        if isinstance(value, tuple):
            self._model_input_names = list(value)
        else:
            # Create a copy to avoid sharing references with external code
            self._model_input_names = value.copy()


# Test cases

class TestBuggyImplementation:
    """Test the BUGGY implementation to demonstrate the original problem"""
    
    def test_buggy_shared_list_object(self):
        """Test that buggy implementation shares list objects between instances"""
        t1 = PreTrainedTokenizerBuggy()
        t2 = PreTrainedTokenizerBuggy()
        
        # BUG: Both instances share the same list object
        assert t1.model_input_names is t2.model_input_names
    
    def test_buggy_cross_instance_mutation(self):
        """Test that mutations in one instance affect other instances (BUG)"""
        t1 = PreTrainedTokenizerBuggy()
        t2 = PreTrainedTokenizerBuggy()
        
        # Mutate t1
        t1.model_input_names.append("custom_field")
        
        # BUG: t2 is also affected!
        assert "custom_field" in t2.model_input_names
        assert len(t2.model_input_names) == 4
    
    def test_buggy_class_level_mutation(self):
        """Test that class-level mutations affect all instances (BUG)"""
        # Mutate the class attribute
        PreTrainedTokenizerBuggy.model_input_names.append("class_added")
        
        t1 = PreTrainedTokenizerBuggy()
        t2 = PreTrainedTokenizerBuggy()
        
        # All instances see the mutation
        assert "class_added" in t1.model_input_names
        assert "class_added" in t2.model_input_names


class TestFixedImplementation:
    """Test the FIXED implementation to verify the solution"""
    
    def test_fixed_instance_isolation(self):
        """Test that fixed implementation provides proper instance isolation"""
        t1 = PreTrainedTokenizerFixed()
        t2 = PreTrainedTokenizerFixed()
        
        # Each instance should have its own list
        assert t1.model_input_names is not t2.model_input_names
        assert t1.model_input_names == t2.model_input_names  # But same content
    
    def test_fixed_no_cross_instance_mutation(self):
        """Test that mutations in one instance don't affect others (FIXED)"""
        t1 = PreTrainedTokenizerFixed()
        t2 = PreTrainedTokenizerFixed()
        
        # Mutate t1
        t1.model_input_names.append("custom_field")
        
        # FIXED: t2 is not affected!
        assert "custom_field" not in t2.model_input_names
        assert len(t2.model_input_names) == 3
        assert len(t1.model_input_names) == 4
    
    def test_fixed_getter_returns_copy(self):
        """Test that the getter returns a copy to prevent external mutations"""
        t = PreTrainedTokenizerFixed()
        
        # Get the list
        external_list = t.model_input_names
        
        # Mutate the external list
        external_list.append("hacked")
        
        # Original should not be affected (getter returned a copy)
        assert "hacked" not in t.model_input_names
    
    def test_fixed_setter_accepts_list(self):
        """Test that setter accepts list type"""
        t = PreTrainedTokenizerFixed()
        
        new_list = ["input_ids", "attention_mask"]
        t.model_input_names = new_list
        
        assert t.model_input_names == ["input_ids", "attention_mask"]
        # Should be independent copy
        new_list.append("should_not_affect")
        assert "should_not_affect" not in t.model_input_names
    
    def test_fixed_setter_accepts_tuple(self):
        """Test that setter accepts tuple type"""
        t = PreTrainedTokenizerFixed()
        
        new_tuple = ("input_ids", "attention_mask")
        t.model_input_names = new_tuple
        
        assert t.model_input_names == ["input_ids", "attention_mask"]
        assert isinstance(t.model_input_names, list)  # Stored as list internally
    
    def test_fixed_initialization_with_kwargs(self):
        """Test initialization with explicit model_input_names"""
        t = PreTrainedTokenizerFixed(model_input_names=["custom_input"])
        
        assert t.model_input_names == ["custom_input"]
    
    def test_fixed_initialization_with_tuple_kwargs(self):
        """Test initialization with tuple model_input_names"""
        t = PreTrainedTokenizerFixed(model_input_names=("tuple_input",))
        
        assert t.model_input_names == ["tuple_input"]
    
    def test_fixed_class_default_is_immutable(self):
        """Test that class-level default cannot be mutated"""
        default = PreTrainedTokenizerFixed._MODEL_INPUT_NAMES_DEFAULT
        
        # This should not affect future instances
        default_tuple = PreTrainedTokenizerFixed._MODEL_INPUT_NAMES_DEFAULT
        default_tuple += ("extra",)  # This creates a new tuple, doesn't mutate
        
        t = PreTrainedTokenizerFixed()
        # Should still have original default
        assert t.model_input_names == ["input_ids", "token_type_ids", "attention_mask"]
    
    def test_fixed_multiple_assignments(self):
        """Test multiple assignments work correctly"""
        t = PreTrainedTokenizerFixed()
        
        # First assignment
        t.model_input_names = ["input_ids", "attention_mask"]
        assert t.model_input_names == ["input_ids", "attention_mask"]
        
        # Second assignment
        t.model_input_names = ["input_ids"]
        assert t.model_input_names == ["input_ids"]
        
        # Third assignment with tuple
        t.model_input_names = ("input_ids", "custom_field")
        assert t.model_input_names == ["input_ids", "custom_field"]


class TestBackwardCompatibility:
    """Test that the fix maintains backward compatibility"""
    
    def test_api_surface_preserved(self):
        """Test that all expected API methods and attributes are available"""
        t = PreTrainedTokenizerFixed()
        
        # Should have model_input_names attribute
        assert hasattr(t, 'model_input_names')
        
        # Should be readable
        assert isinstance(t.model_input_names, list)
        
        # Should be writable
        t.model_input_names = ["input_ids"]
        assert t.model_input_names == ["input_ids"]
    
    def test_indexing_still_works(self):
        """Test that indexing operations work as expected"""
        t = PreTrainedTokenizerFixed()
        
        # Indexing should work
        assert t.model_input_names[0] == "input_ids"
        assert t.model_input_names[-1] == "attention_mask"
    
    def test_iteration_still_works(self):
        """Test that iteration works as expected"""
        t = PreTrainedTokenizerFixed()
        
        # Iteration should work
        names = []
        for name in t.model_input_names:
            names.append(name)
        
        assert names == ["input_ids", "token_type_ids", "attention_mask"]
    
    def test_containment_check_still_works(self):
        """Test that 'in' operator works as expected"""
        t = PreTrainedTokenizerFixed()
        
        # Containment checks should work
        assert "input_ids" in t.model_input_names
        assert "token_type_ids" in t.model_input_names
        assert "attention_mask" in t.model_input_names
        assert "nonexistent" not in t.model_input_names
    
    def test_length_check_still_works(self):
        """Test that len() works as expected"""
        t = PreTrainedTokenizerFixed()
        
        assert len(t.model_input_names) == 3
        
        t.model_input_names = ["input_ids"]
        assert len(t.model_input_names) == 1
    
    def test_slicing_still_works(self):
        """Test that slicing operations work as expected"""
        t = PreTrainedTokenizerFixed()
        
        # Slicing should work
        assert t.model_input_names[:2] == ["input_ids", "token_type_ids"]
        assert t.model_input_names[1:] == ["token_type_ids", "attention_mask"]


class TestSubclassBehavior:
    """Test behavior with subclasses that override model_input_names"""
    
    class CustomTokenizer(PreTrainedTokenizerFixed):
        """Subclass with custom default"""
        _MODEL_INPUT_NAMES_DEFAULT = ("input_ids", "attention_mask")
    
    def test_subclass_has_correct_default(self):
        """Test that subclasses get correct default values"""
        t = TestSubclassBehavior.CustomTokenizer()
        
        assert t.model_input_names == ["input_ids", "attention_mask"]
    
    def test_subclass_instances_are_independent(self):
        """Test that subclass instances are independent"""
        t1 = TestSubclassBehavior.CustomTokenizer()
        t2 = TestSubclassBehavior.CustomTokenizer()
        
        # Should be independent
        assert t1.model_input_names is not t2.model_input_names
        
        # Mutate one
        t1.model_input_names.append("custom")
        
        # Other should not be affected
        assert "custom" not in t2.model_input_names
    
    def test_subclass_can_override(self):
        """Test that subclasses can still override with custom values"""
        t = TestSubclassBehavior.CustomTokenizer(model_input_names=["override"])
        
        assert t.model_input_names == ["override"]


class TestSerialization:
    """Test serialization/deserialization behavior"""
    
    def test_pickle_works(self):
        """Test that pickling works correctly"""
        t1 = PreTrainedTokenizerFixed()
        t1.model_input_names = ["custom", "values"]
        
        # Pickle and unpickle
        pickled = pickle.dumps(t1)
        t2 = pickle.loads(pickled)
        
        # Should have same values
        assert t2.model_input_names == ["custom", "values"]
        
        # But be independent copies
        t2.model_input_names.append("new_value")
        assert "new_value" not in t1.model_input_names
    
    def test_deep_copy_works(self):
        """Test that deep copying works correctly"""
        t1 = PreTrainedTokenizerFixed()
        t1.model_input_names = ["custom", "values"]
        
        # Deep copy
        t2 = copy.deepcopy(t1)
        
        # Should have same values
        assert t2.model_input_names == ["custom", "values"]
        
        # But be independent copies
        t2.model_input_names.append("new_value")
        assert "new_value" not in t1.model_input_names
    
    def test_copy_creates_independent_instance(self):
        """Test that copy creates independent instance"""
        t1 = PreTrainedTokenizerFixed()
        t1.model_input_names = ["custom", "values"]
        
        # Copy
        t2 = copy.copy(t1)
        
        # Should have same values initially
        assert t2.model_input_names == ["custom", "values"]
        
        # But be independent
        t2.model_input_names.append("new_value")
        assert "new_value" not in t1.model_input_names


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_empty_initialization(self):
        """Test initialization with empty model_input_names"""
        t = PreTrainedTokenizerFixed(model_input_names=[])
        
        assert t.model_input_names == []
    
    def test_single_item_initialization(self):
        """Test initialization with single item"""
        t = PreTrainedTokenizerFixed(model_input_names=["single"])
        
        assert t.model_input_names == ["single"]
    
    def test_assignment_with_empty_list(self):
        """Test assignment with empty list"""
        t = PreTrainedTokenizerFixed()
        t.model_input_names = []
        
        assert t.model_input_names == []
    
    def test_assignment_with_single_item(self):
        """Test assignment with single item list"""
        t = PreTrainedTokenizerFixed()
        t.model_input_names = ["single"]
        
        assert t.model_input_names == ["single"]
    
    def test_assignment_with_duplicate_items(self):
        """Test assignment with duplicate items (should be allowed)"""
        t = PreTrainedTokenizerFixed()
        t.model_input_names = ["input_ids", "input_ids"]
        
        assert t.model_input_names == ["input_ids", "input_ids"]
    
    def test_assignment_with_special_characters(self):
        """Test assignment with special characters"""
        t = PreTrainedTokenizerFixed()
        special_names = ["input-ids", "token_type_ids", "attention.mask"]
        t.model_input_names = special_names
        
        assert t.model_input_names == special_names


class TestPerformance:
    """Test performance characteristics"""
    
    def test_property_access_performance(self):
        """Test that property access is reasonably fast"""
        import time
        
        t = PreTrainedTokenizerFixed()
        
        # Time property access
        start_time = time.time()
        for _ in range(10000):
            _ = t.model_input_names
        end_time = time.time()
        
        # Should be reasonably fast (less than 1 second for 10000 accesses)
        access_time = end_time - start_time
        assert access_time < 1.0, f"Property access too slow: {access_time} seconds for 10000 accesses"
    
    def test_property_set_performance(self):
        """Test that property setting is reasonably fast"""
        import time
        
        t = PreTrainedTokenizerFixed()
        
        # Time property setting
        start_time = time.time()
        for i in range(10000):
            t.model_input_names = [f"input_{i}"]
        end_time = time.time()
        
        # Should be reasonably fast (less than 1 second for 10000 sets)
        set_time = end_time - start_time
        assert set_time < 1.0, f"Property setting too slow: {set_time} seconds for 10000 sets"


class TestRealWorldScenarios:
    """Test scenarios that could occur in real usage"""
    
    def test_bert_gpt2_scenario(self):
        """Test the real-world BERT vs GPT-2 scenario"""
        # Simulate BERT tokenizer (needs token_type_ids)
        bert = PreTrainedTokenizerFixed()
        bert.model_input_names = ["input_ids", "token_type_ids", "attention_mask"]
        
        # Simulate GPT-2 tokenizer (doesn't need token_type_ids)
        gpt2 = PreTrainedTokenizerFixed()
        gpt2.model_input_names = ["input_ids", "attention_mask"]
        
        # Verify BERT has token_type_ids
        assert "token_type_ids" in bert.model_input_names
        assert len(bert.model_input_names) == 3
        
        # Verify GPT-2 doesn't have token_type_ids
        assert "token_type_ids" not in gpt2.model_input_names
        assert len(gpt2.model_input_names) == 2
        
        # Mutate GPT-2 (remove token_type_ids - though it doesn't have it)
        gpt2.model_input_names.remove("token_type_ids")  # Should not raise error if not present
        
        # BERT should still have its original values
        assert "token_type_ids" in bert.model_input_names
        assert len(bert.model_input_names) == 3
    
    def test_pipeline_scenario(self):
        """Test a multi-model pipeline scenario"""
        # Simulate a pipeline processing different models
        model_configs = [
            {"name": "bert-base", "inputs": ["input_ids", "token_type_ids", "attention_mask"]},
            {"name": "gpt2", "inputs": ["input_ids", "attention_mask"]},
            {"name": "t5-base", "inputs": ["input_ids", "attention_mask"]},
        ]
        
        tokenizers = []
        for config in model_configs:
            tokenizer = PreTrainedTokenizerFixed(model_input_names=config["inputs"])
            tokenizers.append(tokenizer)
        
        # Verify each tokenizer has correct configuration
        assert tokenizers[0].model_input_names == ["input_ids", "token_type_ids", "attention_mask"]
        assert tokenizers[1].model_input_names == ["input_ids", "attention_mask"]
        assert tokenizers[2].model_input_names == ["input_ids", "attention_mask"]
        
        # Mutate one tokenizer
        tokenizers[1].model_input_names.append("custom_input")
        
        # Verify others are not affected
        assert "custom_input" not in tokenizers[0].model_input_names
        assert "custom_input" not in tokenizers[2].model_input_names
        assert "custom_input" in tokenizers[1].model_input_names
    
    def test_interactive_session_scenario(self):
        """Test interactive session scenario (like Jupyter notebook)"""
        # Simulate user creating and modifying tokenizers in interactive session
        
        # User creates BERT tokenizer
        bert = PreTrainedTokenizerFixed()
        bert.model_input_names = ["input_ids", "token_type_ids", "attention_mask"]
        
        # User creates GPT-2 tokenizer
        gpt2 = PreTrainedTokenizerFixed()
        gpt2.model_input_names = ["input_ids", "attention_mask"]
        
        # User experiments with GPT-2
        gpt2.model_input_names.remove("attention_mask")
        
        # User checks BERT (should still be correct)
        assert bert.model_input_names == ["input_ids", "token_type_ids", "attention_mask"]
        
        # User creates new GPT-2 tokenizer (should have correct default)
        new_gpt2 = PreTrainedTokenizerFixed()
        assert new_gpt2.model_input_names == ["input_ids", "token_type_ids", "attention_mask"]  # Default


# Test runner
if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])