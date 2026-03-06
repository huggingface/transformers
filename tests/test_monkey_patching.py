# Copyright 2026 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import threading
import unittest

from transformers import is_torch_available
from transformers.monkey_patching import (
    apply_patches,
    clear_patch_mapping,
    get_patch_mapping,
    patch_output_recorders,
    register_patch_mapping,
    unregister_patch_mapping,
)
from transformers.testing_utils import require_torch
from transformers.utils.output_capturing import OutputRecorder


if is_torch_available():
    import torch.nn as nn


@require_torch
class MonkeyPatchTest(unittest.TestCase):
    def setUp(self):
        """Clear any existing patches before each test."""
        clear_patch_mapping()

    def tearDown(self):
        """Clean up patches after each test."""
        clear_patch_mapping()

    def test_register_patch_mapping(self):
        """Test basic registration of monkey patches."""

        class CustomModule(nn.Module):
            pass

        # Register a patch
        register_patch_mapping(mapping={"TestModule": CustomModule})

        # Verify it was registered
        mapping = get_patch_mapping()
        self.assertIn("TestModule", mapping)
        self.assertEqual(mapping["TestModule"], CustomModule)

    def test_register_duplicate_with_overwrite(self):
        """Test that registering a duplicate class with overwrite=True works."""

        class CustomModule1(nn.Module):
            pass

        class CustomModule2(nn.Module):
            pass

        # Register initial patch
        register_patch_mapping(mapping={"TestModule": CustomModule1})

        # Overwrite with new patch
        register_patch_mapping(mapping={"TestModule": CustomModule2}, overwrite=True)

        # Verify the new patch is registered
        mapping = get_patch_mapping()
        self.assertEqual(mapping["TestModule"], CustomModule2)

    def test_register_non_nn_module_raises_error(self):
        """Test that registering a non-nn.Module class raises TypeError."""

        class NotAModule:
            pass

        # Try to register a non-nn.Module class
        with self.assertRaises(TypeError) as context:
            register_patch_mapping(mapping={"TestModule": NotAModule})

        self.assertIn("must be a subclass of nn.Module", str(context.exception))

    def test_unregister_patch_mapping(self):
        """Test unregistering monkey patches."""

        class CustomModule(nn.Module):
            pass

        # Register and then unregister
        register_patch_mapping(mapping={"TestModule": CustomModule})
        unregister_patch_mapping(["TestModule"])

        # Verify it was unregistered
        mapping = get_patch_mapping()
        self.assertNotIn("TestModule", mapping)

    def test_unregister_nonexistent_class(self):
        """Test unregistering a class that doesn't exist raises an error."""
        # This should raise an error
        with self.assertRaises(ValueError) as context:
            unregister_patch_mapping(["NonexistentModule"])

        self.assertIn("not found in monkey patch mapping cache", str(context.exception))
        self.assertIn("Cannot unregister", str(context.exception))

    def test_clear_patch_mapping(self):
        """Test clearing all monkey patches."""

        class CustomModule1(nn.Module):
            pass

        class CustomModule2(nn.Module):
            pass

        # Register multiple patches
        register_patch_mapping(mapping={"TestModule1": CustomModule1, "TestModule2": CustomModule2})

        # Clear all patches
        clear_patch_mapping()

        # Verify all were cleared
        mapping = get_patch_mapping()
        self.assertEqual(len(mapping), 0)

    def test_get_patch_mapping_returns_copy(self):
        """Test that get_patch_mapping returns a copy, not the original."""

        class CustomModule(nn.Module):
            pass

        register_patch_mapping(mapping={"TestModule": CustomModule})

        # Get mapping and modify it
        mapping = get_patch_mapping()
        mapping["NewModule"] = CustomModule

        # Verify the internal cache was not modified
        internal_mapping = get_patch_mapping()
        self.assertNotIn("NewModule", internal_mapping)

    def test_apply_patches_context_manager(self):
        """Test that apply_patches context manager works correctly."""

        class CustomLinear(nn.Linear):
            pass

        # Create a dummy module in transformers namespace for testing
        import types

        test_module = types.ModuleType("transformers.test_module")
        test_module.Linear = nn.Linear
        sys.modules["transformers.test_module"] = test_module

        try:
            # Register patch
            register_patch_mapping(mapping={"Linear": CustomLinear})

            # Outside context, original class should be used
            self.assertEqual(test_module.Linear, nn.Linear)

            # Inside context, patched class should be used
            with apply_patches():
                self.assertEqual(test_module.Linear, CustomLinear)

            # Outside context again, original class should be restored
            self.assertEqual(test_module.Linear, nn.Linear)

        finally:
            # Clean up the test module
            del sys.modules["transformers.test_module"]

    def test_thread_safety_concurrent_access(self):
        """Test that concurrent reads and writes are thread-safe."""

        class CustomModule(nn.Module):
            pass

        results = []

        def read_mapping():
            for _ in range(100):
                mapping = get_patch_mapping()
                results.append(len(mapping))

        def write_mapping():
            for i in range(100):
                mapping = {f"Module{i}": CustomModule}
                register_patch_mapping(mapping=mapping)

        # Create threads for reading and writing
        read_thread = threading.Thread(target=read_mapping)
        write_thread = threading.Thread(target=write_mapping)

        read_thread.start()
        write_thread.start()

        read_thread.join()
        write_thread.join()

        # Test should complete without deadlocks or errors
        self.assertEqual(len(results), 100)

    def test_patch_output_recorders_with_output_recorder_instance(self):
        """Test patching output recorders that are OutputRecorder instances."""

        class OriginalModule(nn.Module):
            pass

        class ReplacementModule(nn.Module):
            pass

        class TestModel(nn.Module):
            # Simulate _can_record_outputs with OutputRecorder
            _can_record_outputs = {"output": OutputRecorder(OriginalModule)}

            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

        model = TestModel()

        # Register patch
        register_patch_mapping(mapping={"OriginalModule": ReplacementModule})

        # Patch output recorders
        patch_output_recorders(model)

        # Verify the recorder's target_class was updated
        recorder = model._can_record_outputs["output"]
        self.assertEqual(recorder.target_class, ReplacementModule)

    def test_patch_output_recorders_with_class_type(self):
        """Test patching output recorders that are class types directly."""

        class OriginalModule(nn.Module):
            pass

        class ReplacementModule(nn.Module):
            pass

        class TestModel(nn.Module):
            # Simulate _can_record_outputs with class type directly
            _can_record_outputs = {"output": OriginalModule}

            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

        model = TestModel()

        # Register patch
        register_patch_mapping(mapping={"OriginalModule": ReplacementModule})

        # Patch output recorders
        patch_output_recorders(model)

        # Verify the class was updated
        self.assertEqual(model._can_record_outputs["output"], ReplacementModule)

    def test_pattern_matching_wildcard(self):
        """Test pattern matching with .* regex (matches any characters)."""

        class CustomAttention(nn.Module):
            pass

        # Register with pattern
        register_patch_mapping(mapping={".*Attention": CustomAttention})

        # Create test modules with different attention classes
        import types

        test_module = types.ModuleType("transformers.test_pattern")
        test_module.BertAttention = nn.Module
        test_module.GPT2Attention = nn.Module
        test_module.T5Attention = nn.Module
        test_module.SomethingElse = nn.Module
        sys.modules["transformers.test_pattern"] = test_module

        try:
            with apply_patches():
                # All *Attention classes should be patched
                self.assertEqual(test_module.BertAttention, CustomAttention)
                self.assertEqual(test_module.GPT2Attention, CustomAttention)
                self.assertEqual(test_module.T5Attention, CustomAttention)
                # Non-matching class should not be patched
                self.assertNotEqual(test_module.SomethingElse, CustomAttention)
        finally:
            del sys.modules["transformers.test_pattern"]

    def test_exact_match_precedence_over_pattern(self):
        """Test that exact matches take precedence over patterns."""

        class PatternReplacement(nn.Module):
            pass

        class ExactReplacement(nn.Module):
            pass

        # Register both pattern and exact match
        register_patch_mapping(mapping={".*Attention": PatternReplacement})
        register_patch_mapping(mapping={"BertAttention": ExactReplacement})

        import types

        test_module = types.ModuleType("transformers.test_precedence")
        test_module.BertAttention = nn.Module
        test_module.GPT2Attention = nn.Module
        sys.modules["transformers.test_precedence"] = test_module

        try:
            with apply_patches():
                # Exact match should take precedence
                self.assertEqual(test_module.BertAttention, ExactReplacement)
                # Pattern should still match others
                self.assertEqual(test_module.GPT2Attention, PatternReplacement)
        finally:
            del sys.modules["transformers.test_precedence"]

    def test_pattern_with_output_recorders(self):
        """Test pattern matching works with output recorders."""

        class OriginalAttention(nn.Module):
            pass

        class ReplacementAttention(nn.Module):
            pass

        class TestModel(nn.Module):
            _can_record_outputs = {"output": OutputRecorder(OriginalAttention)}

            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

        model = TestModel()

        # Register with pattern
        register_patch_mapping(mapping={".*Attention": ReplacementAttention})

        # Patch output recorders
        patch_output_recorders(model)

        # Verify the recorder's target_class was updated via pattern matching
        recorder = model._can_record_outputs["output"]
        self.assertEqual(recorder.target_class, ReplacementAttention)


if __name__ == "__main__":
    unittest.main()
