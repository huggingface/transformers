# Copyright 2024 HuggingFace Inc.
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
"""Unit tests for modeling_utils helpers (non-model-specific)."""

import tempfile
import unittest

import torch
import torch.nn as nn

from transformers import PreTrainedConfig, PreTrainedModel
from transformers.testing_utils import require_torch


# ---------------------------------------------------------------------------
# Minimal model fixtures for buffer-related tests
# ---------------------------------------------------------------------------

# Sentinel value that the non-persistent buffer is always initialized to.
_SENTINEL = 42.0


class _TinyConfig(PreTrainedConfig):
    """Minimal config for a tiny test model."""

    model_type = "tiny_test_non_persistent_buffer"

    def __init__(self, hidden_size: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size


class _TinyModelWithNonPersistentBuffer(PreTrainedModel):
    """
    Minimal model that holds:
      - A regular Parameter (so save_pretrained / from_pretrained round-trips work).
      - A *non-persistent* buffer initialised to a known sentinel value (42.0).

    v5 design contract for _init_weights
    -------------------------------------
    When low_cpu_mem_usage=True (default in v5), ALL tensors begin on the meta
    device.  After weight loading, non-persistent buffers are moved from meta to CPU
    as zeros (after the fix to _move_missing_keys_from_meta_to_device), then
    _initialize_missing_keys() calls _init_weights on every sub-module so that
    models can restore the correct values.

    This model honours that contract by re-filling non_persistent_buf in
    _init_weights.  Any real model with non-zero non-persistent buffers must do the
    same to survive a from_pretrained round-trip on the meta-device path.
    """

    config_class = _TinyConfig
    # Silence "missing weights" warning — the buffer is intentionally absent from
    # checkpoints (non-persistent buffers are never saved).
    _keys_to_ignore_on_load_missing = ["non_persistent_buf"]

    def __init__(self, config: _TinyConfig):
        super().__init__(config)
        # A real parameter so the model has something to save / load.
        self.weight = nn.Parameter(torch.ones(config.hidden_size))

        # Non-persistent buffer initialised to a recognisable sentinel value.
        # persistent=False means it is NOT saved to / loaded from state_dict.
        self.register_buffer(
            "non_persistent_buf",
            torch.full((config.hidden_size,), fill_value=_SENTINEL),
            persistent=False,
        )
        # Required: every concrete PreTrainedModel must call post_init() explicitly.
        # It is NOT automatically called by PreTrainedModel.__init__.
        self.post_init()

    def _init_weights(self, module):
        """
        Re-initialize buffers after meta-device -> real-device migration.

        For the meta-device loading path (low_cpu_mem_usage=True):
          Non-persistent buffers arrive here as zeros (after the _move fix).
          This method restores the sentinel value, which is the correct pattern
          for any non-persistent buffer that must hold a non-zero constant.
        """
        if module is self and hasattr(module, "non_persistent_buf"):
            module.non_persistent_buf.fill_(_SENTINEL)

    def forward(self, x=None):
        return self.weight


@require_torch
class NonPersistentBufferRegressionTest(unittest.TestCase):
    """
    Regression tests for GitHub issue #44534.

    Bug summary
    -----------
    _move_missing_keys_from_meta_to_device() called torch.empty_like() on ALL
    non-persistent buffers regardless of whether they were already on a real device.
    torch.empty returns *uninitialized* memory, silently corrupting positional
    encodings, attention masks, and any carefully-set buffer value.

    Two scenarios are fixed:

    Scenario A - CPU path (low_cpu_mem_usage=False)
        Buffer is initialized on CPU in __init__.  The old code overwrote it with
        random garbage.  Fix: guard skips buffers already on a real (non-meta) device.

    Scenario B - meta-device path (low_cpu_mem_usage=True, v5 default)
        Buffer is on meta during init (no real data).  The old code used
        torch.empty_like when moving to CPU -> garbage.  Fix: use torch.zeros_like
        (deterministic zeros), then _init_weights restores the correct value.
    """

    def _make_model(self):
        return _TinyModelWithNonPersistentBuffer(_TinyConfig(hidden_size=8))

    # ------------------------------------------------------------------
    # Sanity checks (no from_pretrained)
    # ------------------------------------------------------------------

    def test_non_persistent_buffer_not_in_state_dict(self):
        """Non-persistent buffer must NOT appear in state_dict() — PyTorch invariant."""
        model = self._make_model()
        self.assertNotIn("non_persistent_buf", model.state_dict())

    def test_non_persistent_buffer_present_in_named_buffers(self):
        """Buffer must appear in named_buffers() even though absent from state_dict()."""
        model = self._make_model()
        self.assertIn("non_persistent_buf", {n for n, _ in model.named_buffers()})

    def test_named_non_persistent_buffers_helper(self):
        """named_non_persistent_buffers() must yield exactly persistent=False buffers."""
        model = self._make_model()
        non_persistent = dict(model.named_non_persistent_buffers())
        self.assertIn("non_persistent_buf", non_persistent)
        all_buffers = dict(model.named_buffers())
        for name in non_persistent:
            self.assertIn(name, all_buffers)

    # ------------------------------------------------------------------
    # Scenario A: CPU path (low_cpu_mem_usage=False)
    # Fix: guard skips buffers already on a real device -> value preserved.
    # ------------------------------------------------------------------

    def test_buffer_preserved_cpu_path(self):
        """
        Scenario A (CPU path): the buffer's sentinel value must survive
        save_pretrained -> from_pretrained with low_cpu_mem_usage=False.

        Before the fix: torch.empty_like unconditionally overwrote the buffer
        with random garbage even though it was already on CPU.
        After the fix:  guard detects device != 'meta', skips the buffer.
        """
        model = self._make_model()
        expected = torch.full((8,), _SENTINEL)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            reloaded = _TinyModelWithNonPersistentBuffer.from_pretrained(tmp_dir, low_cpu_mem_usage=False)

        actual = reloaded.non_persistent_buf
        self.assertTrue(
            torch.equal(actual, expected),
            f"[CPU path / issue #44534] Buffer corrupted.\n"
            f"  expected: {expected.tolist()}\n"
            f"  got:      {actual.tolist()}",
        )

    def test_buffer_device_correct_cpu_path(self):
        """Scenario A: buffer must be on the same device as the model weights."""
        model = self._make_model()
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            reloaded = _TinyModelWithNonPersistentBuffer.from_pretrained(tmp_dir, low_cpu_mem_usage=False)
        self.assertEqual(reloaded.non_persistent_buf.device, reloaded.weight.device)

    # ------------------------------------------------------------------
    # Scenario B: meta-device path (low_cpu_mem_usage=True, v5 default)
    # Fix: zeros_like (not empty_like) + _init_weights restores the value.
    # ------------------------------------------------------------------

    def test_buffer_preserved_meta_path(self):
        """
        Scenario B (meta path): the buffer's sentinel value must survive
        save_pretrained -> from_pretrained with low_cpu_mem_usage=True (v5 default).

        Fix flow:
          1. Model __init__ on meta device  -> buffer has no data.
          2. Checkpoint weights loaded.
          3. _move_missing_keys_from_meta_to_device -> torch.zeros_like (not garbage).
          4. _initialize_missing_keys -> _init_weights(self) -> buffer = 42.0.

        This test confirms that both parts of the fix work together.
        """
        model = self._make_model()
        expected = torch.full((8,), _SENTINEL)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            reloaded = _TinyModelWithNonPersistentBuffer.from_pretrained(tmp_dir, low_cpu_mem_usage=True)

        actual = reloaded.non_persistent_buf
        self.assertTrue(
            torch.equal(actual, expected),
            f"[Meta path / issue #44534] Buffer corrupted.\n"
            f"  expected: {expected.tolist()}\n"
            f"  got:      {actual.tolist()}",
        )

    def test_buffer_not_garbage_meta_path(self):
        """Scenario B: every buffer element must be finite (not random uninitialized memory)."""
        model = self._make_model()
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            reloaded = _TinyModelWithNonPersistentBuffer.from_pretrained(tmp_dir, low_cpu_mem_usage=True)
        buf = reloaded.non_persistent_buf
        self.assertTrue(
            torch.isfinite(buf).all(),
            f"[Meta path] Buffer has non-finite (garbage) values: {buf.tolist()}",
        )

    def test_buffer_preserved_empty_state_dict(self):
        """
        Even with an empty state_dict (simulates a model with no pretrained weights),
        the non-persistent buffer must have the correct value after _init_weights runs.
        """
        config = _TinyConfig(hidden_size=8)
        model = _TinyModelWithNonPersistentBuffer.from_pretrained(None, config=config, state_dict={})
        expected = torch.full((8,), _SENTINEL)
        self.assertTrue(
            torch.equal(model.non_persistent_buf, expected),
            f"Buffer corrupted with empty state_dict.\n"
            f"  expected: {expected.tolist()}\n"
            f"  got:      {model.non_persistent_buf.tolist()}",
        )


if __name__ == "__main__":
    unittest.main()
