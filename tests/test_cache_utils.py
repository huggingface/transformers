import pytest
import torch

from transformers.cache_utils import StaticLayer, StaticSlidingWindowLayer


def _init_layer(layer, batch_size=2, num_heads=4, head_dim=8, seq_len=5):
    """Helper to lazily initialize a layer by feeding it key/value states."""
    keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
    values = torch.randn(batch_size, num_heads, seq_len, head_dim)
    layer.update(keys, values)
    return layer


class TestStaticLayerCrop:
    def test_crop_basic(self):
        layer = StaticLayer(max_cache_len=10)
        _init_layer(layer, seq_len=7)
        assert layer.cumulative_length.item() == 7

        layer.crop(4)

        assert layer.cumulative_length.item() == 4
        assert (layer.keys[:, :, 4:, :] == 0).all()
        assert (layer.values[:, :, 4:, :] == 0).all()
        assert (layer.keys[:, :, :4, :] != 0).any()

    def test_crop_negative(self):
        layer = StaticLayer(max_cache_len=10)
        _init_layer(layer, seq_len=6)
        assert layer.cumulative_length.item() == 6

        layer.crop(-2)

        assert layer.cumulative_length.item() == 4
        assert (layer.keys[:, :, 4:, :] == 0).all()
        assert (layer.values[:, :, 4:, :] == 0).all()

    def test_crop_noop_when_shorter(self):
        layer = StaticLayer(max_cache_len=10)
        _init_layer(layer, seq_len=3)
        original_keys = layer.keys.clone()

        layer.crop(5)

        assert layer.cumulative_length.item() == 3
        assert torch.equal(layer.keys, original_keys)

    def test_crop_noop_when_equal(self):
        layer = StaticLayer(max_cache_len=10)
        _init_layer(layer, seq_len=5)
        original_keys = layer.keys.clone()

        layer.crop(5)

        assert layer.cumulative_length.item() == 5
        assert torch.equal(layer.keys, original_keys)

    def test_crop_uninitialized(self):
        layer = StaticLayer(max_cache_len=10)
        layer.crop(5)
        assert not layer.is_initialized

    def test_crop_preserves_tensor_identity(self):
        """Static tensors must keep the same object identity (for torch.compile)."""
        layer = StaticLayer(max_cache_len=10)
        _init_layer(layer, seq_len=7)
        keys_id = id(layer.keys)
        values_id = id(layer.values)
        cumlen_id = id(layer.cumulative_length)

        layer.crop(3)

        assert id(layer.keys) == keys_id
        assert id(layer.values) == values_id
        assert id(layer.cumulative_length) == cumlen_id

    def test_crop_then_update(self):
        """After cropping, new tokens should be written at the correct position."""
        layer = StaticLayer(max_cache_len=10)
        _init_layer(layer, seq_len=5)
        layer.crop(3)

        new_keys = torch.ones(2, 4, 2, 8)
        new_values = torch.ones(2, 4, 2, 8)
        layer.update(new_keys, new_values)

        assert layer.cumulative_length.item() == 5
        assert (layer.keys[:, :, 3:5, :] == 1).all()

    def test_crop_negative_overflow(self):
        """Negative crop exceeding current length clamps to 0."""
        layer = StaticLayer(max_cache_len=10)
        _init_layer(layer, seq_len=3)

        layer.crop(-10)

        assert layer.cumulative_length.item() == 0
        assert (layer.keys == 0).all()
        assert (layer.values == 0).all()


class TestStaticSlidingWindowLayerCrop:
    def test_crop_basic(self):
        layer = StaticSlidingWindowLayer(max_cache_len=20, sliding_window=10)
        _init_layer(layer, seq_len=5)
        assert layer.cumulative_length_int == 5

        layer.crop(3)

        assert layer.cumulative_length_int == 3
        assert layer.cumulative_length.item() == 3
        assert (layer.keys[:, :, 3:, :] == 0).all()

    def test_crop_raises_when_full(self):
        layer = StaticSlidingWindowLayer(max_cache_len=20, sliding_window=6)
        _init_layer(layer, seq_len=4)
        new_keys = torch.randn(2, 4, 3, 8)
        new_values = torch.randn(2, 4, 3, 8)
        layer.update(new_keys, new_values)

        with pytest.raises(ValueError, match="Cannot `crop`"):
            layer.crop(3)

    def test_crop_negative(self):
        layer = StaticSlidingWindowLayer(max_cache_len=20, sliding_window=10)
        _init_layer(layer, seq_len=6)

        layer.crop(-2)

        assert layer.cumulative_length_int == 4
        assert layer.cumulative_length.item() == 4
