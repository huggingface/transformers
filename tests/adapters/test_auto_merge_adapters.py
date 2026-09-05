import pytest
from transformers.adapters.auto_merge_adapters import AutoMergeAdapters

def test_merge_no_adapters():
    with pytest.raises(ValueError):
        AutoMergeAdapters.merge(None, [])
