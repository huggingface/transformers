import torch

from transformers.trainer_pt_utils import nested_truncate


def test_nested_truncate_tuple_of_lists_truncates_at_sample_level():
    """
    Regression test for tuple[list[Tensor], ...] structures.

    When truncating the last batch, truncation must happen at the
    sample/list level and must not truncate tensor dimensions or
    drop tuple elements.
    """
    remainder = 1

    labels = (
        [torch.randn(6, 4), torch.randn(3, 4)],
        [torch.randint(0, 5, (6,)), torch.randint(0, 5, (3,))],
    )

    truncated = nested_truncate(labels, remainder)

    assert isinstance(truncated, tuple)
    assert len(truncated) == 2

    assert isinstance(truncated[0], list)
    assert isinstance(truncated[1], list)

    assert len(truncated[0]) == remainder
    assert len(truncated[1]) == remainder
