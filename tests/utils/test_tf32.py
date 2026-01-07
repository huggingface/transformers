import torch
from packaging import version

from transformers.utils.import_utils import (
    enable_tf32,
    get_torch_version,
)


def test_enable_tf32():
    torch_version = version.parse(get_torch_version())

    if torch_version >= version.parse("2.9.0"):
        original = torch.backends.fp32_precision

        enable_tf32(True)
        assert torch.backends.fp32_precision in ("tf32", "ieee", "none")

        enable_tf32(False)
        assert torch.backends.fp32_precision in ("ieee", "none")

        torch.backends.fp32_precision = original

    else:
        orig_matmul = torch.backends.cuda.matmul.allow_tf32
        orig_cudnn = torch.backends.cudnn.allow_tf32

        enable_tf32(True)
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True

        enable_tf32(False)
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False

        torch.backends.cuda.matmul.allow_tf32 = orig_matmul
        torch.backends.cudnn.allow_tf32 = orig_cudnn
