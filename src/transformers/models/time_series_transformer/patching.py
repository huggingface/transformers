import torch
from torch import nn


def test_patch():
    # create random tensor with size bsz * channel * time
    x = torch.rand(128, 1, 336)
    time_length = 336
    patch_len = 16
    stride = 8
    x = patch(x, stride=stride, patch_size=patch_len, dim=-1)

    num_of_patches = int((time_length - patch_len)/stride + 2)
    assert x.size() == torch.Size([128, 1, 16, 42])
    assert num_of_patches == 42


def patch(x, stride, patch_size, dim):
    """
    Args:
        x (torch.Tensor): input tensor, shape (bsz, time, channel)
        dim (int): dimension to be patched
        stride (int): stride of patching
        patch_size (int): size of patching

    Returns:
        torch.Tensor: patched tensor, shape
    """
    padding_patch_layer = nn.ReplicationPad1d((0, stride))
    x = padding_patch_layer(x)
    x_unfold = x.unfold(dimension=dim, size=patch_size, step=stride)  # z: [bs x nvars x patch_num x patch_len]
    x = x_unfold.permute(0, 1, 3, 2)  # x: [bs x nvars x patch_len x patch_num]
    return x
