import torch
from torch import nn


class Patch(nn.Module):
    """
    A class to patchify the time series sequence into different patches
    """
    def __init__(self,
                 seq_len: int,
                 patch_len: int,
                 stride: int,
                 padding: bool = False  # TODO: use this to set whether we want to pad zeros to the sequence
                 ):
        super().__init__()

        assert (seq_len > patch_len), f'Sequence length ({seq_len}) has to be greater than the patch length ({patch_len})'

        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride

        # get the number of patches
        self.num_patch = (max(seq_len, patch_len) - patch_len) // stride + 1
        tgt_len = patch_len + stride * (self.num_patch - 1)
        self.s_begin = seq_len - tgt_len

    def forward(self, x: torch.Tensor):
        """

        Args:
            x (torch.Tensor, required): Input of shape [bs x ... x seq_len x n_vars]
        Returns:
            z: output tensor data [bs x ... x n_vars x num_patch x patch_len]
        """
        seq_len = x.shape[-2]
        assert (seq_len == self.seq_len), f"Input sequence length ({seq_len}) doesn't match model ({self.seq_len})."

        # x = x[:, :, self.s_begin:, :]  # xb: [bs x ... x tgt_len x nvars]
        z = x.transpose(0, -2)[self.s_begin:]    # z: [tgt_len x ... x bs x n_vars]
        z = z.transpose(0, -2).contiguous()     # z: [bs x ... x tgt_len x n_vars]  # TODO: need a better solution
        z = z.unfold(dimension=-2, size=self.patch_len, step=self.stride)  # xb: [bs x ... x num_patch x n_vars x patch_len]
        z = z.transpose(-2, -3).contiguous()  # xb: [bs x ... x n_vars x num_patch x patch_len]
        return z


