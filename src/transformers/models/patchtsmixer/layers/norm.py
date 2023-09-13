import torch
from torch import nn

from .basics import Transpose


class NormLayer(nn.Module):
    def __init__(
        self,
        norm_mlp="LayerNorm",
        mode="common_channel",
        num_features=16,
    ):
        super().__init__()
        self.norm_mlp = norm_mlp
        self.mode = mode
        self.num_features = num_features
        if "batch" in norm_mlp.lower():
            self.norm = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(num_features), Transpose(1, 2))
        else:
            self.norm = nn.LayerNorm(num_features)

    def forward(self, x):
        if "batch" in self.norm_mlp.lower():
            if self.mode in ["common_channel", "mix_channel"]:
                # reshape the data
                x_tmp = torch.reshape(
                    x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
                )  # x_tmp: [batch_size*n_vars, num_patches, num_features]
            else:
                x_tmp = x
            x_tmp = self.norm(x_tmp)  # x_tmp: [batch_size*n_vars, num_patches, num_features]
            # put back data to the original shape
            if self.mode in ["common_channel", "mix_channel"]:
                x = torch.reshape(x_tmp, (x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
            else:
                x = x_tmp
        else:
            x = self.norm(x)

        return x
