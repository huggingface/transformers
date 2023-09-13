import inspect

import torch
from torch import nn


def mask_data(z, mask):
    """
    z: [bs x n_vars x num_patch x num_features] mask: [bs x n_vars x num_patch] (bool)
    """
    # mask the z based on mask flag
    # print("z",z.shape)
    # print(mask.shape)
    # print(mask)
    inv_mask = ~mask  # [bs x n_vars x num_patch] # swap 0,1
    # after this step, masked position will have 0
    inv_mask = inv_mask.unsqueeze(-1).repeat(1, 1, 1, z.shape[-1])  # [bs x n_vars x num_patch x num_features]
    # print(inv_mask.shape)
    # print(inv_mask)
    z = z * inv_mask  # [bs x n_vars x num_patch x num_features]
    # print(z.shape)
    # print(z)
    return z


def mask_data_6d(z, mask):
    """
    z: [bs x ts_group1 x ts_group2 x channels x num_patch x num_features] mask: [bs x ts_group1 x ts_group2 x channels
    x num_patch] (bool)
    """
    inv_mask = ~mask  # [bs x ts_group1 x ts_group2 x channels x num_patch] # swap 0,1
    # after this step, masked position will have 0
    inv_mask = inv_mask.unsqueeze(-1).repeat(
        1, 1, 1, 1, 1, z.shape[-1]
    )  # [bs x ts_group1 x ts_group2 x channels x num_patch x num_features]
    z = z * inv_mask  # [bs x ts_group1 x ts_group2 x channels x num_patch x num_features]
    # print(z.shape)
    # print(z)
    return z


class ResidualAddMAE(nn.Module):
    def __init__(self, patch_size, num_features):
        super().__init__()
        self.patcher = nn.Linear(patch_size, num_features)

    def forward(self, z, raw_data):
        """
        raw_data: tensor [bs x n_vars x num_patch x patch_len] z: [bs x n_vars x num_patch x num_features]
        """
        # raw_data = raw_data.transpose(1,2) # [bs  x n_vars x num_patch x patch_len]
        return z + self.patcher(raw_data)  # [bs x n_vars x num_patch x num_features]


class InjectRevinStatistics4D(nn.Module):
    def __init__(self, num_features, num_patches, expansion=2):
        super().__init__()
        self.inverse_transform = nn.Sequential(
            nn.Linear(num_features + 2, expansion * num_features),
            nn.Linear(expansion * num_features, num_features),
        )

        self.map_scale = nn.Sequential(nn.Linear(2, 2 * expansion), nn.Linear(2 * expansion, 2))
        self.num_patches = num_patches

    def forward(self, z, revin_statistics):
        """
        # revin_mean,revin_stddev: [bs x 1 x n_channels] z: [bs x in_channels x num_patch x num_features]

        output: [bs x in_channels x num_patch x num_features]
        """

        revin_mean, revin_stdev = revin_statistics
        revin_mean = revin_mean.transpose(-1, -2)  # [bs x n_channels x 1 ]
        revin_mean = revin_mean.unsqueeze(-2)  # [bs x n_channels x 1 x 1]
        revin_mean = revin_mean.repeat(1, 1, self.num_patches, 1)  # [bs x n_channels x num_patch x 1]

        revin_stdev = revin_stdev.transpose(-1, -2)  # [bs x n_channels x 1 ]
        revin_stdev = revin_stdev.unsqueeze(-2)  # [bs x n_channels x 1 x 1]
        revin_stdev = revin_stdev.repeat(1, 1, self.num_patches, 1)  # [bs x n_channels x num_patch x 1]

        revin_full = torch.cat([revin_mean, revin_stdev], dim=-1)  # [bs x n_channels x num_patch x 2]

        revin_full = self.map_scale(revin_full)  # [bs x n_channels x num_patch x 2]

        z = torch.cat([z, revin_full], dim=-1)  # [bs x channels x num_patch x num_features+2]
        z = self.inverse_transform(z)  # [bs x channels x num_patch x num_features]

        return z


class PatchForecastReshape(nn.Module):
    def __init__(self, num_patches, out_patches):
        super().__init__()
        self.patch_mix = nn.Linear(num_patches, out_patches)

    def forward(self, z):
        """
        z: [bs x n_vars x num_patch x num_features]
        """
        z = z.transpose(-1, -2)  # [bs x n_vars x num_features x num_patches]
        z = self.patch_mix(z)  # [bs x n_vars x num_features x out_patches]
        z = z.transpose(-1, -2)  # [bs x n_vars x out_patches x num_features]
        return z


def get_class_params_via_inspect(input_frame):
    class_params = {}
    args, _, _, values = inspect.getargvalues(input_frame)

    for k in args:
        if k != "self":
            class_params[k] = values[k]

    if "kwargs" in values:
        class_params.update(values["kwargs"])
    return class_params
