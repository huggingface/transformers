import torch
from torch import nn
import inspect
def mask_data(z, mask):
    """
    z: [bs x n_vars x num_patch x num_features]
    mask: [bs x n_vars x num_patch] (bool)
    """
    # mask the z based on mask flag
    # print("z",z.shape)
    # print(mask.shape)
    # print(mask)
    inv_mask = ~mask  # [bs x n_vars x num_patch] # swap 0,1
    # after this step, masked position will have 0
    inv_mask = inv_mask.unsqueeze(-1).repeat(
        1, 1, 1, z.shape[-1]
    )  # [bs x n_vars x num_patch x num_features]
    # print(inv_mask.shape)
    # print(inv_mask)
    z = z * inv_mask  # [bs x n_vars x num_patch x num_features]
    # print(z.shape)
    # print(z)
    return z


def mask_data_6d(z, mask):
    """
    z: [bs x ts_group1 x ts_group2 x channels x num_patch x num_features]
    mask: [bs x ts_group1 x ts_group2 x channels x num_patch] (bool)
    """
    inv_mask = ~mask  # [bs x ts_group1 x ts_group2 x channels x num_patch] # swap 0,1
    # after this step, masked position will have 0
    inv_mask = inv_mask.unsqueeze(-1).repeat(
        1, 1, 1, 1, 1, z.shape[-1]
    )  # [bs x ts_group1 x ts_group2 x channels x num_patch x num_features]
    z = (
        z * inv_mask
    )  # [bs x ts_group1 x ts_group2 x channels x num_patch x num_features]
    # print(z.shape)
    # print(z)
    return z


class ResidualAddMAE(nn.Module):
    def __init__(self, patch_size, num_features):
        super().__init__()
        self.patcher = nn.Linear(patch_size, num_features)

    def forward(self, z, raw_data):
        """
        raw_data: tensor [bs x n_vars x num_patch  x patch_len]
        z: [bs x n_vars x num_patch x num_features]
        """
        # raw_data = raw_data.transpose(1,2) # [bs  x n_vars x num_patch x patch_len]
        return z + self.patcher(raw_data)  # [bs x n_vars x num_patch x num_features]


class InjectRevinStatistics4D(nn.Module):
    def __init__(self, num_features, num_patches, expansion = 2):
        super().__init__()
        self.inverse_transform = nn.Sequential(
                                nn.Linear(num_features+2, expansion*num_features),
                                nn.Linear(expansion*num_features, num_features),
                                )

        self.map_scale = nn.Sequential(nn.Linear(2,2*expansion),nn.Linear(2*expansion,2))
        self.num_patches = num_patches

    def forward(self, z, revin_statistics):
        """
        # revin_mean,revin_stddev: [bs x 1 x n_channels]
        z: [bs x in_channels x num_patch x num_features]

        output: [bs x in_channels x num_patch x num_features]
        """
        
        revin_mean, revin_stdev = revin_statistics
        revin_mean = revin_mean.transpose(-1,-2) # [bs x n_channels x 1 ]
        revin_mean = revin_mean.unsqueeze(-2) # [bs x n_channels x 1 x 1]
        revin_mean = revin_mean.repeat(1,1,self.num_patches,1) # [bs x n_channels x num_patch x 1]

        revin_stdev = revin_stdev.transpose(-1,-2) # [bs x n_channels x 1 ]
        revin_stdev = revin_stdev.unsqueeze(-2) # [bs x n_channels x 1 x 1]
        revin_stdev = revin_stdev.repeat(1,1,self.num_patches,1) # [bs x n_channels x num_patch x 1]

        revin_full = torch.cat([revin_mean,revin_stdev],dim = -1) # [bs x n_channels x num_patch x 2]

        revin_full = self.map_scale(revin_full) # [bs x n_channels x num_patch x 2]

        z  = torch.cat([z,revin_full], dim = -1) # [bs x channels x num_patch x num_features+2]
        z = self.inverse_transform(z) # [bs x channels x num_patch x num_features]
        
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


class PatchPredictionHead(nn.Module):
    """PredictionHead

    Args:
        num_patches (int): Number of patches to segment
        patch_size (int, optional): Patch length. Defaults to 16.
        in_channels (int, optional): Number of input variables. Defaults to 3.
        num_features (int, optional): Hidden feature size. Defaults to 16.
        expansion_factor (int, optional): Expansion factor to use inside MLP. Defaults to 2.
        head_dropout (float, optional): Head Dropout rate. Defaults to 0.2.
        forecast_len (int, optional): Forecast Length. Defaults to 16.
        mode (str, optional): Mixer Mode. Determines how to process the channels. Allowed values: flatten,
            common_channel, mix_channel. In flatten, patch embedding encodes the patch information across all channels.
            In common_channel mode, patch embedding is independent of channels (Channel Independece). In mix_channel,
            we follow channel independence, but in addition to patch and feature mixing, we also do channel mixing.
            Defaults to "common_channel".
        time_hierarchy (bool, optional): Enable time hierarchy in heads. Defaults to False.
        teacher_forcing (bool, optional): Enable teacher forcing in heads while using teacher forcing. Defaults to True.
        head_avg_pool (bool, optional): Use Head avg pool across patches instead of standard flatten (not preferred). Defaults to False.
        head_attn (bool, optional): Enable gated attention in head. (Not preferred) Defaults to False.
        th_mode (str, optional): Mode to mix the hierarchy signals. Allowed values are plain, reconcile.
            reconcile not preferred based on current experiments. Defaults to "reconcile".
    """

    def __init__(
        self,
        num_patches: int,
        in_channels: int = 3,
        patch_size: int = 16,
        num_features: int = 16,
        forecast_len: int = 16,
        head_dropout: float = 0.2,
        mode: str = "common_channel",  # flatten, common_channel, mix_channel
        time_hierarchy: bool = True,
        th_mode: str = "reconcile",  # plain, reconcile
        expansion_factor: int = 2,
        teacher_forcing: bool = False,
        forecast_channel_mixing=False,
        cm_gated_attn=True,
        cm_teacher_forcing=False,
        channel_context_length=0,
    ):
        super().__init__()
        self.forecast_len = forecast_len
        self.nvars = in_channels
        self.num_features = num_features
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.time_hierarchy = time_hierarchy
        self.mode = mode
        self.expansion_factor = expansion_factor
        self.teacher_forcing = teacher_forcing
        self.th_mode = th_mode
        self.forecast_channel_mixing = forecast_channel_mixing
        self.cm_gated_attn = cm_gated_attn
        self.cm_teacher_forcing = cm_teacher_forcing

        if num_patches * patch_size != forecast_len:
            raise Exception("num_patches*patch_size should be equal to forecast_len")

        if self.forecast_channel_mixing:
            if self.mode not in ["common_channel", "mix_channel"]:
                raise Exception(
                    "Forecast channel mixing can be enabled only when backbone mode is common_channel or mix_channel"
                )

            self.fcm = ForecastChannelMixer(
                forecast_channels=in_channels,
                cm_expansion_factor=expansion_factor,
                cm_gated_attn=self.cm_gated_attn,
                cm_dropout=head_dropout,
                cm_teacher_forcing=self.cm_teacher_forcing,
                channel_context_length=channel_context_length,
            )

        if self.mode in ["common_channel", "mix_channel"]:
            self.flatten = nn.Flatten(start_dim=-2)
            self.base_forecast_block = nn.Sequential(
                nn.Dropout(head_dropout),
                nn.Linear(num_features, patch_size),
            )

        else:
            self.base_forecast_block = nn.Sequential(
                nn.Dropout(head_dropout),
                nn.Linear((num_features * in_channels), patch_size * in_channels),
            )
            self.flatten = nn.Flatten(start_dim=1)

        if self.time_hierarchy:
            if self.mode not in ["common_channel", "mix_channel"]:
                raise Exception(
                    "Hierarchy Tuner can be enabled only when backbone mode is common_channel or mix_channel"
                )

            self.ht_block = HierarchyPredictionTuner(
                forecast_len=forecast_len,
                patch_size=patch_size,
                num_features=num_features,
                num_patches=num_patches,
                head_dropout=head_dropout,
                teacher_forcing=teacher_forcing,
                th_mode=th_mode,
                hier_model_type="linear",
            )

    def forward(self, x, y=None):
        """
        # x: [bs x num_patch x num_features] flatten mode or
            [bs x n_vars x num_patch x num_features] common_channel/mix_channel

        time_hierarchy enabled (teacher forcing during training):
            y: ([bs x forecast_patches x nvars],[bs x forecast_len x nvars]) # y_hier, actual_y
        no time hierarchy:
            y: None

        Output:

        time_hierarchy enabled (teacher forcing during training):
            output: ([bs x forecast_patches x nvars],[bs x forecast_len x nvars]) # y_hier, actual_y
        no time hierarchy:
            output: [bs x forecast_len x nvars]

        """

        if self.mode in ["common_channel", "mix_channel"]:
            base_x = self.flatten(x)  # [bs x n_vars x num_patch * num_features]
            # base_x = torch.reshape(
            #     x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
            # )  # [bs x n_vars x num_patch * num_features]

            forecast = self.base_forecast_block(
                x
            )  # [bs x n_vars x num_patch x patch_size]

            forecast = self.flatten(
                forecast
            )  # [bs x n_vars x forecast_len(num_patch * patch_size)]
            # forecast = torch.reshape(
            #     forecast, (forecast.shape[0],
            #                forecast.shape[1],
            #                forecast.shape[2] * forecast.shape[3])
            # )  # [bs x n_vars x forecast_len(num_patch * patch_size)]

            forecast = forecast.transpose(-1, -2)  # [bs x forecast_len x n_vars]

            if self.forecast_channel_mixing:
                forecast = self.fcm(x=forecast, y=y)  # y: [bs x forecast_len x n_vars]

            if self.time_hierarchy:
                h_pred, forecast = self.ht_block(
                    x=base_x, base_forecast=forecast, y=y
                )  # [bs x forecast_patches x nvars],[bs x forecast_len x nvars]

            if self.time_hierarchy:
                return h_pred, forecast
            else:
                return forecast

        else:
            forecast = self.base_forecast_block(
                x
            )  # bs x num_patch x patch_size*in_channels
            forecast = forecast.reshape(
                -1, self.num_patches, self.patch_size, self.nvars
            )
            # bs x num_patch x patch_size x in_channels

            forecast = forecast.reshape(
                -1, self.num_patches * self.patch_size, self.nvars
            )
            # bs x forecast_len(num_patch * patch_size) x in_channels

            return forecast




def get_class_params_via_inspect(input_frame):
    class_params = {}
    args, _, _, values = inspect.getargvalues(input_frame)

    for k in args:
        if k != "self":
            class_params[k] = values[k]

    if "kwargs" in values:
        class_params.update(values["kwargs"])
    return class_params