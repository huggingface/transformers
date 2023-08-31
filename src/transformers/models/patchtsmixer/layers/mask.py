import torch
from torch import nn, Tensor

class PatchMasking(nn.Module):
    def __init__(self, 
                mask_type: str = "random",
                mask_ratio=0.5,
                mask_patches: list = [2, 3],
                mask_patch_ratios: list = [1, 1],
                channel_consistent_masking: bool = True,
                d_size: str = "4D",
                cv_channel_indices: list = None,
                mask_value=0, ):
        """PatchMasking: Class to random or forcast masking.

        Args:
            mask_type (str, optional): Masking type. Allowed values are random, forecast. Defaults to random.
            mask_ratio (float, optional): Mask ratio.
            mask_patches (list, optional): List of patch lengths to mask in the end of the data.
            mask_patch_ratios (list, optional): List of weights to use for each patch length. For Ex. 
            if patch_lengths is [5,4] and mix_ratio is [1,1], then equal weights to both patch lengths. Defaults to None.
            cv_channel_indices (list, optional): Control Variable channel indices. These channels will not be masked. Defaults to None.
            channel_consistent_masking (bool, optional): When true, masking will be same across all channels of a timeseries. Otherwise, masking positions will vary across channels. Defaults to True.
            d_size (str, optional): Input data size. Allowed values: 4D, 6D. Defaults to "4D".
            mask_value (int, optional): Value to use for masking. Defaults to 0.
        """

        self.mask_ratio = mask_ratio
        self.channel_consistent_masking = channel_consistent_masking
        self.d_size = d_size
        self.mask_type = mask_type
        self.mask_patches = mask_patches
        self.mask_patch_ratios = mask_patch_ratios
        self.cv_channel_indices = cv_channel_indices
        self.mask_value = mask_value
        if self.cv_channel_indices is not None:
            self.cv_channel_indices.sort()
        
        super().__init__()

    def forward(self, x: Tensor):
        
        """
        Input:
            x: patched input
                4D: [bs x n_vars x num_patch  x patch_len]
                6D: [bs x tsg1 x tsg2 x n_vars x num_patch  x patch_len]
        
        Output:
            x_mask: Masked patched input
                4D: [bs x n_vars x num_patch  x patch_len]
                6D: [bs x tsg1 x tsg2 x n_vars x num_patch  x patch_len]
            mask: bool tensor indicating True on masked points
                4D: [bs x n_vars x num_patch]
                6D: [bs x tsg1 x tsg2 x n_vars x num_patch]
        """

        if self.mask_type == "random":
            x_mask, mask = cv_random_masking(xb=x,
                                              mask_ratio=self.mask_ratio,
                                              cv_channel_indices=self.cv_channel_indices,
                                              channel_consistent_masking=self.channel_consistent_masking,
                                              d_size=self.d_size,
                                              mask_value=self.mask_value)  
            

        elif self.mask_type == "forecast":
            x_mask, mask = multi_forecast_masking(xb=x,
                                                   patch_lengths=self.mask_patches,
                                                   mix_ratio=self.mask_patch_ratios,
                                                   cv_channel_indices=self.cv_channel_indices,
                                                   d_size=self.d_size,
                                                   mask_value=self.mask_value)

        else:
            raise Exception("Invalid mask type")
        # 4D: xb_mask: [bs x n_vars x num_patch  x patch_len] # mask: [bs x n_vars x num_patch]
        # 6D: xb_mask: [bs x tsg1 x tsg2 x n_vars x num_patch  x patch_len] # mask: [bs x tsg1 x tsg2 x n_vars x num_patch]
        
        mask = mask.bool()  # mask: [bs x n_vars x num_patch]

        return x_mask, mask
    



def cv_random_masking(xb: Tensor, 
    mask_ratio: float, 
    cv_channel_indices:list = None, 
    channel_consistent_masking: bool = True,
    d_size = "4D",
    mask_value = 0):
    """cv_random_masking: Mask the input considering the control variables.

    Args:
        xb (Tensor): Input to mask [ bs x nvars x num_patch x patch_len] or [ bs x tsg1 x tag2 x nvars x num_patch x patch_len]
        mask_ratio (float): Mask ratio.
        cv_channel_indices (list, optional): Control Variable channel indices. These channels will not be masked. Defaults to None.
        channel_consistent_masking (bool, optional): When true, masking will be same across all channels of a timeseries. Otherwise, masking positions will vary across channels. Defaults to True.
        d_size (str, optional): Input data size. Allowed values: 4D, 6D. Defaults to "4D".
        mask_value (int, optional): Value to use for masking. Defaults to 0.

    Returns:
        Tensor: xb_mask, masked input, same shape as input
        Tensor: Mask tensor of shape [bs x c x n] or [bs x tsg1 x tsg2 x c x n]
    """    
    if d_size == "4D":
        bs, nvars, L, D = xb.shape
    elif d_size == "6D":
        bs, tg1, tg2, nvars, L, D = xb.shape

    len_keep = int(L * (1 - mask_ratio))
    
    if d_size == "4D":
        if channel_consistent_masking:
            noise = torch.rand(bs, 1,  L, device=xb.device)  # noise in [0, 1], bs x 1 x  L
            noise = noise.repeat(1, nvars, 1)            # bs x nvars x L
        else:
            noise = torch.rand(bs, nvars, L, device=xb.device)  # noise in [0, 1], bs x nvars x L
        
        mask = torch.ones(bs, nvars, L, device=xb.device)                                  # mask: [bs x nvars x num_patch]
        mask[:, :, :len_keep] = 0


    elif d_size == "6D":
        if channel_consistent_masking:
            noise = torch.rand(bs, 1, 1, 1, L, device=xb.device)  # noise in [0, 1], bs 1 x 1 x 1 x  L
            noise = noise.repeat(1,tg1, tg2, nvars, 1)            # bs x tg1 x tg2 x  nvars x L
        else:
            noise = torch.rand(bs, tg1, tg2, nvars, L, device=xb.device)  # noise in [0, 1], bs x tg1 x tg2 x nvars x L

        mask = torch.ones(bs, tg1, tg2, nvars, L, device=xb.device)                                  # mask: [bs x tg1 x tg2 x nvars x num_patch]
        mask[:, :, :, :, :len_keep] = 0


    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=-1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=-1)     # ids_restore: [bs x nvars x L]
    mask = torch.gather(mask, dim=-1, index=ids_restore)    

    if d_size == "4D":
        mask = mask.unsqueeze(-1).repeat(1,1,1,D) # mask: [bs x nvars x num_patch x patch_len]
        if cv_channel_indices is not None:
            mask[:,cv_channel_indices,:,:] = 0
    elif d_size == "6D":
        mask = mask.unsqueeze(-1).repeat(1,1,1,1,1,D) # mask: [bs x tg1 x tg2 x nvars x num_patch x patch_len]
        if cv_channel_indices is not None:
            mask[:,:,:,cv_channel_indices,:,:] = 0
        
    xb_mask = xb.masked_fill(mask.bool(), mask_value)
    return xb_mask, mask[...,0]

def multi_forecast_masking(xb: Tensor,
    patch_lengths: list,
    mix_ratio: list = None,
    cv_channel_indices: list = None,
    d_size: str = "4D",
    mask_value: int = 0):
    """multi_forecast_masking Mask last K patches where K is from the patch_lengths list.
    For every batch, distribute the patch lengths based on mix_ratio 
    Ignore masks for column indices mentioned in cv_channel_indices

    Args:
        xb (Tensor): Input to mask [ bs x nvars x num_patch x patch_len] or [ bs x tsg1 x tag2 x nvars x num_patch x patch_len]
        patch_lengths (list): List of patch lengths to mask in the end of the data.
        mix_ratio (list, optional): List of weights to use for each patch length. For Ex. 
            if patch_lengths is [5,4] and mix_ratio is [1,1], then equal weights to both patch lengths. Defaults to None.
        cv_channel_indices (list, optional): Control Variable channel indices. These channels will not be masked. Defaults to None.
        d_size (str, optional): Input data size. Allowed values: 4D, 6D. Defaults to "4D".
        mask_value (int, optional): Value to use for masking. Defaults to 0.

    Returns:
        Tensor: xb_mask, masked input, same shape as input
        Tensor: Mask tensor of shape [bs x c x n] or [bs x tsg1 x tsg2 x c x n]
    """    
    if mix_ratio is None:
        mix_ratio = [1 for t in patch_lengths]

    if d_size == "4D":
        bs, nvars, L, D = xb.shape
        mask = torch.zeros(bs,nvars,L,device=xb.device)
    elif d_size == "6D":
        bs, tg1, tg2, nvars, L, D = xb.shape
        mask = torch.zeros(bs, tg1, tg2, nvars, L,device=xb.device)
    
    t_list = []
    total_length = 0
    total_ratio = sum(mix_ratio)

    for i,j in zip(patch_lengths, mix_ratio):
        if i<=0 or i>=L:
            raise Exception("masked_patch_len should be greater than 0 and less than total patches.")
        temp_len = (int(bs*j/total_ratio))
        t_list.append([i,j,temp_len])
        total_length+=temp_len
        
    t_list = sorted(t_list, key=lambda x: x[2])

    if total_length<bs:
        t_list[0][2] = t_list[0][2]+(bs-total_length)
    elif total_length >bs:
        t_list[-1][2] = t_list[-1][2]+(total_length-bs)

    b1 = 0
    for p, r, l in t_list:
        b2 = b1 + l

        if d_size == "4D":
            mask[b1:b2,:,-p:] = 1
        elif d_size == "6D":
            mask[b1:b2,:,:,:,-p:] = 1

        b1 = b2
    
    perm = torch.randperm(mask.shape[0])
    mask = mask[perm]
    if d_size == "4D":
        mask = mask.unsqueeze(-1).repeat(1,1,1,D) # mask: [bs x nvars x num_patch x patch_len]
        if cv_channel_indices is not None:
            mask[:,cv_channel_indices,:,:] = 0
    elif d_size == "6D":
        mask = mask.unsqueeze(-1).repeat(1,1,1,1,1,D) # mask: [bs x tg1 x tg2 x nvars x num_patch x patch_len]
        if cv_channel_indices is not None:
            mask[:,:,:,cv_channel_indices,:,:] = 0
    
    xb_mask = xb.masked_fill(mask.bool(), mask_value)
    return xb_mask, mask[...,0]
        
