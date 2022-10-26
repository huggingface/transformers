# Copyright 2021 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from Bio.SVDSuperimposer import SVDSuperimposer
import numpy as np
import torch


def _superimpose_np(reference, coords):
    """
        Superimposes coordinates onto a reference by minimizing RMSD using SVD.

        Args:
            reference:
                [N, 3] reference array
            coords:
                [N, 3] array
        Returns:
            A tuple of [N, 3] superimposed coords and the final RMSD.
    """
    sup = SVDSuperimposer()
    sup.set(reference, coords)
    sup.run()
    return sup.get_transformed(), sup.get_rms()


def _superimpose_single(reference, coords):
    reference_np = reference.detach().cpu().numpy()    
    coords_np = coords.detach().cpu().numpy()
    superimposed, rmsd = _superimpose_np(reference_np, coords_np)
    return coords.new_tensor(superimposed), coords.new_tensor(rmsd)


def superimpose(reference, coords, mask):
    """
        Superimposes coordinates onto a reference by minimizing RMSD using SVD.

        Args:
            reference:
                [*, N, 3] reference tensor
            coords:
                [*, N, 3] tensor
            mask:
                [*, N] tensor
        Returns:
            A tuple of [*, N, 3] superimposed coords and [*] final RMSDs.
    """
    def select_unmasked_coords(coords, mask):
        return torch.masked_select(
            coords,
            (mask > 0.)[..., None],
        ).reshape(-1, 3)

    batch_dims = reference.shape[:-2]
    flat_reference = reference.reshape((-1,) + reference.shape[-2:])
    flat_coords = coords.reshape((-1,) + reference.shape[-2:])
    flat_mask = mask.reshape((-1,) + mask.shape[-1:])
    superimposed_list = []
    rmsds = []
    for r, c, m in zip(flat_reference, flat_coords, flat_mask):
        r_unmasked_coords = select_unmasked_coords(r, m)
        c_unmasked_coords = select_unmasked_coords(c, m)
        superimposed, rmsd = _superimpose_single(
            r_unmasked_coords, 
            c_unmasked_coords
        )

        # This is very inelegant, but idk how else to invert the masking
        # procedure.
        count = 0
        superimposed_full_size = torch.zeros_like(r)
        for i, unmasked in enumerate(m):
            if(unmasked):
                superimposed_full_size[i] = superimposed[count]
                count += 1

        superimposed_list.append(superimposed_full_size)
        rmsds.append(rmsd)

    superimposed_stacked = torch.stack(superimposed_list, dim=0)
    rmsds_stacked = torch.stack(rmsds, dim=0)

    superimposed_reshaped = superimposed_stacked.reshape(
        batch_dims + coords.shape[-2:]
    )
    rmsds_reshaped = rmsds_stacked.reshape(
        batch_dims
    )

    return superimposed_reshaped, rmsds_reshaped
