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
import torch


def drmsd(structure_1, structure_2, mask=None):
    def prep_d(structure):
        d = structure[..., :, None, :] - structure[..., None, :, :]
        d = d ** 2
        d = torch.sqrt(torch.sum(d, dim=-1))
        return d

    d1 = prep_d(structure_1)
    d2 = prep_d(structure_2)

    drmsd = d1 - d2
    drmsd = drmsd ** 2
    if(mask is not None):
        drmsd = drmsd * (mask[..., None] * mask[..., None, :])
    drmsd = torch.sum(drmsd, dim=(-1, -2))
    n = d1.shape[-1] if mask is None else torch.sum(mask, dim=-1)
    drmsd = drmsd * (1 / (n * (n - 1))) if n > 1 else (drmsd * 0.)
    drmsd = torch.sqrt(drmsd)

    return drmsd


def drmsd_np(structure_1, structure_2, mask=None):
    structure_1 = torch.tensor(structure_1)
    structure_2 = torch.tensor(structure_2)
    if(mask is not None):
        mask = torch.tensor(mask)

    return drmsd(structure_1, structure_2, mask)


def gdt(p1, p2, mask, cutoffs):
    n = torch.sum(mask, dim=-1)
    
    p1 = p1.float()
    p2 = p2.float()
    distances = torch.sqrt(torch.sum((p1 - p2)**2, dim=-1))
    scores = []
    for c in cutoffs:
        score = torch.sum((distances <= c) * mask, dim=-1) / n
        score = torch.mean(score)
        scores.append(score)

    return sum(scores) / len(scores)


def gdt_ts(p1, p2, mask):
    return gdt(p1, p2, mask, [1., 2., 4., 8.])


def gdt_ha(p1, p2, mask):
    return gdt(p1, p2, mask, [0.5, 1., 2., 4.])

