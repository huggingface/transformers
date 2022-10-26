# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
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

from functools import partialmethod
from typing import Optional

import torch
import torch.nn as nn

from .openfold_primitives import Linear, LayerNorm
from .openfold_utils.precision_utils import is_fp16_enabled
from .openfold_utils.tensor_utils import add, permute_final_dims


class TriangleMultiplicativeUpdate(nn.Module):
    """
    Implements Algorithms 11 and 12.
    """
    def __init__(self, c_z, c_hidden, _outgoing=True):
        """
        Args:
            c_z:
                Input channel dimension
            c:
                Hidden channel dimension
        """
        super(TriangleMultiplicativeUpdate, self).__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self._outgoing = _outgoing

        self.linear_a_p = Linear(self.c_z, self.c_hidden)
        self.linear_a_g = Linear(self.c_z, self.c_hidden, init="gating")
        self.linear_b_p = Linear(self.c_z, self.c_hidden)
        self.linear_b_g = Linear(self.c_z, self.c_hidden, init="gating")
        self.linear_g = Linear(self.c_z, self.c_z, init="gating")
        self.linear_z = Linear(self.c_hidden, self.c_z, init="final")

        self.layer_norm_in = LayerNorm(self.c_z)
        self.layer_norm_out = LayerNorm(self.c_hidden)

        self.sigmoid = nn.Sigmoid()

    def _combine_projections(self,
        a: torch.Tensor,
        b: torch.Tensor,
        _inplace_chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        if(self._outgoing):
            a = permute_final_dims(a, (2, 0, 1))
            b = permute_final_dims(b, (2, 1, 0))
        else:
            a = permute_final_dims(a, (2, 1, 0))
            b = permute_final_dims(b,  (2, 0, 1))

        if(_inplace_chunk_size is not None):
            # To be replaced by torch vmap
            for i in range(0, a.shape[-3], _inplace_chunk_size):
                a_chunk = a[..., i: i + _inplace_chunk_size, :, :]
                b_chunk = b[..., i: i + _inplace_chunk_size, :, :]
                a[..., i: i + _inplace_chunk_size, :, :] = (
                    torch.matmul(
                        a_chunk,
                        b_chunk,
                    )
                )

            p = a
        else:
            p = torch.matmul(a, b)

        return permute_final_dims(p, (1, 2, 0))

    def _inference_forward(self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        inplace_chunk_size: Optional[int] = None,
        with_add: bool = True,
    ):
        """
        Args:
            z:
                A [*, N, N, C_z] pair representation
            mask:
                A [*, N, N] pair mask
            inplace_chunk_size:
                Size of chunks used in the main computation. Increase to trade
                memory for speed.
            with_add:
                If True, z is overwritten with (z + update). Otherwise, it is
                overwritten with (update).
        Returns:
            A reference to the overwritten z

        More memory-efficient, inference-only version of the forward function.
        Uses in-place operations, fusion of the addition that happens after
        this module in the Evoformer, a smidge of recomputation, and 
        a cache of overwritten values to lower peak memory consumption of this
        module from 5x the size of the input tensor z to 2.5x its size. Useful
        for inference on extremely long sequences. 
        
        It works as follows. We will make reference to variables used in the
        default forward implementation below. Naively, triangle multiplication
        attention requires the manifestation of 5 tensors the size of z:
        1) z, the "square" input tensor, 2) a, the first projection of z, 
        3) b, the second projection of b, 4) g, a z-sized mask, and 5) a 
        z-sized tensor for intermediate computations. For large N, this is 
        prohibitively expensive; for N=4000, for example, z is more than 8GB 
        alone. To avoid this problem, we compute b, g, and all intermediate
        tensors in small chunks, noting that the chunks required to compute a
        chunk of the output depend only on the tensor a and corresponding 
        vertical and horizontal chunks of z. This suggests an algorithm that 
        loops over pairs of chunks of z: hereafter "columns" and "rows" of
        z, even though each "column" and "row" in fact contains
        inplace_chunk_size contiguous true columns and rows of z. Writing 
        output chunks to a new tensor would bring total memory consumption
        down to 3x the size of z. However, more memory can be saved by writing
        output chunks directly to z in-place. WLOG, we choose to write output
        chunks vertically, overwriting the ith "column" of z at the end of
        the ith iteration of the main loop. Despite this overwriting, the 
        ith column is always one column ahead of previously overwritten columns 
        and can be recovered directly from z. After the first iteration,
        however, the ith row of z is always at least partially overwritten. For
        this reason, we introduce the z-cache, a tensor one-half the size of 
        z. The z-cache initially contains the left half (2nd and 3rd quadrants)
        of z. For 0 < i < N/2, the missing left part of the ith row of z is
        recovered from this cache at the beginning of the ith iteration. Once i 
        exceeds n/2, the cache is "reoriented" to encompass the 3rd and 4th 
        quadrants of z instead. Though the 3rd quadrant of the original z is 
        entirely overwritten at this point, it can be recovered from the z-cache 
        itself. Thereafter, the ith row of z can be recovered in its entirety 
        from the reoriented z-cache. After the final iteration, z has been 
        completely overwritten and contains the triangular multiplicative 
        update. If with_add is True, it instead contains the sum of z and the
        triangular multiplicative update. In either case, peak memory 
        consumption is just 2.5x the size of z, disregarding memory used for 
        chunks and other small variables.
        """
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        mask = mask.unsqueeze(-1)
       
        def compute_projection_helper(pair, mask, a=True):
            if(a):
                linear_g = self.linear_a_g
                linear_p = self.linear_a_p
            else:
                linear_g = self.linear_b_g
                linear_p = self.linear_b_p
            
            pair = self.layer_norm_in(pair)
            p = linear_g(pair)
            p.sigmoid_()
            p *= linear_p(pair)
            p *= mask
            p = permute_final_dims(p, (2, 0, 1))
            return p

        def compute_projection(pair, mask, a=True, chunked=True): 
            need_transpose = self._outgoing ^ a
            if(not chunked):
                p = compute_projection_helper(pair, mask, a)
                if(need_transpose):
                    p = p.transpose(-1, -2)
            else:
                # This computation is chunked so as not to exceed our 2.5x 
                # budget with a large intermediate tensor
                linear_g = self.linear_a_g if a else self.linear_b_g
                c = linear_g.bias.shape[-1]
                out_shape = pair.shape[:-3] + (c,) + pair.shape[-3:-1]
                p = pair.new_zeros(out_shape)
                for i in range(0, pair.shape[-3], inplace_chunk_size):
                    pair_chunk = pair[..., i: i + inplace_chunk_size, :, :]
                    mask_chunk = mask[..., i: i + inplace_chunk_size, :, :]
                    pair_chunk = compute_projection_helper(
                        pair[..., i: i + inplace_chunk_size, :, :],
                        mask[..., i: i + inplace_chunk_size, :, :], 
                        a,
                    )
                    if(need_transpose):
                        pair_chunk = pair_chunk.transpose(-1, -2)
                        p[..., i: i + inplace_chunk_size] = pair_chunk
                    else:
                        p[..., i: i + inplace_chunk_size, :] = pair_chunk
                    
                    del pair_chunk

            return p

        # We start by fully manifesting a. In addition to the input, this
        # brings total memory consumption to 2x z (disregarding size of chunks)
        # [*, N, N, c]
        a = compute_projection(z, mask, True, chunked=True)

        if(inplace_chunk_size is not None):
            n = a.shape[-1]
            half_n = n // 2 + n % 2
            row_dim = -3
            col_dim = -2
            b_chunk_dim = row_dim if self._outgoing else col_dim
            
            def empty_slicer(t):
                return [slice(None) for _ in t.shape]
            
            def slice_tensor(t, start, end, dim):
                # Slices start:end from the dim dimension of t
                s = empty_slicer(t)
                s[dim] = slice(start, end)
                return t[s]

            def flip_z_cache_(z_cache, z):
                # "Reorient" the z_cache (see below), filling it with quadrants
                # 3---recovered from the z_cache---and 4---recovered from z---
                # of the input tensor z. 
                quadrant_3 = slice_tensor(
                    z_cache, half_n, None, row_dim
                )
                z_cache = z_cache.transpose(row_dim, col_dim)
                
                # If n is odd, we need to shrink the z_cache by one row
                z_cache = z_cache[..., :(n // 2), :, :]

                # Move the 3rd quadrant of z into the 
                first_half_slicer = empty_slicer(z_cache)
                first_half_slicer[col_dim] = slice(0, half_n)
                z_cache[first_half_slicer] = quadrant_3
               
                # Get the fourth quadrant of z
                quadrant_4 = slice_tensor(z, half_n, None, row_dim)
                quadrant_4 = slice_tensor(
                    quadrant_4, half_n, None, col_dim
                )

                # Insert said quadrant into the rotated z-cache
                quadrant_3_slicer = empty_slicer(z_cache)
                quadrant_3_slicer[col_dim] = slice(half_n, None)

                z_cache[quadrant_3_slicer] = quadrant_4

                return z_cache

            # Initialize the z cache to the left half of z.
            z_cache_shape = list(z.shape)
            z_cache_shape[col_dim] = half_n
            z_cache = z.new_zeros(z_cache_shape)
            z_cache_slicer = empty_slicer(z_cache)
            z_cache_slicer[col_dim] = slice(0, half_n)
            z_cache.copy_(z[z_cache_slicer])
            z_cache_rotated = False

            # We need to reorient the z-cache at the halfway point, and we 
            # don't want a single chunk to straddle that point. We contract one
            # of the chunks in the middle to address that problem.
            i_range = list(range(0, half_n, inplace_chunk_size))
            initial_offsets = [
                i_2 - i_1 for i_1, i_2 in zip(i_range, i_range[1:] + [half_n])
            ]
            after_half = list(range(half_n, n, inplace_chunk_size))
            after_half_offsets = [inplace_chunk_size for _ in after_half]
            combined_range_with_offsets = zip(
                i_range + after_half, initial_offsets + after_half_offsets
            )
            for i, offset in combined_range_with_offsets:
                if(not z_cache_rotated and i >= half_n):
                    z_cache = flip_z_cache_(z_cache, z)
                    z_cache_rotated = True

                z_chunk_b = slice_tensor(
                    z, i, i + offset, b_chunk_dim,
                )
                mask_chunk = slice_tensor(
                    mask, i, i + offset, b_chunk_dim,
                )

                z_chunk_b = z_chunk_b.clone()
                if(b_chunk_dim == col_dim):
                    z_chunk_b = slice_tensor(
                        z, i, i + offset, col_dim
                    )
                else: # b_chunk_dim == row_dim
                    # In this case, the b-dimension (b_chunk_dim) is partially 
                    # overwritten at the end of each iteration. We need to 
                    # restore the missing component from the z-cache.
                    if(not z_cache_rotated):
                        z_chunk_slicer = empty_slicer(z_chunk_b)
                        z_chunk_slicer[col_dim] = slice(0, half_n)
                        z_chunk_b[z_chunk_slicer] = slice_tensor(
                            z_cache, i, i + offset, row_dim,
                        )
                    else:
                        z_cache_offset = i - half_n
                        z_chunk_b = slice_tensor(
                            z_cache, 
                            z_cache_offset, z_cache_offset + offset, 
                            row_dim
                        )

                b_chunk = compute_projection(
                    z_chunk_b, mask_chunk, a=False, chunked=False
                )
                del z_chunk_b

                x_chunk = torch.matmul(
                     a,
                     b_chunk,
                )
                x_chunk = permute_final_dims(x_chunk, (1, 2, 0))
                x_chunk = self.layer_norm_out(x_chunk)
                x_chunk = self.linear_z(x_chunk)

                # The g dimension (col_dim) is parallel to and ahead of the 
                # overwrites in z. We can extract the g chunk normally.
                z_chunk_g = slice_tensor(
                    z, i, i + offset, col_dim
                )
                g_chunk = self.linear_g(self.layer_norm_in(z_chunk_g)) 
                g_chunk.sigmoid_()
                del z_chunk_g
                
                x_chunk *= g_chunk

                # Write the columns into z in-place
                z_slicer = empty_slicer(z)
                z_slicer[col_dim] = slice(i, i + offset)
                if(with_add):
                    z[z_slicer] += x_chunk
                else:
                    z[z_slicer] = x_chunk
        else:
            b = compute_projection(z, mask, False, False)
            x = torch.matmul(a, b)
            x = self.layer_norm_out(x)
            x = self.linear_z(x)
            g = self.linear_g(z)
            g.sigmoid_()
            x *= g
            if(with_add):
                z += x
            else:
                z = x

        return z

    def forward(self, 
        z: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        inplace_safe: bool = False,
        _add_with_inplace: bool = False,
        _inplace_chunk_size: Optional[int] = 256,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor
            mask:
                [*, N_res, N_res] input mask
        Returns:
            [*, N_res, N_res, C_z] output tensor
        """
        if(inplace_safe):
            x = self._inference_forward(
                z, 
                mask, 
                inplace_chunk_size=_inplace_chunk_size,
                with_add=_add_with_inplace,
            )
            return x

        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        mask = mask.unsqueeze(-1)
        
        z = self.layer_norm_in(z)
        a = mask
        a = a * self.sigmoid(self.linear_a_g(z)) 
        a = a * self.linear_a_p(z)
        b = mask
        b = b * self.sigmoid(self.linear_b_g(z))
        b = b * self.linear_b_p(z)
        
        if(is_fp16_enabled()): 
            with torch.cuda.amp.autocast(enabled=False):
                x = self._combine_projections(a.float(), b.float())
        else:
            x = self._combine_projections(a, b)
        
        del a, b
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = self.sigmoid(self.linear_g(z))
        x = x * g

        return x


class TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate):
    """
    Implements Algorithm 11.
    """
    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=True)


class TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate):
    """
    Implements Algorithm 12.
    """
    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=False)
