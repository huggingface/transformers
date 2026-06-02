# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Communication primitives for 2D context-parallel distributed operations."""

from typing import Optional

import torch
import torch.distributed as dist

from projects.huggingface.transformers.models.esmfold2.distributed.utils import (
    LayoutMap,
    get_group_rank_from_axial_shift,
)


class One2OneComm:
    """Point-to-point communication with parity-based deadlock avoidance."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        rank_send_to: int,
        rank_recv_from: int,
        parity: Optional[bool] = None,
    ):
        self.group = group
        self.rank = dist.get_rank(self.group)
        self.world_size = dist.get_world_size(self.group)

        if rank_send_to >= self.world_size:
            raise ValueError(f"rank_send_to >= world_size {self.world_size}")
        if rank_recv_from >= self.world_size:
            raise ValueError(f"rank_recv_from >= world_size {self.world_size}")

        is_self_send = rank_send_to == self.rank
        is_self_recv = rank_recv_from == self.rank
        if is_self_send != is_self_recv:
            raise ValueError(
                "Asymmetric send/recv not supported: "
                f"is_self_send={is_self_send}, is_self_recv={is_self_recv}"
            )
        self.is_self_comm = is_self_send
        self._rank_in_group_send_to = rank_send_to
        self._rank_in_group_recv_from = rank_recv_from
        self.parity = parity

        if not self.is_self_comm:
            self.rank_send_to = dist.get_global_rank(self.group, rank_send_to)
            self.rank_recv_from = dist.get_global_rank(self.group, rank_recv_from)
            if self.parity is None:
                self.parity = bool(self.rank % 2)
            self._queue_send_recv = []
            self._work_to_finish = None

    def __deepcopy__(self, memo):
        return One2OneComm(
            self.group,
            self._rank_in_group_send_to,
            self._rank_in_group_recv_from,
            self.parity,
        )

    def _prep_batch_isend_irecv(
        self, to_send: torch.Tensor, to_recv: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.is_self_comm:
            if to_recv is None:
                return to_send.detach().clone()
            to_recv.copy_(to_send)
            return to_recv

        ans = torch.empty_like(to_send) if to_recv is None else to_recv
        if self.parity:
            send_op = dist.P2POp(
                dist.isend, to_send, self.rank_send_to, group=self.group
            )
            recv_op = dist.P2POp(dist.irecv, ans, self.rank_recv_from, group=self.group)
            self._queue_send_recv.append(send_op)
            self._queue_send_recv.append(recv_op)
        else:
            recv_op = dist.P2POp(dist.irecv, ans, self.rank_recv_from, group=self.group)
            send_op = dist.P2POp(
                dist.isend, to_send, self.rank_send_to, group=self.group
            )
            self._queue_send_recv.append(recv_op)
            self._queue_send_recv.append(send_op)
        return ans

    def _dispatch(self):
        if self.is_self_comm:
            return
        if self._work_to_finish is not None:
            raise RuntimeError("Unfinished communication in queue; cannot dispatch new")
        self._work_to_finish = dist.batch_isend_irecv(self._queue_send_recv)

    def wait_until_finished(self):
        if self.is_self_comm:
            return
        if self._work_to_finish is None:
            raise RuntimeError("Cannot wait without unfinished communication")
        for work in self._work_to_finish:
            work.wait()
        self._work_to_finish = None
        self._queue_send_recv = []

    def enqueue_to_dispatch(
        self, to_send: torch.Tensor, to_recv: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        recv = self._prep_batch_isend_irecv(to_send, to_recv)
        if self.is_self_comm:
            return recv
        self._dispatch()
        return recv


class TransposeComm(One2OneComm):
    """Transposes data between (i,j) and (j,i) on a square grid."""

    def __init__(self, process_group: dist.ProcessGroup, group_layout: LayoutMap):
        if group_layout.shape is None:
            raise ValueError("group_layout must have a shape")
        self.world_size = dist.get_world_size(process_group)
        if self.world_size != group_layout.numel:
            raise ValueError("Inconsistent world_size and group_layout.numel")
        if len(group_layout.shape) != 2:
            raise ValueError(f"{self.__class__} only supports 2D group layout")
        if group_layout.shape[0] != group_layout.shape[1]:
            raise ValueError(f"group_layout.shape {group_layout.shape} is not square")

        self.group_layout = group_layout
        self.global_rank = dist.get_rank()
        self.group_rank = dist.get_rank(process_group)
        self.rank_coords: tuple[int, ...] = self.group_layout.unravel(self.group_rank)

        transpose_group_rank = self.group_layout(self.rank_coords[::-1])
        self.transpose_rank = dist.get_global_rank(process_group, transpose_group_rank)
        self.parity_transpose = self.rank_coords[0] < self.rank_coords[1]
        super().__init__(
            process_group,
            transpose_group_rank,
            transpose_group_rank,
            parity=self.parity_transpose,
        )

    def __deepcopy__(self, memo):
        return TransposeComm(self.group, self.group_layout)


def ternary_parity(my_rank: int, send_rank: int, recv_rank: int) -> bool:
    """Parity to avoid deadlocks: True if my_rank < min(send, recv)."""
    return my_rank < min(send_rank, recv_rank)


class Ring2DComm:
    """Ring communication on a 2D grid for TriangleMult and similar operations.

    Sets up:
    - Transpose communication (send (i,j) to (j,i))
    - Row-wise ring (left shift per row)
    - Column-wise ring (up shift per column)
    - Initial offset shifts (row i shifts left by i; col j shifts up by j)
    """

    def __init__(
        self,
        group_2d: dist.ProcessGroup,
        group_col: dist.ProcessGroup,
        group_layout: LayoutMap,
    ):
        self.group_2d = group_2d
        self.group_col = group_col
        self.group_layout = group_layout
        ranks_group_2d = set(dist.get_process_group_ranks(self.group_2d))
        ranks_group_col = set(dist.get_process_group_ranks(self.group_col))

        if not ranks_group_col.issubset(ranks_group_2d):
            raise ValueError("group_col ranks are not a subset of group_2d ranks")

        self.size_2d = dist.get_world_size(self.group_2d)
        if self.size_2d != self.group_layout.numel:
            raise ValueError(
                f"group_2d size {self.size_2d} != group_layout.numel {self.group_layout.numel}"
            )
        if self.group_layout.shape[0] != self.group_layout.shape[1]:
            raise ValueError(
                f"group_layout.shape {self.group_layout.shape} is not square"
            )

        self.rank_2d = dist.get_rank(self.group_2d)
        self.coord_2d = self.group_layout.unravel(self.rank_2d)

        self.comm_2d_trans = TransposeComm(self.group_2d, self.group_layout)

        # Initial row shift: row i shifts left by i
        self.send_rank_row_init = get_group_rank_from_axial_shift(
            self.coord_2d, 1, -self.coord_2d[0], self.group_layout
        )
        self.recv_rank_row_init = get_group_rank_from_axial_shift(
            self.coord_2d, 1, self.coord_2d[0], self.group_layout
        )
        self.comm_row_init = One2OneComm(
            self.group_2d,
            self.send_rank_row_init,
            self.recv_rank_row_init,
            parity=ternary_parity(
                self.rank_2d, self.send_rank_row_init, self.recv_rank_row_init
            ),
        )

        # Subsequent row shifts: left by 1
        self.send_rank_row = get_group_rank_from_axial_shift(
            self.coord_2d, 1, -1, self.group_layout
        )
        self.recv_rank_row = get_group_rank_from_axial_shift(
            self.coord_2d, 1, 1, self.group_layout
        )
        self.comm_row = One2OneComm(
            self.group_2d,
            self.send_rank_row,
            self.recv_rank_row,
            parity=ternary_parity(self.rank_2d, self.send_rank_row, self.recv_rank_row),
        )

        # Initial col shift: col j shifts up by j
        self.send_rank_col_init = get_group_rank_from_axial_shift(
            self.coord_2d, 0, -self.coord_2d[1], self.group_layout
        )
        self.recv_rank_col_init = get_group_rank_from_axial_shift(
            self.coord_2d, 0, self.coord_2d[1], self.group_layout
        )
        self.comm_col_init = One2OneComm(
            self.group_2d,
            self.send_rank_col_init,
            self.recv_rank_col_init,
            parity=ternary_parity(
                self.rank_2d, self.send_rank_col_init, self.recv_rank_col_init
            ),
        )

        # Subsequent col shifts: up by 1
        self.send_rank_col = get_group_rank_from_axial_shift(
            self.coord_2d, 0, -1, self.group_layout
        )
        self.recv_rank_col = get_group_rank_from_axial_shift(
            self.coord_2d, 0, 1, self.group_layout
        )
        self.comm_col = One2OneComm(
            self.group_2d,
            self.send_rank_col,
            self.recv_rank_col,
            parity=ternary_parity(self.rank_2d, self.send_rank_col, self.recv_rank_col),
        )

        # Fused transpose + initial shift for backward pass
        coords_t = self.coord_2d[::-1]
        self.send_rank_transpose_row_init = get_group_rank_from_axial_shift(
            coords_t, 1, -coords_t[0], self.group_layout
        )
        recv_rank_transpose_row_init = get_group_rank_from_axial_shift(
            self.coord_2d, 1, self.coord_2d[0], self.group_layout
        )
        self.recv_rank_transpose_row_init = self.group_layout(
            self.group_layout.unravel(recv_rank_transpose_row_init)[::-1]
        )
        self.comm_transpose_row_init = One2OneComm(
            self.group_2d,
            self.send_rank_transpose_row_init,
            self.recv_rank_transpose_row_init,
            parity=ternary_parity(
                self.rank_2d,
                self.send_rank_transpose_row_init,
                self.recv_rank_transpose_row_init,
            ),
        )

        self.send_rank_transpose_col_init = get_group_rank_from_axial_shift(
            coords_t, 0, -coords_t[1], self.group_layout
        )
        recv_rank_transpose_col_init = get_group_rank_from_axial_shift(
            self.coord_2d, 0, self.coord_2d[1], self.group_layout
        )
        self.recv_rank_transpose_col_init = self.group_layout(
            self.group_layout.unravel(recv_rank_transpose_col_init)[::-1]
        )
        self.comm_transpose_col_init = One2OneComm(
            self.group_2d,
            self.send_rank_transpose_col_init,
            self.recv_rank_transpose_col_init,
            parity=ternary_parity(
                self.rank_2d,
                self.send_rank_transpose_col_init,
                self.recv_rank_transpose_col_init,
            ),
        )


class AttentionPairBiasComm:
    """Communication setup for ring attention with pair bias.

    Manages transpose comms for K/V/mask and ring shift comms for K/V/Z.
    """

    def __init__(
        self,
        process_group: dist.ProcessGroup,
        group_layout: LayoutMap,
        cp_axis_0_group: dist.ProcessGroup,
        cp_axis_1_group: dist.ProcessGroup,
    ):
        self.process_group = process_group
        self.cp_axis_0_group = cp_axis_0_group
        self.cp_axis_1_group = cp_axis_1_group

        if group_layout.shape is None:
            raise ValueError("group_layout must have a shape")
        self.world_size = dist.get_world_size(self.process_group)
        if self.world_size != group_layout.numel:
            raise ValueError("Inconsistent world_size and group_layout.numel")
        if len(group_layout.shape) != 2:
            raise ValueError(f"{self.__class__} only supports 2D group layout")
        if group_layout.shape[0] != group_layout.shape[1]:
            raise ValueError(f"group_layout.shape {group_layout.shape} is not square")

        self.group_layout = group_layout
        self.global_rank = dist.get_rank()
        self.group_rank = dist.get_rank(self.process_group)
        self.rank_coords: tuple[int, ...] = self.group_layout.unravel(self.group_rank)

        self.comm_transpose_k = TransposeComm(self.process_group, self.group_layout)
        self.comm_transpose_v = TransposeComm(self.process_group, self.group_layout)
        self.comm_transpose_mask = TransposeComm(self.process_group, self.group_layout)

        self.send_rank_kvz = get_group_rank_from_axial_shift(
            self.rank_coords, 1, 1, self.group_layout
        )
        self.recv_rank_kvz = get_group_rank_from_axial_shift(
            self.rank_coords, 1, -1, self.group_layout
        )
        self.parity = self.rank_coords[1] % 2 == 1
        self.comm_k = One2OneComm(
            self.process_group,
            self.send_rank_kvz,
            self.recv_rank_kvz,
            parity=self.parity,
        )
        self.comm_v = One2OneComm(
            self.process_group,
            self.send_rank_kvz,
            self.recv_rank_kvz,
            parity=self.parity,
        )
        self.comm_z = One2OneComm(
            self.process_group,
            self.send_rank_kvz,
            self.recv_rank_kvz,
            parity=self.parity,
        )

    def __deepcopy__(self, memo):
        return AttentionPairBiasComm(
            self.process_group,
            self.group_layout,
            self.cp_axis_0_group,
            self.cp_axis_1_group,
        )
