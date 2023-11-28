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

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def rot_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication of two rotation matrix tensors. Written out by hand to avoid AMP downcasting.

    Args:
        a: [*, 3, 3] left multiplicand
        b: [*, 3, 3] right multiplicand
    Returns:
        The product ab
    """

    def row_mul(i: int) -> torch.Tensor:
        return torch.stack(
            [
                a[..., i, 0] * b[..., 0, 0] + a[..., i, 1] * b[..., 1, 0] + a[..., i, 2] * b[..., 2, 0],
                a[..., i, 0] * b[..., 0, 1] + a[..., i, 1] * b[..., 1, 1] + a[..., i, 2] * b[..., 2, 1],
                a[..., i, 0] * b[..., 0, 2] + a[..., i, 1] * b[..., 1, 2] + a[..., i, 2] * b[..., 2, 2],
            ],
            dim=-1,
        )

    return torch.stack(
        [
            row_mul(0),
            row_mul(1),
            row_mul(2),
        ],
        dim=-2,
    )


def rot_vec_mul(r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Applies a rotation to a vector. Written out by hand to avoid transfer to avoid AMP downcasting.

    Args:
        r: [*, 3, 3] rotation matrices
        t: [*, 3] coordinate tensors
    Returns:
        [*, 3] rotated coordinates
    """
    x, y, z = torch.unbind(t, dim=-1)
    return torch.stack(
        [
            r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z,
            r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z,
            r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z,
        ],
        dim=-1,
    )


@lru_cache(maxsize=None)
def identity_rot_mats(
    batch_dims: Tuple[int, ...],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = True,
) -> torch.Tensor:
    rots = torch.eye(3, dtype=dtype, device=device, requires_grad=requires_grad)
    rots = rots.view(*((1,) * len(batch_dims)), 3, 3)
    rots = rots.expand(*batch_dims, -1, -1)
    rots = rots.contiguous()

    return rots


@lru_cache(maxsize=None)
def identity_trans(
    batch_dims: Tuple[int, ...],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = True,
) -> torch.Tensor:
    trans = torch.zeros((*batch_dims, 3), dtype=dtype, device=device, requires_grad=requires_grad)
    return trans


@lru_cache(maxsize=None)
def identity_quats(
    batch_dims: Tuple[int, ...],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = True,
) -> torch.Tensor:
    quat = torch.zeros((*batch_dims, 4), dtype=dtype, device=device, requires_grad=requires_grad)

    with torch.no_grad():
        quat[..., 0] = 1

    return quat


_quat_elements: List[str] = ["a", "b", "c", "d"]
_qtr_keys: List[str] = [l1 + l2 for l1 in _quat_elements for l2 in _quat_elements]
_qtr_ind_dict: Dict[str, int] = {key: ind for ind, key in enumerate(_qtr_keys)}


def _to_mat(pairs: List[Tuple[str, int]]) -> np.ndarray:
    mat = np.zeros((4, 4))
    for key, value in pairs:
        ind = _qtr_ind_dict[key]
        mat[ind // 4][ind % 4] = value

    return mat


_QTR_MAT = np.zeros((4, 4, 3, 3))
_QTR_MAT[..., 0, 0] = _to_mat([("aa", 1), ("bb", 1), ("cc", -1), ("dd", -1)])
_QTR_MAT[..., 0, 1] = _to_mat([("bc", 2), ("ad", -2)])
_QTR_MAT[..., 0, 2] = _to_mat([("bd", 2), ("ac", 2)])
_QTR_MAT[..., 1, 0] = _to_mat([("bc", 2), ("ad", 2)])
_QTR_MAT[..., 1, 1] = _to_mat([("aa", 1), ("bb", -1), ("cc", 1), ("dd", -1)])
_QTR_MAT[..., 1, 2] = _to_mat([("cd", 2), ("ab", -2)])
_QTR_MAT[..., 2, 0] = _to_mat([("bd", 2), ("ac", -2)])
_QTR_MAT[..., 2, 1] = _to_mat([("cd", 2), ("ab", 2)])
_QTR_MAT[..., 2, 2] = _to_mat([("aa", 1), ("bb", -1), ("cc", -1), ("dd", 1)])


def quat_to_rot(quat: torch.Tensor) -> torch.Tensor:
    """
    Converts a quaternion to a rotation matrix.

    Args:
        quat: [*, 4] quaternions
    Returns:
        [*, 3, 3] rotation matrices
    """
    # [*, 4, 4]
    quat = quat[..., None] * quat[..., None, :]

    # [4, 4, 3, 3]
    mat = _get_quat("_QTR_MAT", dtype=quat.dtype, device=quat.device)

    # [*, 4, 4, 3, 3]
    shaped_qtr_mat = mat.view((1,) * len(quat.shape[:-2]) + mat.shape)
    quat = quat[..., None, None] * shaped_qtr_mat

    # [*, 3, 3]
    return torch.sum(quat, dim=(-3, -4))


def rot_to_quat(rot: torch.Tensor) -> torch.Tensor:
    if rot.shape[-2:] != (3, 3):
        raise ValueError("Input rotation is incorrectly shaped")

    [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = [[rot[..., i, j] for j in range(3)] for i in range(3)]

    k = [
        [
            xx + yy + zz,
            zy - yz,
            xz - zx,
            yx - xy,
        ],
        [
            zy - yz,
            xx - yy - zz,
            xy + yx,
            xz + zx,
        ],
        [
            xz - zx,
            xy + yx,
            yy - xx - zz,
            yz + zy,
        ],
        [
            yx - xy,
            xz + zx,
            yz + zy,
            zz - xx - yy,
        ],
    ]

    _, vectors = torch.linalg.eigh((1.0 / 3.0) * torch.stack([torch.stack(t, dim=-1) for t in k], dim=-2))
    return vectors[..., -1]


_QUAT_MULTIPLY = np.zeros((4, 4, 4))
_QUAT_MULTIPLY[:, :, 0] = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]

_QUAT_MULTIPLY[:, :, 1] = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]]

_QUAT_MULTIPLY[:, :, 2] = [[0, 0, 1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0]]

_QUAT_MULTIPLY[:, :, 3] = [[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0]]

_QUAT_MULTIPLY_BY_VEC = _QUAT_MULTIPLY[:, 1:, :]

_CACHED_QUATS: Dict[str, np.ndarray] = {
    "_QTR_MAT": _QTR_MAT,
    "_QUAT_MULTIPLY": _QUAT_MULTIPLY,
    "_QUAT_MULTIPLY_BY_VEC": _QUAT_MULTIPLY_BY_VEC,
}


@lru_cache(maxsize=None)
def _get_quat(quat_key: str, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(_CACHED_QUATS[quat_key], dtype=dtype, device=device)


def quat_multiply(quat1: torch.Tensor, quat2: torch.Tensor) -> torch.Tensor:
    """Multiply a quaternion by another quaternion."""
    mat = _get_quat("_QUAT_MULTIPLY", dtype=quat1.dtype, device=quat1.device)
    reshaped_mat = mat.view((1,) * len(quat1.shape[:-1]) + mat.shape)
    return torch.sum(reshaped_mat * quat1[..., :, None, None] * quat2[..., None, :, None], dim=(-3, -2))


def quat_multiply_by_vec(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Multiply a quaternion by a pure-vector quaternion."""
    mat = _get_quat("_QUAT_MULTIPLY_BY_VEC", dtype=quat.dtype, device=quat.device)
    reshaped_mat = mat.view((1,) * len(quat.shape[:-1]) + mat.shape)
    return torch.sum(reshaped_mat * quat[..., :, None, None] * vec[..., None, :, None], dim=(-3, -2))


def invert_rot_mat(rot_mat: torch.Tensor) -> torch.Tensor:
    return rot_mat.transpose(-1, -2)


def invert_quat(quat: torch.Tensor) -> torch.Tensor:
    quat_prime = quat.clone()
    quat_prime[..., 1:] *= -1
    inv = quat_prime / torch.sum(quat**2, dim=-1, keepdim=True)
    return inv


class Rotation:
    """
    A 3D rotation. Depending on how the object is initialized, the rotation is represented by either a rotation matrix
    or a quaternion, though both formats are made available by helper functions. To simplify gradient computation, the
    underlying format of the rotation cannot be changed in-place. Like Rigid, the class is designed to mimic the
    behavior of a torch Tensor, almost as if each Rotation object were a tensor of rotations, in one format or another.
    """

    def __init__(
        self,
        rot_mats: Optional[torch.Tensor] = None,
        quats: Optional[torch.Tensor] = None,
        normalize_quats: bool = True,
    ):
        """
        Args:
            rot_mats:
                A [*, 3, 3] rotation matrix tensor. Mutually exclusive with quats
            quats:
                A [*, 4] quaternion. Mutually exclusive with rot_mats. If normalize_quats is not True, must be a unit
                quaternion
            normalize_quats:
                If quats is specified, whether to normalize quats
        """
        if (rot_mats is None and quats is None) or (rot_mats is not None and quats is not None):
            raise ValueError("Exactly one input argument must be specified")

        if (rot_mats is not None and rot_mats.shape[-2:] != (3, 3)) or (quats is not None and quats.shape[-1] != 4):
            raise ValueError("Incorrectly shaped rotation matrix or quaternion")

        # Force full-precision
        if quats is not None:
            quats = quats.to(dtype=torch.float32)
        if rot_mats is not None:
            rot_mats = rot_mats.to(dtype=torch.float32)

        if quats is not None and normalize_quats:
            quats = quats / torch.linalg.norm(quats, dim=-1, keepdim=True)

        self._rot_mats = rot_mats
        self._quats = quats

    @staticmethod
    def identity(
        shape,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = True,
        fmt: str = "quat",
    ) -> Rotation:
        """
        Returns an identity Rotation.

        Args:
            shape:
                The "shape" of the resulting Rotation object. See documentation for the shape property
            dtype:
                The torch dtype for the rotation
            device:
                The torch device for the new rotation
            requires_grad:
                Whether the underlying tensors in the new rotation object should require gradient computation
            fmt:
                One of "quat" or "rot_mat". Determines the underlying format of the new object's rotation
        Returns:
            A new identity rotation
        """
        if fmt == "rot_mat":
            rot_mats = identity_rot_mats(
                shape,
                dtype,
                device,
                requires_grad,
            )
            return Rotation(rot_mats=rot_mats, quats=None)
        elif fmt == "quat":
            quats = identity_quats(shape, dtype, device, requires_grad)
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError(f"Invalid format: f{fmt}")

    # Magic methods

    def __getitem__(self, index: Any) -> Rotation:
        """
        Allows torch-style indexing over the virtual shape of the rotation object. See documentation for the shape
        property.

        Args:
            index:
                A torch index. E.g. (1, 3, 2), or (slice(None,))
        Returns:
            The indexed rotation
        """
        if type(index) != tuple:
            index = (index,)

        if self._rot_mats is not None:
            rot_mats = self._rot_mats[index + (slice(None), slice(None))]
            return Rotation(rot_mats=rot_mats)
        elif self._quats is not None:
            quats = self._quats[index + (slice(None),)]
            return Rotation(quats=quats, normalize_quats=False)
        else:
            raise ValueError("Both rotations are None")

    def __mul__(self, right: torch.Tensor) -> Rotation:
        """
        Pointwise left multiplication of the rotation with a tensor. Can be used to e.g. mask the Rotation.

        Args:
            right:
                The tensor multiplicand
        Returns:
            The product
        """
        if not (isinstance(right, torch.Tensor)):
            raise TypeError("The other multiplicand must be a Tensor")

        if self._rot_mats is not None:
            rot_mats = self._rot_mats * right[..., None, None]
            return Rotation(rot_mats=rot_mats, quats=None)
        elif self._quats is not None:
            quats = self._quats * right[..., None]
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError("Both rotations are None")

    def __rmul__(self, left: torch.Tensor) -> Rotation:
        """
        Reverse pointwise multiplication of the rotation with a tensor.

        Args:
            left:
                The left multiplicand
        Returns:
            The product
        """
        return self.__mul__(left)

    # Properties

    @property
    def shape(self) -> torch.Size:
        """
        Returns the virtual shape of the rotation object. This shape is defined as the batch dimensions of the
        underlying rotation matrix or quaternion. If the Rotation was initialized with a [10, 3, 3] rotation matrix
        tensor, for example, the resulting shape would be [10].

        Returns:
            The virtual shape of the rotation object
        """
        if self._rot_mats is not None:
            return self._rot_mats.shape[:-2]
        elif self._quats is not None:
            return self._quats.shape[:-1]
        else:
            raise ValueError("Both rotations are None")

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the dtype of the underlying rotation.

        Returns:
            The dtype of the underlying rotation
        """
        if self._rot_mats is not None:
            return self._rot_mats.dtype
        elif self._quats is not None:
            return self._quats.dtype
        else:
            raise ValueError("Both rotations are None")

    @property
    def device(self) -> torch.device:
        """
        The device of the underlying rotation

        Returns:
            The device of the underlying rotation
        """
        if self._rot_mats is not None:
            return self._rot_mats.device
        elif self._quats is not None:
            return self._quats.device
        else:
            raise ValueError("Both rotations are None")

    @property
    def requires_grad(self) -> bool:
        """
        Returns the requires_grad property of the underlying rotation

        Returns:
            The requires_grad property of the underlying tensor
        """
        if self._rot_mats is not None:
            return self._rot_mats.requires_grad
        elif self._quats is not None:
            return self._quats.requires_grad
        else:
            raise ValueError("Both rotations are None")

    def get_rot_mats(self) -> torch.Tensor:
        """
        Returns the underlying rotation as a rotation matrix tensor.

        Returns:
            The rotation as a rotation matrix tensor
        """
        if self._rot_mats is not None:
            return self._rot_mats
        elif self._quats is not None:
            return quat_to_rot(self._quats)
        else:
            raise ValueError("Both rotations are None")

    def get_quats(self) -> torch.Tensor:
        """
        Returns the underlying rotation as a quaternion tensor.

        Depending on whether the Rotation was initialized with a quaternion, this function may call torch.linalg.eigh.

        Returns:
            The rotation as a quaternion tensor.
        """
        if self._rot_mats is not None:
            return rot_to_quat(self._rot_mats)
        elif self._quats is not None:
            return self._quats
        else:
            raise ValueError("Both rotations are None")

    def get_cur_rot(self) -> torch.Tensor:
        """
        Return the underlying rotation in its current form

        Returns:
            The stored rotation
        """
        if self._rot_mats is not None:
            return self._rot_mats
        elif self._quats is not None:
            return self._quats
        else:
            raise ValueError("Both rotations are None")

    # Rotation functions

    def compose_q_update_vec(self, q_update_vec: torch.Tensor, normalize_quats: bool = True) -> Rotation:
        """
        Returns a new quaternion Rotation after updating the current object's underlying rotation with a quaternion
        update, formatted as a [*, 3] tensor whose final three columns represent x, y, z such that (1, x, y, z) is the
        desired (not necessarily unit) quaternion update.

        Args:
            q_update_vec:
                A [*, 3] quaternion update tensor
            normalize_quats:
                Whether to normalize the output quaternion
        Returns:
            An updated Rotation
        """
        quats = self.get_quats()
        new_quats = quats + quat_multiply_by_vec(quats, q_update_vec)
        return Rotation(
            rot_mats=None,
            quats=new_quats,
            normalize_quats=normalize_quats,
        )

    def compose_r(self, r: Rotation) -> Rotation:
        """
        Compose the rotation matrices of the current Rotation object with those of another.

        Args:
            r:
                An update rotation object
        Returns:
            An updated rotation object
        """
        r1 = self.get_rot_mats()
        r2 = r.get_rot_mats()
        new_rot_mats = rot_matmul(r1, r2)
        return Rotation(rot_mats=new_rot_mats, quats=None)

    def compose_q(self, r: Rotation, normalize_quats: bool = True) -> Rotation:
        """
        Compose the quaternions of the current Rotation object with those of another.

        Depending on whether either Rotation was initialized with quaternions, this function may call
        torch.linalg.eigh.

        Args:
            r:
                An update rotation object
        Returns:
            An updated rotation object
        """
        q1 = self.get_quats()
        q2 = r.get_quats()
        new_quats = quat_multiply(q1, q2)
        return Rotation(rot_mats=None, quats=new_quats, normalize_quats=normalize_quats)

    def apply(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Apply the current Rotation as a rotation matrix to a set of 3D coordinates.

        Args:
            pts:
                A [*, 3] set of points
        Returns:
            [*, 3] rotated points
        """
        rot_mats = self.get_rot_mats()
        return rot_vec_mul(rot_mats, pts)

    def invert_apply(self, pts: torch.Tensor) -> torch.Tensor:
        """
        The inverse of the apply() method.

        Args:
            pts:
                A [*, 3] set of points
        Returns:
            [*, 3] inverse-rotated points
        """
        rot_mats = self.get_rot_mats()
        inv_rot_mats = invert_rot_mat(rot_mats)
        return rot_vec_mul(inv_rot_mats, pts)

    def invert(self) -> Rotation:
        """
        Returns the inverse of the current Rotation.

        Returns:
            The inverse of the current Rotation
        """
        if self._rot_mats is not None:
            return Rotation(rot_mats=invert_rot_mat(self._rot_mats), quats=None)
        elif self._quats is not None:
            return Rotation(
                rot_mats=None,
                quats=invert_quat(self._quats),
                normalize_quats=False,
            )
        else:
            raise ValueError("Both rotations are None")

    # "Tensor" stuff

    def unsqueeze(self, dim: int) -> Rotation:
        """
        Analogous to torch.unsqueeze. The dimension is relative to the shape of the Rotation object.

        Args:
            dim: A positive or negative dimension index.
        Returns:
            The unsqueezed Rotation.
        """
        if dim >= len(self.shape):
            raise ValueError("Invalid dimension")

        if self._rot_mats is not None:
            rot_mats = self._rot_mats.unsqueeze(dim if dim >= 0 else dim - 2)
            return Rotation(rot_mats=rot_mats, quats=None)
        elif self._quats is not None:
            quats = self._quats.unsqueeze(dim if dim >= 0 else dim - 1)
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError("Both rotations are None")

    @staticmethod
    def cat(rs: Sequence[Rotation], dim: int) -> Rotation:
        """
        Concatenates rotations along one of the batch dimensions. Analogous to torch.cat().

        Note that the output of this operation is always a rotation matrix, regardless of the format of input
        rotations.

        Args:
            rs:
                A list of rotation objects
            dim:
                The dimension along which the rotations should be concatenated
        Returns:
            A concatenated Rotation object in rotation matrix format
        """
        rot_mats = torch.cat(
            [r.get_rot_mats() for r in rs],
            dim=dim if dim >= 0 else dim - 2,
        )

        return Rotation(rot_mats=rot_mats, quats=None)

    def map_tensor_fn(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> Rotation:
        """
        Apply a Tensor -> Tensor function to underlying rotation tensors, mapping over the rotation dimension(s). Can
        be used e.g. to sum out a one-hot batch dimension.

        Args:
            fn:
                A Tensor -> Tensor function to be mapped over the Rotation
        Returns:
            The transformed Rotation object
        """
        if self._rot_mats is not None:
            rot_mats = self._rot_mats.view(self._rot_mats.shape[:-2] + (9,))
            rot_mats = torch.stack(list(map(fn, torch.unbind(rot_mats, dim=-1))), dim=-1)
            rot_mats = rot_mats.view(rot_mats.shape[:-1] + (3, 3))
            return Rotation(rot_mats=rot_mats, quats=None)
        elif self._quats is not None:
            quats = torch.stack(list(map(fn, torch.unbind(self._quats, dim=-1))), dim=-1)
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError("Both rotations are None")

    def cuda(self) -> Rotation:
        """
        Analogous to the cuda() method of torch Tensors

        Returns:
            A copy of the Rotation in CUDA memory
        """
        if self._rot_mats is not None:
            return Rotation(rot_mats=self._rot_mats.cuda(), quats=None)
        elif self._quats is not None:
            return Rotation(rot_mats=None, quats=self._quats.cuda(), normalize_quats=False)
        else:
            raise ValueError("Both rotations are None")

    def to(self, device: Optional[torch.device], dtype: Optional[torch.dtype]) -> Rotation:
        """
        Analogous to the to() method of torch Tensors

        Args:
            device:
                A torch device
            dtype:
                A torch dtype
        Returns:
            A copy of the Rotation using the new device and dtype
        """
        if self._rot_mats is not None:
            return Rotation(
                rot_mats=self._rot_mats.to(device=device, dtype=dtype),
                quats=None,
            )
        elif self._quats is not None:
            return Rotation(
                rot_mats=None,
                quats=self._quats.to(device=device, dtype=dtype),
                normalize_quats=False,
            )
        else:
            raise ValueError("Both rotations are None")

    def detach(self) -> Rotation:
        """
        Returns a copy of the Rotation whose underlying Tensor has been detached from its torch graph.

        Returns:
            A copy of the Rotation whose underlying Tensor has been detached from its torch graph
        """
        if self._rot_mats is not None:
            return Rotation(rot_mats=self._rot_mats.detach(), quats=None)
        elif self._quats is not None:
            return Rotation(
                rot_mats=None,
                quats=self._quats.detach(),
                normalize_quats=False,
            )
        else:
            raise ValueError("Both rotations are None")


class Rigid:
    """
    A class representing a rigid transformation. Little more than a wrapper around two objects: a Rotation object and a
    [*, 3] translation Designed to behave approximately like a single torch tensor with the shape of the shared batch
    dimensions of its component parts.
    """

    def __init__(self, rots: Optional[Rotation], trans: Optional[torch.Tensor]):
        """
        Args:
            rots: A [*, 3, 3] rotation tensor
            trans: A corresponding [*, 3] translation tensor
        """
        # (we need device, dtype, etc. from at least one input)

        batch_dims, dtype, device, requires_grad = None, None, None, None
        if trans is not None:
            batch_dims = trans.shape[:-1]
            dtype = trans.dtype
            device = trans.device
            requires_grad = trans.requires_grad
        elif rots is not None:
            batch_dims = rots.shape
            dtype = rots.dtype
            device = rots.device
            requires_grad = rots.requires_grad
        else:
            raise ValueError("At least one input argument must be specified")

        if rots is None:
            rots = Rotation.identity(
                batch_dims,
                dtype,
                device,
                requires_grad,
            )
        elif trans is None:
            trans = identity_trans(
                batch_dims,
                dtype,
                device,
                requires_grad,
            )

        assert rots is not None
        assert trans is not None

        if (rots.shape != trans.shape[:-1]) or (rots.device != trans.device):
            raise ValueError("Rots and trans incompatible")

        # Force full precision. Happens to the rotations automatically.
        trans = trans.to(dtype=torch.float32)

        self._rots = rots
        self._trans = trans

    @staticmethod
    def identity(
        shape: Tuple[int, ...],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = True,
        fmt: str = "quat",
    ) -> Rigid:
        """
        Constructs an identity transformation.

        Args:
            shape:
                The desired shape
            dtype:
                The dtype of both internal tensors
            device:
                The device of both internal tensors
            requires_grad:
                Whether grad should be enabled for the internal tensors
        Returns:
            The identity transformation
        """
        return Rigid(
            Rotation.identity(shape, dtype, device, requires_grad, fmt=fmt),
            identity_trans(shape, dtype, device, requires_grad),
        )

    def __getitem__(self, index: Any) -> Rigid:
        """
        Indexes the affine transformation with PyTorch-style indices. The index is applied to the shared dimensions of
        both the rotation and the translation.

        E.g.::

            r = Rotation(rot_mats=torch.rand(10, 10, 3, 3), quats=None) t = Rigid(r, torch.rand(10, 10, 3)) indexed =
            t[3, 4:6] assert(indexed.shape == (2,)) assert(indexed.get_rots().shape == (2,))
            assert(indexed.get_trans().shape == (2, 3))

        Args:
            index: A standard torch tensor index. E.g. 8, (10, None, 3),
            or (3, slice(0, 1, None))
        Returns:
            The indexed tensor
        """
        if type(index) != tuple:
            index = (index,)

        return Rigid(
            self._rots[index],
            self._trans[index + (slice(None),)],
        )

    def __mul__(self, right: torch.Tensor) -> Rigid:
        """
        Pointwise left multiplication of the transformation with a tensor. Can be used to e.g. mask the Rigid.

        Args:
            right:
                The tensor multiplicand
        Returns:
            The product
        """
        if not (isinstance(right, torch.Tensor)):
            raise TypeError("The other multiplicand must be a Tensor")

        new_rots = self._rots * right
        new_trans = self._trans * right[..., None]

        return Rigid(new_rots, new_trans)

    def __rmul__(self, left: torch.Tensor) -> Rigid:
        """
        Reverse pointwise multiplication of the transformation with a tensor.

        Args:
            left:
                The left multiplicand
        Returns:
            The product
        """
        return self.__mul__(left)

    @property
    def shape(self) -> torch.Size:
        """
        Returns the shape of the shared dimensions of the rotation and the translation.

        Returns:
            The shape of the transformation
        """
        return self._trans.shape[:-1]

    @property
    def device(self) -> torch.device:
        """
        Returns the device on which the Rigid's tensors are located.

        Returns:
            The device on which the Rigid's tensors are located
        """
        return self._trans.device

    def get_rots(self) -> Rotation:
        """
        Getter for the rotation.

        Returns:
            The rotation object
        """
        return self._rots

    def get_trans(self) -> torch.Tensor:
        """
        Getter for the translation.

        Returns:
            The stored translation
        """
        return self._trans

    def compose_q_update_vec(self, q_update_vec: torch.Tensor) -> Rigid:
        """
        Composes the transformation with a quaternion update vector of shape [*, 6], where the final 6 columns
        represent the x, y, and z values of a quaternion of form (1, x, y, z) followed by a 3D translation.

        Args:
            q_vec: The quaternion update vector.
        Returns:
            The composed transformation.
        """
        q_vec, t_vec = q_update_vec[..., :3], q_update_vec[..., 3:]
        new_rots = self._rots.compose_q_update_vec(q_vec)

        trans_update = self._rots.apply(t_vec)
        new_translation = self._trans + trans_update

        return Rigid(new_rots, new_translation)

    def compose(self, r: Rigid) -> Rigid:
        """
        Composes the current rigid object with another.

        Args:
            r:
                Another Rigid object
        Returns:
            The composition of the two transformations
        """
        new_rot = self._rots.compose_r(r._rots)
        new_trans = self._rots.apply(r._trans) + self._trans
        return Rigid(new_rot, new_trans)

    def apply(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Applies the transformation to a coordinate tensor.

        Args:
            pts: A [*, 3] coordinate tensor.
        Returns:
            The transformed points.
        """
        rotated = self._rots.apply(pts)
        return rotated + self._trans

    def invert_apply(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Applies the inverse of the transformation to a coordinate tensor.

        Args:
            pts: A [*, 3] coordinate tensor
        Returns:
            The transformed points.
        """
        pts = pts - self._trans
        return self._rots.invert_apply(pts)

    def invert(self) -> Rigid:
        """
        Inverts the transformation.

        Returns:
            The inverse transformation.
        """
        rot_inv = self._rots.invert()
        trn_inv = rot_inv.apply(self._trans)

        return Rigid(rot_inv, -1 * trn_inv)

    def map_tensor_fn(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> Rigid:
        """
        Apply a Tensor -> Tensor function to underlying translation and rotation tensors, mapping over the
        translation/rotation dimensions respectively.

        Args:
            fn:
                A Tensor -> Tensor function to be mapped over the Rigid
        Returns:
            The transformed Rigid object
        """
        new_rots = self._rots.map_tensor_fn(fn)
        new_trans = torch.stack(list(map(fn, torch.unbind(self._trans, dim=-1))), dim=-1)

        return Rigid(new_rots, new_trans)

    def to_tensor_4x4(self) -> torch.Tensor:
        """
        Converts a transformation to a homogenous transformation tensor.

        Returns:
            A [*, 4, 4] homogenous transformation tensor
        """
        tensor = self._trans.new_zeros((*self.shape, 4, 4))
        tensor[..., :3, :3] = self._rots.get_rot_mats()
        tensor[..., :3, 3] = self._trans
        tensor[..., 3, 3] = 1
        return tensor

    @staticmethod
    def from_tensor_4x4(t: torch.Tensor) -> Rigid:
        """
        Constructs a transformation from a homogenous transformation tensor.

        Args:
            t: [*, 4, 4] homogenous transformation tensor
        Returns:
            T object with shape [*]
        """
        if t.shape[-2:] != (4, 4):
            raise ValueError("Incorrectly shaped input tensor")

        rots = Rotation(rot_mats=t[..., :3, :3], quats=None)
        trans = t[..., :3, 3]

        return Rigid(rots, trans)

    def to_tensor_7(self) -> torch.Tensor:
        """
        Converts a transformation to a tensor with 7 final columns, four for the quaternion followed by three for the
        translation.

        Returns:
            A [*, 7] tensor representation of the transformation
        """
        tensor = self._trans.new_zeros((*self.shape, 7))
        tensor[..., :4] = self._rots.get_quats()
        tensor[..., 4:] = self._trans

        return tensor

    @staticmethod
    def from_tensor_7(t: torch.Tensor, normalize_quats: bool = False) -> Rigid:
        if t.shape[-1] != 7:
            raise ValueError("Incorrectly shaped input tensor")

        quats, trans = t[..., :4], t[..., 4:]

        rots = Rotation(rot_mats=None, quats=quats, normalize_quats=normalize_quats)

        return Rigid(rots, trans)

    @staticmethod
    def from_3_points(
        p_neg_x_axis: torch.Tensor, origin: torch.Tensor, p_xy_plane: torch.Tensor, eps: float = 1e-8
    ) -> Rigid:
        """
        Implements algorithm 21. Constructs transformations from sets of 3 points using the Gram-Schmidt algorithm.

        Args:
            p_neg_x_axis: [*, 3] coordinates
            origin: [*, 3] coordinates used as frame origins
            p_xy_plane: [*, 3] coordinates
            eps: Small epsilon value
        Returns:
            A transformation object of shape [*]
        """
        p_neg_x_axis_unbound = torch.unbind(p_neg_x_axis, dim=-1)
        origin_unbound = torch.unbind(origin, dim=-1)
        p_xy_plane_unbound = torch.unbind(p_xy_plane, dim=-1)

        e0 = [c1 - c2 for c1, c2 in zip(origin_unbound, p_neg_x_axis_unbound)]
        e1 = [c1 - c2 for c1, c2 in zip(p_xy_plane_unbound, origin_unbound)]

        denom = torch.sqrt(sum(c * c for c in e0) + eps * torch.ones_like(e0[0]))
        e0 = [c / denom for c in e0]
        dot = sum((c1 * c2 for c1, c2 in zip(e0, e1)))
        e1 = [c2 - c1 * dot for c1, c2 in zip(e0, e1)]
        denom = torch.sqrt(sum((c * c for c in e1)) + eps * torch.ones_like(e1[0]))
        e1 = [c / denom for c in e1]
        e2 = [
            e0[1] * e1[2] - e0[2] * e1[1],
            e0[2] * e1[0] - e0[0] * e1[2],
            e0[0] * e1[1] - e0[1] * e1[0],
        ]

        rots = torch.stack([c for tup in zip(e0, e1, e2) for c in tup], dim=-1)
        rots = rots.reshape(rots.shape[:-1] + (3, 3))

        rot_obj = Rotation(rot_mats=rots, quats=None)

        return Rigid(rot_obj, torch.stack(origin_unbound, dim=-1))

    def unsqueeze(self, dim: int) -> Rigid:
        """
        Analogous to torch.unsqueeze. The dimension is relative to the shared dimensions of the rotation/translation.

        Args:
            dim: A positive or negative dimension index.
        Returns:
            The unsqueezed transformation.
        """
        if dim >= len(self.shape):
            raise ValueError("Invalid dimension")
        rots = self._rots.unsqueeze(dim)
        trans = self._trans.unsqueeze(dim if dim >= 0 else dim - 1)

        return Rigid(rots, trans)

    @staticmethod
    def cat(ts: Sequence[Rigid], dim: int) -> Rigid:
        """
        Concatenates transformations along a new dimension.

        Args:
            ts:
                A list of T objects
            dim:
                The dimension along which the transformations should be concatenated
        Returns:
            A concatenated transformation object
        """
        rots = Rotation.cat([t._rots for t in ts], dim)
        trans = torch.cat([t._trans for t in ts], dim=dim if dim >= 0 else dim - 1)

        return Rigid(rots, trans)

    def apply_rot_fn(self, fn: Callable[[Rotation], Rotation]) -> Rigid:
        """
        Applies a Rotation -> Rotation function to the stored rotation object.

        Args:
            fn: A function of type Rotation -> Rotation
        Returns:
            A transformation object with a transformed rotation.
        """
        return Rigid(fn(self._rots), self._trans)

    def apply_trans_fn(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> Rigid:
        """
        Applies a Tensor -> Tensor function to the stored translation.

        Args:
            fn:
                A function of type Tensor -> Tensor to be applied to the translation
        Returns:
            A transformation object with a transformed translation.
        """
        return Rigid(self._rots, fn(self._trans))

    def scale_translation(self, trans_scale_factor: float) -> Rigid:
        """
        Scales the translation by a constant factor.

        Args:
            trans_scale_factor:
                The constant factor
        Returns:
            A transformation object with a scaled translation.
        """
        return self.apply_trans_fn(lambda t: t * trans_scale_factor)

    def stop_rot_gradient(self) -> Rigid:
        """
        Detaches the underlying rotation object

        Returns:
            A transformation object with detached rotations
        """
        return self.apply_rot_fn(lambda r: r.detach())

    @staticmethod
    def make_transform_from_reference(
        n_xyz: torch.Tensor, ca_xyz: torch.Tensor, c_xyz: torch.Tensor, eps: float = 1e-20
    ) -> Rigid:
        """
        Returns a transformation object from reference coordinates.

        Note that this method does not take care of symmetries. If you provide the atom positions in the non-standard
        way, the N atom will end up not at [-0.527250, 1.359329, 0.0] but instead at [-0.527250, -1.359329, 0.0]. You
        need to take care of such cases in your code.

        Args:
            n_xyz: A [*, 3] tensor of nitrogen xyz coordinates.
            ca_xyz: A [*, 3] tensor of carbon alpha xyz coordinates.
            c_xyz: A [*, 3] tensor of carbon xyz coordinates.
        Returns:
            A transformation object. After applying the translation and rotation to the reference backbone, the
            coordinates will approximately equal to the input coordinates.
        """
        translation = -1 * ca_xyz
        n_xyz = n_xyz + translation
        c_xyz = c_xyz + translation

        c_x, c_y, c_z = [c_xyz[..., i] for i in range(3)]
        norm = torch.sqrt(eps + c_x**2 + c_y**2)
        sin_c1 = -c_y / norm
        cos_c1 = c_x / norm

        c1_rots = sin_c1.new_zeros((*sin_c1.shape, 3, 3))
        c1_rots[..., 0, 0] = cos_c1
        c1_rots[..., 0, 1] = -1 * sin_c1
        c1_rots[..., 1, 0] = sin_c1
        c1_rots[..., 1, 1] = cos_c1
        c1_rots[..., 2, 2] = 1

        norm = torch.sqrt(eps + c_x**2 + c_y**2 + c_z**2)
        sin_c2 = c_z / norm
        cos_c2 = torch.sqrt(c_x**2 + c_y**2) / norm

        c2_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
        c2_rots[..., 0, 0] = cos_c2
        c2_rots[..., 0, 2] = sin_c2
        c2_rots[..., 1, 1] = 1
        c2_rots[..., 2, 0] = -1 * sin_c2
        c2_rots[..., 2, 2] = cos_c2

        c_rots = rot_matmul(c2_rots, c1_rots)
        n_xyz = rot_vec_mul(c_rots, n_xyz)

        _, n_y, n_z = [n_xyz[..., i] for i in range(3)]
        norm = torch.sqrt(eps + n_y**2 + n_z**2)
        sin_n = -n_z / norm
        cos_n = n_y / norm

        n_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
        n_rots[..., 0, 0] = 1
        n_rots[..., 1, 1] = cos_n
        n_rots[..., 1, 2] = -1 * sin_n
        n_rots[..., 2, 1] = sin_n
        n_rots[..., 2, 2] = cos_n

        rots = rot_matmul(n_rots, c_rots)

        rots = rots.transpose(-1, -2)
        translation = -1 * translation

        rot_obj = Rotation(rot_mats=rots, quats=None)

        return Rigid(rot_obj, translation)

    def cuda(self) -> Rigid:
        """
        Moves the transformation object to GPU memory

        Returns:
            A version of the transformation on GPU
        """
        return Rigid(self._rots.cuda(), self._trans.cuda())
