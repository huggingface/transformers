# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-import
"""Runtime NDArray api"""
from __future__ import absolute_import

import sys
import ctypes
import numpy as np
from .base import _LIB, check_call, c_array, string_types, _FFI_MODE, c_str
from .runtime_ctypes import TVMType, TVMContext, TVMArray, TVMArrayHandle
from .runtime_ctypes import TypeCode, tvm_shape_index_t


IMPORT_EXCEPT = RuntimeError if _FFI_MODE == "cython" else ImportError

try:
    # pylint: disable=wrong-import-position
    if _FFI_MODE == "ctypes":
        raise ImportError()
    if sys.version_info >= (3, 0):
        from ._cy3.core import _set_class_ndarray, _make_array, _from_dlpack
        from ._cy3.core import NDArrayBase as _NDArrayBase
        from ._cy3.core import _reg_extension, _reg_ndarray
    else:
        from ._cy2.core import _set_class_ndarray, _make_array, _from_dlpack
        from ._cy2.core import NDArrayBase as _NDArrayBase
        from ._cy2.core import _reg_extension, _reg_ndarray
except IMPORT_EXCEPT:
    # pylint: disable=wrong-import-position
    from ._ctypes.ndarray import _set_class_ndarray, _make_array, _from_dlpack
    from ._ctypes.ndarray import NDArrayBase as _NDArrayBase
    from ._ctypes.ndarray import _reg_extension, _reg_ndarray


def context(dev_type, dev_id=0):
    """Construct a TVM context with given device type and id.

    Parameters
    ----------
    dev_type: int or str
        The device type mask or name of the device.

    dev_id : int, optional
        The integer device id

    Returns
    -------
    ctx: TVMContext
        The corresponding context.

    Examples
    --------
    Context can be used to create reflection of context by
    string representation of the device type.

    .. code-block:: python

      assert tvm.context("cpu", 1) == tvm.cpu(1)
      assert tvm.context("gpu", 0) == tvm.gpu(0)
      assert tvm.context("cuda", 0) == tvm.gpu(0)
    """
    if isinstance(dev_type, string_types):
        dev_type = dev_type.split()[0]
        if dev_type not in TVMContext.STR2MASK:
            raise ValueError("Unknown device type %s" % dev_type)
        dev_type = TVMContext.STR2MASK[dev_type]
    return TVMContext(dev_type, dev_id)


def numpyasarray(np_data):
    """Return a TVMArray representation of a numpy array.
    """
    data = np_data
    assert data.flags['C_CONTIGUOUS']
    arr = TVMArray()
    shape = c_array(tvm_shape_index_t, data.shape)
    arr.data = data.ctypes.data_as(ctypes.c_void_p)
    arr.shape = shape
    arr.strides = None
    arr.dtype = TVMType(np.dtype(data.dtype).name)
    arr.ndim = data.ndim
    # CPU device
    arr.ctx = context(1, 0)
    return arr, shape


def empty(shape, dtype="float32", ctx=context(1, 0)):
    """Create an empty array given shape and device

    Parameters
    ----------
    shape : tuple of int
        The shape of the array

    dtype : type or str
        The data type of the array.

    ctx : TVMContext
        The context of the array

    Returns
    -------
    arr : tvm.nd.NDArray
        The array tvm supported.
    """
    shape = c_array(tvm_shape_index_t, shape)
    ndim = ctypes.c_int(len(shape))
    handle = TVMArrayHandle()
    dtype = TVMType(dtype)
    check_call(_LIB.TVMArrayAlloc(
        shape, ndim,
        ctypes.c_int(dtype.type_code),
        ctypes.c_int(dtype.bits),
        ctypes.c_int(dtype.lanes),
        ctx.device_type,
        ctx.device_id,
        ctypes.byref(handle)))
    return _make_array(handle, False, False)


def from_dlpack(dltensor):
    """Produce an array from a DLPack tensor without memory copy.
    Retreives the underlying DLPack tensor's pointer to create an array from the
    data. Removes the original DLPack tensor's destructor as now the array is
    responsible for destruction.

    Parameters
    ----------
    dltensor : DLPack tensor
        Input DLManagedTensor, can only be consumed once.

    Returns
    -------
    arr: tvm.nd.NDArray
        The array view of the tensor data.
    """
    return _from_dlpack(dltensor)


class NDArrayBase(_NDArrayBase):
    """A simple Device/CPU Array object in runtime."""
    @property
    def shape(self):
        """Shape of this array"""
        return tuple(self.handle.contents.shape[i] for i in range(self.handle.contents.ndim))

    @property
    def dtype(self):
        """Type of this array"""
        return str(self.handle.contents.dtype)

    @property
    def ctx(self):
        """context of this array"""
        return self.handle.contents.ctx

    @property
    def context(self):
        """context of this array"""
        return self.ctx

    def __hash__(self):
        return ctypes.cast(self.handle, ctypes.c_void_p).value

    def __eq__(self, other):
        return self.same_as(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def same_as(self, other):
        """Check object identity equality

        Parameters
        ----------
        other : object
            The other object to compare to

        Returns
        -------
        same : bool
            Whether other is same as self.
        """
        if not isinstance(other, NDArrayBase):
            return False
        return self.__hash__() == other.__hash__()

    def __setitem__(self, in_slice, value):
        """Set ndarray value"""
        if (not isinstance(in_slice, slice) or
                in_slice.start is not None
                or in_slice.stop is not None):
            raise ValueError('Array only support set from numpy array')
        if isinstance(value, NDArrayBase):
            if value.handle is not self.handle:
                value.copyto(self)
        elif isinstance(value, (np.ndarray, np.generic)):
            self.copyfrom(value)
        else:
            raise TypeError('type %s not supported' % str(type(value)))

    def copyfrom(self, source_array):
        """Peform an synchronize copy from the array.

        Parameters
        ----------
        source_array : array_like
            The data source we should like to copy from.

        Returns
        -------
        arr : NDArray
            Reference to self.
        """
        if isinstance(source_array, NDArrayBase):
            source_array.copyto(self)
            return self

        if not isinstance(source_array, np.ndarray):
            try:
                source_array = np.array(source_array, dtype=self.dtype)
            except:
                raise TypeError('array must be an array_like data,' +
                                'type %s is not supported' % str(type(source_array)))
        t = TVMType(self.dtype)
        shape, dtype = self.shape, self.dtype
        if t.lanes > 1:
            shape = shape + (t.lanes,)
            t.lanes = 1
            dtype = str(t)

        if source_array.shape != shape:
            raise ValueError("array shape do not match the shape of NDArray {0} vs {1}".format(
                source_array.shape, shape))
        source_array = np.ascontiguousarray(source_array, dtype=dtype)
        assert source_array.flags['C_CONTIGUOUS']
        data = source_array.ctypes.data_as(ctypes.c_void_p)
        nbytes = ctypes.c_size_t(source_array.size * source_array.dtype.itemsize)
        check_call(_LIB.TVMArrayCopyFromBytes(self.handle, data, nbytes))
        return self

    def __repr__(self):
        res = "<tvm.NDArray shape={0}, {1}>\n".format(self.shape, self.context)
        res += self.asnumpy().__repr__()
        return res

    def __str__(self):
        return str(self.asnumpy())

    def asnumpy(self):
        """Convert this array to numpy array

        Returns
        -------
        np_arr : numpy.ndarray
            The corresponding numpy array.
        """
        t = TVMType(self.dtype)
        shape, dtype = self.shape, self.dtype
        if t.lanes > 1:
            shape = shape + (t.lanes,)
            t.lanes = 1
            dtype = str(t)
        np_arr = np.empty(shape, dtype=dtype)
        assert np_arr.flags['C_CONTIGUOUS']
        data = np_arr.ctypes.data_as(ctypes.c_void_p)
        nbytes = ctypes.c_size_t(np_arr.size * np_arr.dtype.itemsize)
        check_call(_LIB.TVMArrayCopyToBytes(self.handle, data, nbytes))
        return np_arr

    def copyto(self, target):
        """Copy array to target

        Parameters
        ----------
        target : NDArray
            The target array to be copied, must have same shape as this array.
        """
        if isinstance(target, TVMContext):
            target = empty(self.shape, self.dtype, target)
        if isinstance(target, NDArrayBase):
            check_call(_LIB.TVMArrayCopyFromTo(
                self.handle, target.handle, None))
        else:
            raise ValueError("Unsupported target type %s" % str(type(target)))
        return target


def free_extension_handle(handle, type_code):
    """Free c++ extension type handle

    Parameters
    ----------
    handle : ctypes.c_void_p
        The handle to the extension type.

    type_code : int
         The tyoe code
    """
    check_call(_LIB.TVMExtTypeFree(handle, ctypes.c_int(type_code)))


def register_extension(cls, fcreate=None):
    """Register a extension class to TVM.

    After the class is registered, the class will be able
    to directly pass as Function argument generated by TVM.

    Parameters
    ----------
    cls : class
        The class object to be registered as extension.

    fcreate : function, optional
        The creation function to create a class object given handle value.

    Note
    ----
    The registered class is requires one property: _tvm_handle.

    If the registered class is a subclass of NDArray,
    it is required to have a class attribute _array_type_code.
    Otherwise, it is required to have a class attribute _tvm_tcode.

    - ```_tvm_handle``` returns integer represents the address of the handle.
    - ```_tvm_tcode``` or ```_array_type_code``` gives integer represents type
      code of the class.

    Returns
    -------
    cls : class
        The class being registered.

    Example
    -------
    The following code registers user defined class
    MyTensor to be DLTensor compatible.

    .. code-block:: python

       @tvm.register_extension
       class MyTensor(object):
           _tvm_tcode = tvm.TypeCode.ARRAY_HANDLE

           def __init__(self):
               self.handle = _LIB.NewDLTensor()

           @property
           def _tvm_handle(self):
               return self.handle.value
    """
    if issubclass(cls, _NDArrayBase):
        assert fcreate is not None
        assert hasattr(cls, "_array_type_code")
        _reg_ndarray(cls, fcreate)
    else:
        assert hasattr(cls, "_tvm_tcode")
        if fcreate and cls._tvm_tcode < TypeCode.EXT_BEGIN:
            raise ValueError("Cannot register create when extension tcode is same as buildin")
        _reg_extension(cls, fcreate)
    return cls
