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
# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-branches, global-statement
"""Function configuration API."""
from __future__ import absolute_import

import ctypes
import traceback
from numbers import Number, Integral

from ..base import _LIB, get_last_ffi_error, py2cerror
from ..base import c_str, string_types
from ..node_generic import convert_to_node, NodeGeneric
from ..runtime_ctypes import TVMType, TVMByteArray, TVMContext
from . import ndarray as _nd
from .ndarray import NDArrayBase, _make_array
from .types import TVMValue, TypeCode
from .types import TVMPackedCFunc, TVMCFuncFinalizer
from .types import RETURN_SWITCH, C_TO_PY_ARG_SWITCH, _wrap_arg_func, _ctx_to_int64
from .node import NodeBase
from . import node as _node

FunctionHandle = ctypes.c_void_p
ModuleHandle = ctypes.c_void_p
ObjectHandle = ctypes.c_void_p
TVMRetValueHandle = ctypes.c_void_p

def _ctypes_free_resource(rhandle):
    """callback to free resources when it it not needed."""
    pyobj = ctypes.cast(rhandle, ctypes.py_object)
    ctypes.pythonapi.Py_DecRef(pyobj)

# Global callback that is always alive
TVM_FREE_PYOBJ = TVMCFuncFinalizer(_ctypes_free_resource)
ctypes.pythonapi.Py_IncRef(ctypes.py_object(TVM_FREE_PYOBJ))

def convert_to_tvm_func(pyfunc):
    """Convert a python function to TVM function

    Parameters
    ----------
    pyfunc : python function
        The python function to be converted.

    Returns
    -------
    tvmfunc: tvm.nd.Function
        The converted tvm function.
    """
    local_pyfunc = pyfunc
    def cfun(args, type_codes, num_args, ret, _):
        """ ctypes function """
        num_args = num_args.value if isinstance(num_args, ctypes.c_int) else num_args
        pyargs = (C_TO_PY_ARG_SWITCH[type_codes[i]](args[i]) for i in range(num_args))
        # pylint: disable=broad-except
        try:
            rv = local_pyfunc(*pyargs)
        except Exception:
            msg = traceback.format_exc()
            msg = py2cerror(msg)
            _LIB.TVMAPISetLastError(c_str(msg))
            return -1

        if rv is not None:
            if isinstance(rv, tuple):
                raise ValueError("PackedFunction can only support one return value")
            temp_args = []
            values, tcodes, _ = _make_tvm_args((rv,), temp_args)
            if not isinstance(ret, TVMRetValueHandle):
                ret = TVMRetValueHandle(ret)
            if _LIB.TVMCFuncSetReturn(ret, values, tcodes, ctypes.c_int(1)) != 0:
                raise get_last_ffi_error()
            _ = temp_args
            _ = rv
        return 0

    handle = FunctionHandle()
    f = TVMPackedCFunc(cfun)
    # NOTE: We will need to use python-api to increase ref count of the f
    # TVM_FREE_PYOBJ will be called after it is no longer needed.
    pyobj = ctypes.py_object(f)
    ctypes.pythonapi.Py_IncRef(pyobj)
    if _LIB.TVMFuncCreateFromCFunc(
            f, pyobj, TVM_FREE_PYOBJ, ctypes.byref(handle)) != 0:
        raise get_last_ffi_error()
    return _CLASS_FUNCTION(handle, False)


def _make_tvm_args(args, temp_args):
    """Pack arguments into c args tvm call accept"""
    num_args = len(args)
    values = (TVMValue * num_args)()
    type_codes = (ctypes.c_int * num_args)()
    for i, arg in enumerate(args):
        if isinstance(arg, NodeBase):
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.NODE_HANDLE
        elif arg is None:
            values[i].v_handle = None
            type_codes[i] = TypeCode.NULL
        elif isinstance(arg, NDArrayBase):
            values[i].v_handle = ctypes.cast(arg.handle, ctypes.c_void_p)
            type_codes[i] = (TypeCode.NDARRAY_CONTAINER
                             if not arg.is_view else TypeCode.ARRAY_HANDLE)
        elif isinstance(arg, _nd._TVM_COMPATS):
            values[i].v_handle = ctypes.c_void_p(arg._tvm_handle)
            type_codes[i] = arg.__class__._tvm_tcode
        elif isinstance(arg, Integral):
            values[i].v_int64 = arg
            type_codes[i] = TypeCode.INT
        elif isinstance(arg, Number):
            values[i].v_float64 = arg
            type_codes[i] = TypeCode.FLOAT
        elif isinstance(arg, TVMType):
            values[i].v_str = c_str(str(arg))
            type_codes[i] = TypeCode.STR
        elif isinstance(arg, TVMContext):
            values[i].v_int64 = _ctx_to_int64(arg)
            type_codes[i] = TypeCode.TVM_CONTEXT
        elif isinstance(arg, bytearray):
            arr = TVMByteArray()
            arr.data = ctypes.cast(
                (ctypes.c_byte * len(arg)).from_buffer(arg),
                ctypes.POINTER(ctypes.c_byte))
            arr.size = len(arg)
            values[i].v_handle = ctypes.c_void_p(ctypes.addressof(arr))
            temp_args.append(arr)
            type_codes[i] = TypeCode.BYTES
        elif isinstance(arg, string_types):
            values[i].v_str = c_str(arg)
            type_codes[i] = TypeCode.STR
        elif isinstance(arg, (list, tuple, dict, NodeGeneric)):
            arg = convert_to_node(arg)
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.NODE_HANDLE
            temp_args.append(arg)
        elif isinstance(arg, _CLASS_MODULE):
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.MODULE_HANDLE
        elif isinstance(arg, FunctionBase):
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.FUNC_HANDLE
        elif isinstance(arg, ctypes.c_void_p):
            values[i].v_handle = arg
            type_codes[i] = TypeCode.HANDLE
        elif callable(arg):
            arg = convert_to_tvm_func(arg)
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.FUNC_HANDLE
            temp_args.append(arg)
        elif isinstance(arg, _CLASS_OBJECT):
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.OBJECT_CELL
        else:
            raise TypeError("Don't know how to handle type %s" % type(arg))
    return values, type_codes, num_args


class FunctionBase(object):
    """Function base."""
    __slots__ = ["handle", "is_global"]
    # pylint: disable=no-member
    def __init__(self, handle, is_global):
        """Initialize the function with handle

        Parameters
        ----------
        handle : FunctionHandle
            the handle to the underlying function.

        is_global : bool
            Whether this is a global function in python
        """
        self.handle = handle
        self.is_global = is_global

    def __del__(self):
        if not self.is_global and _LIB is not None:
            if _LIB.TVMFuncFree(self.handle) != 0:
                raise get_last_ffi_error()

    def __call__(self, *args):
        """Call the function with positional arguments

        args : list
           The positional arguments to the function call.
        """
        temp_args = []
        values, tcodes, num_args = _make_tvm_args(args, temp_args)
        ret_val = TVMValue()
        ret_tcode = ctypes.c_int()
        if _LIB.TVMFuncCall(
                self.handle, values, tcodes, ctypes.c_int(num_args),
                ctypes.byref(ret_val), ctypes.byref(ret_tcode)) != 0:
            raise get_last_ffi_error()
        _ = temp_args
        _ = args
        return RETURN_SWITCH[ret_tcode.value](ret_val)


def __init_handle_by_constructor__(fconstructor, args):
    """Initialize handle by constructor"""
    temp_args = []
    values, tcodes, num_args = _make_tvm_args(args, temp_args)
    ret_val = TVMValue()
    ret_tcode = ctypes.c_int()
    if _LIB.TVMFuncCall(
            fconstructor.handle, values, tcodes, ctypes.c_int(num_args),
            ctypes.byref(ret_val), ctypes.byref(ret_tcode)) != 0:
        raise get_last_ffi_error()
    _ = temp_args
    _ = args
    assert ret_tcode.value == TypeCode.NODE_HANDLE
    handle = ret_val.v_handle
    return handle


def _return_module(x):
    """Return function"""
    handle = x.v_handle
    if not isinstance(handle, ModuleHandle):
        handle = ModuleHandle(handle)
    return _CLASS_MODULE(handle)


def _handle_return_func(x):
    """Return function"""
    handle = x.v_handle
    if not isinstance(handle, FunctionHandle):
        handle = FunctionHandle(handle)
    return _CLASS_FUNCTION(handle, False)

# setup return handle for function type
_node.__init_by_constructor__ = __init_handle_by_constructor__
RETURN_SWITCH[TypeCode.FUNC_HANDLE] = _handle_return_func
RETURN_SWITCH[TypeCode.MODULE_HANDLE] = _return_module
RETURN_SWITCH[TypeCode.NDARRAY_CONTAINER] = lambda x: _make_array(x.v_handle, False, True)
C_TO_PY_ARG_SWITCH[TypeCode.FUNC_HANDLE] = _wrap_arg_func(
    _handle_return_func, TypeCode.FUNC_HANDLE)
C_TO_PY_ARG_SWITCH[TypeCode.MODULE_HANDLE] = _wrap_arg_func(
    _return_module, TypeCode.MODULE_HANDLE)
C_TO_PY_ARG_SWITCH[TypeCode.ARRAY_HANDLE] = lambda x: _make_array(x.v_handle, True, False)
C_TO_PY_ARG_SWITCH[TypeCode.NDARRAY_CONTAINER] = lambda x: _make_array(x.v_handle, False, True)

_CLASS_MODULE = None
_CLASS_FUNCTION = None
_CLASS_OBJECT = None

def _set_class_module(module_class):
    """Initialize the module."""
    global _CLASS_MODULE
    _CLASS_MODULE = module_class

def _set_class_function(func_class):
    global _CLASS_FUNCTION
    _CLASS_FUNCTION = func_class

def _set_class_object(obj_class):
    global _CLASS_OBJECT
    _CLASS_OBJECT = obj_class
