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
# pylint: disable=invalid-name
"""Runtime Object api"""
from __future__ import absolute_import

import ctypes
from ..base import _LIB, check_call
from .types import TypeCode, RETURN_SWITCH

ObjectHandle = ctypes.c_void_p

"""Maps object type to its constructor"""
OBJECT_TYPE = {}

def _register_object(index, cls):
    """register object class"""
    OBJECT_TYPE[index] = cls


def _return_object(x):
    handle = x.v_handle
    if not isinstance(handle, ObjectHandle):
        handle = ObjectHandle(handle)
    tag = ctypes.c_int()
    check_call(_LIB.TVMGetObjectTag(handle, ctypes.byref(tag)))
    cls = OBJECT_TYPE.get(tag.value, ObjectBase)
    obj = cls(handle)
    return obj

RETURN_SWITCH[TypeCode.OBJECT_CELL] = _return_object


class ObjectBase(object):
    __slots__ = ["handle"]

    def __init__(self, handle):
        self.handle = handle
