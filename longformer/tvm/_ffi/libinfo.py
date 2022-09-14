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
"""Library information."""
from __future__ import absolute_import
import sys
import os

def split_env_var(env_var, split):
    """Splits environment variable string.

    Parameters
    ----------
    env_var : str
        Name of environment variable.

    split : str
        String to split env_var on.

    Returns
    -------
    splits : list(string)
        If env_var exists, split env_var. Otherwise, empty list.
    """
    if os.environ.get(env_var, None):
        return [p.strip() for p in os.environ[env_var].split(split)]
    return []

def find_lib_path(name=None, search_path=None, optional=False):
    """Find dynamic library files.

    Parameters
    ----------
    name : list of str
        List of names to be found.

    Returns
    -------
    lib_path : list(string)
        List of all found path to the libraries
    """
    use_runtime = os.environ.get("TVM_USE_RUNTIME_LIB", False)

    # See https://github.com/dmlc/tvm/issues/281 for some background.

    # NB: This will either be the source directory (if TVM is run
    # inplace) or the install directory (if TVM is installed).
    # An installed TVM's curr_path will look something like:
    #   $PREFIX/lib/python3.6/site-packages/tvm/_ffi
    ffi_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    source_dir = os.path.join(ffi_dir, "..", "..", "..")
    install_lib_dir = os.path.join(ffi_dir, "..", "..", "..", "..")

    dll_path = []

    if os.environ.get('TVM_LIBRARY_PATH', None):
        dll_path.append(os.environ['TVM_LIBRARY_PATH'])

    if sys.platform.startswith('linux'):
        dll_path.extend(split_env_var('LD_LIBRARY_PATH', ':'))
        dll_path.extend(split_env_var('PATH', ':'))
    elif sys.platform.startswith('darwin'):
        dll_path.extend(split_env_var('DYLD_LIBRARY_PATH', ':'))
        dll_path.extend(split_env_var('PATH', ':'))
    elif sys.platform.startswith('win32'):
        dll_path.extend(split_env_var('PATH', ';'))

    # Pip lib directory
    dll_path.append(os.path.join(ffi_dir, ".."))
    # Default cmake build directory
    dll_path.append(os.path.join(source_dir, "build"))
    dll_path.append(os.path.join(source_dir, "build", "Release"))
    # Default make build directory
    dll_path.append(os.path.join(source_dir, "lib"))

    dll_path.append(install_lib_dir)

    dll_path = [os.path.realpath(x) for x in dll_path]
    if search_path is not None:
        if search_path is list:
            dll_path = dll_path + search_path
        else:
            dll_path.append(search_path)
    if name is not None:
        if isinstance(name, list):
            lib_dll_path = []
            for n in name:
                lib_dll_path += [os.path.join(p, n) for p in dll_path]
        else:
            lib_dll_path = [os.path.join(p, name) for p in dll_path]
        runtime_dll_path = []
    else:
        if sys.platform.startswith('win32'):
            lib_dll_path = [os.path.join(p, 'libtvm.dll') for p in dll_path] +\
                           [os.path.join(p, 'tvm.dll') for p in dll_path]
            runtime_dll_path = [os.path.join(p, 'libtvm_runtime.dll') for p in dll_path] +\
                               [os.path.join(p, 'tvm_runtime.dll') for p in dll_path]
        elif sys.platform.startswith('darwin'):
            lib_dll_path = [os.path.join(p, 'libtvm.dylib') for p in dll_path]
            runtime_dll_path = [os.path.join(p, 'libtvm_runtime.dylib') for p in dll_path]
        else:
            lib_dll_path = [os.path.join(p, 'libtvm.so') for p in dll_path]
            runtime_dll_path = [os.path.join(p, 'libtvm_runtime.so') for p in dll_path]

    if not use_runtime:
        # try to find lib_dll_path
        lib_found = [p for p in lib_dll_path if os.path.exists(p) and os.path.isfile(p)]
        lib_found += [p for p in runtime_dll_path if os.path.exists(p) and os.path.isfile(p)]
    else:
        # try to find runtime_dll_path
        use_runtime = True
        lib_found = [p for p in runtime_dll_path if os.path.exists(p) and os.path.isfile(p)]

    if not lib_found:
        message = ('Cannot find the files.\n' +
                   'List of candidates:\n' +
                   str('\n'.join(lib_dll_path + runtime_dll_path)))
        if not optional:
            raise RuntimeError(message)
        return None

    if use_runtime:
        sys.stderr.write("Loading runtime library %s... exec only\n" % lib_found[0])
        sys.stderr.flush()
    return lib_found


def find_include_path(name=None, search_path=None, optional=False):
    """Find header files for C compilation.

    Parameters
    ----------
    name : list of str
        List of directory names to be searched.

    Returns
    -------
    include_path : list(string)
        List of all found paths to header files.
    """
    ffi_dir = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    source_dir = os.path.join(ffi_dir, "..", "..", "..")
    install_include_dir = os.path.join(ffi_dir, "..", "..", "..", "..")
    third_party_dir = os.path.join(source_dir, "3rdparty")

    header_path = []

    if os.environ.get('TVM_INCLUDE_PATH', None):
        header_path.append(os.environ['TVM_INCLUDE_PATH'])

    header_path.append(install_include_dir)
    header_path.append(source_dir)
    header_path.append(third_party_dir)

    header_path = [os.path.abspath(x) for x in header_path]
    if search_path is not None:
        if search_path is list:
            header_path = header_path + search_path
        else:
            header_path.append(search_path)
    if name is not None:
        if isinstance(name, list):
            tvm_include_path = []
            for n in name:
                tvm_include_path += [os.path.join(p, n) for p in header_path]
        else:
            tvm_include_path = [os.path.join(p, name) for p in header_path]
        dlpack_include_path = []
    else:
        tvm_include_path = [os.path.join(p, 'include') for p in header_path]
        dlpack_include_path = [os.path.join(p, 'dlpack/include') for p in header_path]

        # try to find include path
        include_found = [p for p in tvm_include_path if os.path.exists(p) and os.path.isdir(p)]
        include_found += [p for p in dlpack_include_path if os.path.exists(p) and os.path.isdir(p)]

    if not include_found:
        message = ('Cannot find the files.\n' +
                   'List of candidates:\n' +
                   str('\n'.join(tvm_include_path + dlpack_include_path)))
        if not optional:
            raise RuntimeError(message)
        return None

    return include_found


# current version
# We use the version of the incoming release for code
# that is under development.
# The following line is set by tvm/python/update_version.py
__version__ = "0.6.dev"
