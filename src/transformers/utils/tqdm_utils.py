# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Wrapper around tqdm."""

from tqdm import auto as tqdm_lib


class EmptyTqdm:
    """Dummy tqdm which doesn't do anything."""

    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        self._iterator = args[0] if args else None

    def __iter__(self):
        return iter(self._iterator)

    def __getattr__(self, _):
        """Return empty function."""

        def empty_fn(*args, **kwargs):  # pylint: disable=unused-argument
            return

        return empty_fn

    def __enter__(self):
        return self

    def __exit__(self, type_, value, traceback):
        return


_active = True


class _tqdm_cls:
    def __call__(self, *args, **kwargs):
        if _active:
            return tqdm_lib.tqdm(*args, **kwargs)
        else:
            return EmptyTqdm(*args, **kwargs)

    def set_lock(self, *args, **kwargs):
        self._lock = None
        if _active:
            return tqdm_lib.tqdm.set_lock(*args, **kwargs)

    def get_lock(self):
        if _active:
            return tqdm_lib.tqdm.get_lock()


tqdm = _tqdm_cls()


def set_progress_bar_enabled(boolean: bool):
    """Enable/disable tqdm progress bars."""
    global _active
    _active = bool(boolean)


def is_progress_bar_enabled() -> bool:
    """Return a boolean indicating whether tqdm progress bars are enabled."""
    global _active
    return bool(_active)
