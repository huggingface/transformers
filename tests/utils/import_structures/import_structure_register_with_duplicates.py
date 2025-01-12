# Copyright 2024 The HuggingFace Team. All rights reserved.
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

# fmt: off

from transformers.utils.import_utils import export


@export(backends=("torch", "torch"))
class C0:
    def __init__(self):
        pass


@export(backends=("torch", "torch"))
def c0():
    pass


@export(backends=("torch", "torch"))
# That's a statement
class C1:
    def __init__(self):
        pass


@export(backends=("torch", "torch"))
# That's a statement
def c1():
    pass


@export(backends=("torch", "torch"))
# That's a statement
class C2:
    def __init__(self):
        pass


@export(backends=("torch", "torch"))
# That's a statement
def c2():
    pass


@export(
    backends=(
        "torch",
        "torch"
    )
)
# That's a statement
class C3:
    def __init__(self):
        pass


@export(
    backends=(
        "torch",
        "torch"
    )
)
# That's a statement
def c3():
    pass
