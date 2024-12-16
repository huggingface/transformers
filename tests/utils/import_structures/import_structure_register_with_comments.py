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


@export()
# That's a statement
class B0:
    def __init__(self):
        pass


@export()
# That's a statement
def b0():
    pass


@export(backends=("torch", "tf"))
# That's a statement
class B1:
    def __init__(self):
        pass


@export(backends=("torch", "tf"))
# That's a statement
def b1():
    pass


@export(backends=("torch", "tf"))
# That's a statement
class B2:
    def __init__(self):
        pass


@export(backends=("torch", "tf"))
# That's a statement
def b2():
    pass


@export(
    backends=(
        "torch",
        "tf"
    )
)
# That's a statement
class B3:
    def __init__(self):
        pass


@export(
    backends=(
        "torch",
        "tf"
    )
)
# That's a statement
def b3():
    pass
