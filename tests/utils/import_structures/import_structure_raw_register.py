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
class A0:
    def __init__(self):
        pass


@export()
def a0():
    pass


@export(backends=("torch", "tf"))
class A1:
    def __init__(self):
        pass


@export(backends=("torch", "tf"))
def a1():
    pass


@export(
    backends=("torch", "tf")
)
class A2:
    def __init__(self):
        pass


@export(
    backends=("torch", "tf")
)
def a2():
    pass


@export(
    backends=(
        "torch",
        "tf"
    )
)
class A3:
    def __init__(self):
        pass


@export(
    backends=(
            "torch",
            "tf"
    )
)
def a3():
    pass

@export(backends=())
class A4:
    def __init__(self):
        pass
