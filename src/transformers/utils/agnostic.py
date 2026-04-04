# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""
GPU calls that are device-agnostic.
"""

try:
    import torch
except Exception:
    torch = None


class AgnosticGPU:
    @staticmethod
    def configure() -> "AgnosticGPU":
        return (
            NoGPU()
            if torch is None
            else CUDAGPU()
            if torch.cuda.is_available()
            else XPUGPU()
            if (hasattr(torch, "xpu") and torch.xpu.is_available())
            else MPSGPU()
            if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
            else NoGPU()
        )

    name: str

    def is_accelerator_available(self) -> bool:
        return False

    def current_device(self) -> int:
        return 0

    def device_count(self) -> int:
        return 0


class CUDAGPU(AgnosticGPU):
    def __init__(self):
        assert torch is not None
        self.name = "cuda"
        self.is_accelerator_available = torch.cuda.is_available
        self.current_device = torch.cuda.current_device
        self.device_count = torch.cuda.device_count


class XPUGPU(AgnosticGPU):
    def __init__(self):
        assert torch is not None
        self.name = "xpu"
        self.is_accelerator_available = torch.xpu.is_available
        self.current_device = torch.xpu.current_device
        self.device_count = torch.xpu.device_count


class MPSGPU(AgnosticGPU):
    def __init__(self):
        assert torch is not None
        self.name = "mps"
        self.is_accelerator_available = torch.mps.is_available
        # self.current_device = torch.mps.current_device
        self.device_count = torch.mps.device_count


class NoGPU(AgnosticGPU):
    def __init__(self) -> None:
        self.name = "cpu"


gpu = AgnosticGPU.configure()
