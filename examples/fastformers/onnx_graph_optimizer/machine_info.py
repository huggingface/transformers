#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

# It is used to dump machine information for Notebooks

import argparse
import logging
from typing import List, Dict, Union, Tuple
import cpuinfo
import psutil
import json
import sys
import platform
from os import environ
from py3nvml.py3nvml import nvmlInit, nvmlSystemGetDriverVersion, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, \
                            nvmlDeviceGetMemoryInfo, nvmlDeviceGetName, nvmlShutdown, NVMLError


class MachineInfo():
    """ Class encapsulating Machine Info logic. """
    def __init__(self, silent=False, logger=None):
        self.silent = silent

        if logger is None:
            logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s: %(message)s", level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        self.machine_info = None
        try:
            self.machine_info = self.get_machine_info()
        except Exception:
            self.logger.exception("Exception in getting machine info.")
            self.machine_info = None

    def get_machine_info(self):
        """Get machine info in metric format"""
        gpu_info = self.get_gpu_info_by_nvml()
        cpu_info = cpuinfo.get_cpu_info()

        machine_info = {
            "gpu": gpu_info,
            "cpu": self.get_cpu_info(),
            "memory": self.get_memory_info(),
            "python": self._try_get(cpu_info, ["python_version"]),
            "os": platform.platform(),
            "onnxruntime": self.get_onnxruntime_info(),
            "onnxruntime_tools": self.get_onnxruntime_tools_info(),
            "pytorch": self.get_pytorch_info(),
            "tensorflow": self.get_tensorflow_info()
        }
        return machine_info

    def get_memory_info(self) -> Dict:
        """Get memory info"""
        mem = psutil.virtual_memory()
        return {"total": mem.total, "available": mem.available}

    def _try_get(self, cpu_info: Dict, names: List) -> str:
        for name in names:
            if name in cpu_info:
                return cpu_info[name]
        return ""

    def get_cpu_info(self) -> Dict:
        """Get CPU info"""
        cpu_info = cpuinfo.get_cpu_info()

        return {
            "brand": self._try_get(cpu_info, ["brand", "brand_raw"]),
            "cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "hz": self._try_get(cpu_info, ["hz_actual"]),
            "l2_cache": self._try_get(cpu_info, ["l2_cache_size"]),
            "flags": self._try_get(cpu_info, ["flags"]),
            "processor": platform.uname().processor
        }

    def get_gpu_info_by_nvml(self) -> Dict:
        """Get GPU info using nvml"""
        gpu_info_list = []
        driver_version = None
        try:
            nvmlInit()
            driver_version = nvmlSystemGetDriverVersion()
            deviceCount = nvmlDeviceGetCount()
            for i in range(deviceCount):
                handle = nvmlDeviceGetHandleByIndex(i)
                info = nvmlDeviceGetMemoryInfo(handle)
                gpu_info = {}
                gpu_info["memory_total"] = info.total
                gpu_info["memory_available"] = info.free
                gpu_info["name"] = nvmlDeviceGetName(handle)
                gpu_info_list.append(gpu_info)
            nvmlShutdown()
        except NVMLError as error:
            if not self.silent:
                self.logger.error("Error fetching GPU information using nvml: %s", error)
            return None

        result = {"driver_version": driver_version, "devices": gpu_info_list}

        if 'CUDA_VISIBLE_DEVICES' in environ:
            result["cuda_visible"] = environ['CUDA_VISIBLE_DEVICES']
        return result

    def get_onnxruntime_info(self) -> Dict:
        try:
            import onnxruntime
            return {
                "version": onnxruntime.__version__,
                "support_gpu": 'CUDAExecutionProvider' in onnxruntime.get_available_providers()
            }
        except ImportError as error:
            if not self.silent:
                self.logger.exception(error)
            return None
        except Exception as exception:
            if not self.silent:
                self.logger.exception(exception, False)
            return None

    def get_onnxruntime_tools_info(self) -> Dict:
        try:
            import onnxruntime_tools
            return {"version": onnxruntime_tools.__version__}
        except ImportError as error:
            if not self.silent:
                self.logger.exception(error)
            return None
        except Exception as exception:
            if not self.silent:
                self.logger.exception(exception, False)
            return None

    def get_pytorch_info(self) -> Dict:
        try:
            import torch
            return {"version": torch.__version__, "support_gpu": torch.cuda.is_available(), "cuda": torch.version.cuda}
        except ImportError as error:
            if not self.silent:
                self.logger.exception(error)
            return None
        except Exception as exception:
            if not self.silent:
                self.logger.exception(exception, False)
            return None

    def get_tensorflow_info(self) -> Dict:
        try:
            import tensorflow as tf
            return {
                "version": tf.version.VERSION,
                "git_version": tf.version.GIT_VERSION,
                "support_gpu": tf.test.is_built_with_cuda()
            }
        except ImportError as error:
            if not self.silent:
                self.logger.exception(error)
            return None
        except ModuleNotFoundError as error:
            if not self.silent:
                self.logger.exception(error)
            return None


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--silent', required=False, action='store_true', help="Do not print error message")
    parser.set_defaults(silent=False)

    args = parser.parse_args()
    return args


def get_machine_info(silent=True) -> str:
    machine = MachineInfo(silent)
    return json.dumps(machine.machine_info, indent=2)


if __name__ == '__main__':
    args = parse_arguments()
    print(get_machine_info(args.silent))
