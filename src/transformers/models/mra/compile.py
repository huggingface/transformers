from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

curr_path = os.path.dirname(os.path.realpath(__file__))
src_files = ["cuda_kernel.cu", "cuda_launch.cu", "torch_extension.cpp"]
src_files = [os.path.join(curr_path, file) for file in src_files]

setup(name = 'cuda_kernel', ext_modules=[cpp_extension.CUDAExtension(name = "kernel", sources = src_files)],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

