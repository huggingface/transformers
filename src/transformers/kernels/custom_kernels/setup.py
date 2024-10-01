from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sparse_kernels',
    ext_modules=[
        CUDAExtension(
            name='sparse_kernels',
            sources=['sparse_kernels.cpp', 'kernels.cu'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
