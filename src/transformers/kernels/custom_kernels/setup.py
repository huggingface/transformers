from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='your_extension',
    ext_modules=[
        CUDAExtension(
            name='your_extension',
            sources=['your_extension.cpp', 'kernels.cu'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
