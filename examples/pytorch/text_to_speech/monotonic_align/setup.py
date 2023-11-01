from distutils.core import setup

import numpy
from Cython.Build import cythonize


setup(name="monotonic_align", ext_modules=cythonize("core.pyx"), include_dirs=[numpy.get_include()])
