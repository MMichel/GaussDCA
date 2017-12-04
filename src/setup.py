from __future__ import division
from setuptools import setup, Extension

from Cython.Build import cythonize


extension = Extension('_load_data', ['_load_data.pyx'],
                      extra_compile_args="-O2 -march=native -pipe -mtune=native".split(),
                      extra_link_args="-O2 -march=native -pipe -mtune=native".split())
setup(name='ss_load_data', ext_modules=cythonize(extension))


extension = Extension('_gaussdca', ['_gaussdca.pyx'],
                      extra_compile_args='-O2 -march=native -pipe -std=c11'.split(),
                      extra_link_args='-O2 -march=native -pipe -std=c11'.split())
setup(name='_gaussdca', ext_modules=cythonize(extension))


extension = Extension('_gaussdca_parallel', ['_gaussdca_parallel.pyx'],
                      extra_compile_args='-O2 -march=native -pipe -std=c11 -fopenmp'.split(),
                      extra_link_args='-O2 -march=native -pipe -std=c11 -fopenmp'.split())
setup(name='_gaussdca_parallel', ext_modules=cythonize(extension))


extension = Extension('_gaussdca_parallel_opt', ['_gaussdca_parallel_opt.pyx'],
                      extra_compile_args='-O2 -march=native -pipe -std=c11 -fopenmp'.split(),
                      extra_link_args='-O2 -march=native -pipe -std=c11 -fopenmp'.split())
setup(name='_gaussdca_parallel_opt', ext_modules=cythonize(extension))
