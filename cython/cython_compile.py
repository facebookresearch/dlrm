# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: compile .so from python code

from __future__ import absolute_import, division, print_function, unicode_literals

from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension

ext_modules = [
    Extension(
        "data_utils_cython",
        ["data_utils_cython.pyx"],
        extra_compile_args=['-O3'],
        extra_link_args=['-O3'],
    )
]

setup(
    name='data_utils_cython',
    ext_modules=cythonize(ext_modules)
)
