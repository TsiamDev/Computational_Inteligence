# -*- coding: utf-8 -*-
"""
Created on Tue May 31 21:35:19 2022

@author: HomeTheater
"""

from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Hello world app',
    ext_modules=cythonize("PERHAPS_FINAL_TESTBENCH.py"),
    zip_safe=False,
)