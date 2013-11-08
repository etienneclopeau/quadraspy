# -*- coding: utf-8 -*-
"""
Created on Thu Nov 07 10:59:30 2013

@author: clopeau
"""

from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy.distutils.misc_util
include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs()
setup (
       ext_modules = [Extension('imu',
                                ['imu.py','imu.pxd'],
                                 include_dirs=include_dirs)],
       cmdclass = {'build_ext': build_ext}
       )