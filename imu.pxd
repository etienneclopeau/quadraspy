# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:07:31 2013

@author: clopeau
"""
import cython
from libc.stdio cimport *
import numpy as np
cimport numpy as np
from capteurs import Acc, Gyr,Mag

DTYPE = np.float
ctypedef np.float_t DTYPE_t


cdef conj(np.ndarray[double, ndim=1] q)



cdef class IMU:
    cdef double quat0
    cdef double quat1
    cdef double quat2
    cdef double quat3
    cdef double earth_magnetic_field_x
    cdef double earth_magnetic_field_z
    cdef double gyr_b0
    cdef double gyr_b1
    cdef double gyr_b2
    cdef double tbefore
    #cdef bint log
    cdef bint running
    cdef np.ndarray gyr
    cdef np.ndarray acc
    cdef np.ndarray mag
    cdef np.ndarray gyrc
    cdef np.ndarray gyr_ba
    cdef double tcurrent
    cdef object logFile
    cdef Acc accelerometer
    cdef Gyr gyrometer
    cdef Mag magnetometer

    #@cython.boundscheck(False) # turn off boundscheck for this function
    @cython.locals(
                   #tcurrent=cython.double, deltat=cython.double,
                   gyroMeasError=cython.double,gyroMeasDrift=cython.double,
                   beta=cython.double,zeta=cython.double,
                   quata=np.ndarray,
                   #gyr_ba=np.ndarray,
                   halfquat=np.ndarray,
                   twoquat=np.ndarray,                   
                   twoearth_magnetic_field_x=cython.double, twoearth_magnetic_field_z=cython.double, 
                   twoearth_magnetic_field_xquat=np.ndarray,
                   twoearth_magnetic_field_zquat=np.ndarray,
                   twomag_x=cython.double, twomag_y=cython.double, twomag_z=cython.double,
                   norm = cython.double,
                   f_1=cython.double,f_2=cython.double,f_3=cython.double,f_4=cython.double,
                   f_5=cython.double,f_6=cython.double,
                   J_11or24=cython.double,J_12or23=cython.double,J_13or22=cython.double,
                   J_14or21=cython.double,J_32=cython.double,J_33=cython.double,J_41=cython.double,
                   J_42=cython.double,J_43=cython.double,J_44=cython.double,
                   J_51=cython.double,J_52=cython.double,J_53=cython.double,
                   J_54=cython.double,J_61=cython.double,J_62=cython.double,
                   J_63=cython.double,J_64=cython.double,
                   quatHatDot_1=cython.double,quatHatDot_2=cython.double,
                   quatHatDot_3=cython.double,quatHatDot_4=cython.double,
                   gyr_err=np.ndarray,
                   quatDot_omega_1=cython.double,quatDot_omega_2=cython.double,
                   quatDot_omega_3=cython.double,quatDot_omega_4=cython.double,
                   h_x=cython.double,h_y=cython.double,h_z=cython.double,
                   phi=cython.double,theta=cython.double,psi=cython.double )
    cpdef update(self)
    cpdef start(self)
    cpdef run(self)
    cpdef stop(self)


@cython.locals(imu=IMU,
               i = cython.int,
               phi=cython.double,theta=cython.double,psi=cython.double)
cpdef logIMU(bint print_=? , bint log=?)
    
@cython.locals(imu=IMU,
               i = cython.int, t0 = cython.double,
               phi=cython.double,theta=cython.double,psi=cython.double)
cpdef timeIMU(int niter =?)

cpdef plotIMU(char* fileName = ?)

cpdef quat2matrix(np.ndarray[np.double_t, ndim = 1 ] quat)

cpdef plotIMU3d()

cpdef plotIMU3d_2()

    
    
