# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:07:31 2013

@author: clopeau
"""
import cython
import numpy as np
cimport numpy as np

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
    
    #@cython.boundscheck(False) # turn off boundscheck for this function
    @cython.locals(
                   tcurrent=cython.double, deltat=cython.double,
                   gyroMeasError=cython.double,gyroMeasDrift=cython.double,
                   beta=cython.double,zeta=cython.double,
                   quata=np.ndarray,
                   gyr_ba=np.ndarray,
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
    cpdef tuple update(self, np.ndarray[double, ndim=1] acc, np.ndarray[double, ndim=1] mag, np.ndarray[double, ndim=1] gyr)
#        cdef:
#            double tcurrent
#            double deltat
#            double gyroMeasError
#            double gyroMeasDrift
#            double beta
#            double zeta
#            #np.ndarray[np.double_t, dim = 1 ] halfquat
#            #np.ndarray[np.double_t, dim = 1 ]twoquat
#            double twoearth_magnetic_field_x
#            double twoearth_magnetic_field_z
#            #np.ndarray[np.double_t, dim = 1 ] twoearth_magnetic_field_xquat
#            #np.ndarray[np.double_t, dim = 1 ] twoearth_magnetic_field_zquat
#            double twomag_x
#            double twomag_y
#            double twomag_z
#            double f_1
#            double f_2
#            double f_3
#            double f_4
#            double f_5
#            double f_6
#            double J_11or24
#            double J_12or23
#            double J_13or22
#            double J_14or21
#            double J_32
#            double J_33
#            double J_41
#            double J_42
#            double J_43
#            double J_44
#            double J_51
#            double J_52
#            double J_53
#            double J_54
#            double J_61
#            double J_62
#            double J_63
#            double J_64
#            double quatHatDot_1
#            double quatHatDot_2
#            double quatHatDot_3
#            double quatHatDot_4
#            #np.ndarray[np.double_t, dim = 1 ] gyr_err=np.ndarray,
#            double quatDot_omega_1
#            double quatDot_omega_2
#            double quatDot_omega_3
#            double quatDot_omega_4
#            double h_x
#            double h_y
#            double h_z
#            double phi
#            double theta
#            double psi

@cython.locals(imu=IMU,
               i = cython.int, t0 = cython.double,
               ax = cython.double,ay = cython.double,az = cython.double,
               hx = cython.double,hy = cython.double,hz = cython.double,
               gx = cython.double,gy = cython.double,gz = cython.double,
               phi=cython.double,theta=cython.double,psi=cython.double)
cpdef logIMU()
    
@cython.locals(imu=IMU,
               i = cython.int, t0 = cython.double,
               phi=cython.double,theta=cython.double,psi=cython.double)
cpdef timeIMU(int niter =?)

cpdef plotIMU()

cpdef quat2matrix(np.ndarray[np.double_t, ndim = 1 ] quat)

cpdef plotIMU3d()

cpdef plotIMU3d_2()

    
    
