# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:07:31 2013

@author: clopeau
"""
import cython
cimport numpy

import time
from numpy import   array,  sqrt,zeros,cross,arctan,arcsin,radians,degrees
from numpy.linalg import norm as npnorm
from capteurs import getCapteurs
 
def conj(q):
    return array([q[0],-q[1],-q[2],-q[3]])

@cython.cclass
@cython.locals(quat = numpy.ndarrayy[double, ndim=1],
               earth_magnetic_field_x = cython.double,
               earth_magnetic_field_z = cython.double,
               gyrb = numpy.ndarrayy[double, ndim=1],
               tbefore = cython.double)
class IMU():
    """class IMU
    based on http:#www.x-io.co.uk/res/doc/madgwick_internal_report.pdf
    """
    
    def __init__(self):
        self.quat = array([1.,0.,0.,0.]) # quaternion
        self.earth_magnetic_field_x = 1. # orientation of earth magnetic field in ground coordinates
        self.earth_magnetic_field_z = 0. 
        self.gyr_b = zeros(3) # estimated bias of gyrometers
        self.tbefore = time.time()


    @cython.cfunc
    @cython.returns(cython.tuple)
    @cython.locals(acc = cython.tuple, mag = cython.tuple, gyr = cython.tuple,
                   tcurrent=cython.double, deltat=cython.double,
                   gyroMeasError=cython.double,gyroMeasDrift=cython.double,
                   beta=cython.double,zeta=cython.double,
                   halfquat=numpy.ndarrayy[double, ndim=1],
                   twoquat=numpy.ndarrayy[double, ndim=1],
                   twoearth_magnetic_field_x=cython.double, twoearth_magnetic_field_z=cython.double, 
                   twoearth_magnetic_field_xquat=cython.double, twoearth_magnetic_field_zquat=cython.double,
                   twomag_x=cython.double, twomag_y=cython.double, twomag_z=cython.double,
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
                   gyr_err=numpy.ndarrayy[double, ndim=1],
                   quatDot_omega_1=cython.double,quatDot_omega_2=cython.double,
                   quatDot_omega_3=cython.double,quatDot_omega_4=cython.double,
                   h_x=cython.double,h_y=cython.double,h_z=cython.double,
                   phi=cython.double,theta=cython.double,psi=cython.double
                   )
    @cython.boundscheck(False) # turn off boundscheck for this function
    def update(self, acc,mag,gyr):
        """acc, gyr, mag are array(3) mesurement of acceleration, angular rates and magnetic field
        """
        tcurrent = time.time()
        deltat = tcurrent - self.tbefore   # sampling period in seconds (shown as 1 ms)
        gyroMeasError = 3.14159265358979 * (10.0 / 180.0) # gyroscope measurement error in rad/s (shown as 5 deg/s)
        gyroMeasDrift = 3.14159265358979 * (0.2 / 180.0) # gyroscope measurement error in rad/s/s (shown as 0.2f deg/s/s)
        beta = sqrt(3.0 / 4.0) * gyroMeasError # compute beta
        zeta = sqrt(3.0 / 4.0) * gyroMeasDrift # compute zeta


 
        # axulirary variables to avoid reapeated calcualtions
        halfquat = 0.5 * self.quat
        twoquat = 2.0 * self.quat
        twoearth_magnetic_field_x = 2.0 * self.earth_magnetic_field_x
        twoearth_magnetic_field_z = 2.0 * self.earth_magnetic_field_z
        twoearth_magnetic_field_xquat = 2.0 * self.earth_magnetic_field_x * self.quat
        twoearth_magnetic_field_zquat = 2.0 * self.earth_magnetic_field_z * self.quat
        twomag_x = 2.0 * mag[0]
        twomag_y = 2.0 * mag[1]
        twomag_z = 2.0 * mag[2]



       # normalise the accelerometer measurement
        acc /= npnorm(acc)
    
        # normalise the magnetometer measurement
        mag /= npnorm(mag)
    
        # compute the objective function and Jacobian
        f_1 = twoquat[1] * self.quat[3] - twoquat[0] * self.quat[2] - acc[0]
        f_2 = twoquat[0] * self.quat[1] + twoquat[2] * self.quat[3] - acc[1]
        f_3 = 1.0 - twoquat[1] * self.quat[1] - twoquat[2] * self.quat[2] - acc[2]
        f_4 = twoearth_magnetic_field_x * (0.5 - self.quat[2] * self.quat[2] - self.quat[3] * self.quat[3]) + twoearth_magnetic_field_z * (self.quat[1]*self.quat[3] - self.quat[0]*self.quat[2]) - mag[0]
        f_5 = twoearth_magnetic_field_x * (self.quat[1] * self.quat[2] - self.quat[0] * self.quat[3]) + twoearth_magnetic_field_z * (self.quat[0] * self.quat[1] + self.quat[2] * self.quat[3]) - mag[1]
        f_6 = twoearth_magnetic_field_x * (self.quat[0]*self.quat[2] + self.quat[1]*self.quat[3]) + twoearth_magnetic_field_z * (0.5 - self.quat[1] * self.quat[1] - self.quat[2] * self.quat[2]) - mag[2]
        J_11or24 = twoquat[2] # J_11 negated in matrix multiplication
        J_12or23 = 2.0 * self.quat[3]
        J_13or22 = twoquat[0] # J_12 negated in matrix multiplication
        J_14or21 = twoquat[1]
        J_32 = 2.0 * J_14or21 # negated in matrix multiplication
        J_33 = 2.0 * J_11or24 # negated in matrix multiplication
        J_41 = twoearth_magnetic_field_zquat[2] # negated in matrix multiplication
        J_42 = twoearth_magnetic_field_zquat[3]
        J_43 = 2.0 * twoearth_magnetic_field_xquat[2] + twoearth_magnetic_field_zquat[0] # negated in matrix multiplication
        J_44 = 2.0 * twoearth_magnetic_field_xquat[3] - twoearth_magnetic_field_zquat[1] # negated in matrix multiplication
        J_51 = twoearth_magnetic_field_xquat[3] - twoearth_magnetic_field_zquat[1] # negated in matrix multiplication
        J_52 = twoearth_magnetic_field_xquat[2] + twoearth_magnetic_field_zquat[0]
        J_53 = twoearth_magnetic_field_xquat[1] + twoearth_magnetic_field_zquat[3]
        J_54 = twoearth_magnetic_field_xquat[0] - twoearth_magnetic_field_zquat[2] # negated in matrix multiplication
        J_61 = twoearth_magnetic_field_xquat[2]
        J_62 = twoearth_magnetic_field_xquat[3] - 2.0 * twoearth_magnetic_field_zquat[1]
        J_63 = twoearth_magnetic_field_xquat[0] - 2.0 * twoearth_magnetic_field_zquat[2]
        J_64 = twoearth_magnetic_field_xquat[1]
    
        # compute the gradient (matrix multiplication)
        quatHatDot_1 = J_14or21 * f_2 - J_11or24 * f_1 - J_41 * f_4 - J_51 * f_5 + J_61 * f_6
        quatHatDot_2 = J_12or23 * f_1 + J_13or22 * f_2 - J_32 * f_3 + J_42 * f_4 + J_52 * f_5 + J_62 * f_6
        quatHatDot_3 = J_12or23 * f_2 - J_33 * f_3 - J_13or22 * f_1 - J_43 * f_4 + J_53 * f_5 + J_63 * f_6
        quatHatDot_4 = J_14or21 * f_1 + J_11or24 * f_2 - J_44 * f_4 - J_54 * f_5 + J_64 * f_6
        # normalise the gradient to estimate direction of the gyroscope error
        norm = sqrt(quatHatDot_1 * quatHatDot_1 + quatHatDot_2 * quatHatDot_2 + quatHatDot_3 * quatHatDot_3 + quatHatDot_4 * quatHatDot_4)
        quatHatDot_1 = quatHatDot_1 / norm
        quatHatDot_2 = quatHatDot_2 / norm
        quatHatDot_3 = quatHatDot_3 / norm
        quatHatDot_4 = quatHatDot_4 / norm
        # compute angular estimated direction of the gyroscope error
        gyr_err = array([twoquat[0] * quatHatDot_2 - twoquat[1] * quatHatDot_1 - twoquat[2] * quatHatDot_4 + twoquat[3] * quatHatDot_3,
                         twoquat[0] * quatHatDot_3 + twoquat[1] * quatHatDot_4 - twoquat[2] * quatHatDot_1 - twoquat[3] * quatHatDot_2,
                         twoquat[0] * quatHatDot_4 - twoquat[1] * quatHatDot_3 + twoquat[2] * quatHatDot_2 - twoquat[3] * quatHatDot_1])
        # compute and remove the gyroscope baises
        self.gyr_b += gyr_err * deltat * zeta
        gyr -= self.gyr_b
        # compute the quaternion rate measured by gyroscopes
        quatDot_omega_1 = -halfquat[1] * gyr[0] - halfquat[2] * gyr[1] - halfquat[3] * gyr[2]
        quatDot_omega_2 = halfquat[0] * gyr[0] + halfquat[2] * gyr[2] - halfquat[3] * gyr[1]
        quatDot_omega_3 = halfquat[0] * gyr[1] - halfquat[1] * gyr[2] + halfquat[3] * gyr[0]
        quatDot_omega_4 = halfquat[0] * gyr[2] + halfquat[1] * gyr[1] - halfquat[2] * gyr[0]
        # compute then integrate the estimated quaternion rate
        self.quat[0] += (quatDot_omega_1 - (beta * quatHatDot_1)) * deltat
        self.quat[1] += (quatDot_omega_2 - (beta * quatHatDot_2)) * deltat
        self.quat[2] += (quatDot_omega_3 - (beta * quatHatDot_3)) * deltat
        self.quat[3] += (quatDot_omega_4 - (beta * quatHatDot_4)) * deltat
        # normalise quaternion
        self.quat /= npnorm(self.quat)
        # compute flux in the earth frame
        h_x = twomag_x * (0.5 - self.quat[2] * self.quat[2] - self.quat[3] * self.quat[3]) + twomag_y * (self.quat[1]*self.quat[2] - self.quat[0]*self.quat[3]) + twomag_z * (self.quat[1]*self.quat[3] + self.quat[0]*self.quat[2])
        h_y = twomag_x * (self.quat[1]*self.quat[2] + self.quat[0]*self.quat[3]) + twomag_y * (0.5 - self.quat[1] * self.quat[1] - self.quat[3] * self.quat[3]) + twomag_z * (self.quat[2]*self.quat[3] - self.quat[0]*self.quat[1])
        h_z = twomag_x * (self.quat[1]*self.quat[3] - self.quat[0]*self.quat[2]) + twomag_y * (self.quat[2]*self.quat[3] + self.quat[0]*self.quat[1]) + twomag_z * (0.5 - self.quat[1] * self.quat[1] - self.quat[2] * self.quat[2])
        # normalise the flux vector to have only components in the x and z
        self.earth_magnetic_field_x = sqrt((h_x * h_x) + (h_y * h_y))
        self.earth_magnetic_field_z = h_z
        
        self.tbefore = tcurrent

        phi = arctan(2.*(self.quat[0]*self.quat[1]+self.quat[2]*self.quat[3])/(1-2*(self.quat[1]**2+self.quat[2]**2)))
        theta = arcsin(2*(self.quat[0]*self.quat[2]-self.quat[3]*self.quat[1]))
        psi = arctan(2.*(self.quat[0]*self.quat[3]+self.quat[1]*self.quat[2])/(1-2*(self.quat[2]**2+self.quat[3]**2)))
        print phi,theta,psi
        return phi,theta,psi


def logIMU():
    acc, mag, gyr = getCapteurs()
    imu = IMU()
    i=0
    f = open('log_IMU','w')
    while True:
        i+=1
        print i
        ax,ay,az = acc.getAcc()
        hx,hy,hz = mag.getMag()
        gx,gy,gz = gyr.getGyr()
        phi,theta,psi = imu.update([ax,ay,az],[hx,hy,hz],[gx,gy,gz])
        #imu1.update([1,0,0],[0,1,1],[0,0,0])
        f.write('%s %s %s %s %s %s %s %s %s %s %s %s\n'%(ax,ay,az,hx,hy,hz,gx,gy,gz,phi,theta,psi))
    f.close()

def plotIMU():
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    f=open('log_IMU')
    Tlog = list()
    for line in f:
        Tlog.append([float(a) for a in line.split()])
    f.close()
    Tlog = array(Tlog)
    print Tlog[:,2]
    fig2 = plt.figure()
    ax = fig2.add_subplot(311)
    ax.plot(degrees(Tlog[:,9]))
    ax = fig2.add_subplot(312)
    ax.plot(degrees(Tlog[:,10]))
    ax = fig2.add_subplot(313)
    ax.plot(degrees(Tlog[:,11]))

    plt.show()

def quat2matrix(quat):
    q0=quat[0]
    q1=quat[1]
    q2=quat[2]
    q3=quat[3]
    
    mat = array([[q0**2+q1**2-q2**2-q3**2, 2*(q1*q2-q0*q3)         ,2*(q0*q2+q1*q3)],
                 [2*(q1*q2+q0*q3)        , q0**2-q1**2+q2**2-q3**2 ,2*(q2*q3-q0*q1)],
                 [2*(q1*q3-q0*q2)        , 2*(q0*q1+q2*q3)         ,q0**2-q1**2-q2**2+q3**2]])
    return mat

def plotIMU3d():
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.interactive(True)
    
    acc, mag, gyr = getCapteurs()
    imu = IMU()
   
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') 
 
    while True:
        imu.update(acc.getAcc(),mag.getMag(),gyr.getGyr())
        mat = quat2matrix(imu.quat)
        
	for line in ax.lines : line.remove() 
        vx = ax.plot([0,mat[0,0]],[0,mat[0,1]],[0,mat[0,2]])
        vy = ax.plot([0,mat[1,0]],[0,mat[1,1]],[0,mat[1,2]])
        vz = ax.plot([0,mat[2,0]],[0,mat[2,1]],[0,mat[2,2]])
        
        plt.draw()

def plotIMU3d_2():


    acc, mag, gyr = getCapteurs()
    imu = IMU()


    from pyqtgraph.Qt import QtGui
    app = QtGui.QApplication([])

    ## build a QApplication before building other widgets
    #import pyqtgraph as pg
    #pg.mkQApp()
    
    ## make a widget for displaying 3D objects
    import pyqtgraph.opengl as gl
    view = gl.GLViewWidget()
    return
    view.opts['distance'] = 20
    view.show()

    return
    ax = gl.GLAxisItem()
    ax.setSize(5,5,5)
    view.addItem(ax)
    
    b = gl.GLBoxItem()
    view.addItem(b)
    
    ax2 = gl.GLAxisItem()
    ax2.setParentItem(b)

    QtGui.QApplication.instance().exec_()

    return

    while True:
        phi,theta,psi = imu.update(acc.getAcc(),mag.getMag(),gyr.getGyr())
        b.rotate(degrees(phi),1,0,0,local = True)
        b.rotate(degrees(theta),0,0,1,local = True)
        b.rotate(degrees(psi),0,1,0,local = True)
        
    
    
    
if __name__ == "__main__":
    #logIMU()
#    plotIMU()
    plotIMU3d_2()


