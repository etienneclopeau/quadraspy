# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:07:31 2013

@author: clopeau
"""
import time
from numpy import   array,  sqrt,zeros,cross,arctan,arcsin,radians,degrees
from numpy.linalg import norm as npnorm
from capteurs import getCapteurs
 
def conj(q):
    return array([q[0],-q[1],-q[2],-q[3]])

class imu():
    """class IMU
    based on http:#www.x-io.co.uk/res/doc/madgwick_internal_report.pdf
    """
    
    def __init__(self):
        self.quat = array([1.,0.,0.,0.]) # quaternion
        self.earth_magnetic_field_x = 1 # orientation of earth magnetic field in ground coordinates
        self.earth_magnetic_field_z = 0 
        self.gyr_b = zeros(3) # estimated bias of gyrometers
        self.tbefore = time.time()



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


if __name__ == "__main__":
#    acc, mag, gyr = getCapteurs()
#    imu1 = imu()
#    i=0
#    f = open('log_IMU','w')
#    while True:
#        i+=1
#        print i
#        ax,ay,az = acc.getAcc()
#        hx,hy,hz = mag.getMag()
#        gx,gy,gz = gyr.getGyr()
#        phi,theta,psi = imu1.update([ax,ay,az],[hx,hy,hz],[gx,gy,gz])
#        #imu1.update([1,0,0],[0,1,1],[0,0,0])
#        f.write('%s %s %s %s %s %s %s %s %s %s %s %s\n'%(ax,ay,az,hx,hy,hz,gx,gy,gz,phi,theta,psi))
#    f.close()

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


