# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:07:31 2013

@author: clopeau
"""
import threading
from time import gmtime, strftime, time
from numpy import   array,  sqrt,zeros,cross,arctan,arcsin,radians,degrees,arctan2
from numpy.linalg import norm as npnorm
from capteurs import getCapteurs
 
def conj(q):
    return array([q[0],-q[1],-q[2],-q[3]])

class IMU():
    """class IMU
    based on http:#www.x-io.co.uk/res/doc/madgwick_internal_report.pdf
    hearth frame:
        z: vertical vers le haut
        x: horizontal aligned with magnetic field
        y; complete
    euler definition and order
    psi: rotation around z
    theta: rotatioin around y
    phi: rotation around x
    """
    
    def __init__(self, log = True):
        self.quat0 = 1.
        self.quat1 = 0.
        self.quat2 = 0.
        self.quat3 = 0.
        self.earth_magnetic_field_x = 1 # orientation of earth magnetic field in ground coordinates
        self.earth_magnetic_field_z = 0 
        self.gyr_b0 = 0
        self.gyr_b1 = 0
        self.gyr_b2 = 0 # estimated bias of gyrometers
        self.tbefore = time()
        self.log = log
        logFile = strftime("log/_imuLog_%Y%b%d_%Hh%Mm%Ss", gmtime())
        self.logFile = open(logFile,'w')
        self.accelerometer, self.magnetometer, self.gyrometer = getCapteurs()
        self.stop = False


        self.gyrc = array([0.,0.,0.])
        self.gyr_ba = array([0.,0.,0.])
        self.tcurrent = 0.

        self.start()

    def start(self):
        self.thread = threading.Thread(target = self.run)
        self.thread.start()

    def run(self):
        while not self.stop:
            self.update()

    def stop(self):
        self.stop = True

    def update(self):
        """updates the quaternions
        """
        self.tcurrent = time()
        self.gyr = self.gyrometer.getGyr()
        self.acc = accelerometer.getAcc()
        self.mag = self.magnetometer.getMag()

        self.deltat = self.tcurrent - self.tbefore   # sampling period in seconds (shown as 1 ms)
        gyroMeasError = 3.14159265358979 * (10.0 / 180.0) # gyroscope measurement error in rad/s (shown as 5 deg/s)
        gyroMeasDrift = 3.14159265358979 * (0.2 / 180.0) # gyroscope measurement error in rad/s/s (shown as 0.2f deg/s/s)
        beta = sqrt(3.0 / 4.0) * gyroMeasError # compute beta
        zeta = sqrt(3.0 / 4.0) * gyroMeasDrift # compute zeta


 
        # axulirary variables to avoid reapeated calcualtions
        quata = array([self.quat0,self.quat1,self.quat2,self.quat3])
        self.gyr_ba = array([self.gyr_b0, self.gyr_b1, self.gyr_b2])
        halfquat = 0.5 * quata
        twoquat = 2.0 * quata
        twoearth_magnetic_field_x = 2.0 * self.earth_magnetic_field_x
        twoearth_magnetic_field_z = 2.0 * self.earth_magnetic_field_z
        twoearth_magnetic_field_xquat = 2.0 * self.earth_magnetic_field_x * quata
        twoearth_magnetic_field_zquat = 2.0 * self.earth_magnetic_field_z * quata
        twomag_x = 2.0 * self.mag[0]
        twomag_y = 2.0 * self.mag[1]
        twomag_z = 2.0 * self.mag[2]



       # normalise the accelerometer measurement
        self.acc /= npnorm(self.acc)
    
        # normalise the magnetometer measurement
        self.mag /= npnorm(self.mag)
    
        # compute the objective function and Jacobian
        f_1 = twoquat[1] * self.quat3 - twoquat[0] * self.quat2 - self.acc[0]
        f_2 = twoquat[0] * self.quat1 + twoquat[2] * self.quat3 - self.acc[1]
        f_3 = 1.0 - twoquat[1] * self.quat1 - twoquat[2] * self.quat2 - self.acc[2]
        f_4 = twoearth_magnetic_field_x * (0.5 - self.quat2 * self.quat2 - self.quat3 * self.quat3) + twoearth_magnetic_field_z * (self.quat1*self.quat3 - self.quat0*self.quat2) - mag[0]
        f_5 = twoearth_magnetic_field_x * (self.quat1 * self.quat2 - self.quat0 * self.quat3) + twoearth_magnetic_field_z * (self.quat0 * self.quat1 + self.quat2 * self.quat3) - mag[1]
        f_6 = twoearth_magnetic_field_x * (self.quat0*self.quat2 + self.quat1*self.quat3) + twoearth_magnetic_field_z * (0.5 - self.quat1 * self.quat1 - self.quat2 * self.quat2) - mag[2]
        J_11or24 = twoquat[2] # J_11 negated in matrix multiplication
        J_12or23 = 2.0 * self.quat3
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
        
        self.gyr_ba += gyr_err * self.deltat * zeta
        self.gyr_b0, self.gyr_b1, self.gyr_b2 = self.gyr_ba[0],self.gyr_ba[1],self.gyr_ba[2]
        self.gyrc = self.gyr-self.gyr_ba
        # compute the quaternion rate measured by gyroscopes
        quatDot_omega_1 = -halfquat[1] * self.gyrc[0] - halfquat[2] * self.gyrc[1] - halfquat[3] * self.gyrc[2]
        quatDot_omega_2 = halfquat[0] * self.gyrc[0] + halfquat[2] * self.gyrc[2] - halfquat[3] * self.gyrc[1]
        quatDot_omega_3 = halfquat[0] * self.gyrc[1] - halfquat[1] * self.gyrc[2] + halfquat[3] * self.gyrc[0]
        quatDot_omega_4 = halfquat[0] * self.gyrc[2] + halfquat[1] * self.gyrc[1] - halfquat[2] * self.gyrc[0]
        # compute then integrate the estimated quaternion rate
        self.quat0 += (quatDot_omega_1 - (beta * quatHatDot_1)) * self.deltat
        self.quat1 += (quatDot_omega_2 - (beta * quatHatDot_2)) * self.deltat
        self.quat2 += (quatDot_omega_3 - (beta * quatHatDot_3)) * self.deltat
        self.quat3 += (quatDot_omega_4 - (beta * quatHatDot_4)) * self.deltat
        # normalise quaternion 
        norm = sqrt(self.quat0*self.quat0+self.quat1*self.quat1+self.quat2*self.quat2+self.quat3*self.quat3)
        self.quat0 = self.quat0/norm 
        self.quat1 = self.quat1/norm 
        self.quat2 = self.quat2/norm 
        self.quat3 = self.quat3/norm 
	
        # compute flux in the earth frame
        h_x = twomag_x * (0.5 - self.quat2 * self.quat2 - self.quat3 * self.quat3) + twomag_y * (self.quat1*self.quat2 - self.quat0*self.quat3) + twomag_z * (self.quat1*self.quat3 + self.quat0*self.quat2)
        h_y = twomag_x * (self.quat1*self.quat2 + self.quat0*self.quat3) + twomag_y * (0.5 - self.quat1 * self.quat1 - self.quat3 * self.quat3) + twomag_z * (self.quat2*self.quat3 - self.quat0*self.quat1)
        h_z = twomag_x * (self.quat1*self.quat3 - self.quat0*self.quat2) + twomag_y * (self.quat2*self.quat3 + self.quat0*self.quat1) + twomag_z * (0.5 - self.quat1 * self.quat1 - self.quat2 * self.quat2)
        # normalise the flux vector to have only components in the x and z
        self.earth_magnetic_field_x = sqrt((h_x * h_x) + (h_y * h_y))
        self.earth_magnetic_field_z = h_z
        
        self.tbefore = self.tcurrent

        

    def getEuler(self):
        quat0=self.quat0
        quat1=-self.quat1
        quat2=-self.quat2
        quat3=-self.quat3
#        phi = arctan(2.*(self.quat0*self.quat1+self.quat2*self.quat3)/(1-2*(self.quat1**2+self.quat2**2)))
#        theta = arcsin(2*(self.quat0*self.quat2-self.quat3*self.quat1))
#        psi = arctan(2.*(self.quat0*self.quat3+self.quat1*self.quat2)/(1-2*(self.quat2**2+self.quat3**2)))
        psi = arctan2(2.*(quat1*quat2-quat0*quat3), 2.*(quat0**2+quat1**2)-1.)
        theta = -arcsin(2.*(quat1*quat3+quat0*quat2))
        phi = arctan2(2.*(quat2*quat3-quat0*quat1) , 2.*(quat0**2+quat3**2)-1.)
        #print phi,theta,psi
        return psi,theta,phi
    
    def log    
        psi,theta,phi = self.getEuler()          
        self.logFile.write(('%s '*26+'\n')%(self.tcurrent,self.deltat,
                                              self.acc[0],self.acc[1],self.acc[2],
                                              self.mag[0],self.mag[1],self.mag[2],
                                              self.gyr[0],self.gyr[1],self.gyr[2],
                                              self.gyrc[0],self.gyrc[1],self.gyrc[2],
                                              self.gyr_ba[0],self.gyr_ba[1],self.gyr_ba[2], 
                                              self.earth_magnetic_field_x,self.earth_magnetic_field_z ,   
                                              self.quat0,self.quat1,self.quat2,self.quat3,
                                              psi,theta,phi))
        
        


def logIMU(print_ = True, log = False):
    imu = IMU(log = log)
    i=0
    while True:
        i+=1
        #print i
        psi,theta,phi = imu.getEuler()
        if print_ : print '%10.7f %10.7f %10.7f'%(degrees(psi),degrees(theta),degrees(phi))
        time.sleep(0.1)

def timeIMU(niter = 1000):
    
    imu = IMU()
    i=0
    t0 = time()
    while i < niter:
        i+=1
        print i
        psi,theta,phi = imu.getEuler()
    print time() - t0

def plotIMU(fileName = '_log_IMU'):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    f=open(fileName)
    Tlog = list()
    for line in f:
        Tlog.append([float(a) for a in line.split()])
    f.close()
    Tlog = array(Tlog)
    print Tlog[:,2]
    fig2 = plt.figure()
    ax = fig2.add_subplot(311)
    ax.plot(Tlog[:,0],degrees(Tlog[:,-3]))
    ax = fig2.add_subplot(312)
    ax.plot(Tlog[:,0],degrees(Tlog[:,-2]))
    ax = fig2.add_subplot(313)
    ax.plot(Tlog[:,0],degrees(Tlog[:,-1]))

    plt.show()

def quat2matrix(quat):
    q0=quat[0]
    q1=quat[1]
    q2=quat[2]
    q3=quat[3]
    
    mat = array([[2*q0**2 - 1 + 2*q1**2, 2*(q1*q2 + q0*q3)     ,2*(q1*q3 - q0*q2)],
                 [2*(q1*q2 - q0*q3)    , 2*q0**2 - 1 + 2*q2**2 ,2*(q2*q3 + q0*q1)],
                 [2*(q1*q3 + q0*q2)    , 2*(q2*q3 - q0*q1)     ,2*q0**2 - 1 + 2*q3**2]])
                 
    return mat

def plotIMU3d():
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.interactive(True)
    
    
    imu = IMU()
   
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') 
 
    while True:
        imu.getEuler()
        mat = quat2matrix(imu.quat)
        
        for line in ax.lines : line.remove() 
        vx = ax.plot([0,mat[0,0]],[0,mat[0,1]],[0,mat[0,2]])
        vy = ax.plot([0,mat[1,0]],[0,mat[1,1]],[0,mat[1,2]])
        vz = ax.plot([0,mat[2,0]],[0,mat[2,1]],[0,mat[2,2]])
        
        plt.draw()
        time.sleep(0.5)

def plotIMU3d_2():


    
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
        psi,theta,phi= imu.update()
        b.rotate(degrees(phi),1,0,0,local = True)
        b.rotate(degrees(theta),0,0,1,local = True)
        b.rotate(degrees(psi),0,1,0,local = True)

        time.sleep(0.5)
        
    
    
    
if __name__ == "__main__":
    #logIMU()
#    plotIMU()
    plotIMU3d_2()


