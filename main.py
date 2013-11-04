# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 23:22:28 2013

@author: clopeau
"""
from numpy import array

from imu import imu0
from time import sleep
from capteurs import getacc, getmag, getgyr


imu = imu0()

#convegence de l'IMU
for i in range(1000):
    ax,ay,az = getacc()
    mx,my,mz = getmag()
    wx,wy,wz = getgyr()
    imu.update(ax,ay,az,mx,my,mz,wx,wy,wz)

longi0 =   array([[ 1, 1],
                  [-1,-1]])
lateral0 = array([[ 1,-1],
                  [ 1,-1]])
pivot0 =   array([[ 1,-1],
                  [-1, 1]])
                  
def respectAngles(alphac=0.,betac=0.,phic=0.):
    da = alpha - alphac
    db = beta - betac
    dp = phi - phic
    
    dmotors = da*longi0 + db*lateral0 + dp*pivot0
    
def setTotalPower(altc = 1.):
    dalt = alt - altc    
    

