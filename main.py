# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 23:22:28 2013

@author: clopeau
"""
from numpy import array
from time import sleep,time

from imu import IMU
from capteurs import getCapteurs
from motors import Motors


imu = IMU()
acc, mag, gyr = getCapteurs()
motors = Motors()

#convegence de l'IMU
for i in range(1000):
    imu.update(acc.getAcc(),mag.getMag(),gyr.getGyr())

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
    
def getPower(option = 'test'):
    if option == 'test':
        return 0.4
        
    else: return 0.

def getAttitudeRegulation(option = 'test'):
    if option == 'test':
        return array([[ 0,0],
                      [ 0,0]])
    else: return array([[ 0,0],
                       [ 0,0]])

def getDistributedPower():
    
    power = getPower()
    equilibration = getAttitudeRegulation()
    
    return equilibration + power
    

if __name__ == "__main__":
    t0 = time()
    while time() - t0 < 1.:
        distributedPower = getDistributedPower()
        motors.setMotorsSpeed(distributedPower)
        
        


