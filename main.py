# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 23:22:28 2013

@author: clopeau
"""
from numpy import array
from time import sleep,time

from imu import IMU
from motors import Motors
from quad import Quad

imu = IMU()
motors = Motors()
quad = Quad(imu,motors)

#convegence de l'IMU
time.sleep(30)



def testPdecolage():
    """ trouver la puissance de decolage CTRL-C pour couper"""
    p = array([[0.,0.],
               [0.,0.]])
    while True:
        p = p + 0.005
        print p
        motors.setMotorsSpeed(p)
        time.sleep(1)

def teststabilite(p):
    """ test stabilite CTRL-C pour couper"""
    p = p * array([[1.,1.],
                   [1.,1.]])
    while True:
        peq = getDistributedPower(rol = 0.,pitch = 0.,yaw = 0.,option = 'maintainConsign') 
        motors.setMotorsSpeed(peq)
    

if __name__ == "__main__":
    
        
        


