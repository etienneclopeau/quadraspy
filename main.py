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
from pid import PID


imu = IMU()
acc, mag, gyr = getCapteurs()
motors = Motors()
pid_rol = PID(1,0,1)
pid_pitch = PID(1,0,1)
pid_yaw = PID(1,0,1)


#convegence de l'IMU
for i in range(1000):
    imu.update(acc.getAcc(),mag.getMag(),gyr.getGyr())

longi0 =   array([[ 1, 1],
                  [-1,-1]])
lateral0 = array([[ 1,-1],
                  [ 1,-1]])
pivot0 =   array([[ 1,-1],
                  [-1, 1]])
                  
#def respectAngles(rol_c=0.,pitch_c=0.,yaw_c=0.):
#    regul_rol = pid_rol(rol, rol_c)
#    if regul_rol > 0.5 : da = 0.5
#    if regul_rol < 0. : da = 0.
#    regul_pitch = pid_pitch(pitch, pitch_c)
#    if regul_pitch > 0.5 : da = 0.5
#    if regul_pitch < 0. : da = 0.
#    regul_yaw = pid_yaw(yaw, yaw_c)
#    if regul_yaw > 0.5 : da = 0.5
#    if regul_yaw < 0. : da = 0.
#    
#    dmotors = (regul_rol*longi0 + 1) * (regul_pitch*lateral0 + 1) * (regul_yaw*pivot0 + 1)
#    return dmotors
    
def setTotalPower(altc = 1.):
    dalt = alt - altc    
    
def getPower(option = 'test'):
    if option == 'test':
        return 0.4
        
    else: return 0.

def getAttitudeRegulation(option = 'test'):
    if option == 'test':
        return array([[ 1,1],
                      [ 1,1]])
    elif option == 'stayHorizontal':
        regul_rol = pid_rol(rol, rol_c)
        if regul_rol > 0.5 : da = 0.5
        if regul_rol < 0. : da = 0.
        regul_pitch = pid_pitch(pitch, pitch_c)
        if regul_pitch > 0.5 : da = 0.5
        if regul_pitch < 0. : da = 0.
        regul_yaw = pid_yaw(yaw, yaw_c)
        if regul_yaw > 0.5 : da = 0.5
        if regul_yaw < 0. : da = 0.
        
        dmotors = (regul_rol*longi0 + 1) * (regul_pitch*lateral0 + 1) * (regul_yaw*pivot0 + 1)

    else: return array([[ 1,1],
                        [ 1,1]])

def getDistributedPower():
    
    power = getPower()
    equilibration = getAttitudeRegulation()
    
    return equilibration * power
    

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
        equilibration = getAttitudeRegulation(option = 'stayHorizontal')
        peq = p*equilibration 
        motors.setMotorsSpeed(peq)
    

if __name__ == "__main__":
    t0 = time()
    while time() - t0 < 1.:
        distributedPower = getDistributedPower()
        motors.setMotorsSpeed(distributedPower)
        
        


